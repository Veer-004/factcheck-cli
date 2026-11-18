"""``factcheck-audio``: fact-check the claims spoken in an English audio file.

Exit codes (same as ``factcheck`` / ``factcheck-video``):
  0  every checked claim resolved (or --extract-only produced a transcript)
  1  partial result, or nothing to check (no speech, not English, everything unreliable)
  2  invalid input, missing dependency/model, or configuration problem
  3  run failure (including a transcription failure)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional

from .models import AudioLimits

EXIT_OK, EXIT_PARTIAL, EXIT_INVALID, EXIT_FAILURE = 0, 1, 2, 3


def _parser() -> argparse.ArgumentParser:
    from .transcribe import (DEFAULT_COMPUTE_TYPE, DEFAULT_MODEL, DEFAULT_SPEECH_THRESHOLD, DEFAULT_THREADS,
                             MODEL_SIZES_MB)

    d = AudioLimits()
    p = argparse.ArgumentParser(
        prog="factcheck-audio",
        description="Fact-check the claims spoken in an English audio file (mp3, wav, m4a, flac, ogg, opus, ...). "
                    "Speech is transcribed locally with faster-whisper; the transcript then goes through the same "
                    "fact-checking pipeline as `factcheck`. Only English speech is supported.",
    )
    p.add_argument("audio", nargs="?", type=Path, help="audio file")
    p.add_argument("--json-out", type=Path, help="also write the full report as JSON to this file")
    p.add_argument("-q", "--quiet", action="store_true", help="print only the overall verdict line")
    p.add_argument("-v", "--verbose", action="store_true", help="show every segment, confidences and diagnostics")
    p.add_argument("--extract-only", "--transcribe-only", dest="extract_only", action="store_true",
                   help="only transcribe; no fact-check (no API keys needed)")
    p.add_argument("--check", action="store_true", help="show dependency and speech-model status, then exit")

    s = p.add_argument_group("speech recognition (faster-whisper, CPU)")
    s.add_argument("--whisper-model", default=DEFAULT_MODEL,
                   help=f"model size or local model directory (default {DEFAULT_MODEL}; sizes: "
                        + ", ".join(f"{k} ~{v} MB" for k, v in MODEL_SIZES_MB.items())
                        + "; English-only variants such as base.en skip language identification)")
    s.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE,
                   choices=["int8", "int8_float32", "int16", "float32"], help="CPU precision (default int8)")
    s.add_argument("--assume-english", action="store_true",
                   help="skip language identification and transcribe as English (for accented English that is "
                        "misidentified)")
    s.add_argument("--beam-size", type=int, default=5, help="decoding beam size (default 5; 1 is faster)")
    s.add_argument("--no-vad", action="store_true", help="disable voice-activity filtering of silence")
    s.add_argument("--no-word-timestamps", action="store_true", help="segment-level timestamps only (faster)")
    s.add_argument("--speech-threshold", type=float, default=DEFAULT_SPEECH_THRESHOLD,
                   help=f"segment confidence below which speech is not checked (default {DEFAULT_SPEECH_THRESHOLD})")
    s.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help=f"CPU threads for transcription (default {DEFAULT_THREADS} = library default, the fastest "
                        "setting measured on a 12-core laptop CPU)")
    s.add_argument("--allow-download", action="store_true", help="allow downloading the speech model on first use")

    lim = p.add_argument_group("limits")
    lim.add_argument("--max-duration", type=float, default=d.max_duration_s,
                     help=f"maximum audio length in seconds (default {d.max_duration_s:.0f})")
    lim.add_argument("--max-file-mb", type=int, default=d.max_file_mb, help=f"default {d.max_file_mb}")
    lim.add_argument("--extract-timeout", type=float, default=d.extract_timeout_s,
                     help=f"time limit for decoding + transcription in seconds (default {d.extract_timeout_s:.0f})")
    lim.add_argument("--max-text-chars", type=int, default=6000,
                     help="characters of transcript sent to the fact-check (default 6000, max 20000)")

    f = p.add_argument_group("fact-check (same options as factcheck)")
    f.add_argument("--model", default=None,
                   help="Gemini model for the fact-check (default from FACTCHECK_MODEL), as in the other commands; "
                        "the speech model is --whisper-model")
    f.add_argument("--max-claims", type=int, default=None, help="checkable claims per run (default 6)")
    f.add_argument("--search-budget", type=int, default=None, help="Tavily credits per run (default 40)")
    f.add_argument("--timeout", type=float, default=None, help="fact-check time limit in seconds (default 180)")
    return p


def check_status(model: str) -> str:
    from .transcribe import model_status

    lines = []
    for mod, label in (("av", "PyAV (FFmpeg)"), ("faster_whisper", "faster-whisper"), ("ctranslate2", "CTranslate2")):
        lines.append(f"{label:16} {'installed' if importlib.util.find_spec(mod) else 'MISSING'}")
    if importlib.util.find_spec("av"):
        import av

        lines.append(f"{'FFmpeg libs':16} libavcodec {'.'.join(map(str, av.library_versions.get('libavcodec', ())))}")
    lines.append(f"{'speech model':16} {model}: {model_status(model)}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        print(check_status(args.whisper_model))
        return EXIT_OK
    if args.audio is None:
        print("error: an audio file is required (or use --check)", file=sys.stderr)
        return EXIT_INVALID
    if not 0.0 <= args.speech_threshold <= 1.0:
        print("error: --speech-threshold must be between 0 and 1", file=sys.stderr)
        return EXIT_INVALID
    if not 1 <= args.beam_size <= 10 or args.threads < 0:
        print("error: --beam-size must be 1-10 and --threads >= 0", file=sys.stderr)
        return EXIT_INVALID
    if args.json_out is not None and args.json_out.is_dir():
        print(f"error: --json-out points to a directory: {args.json_out}", file=sys.stderr)
        return EXIT_INVALID
    return run(args)


def run(args: argparse.Namespace) -> int:
    # Heavy imports only now, so `--help` and `--check` stay fast.
    from factcheck import envfile
    from factcheck.cli import exit_code_for as text_exit_code
    from factcheck.config import Config, ConfigError
    from factcheck.graph import RunFailure
    from factcheck.security import redact_secrets
    from factcheck.validation import InputError

    from .audio_io import AudioInputError, DependencyMissing
    from .models import TranscriptionStatus
    from .pipeline import verify_audio
    from .report import render_audio_json, render_audio_report
    from .transcribe import EnglishTranscriber, ModelUnavailable

    try:
        limits = AudioLimits(max_file_mb=args.max_file_mb, max_duration_s=args.max_duration,
                             extract_timeout_s=args.extract_timeout)
    except Exception as exc:  # pydantic.ValidationError
        err = exc.errors()[0] if hasattr(exc, "errors") else {"loc": ("?",), "msg": str(exc)}
        print(f"error: invalid limit {err['loc'][0]}: {err['msg']}", file=sys.stderr)
        return EXIT_INVALID
    for warning in envfile.load_env_file():
        print(f"warning: {warning}", file=sys.stderr)

    def make_transcriber():
        return EnglishTranscriber(args.whisper_model, compute_type=args.compute_type,
                                  allow_download=args.allow_download, threads=args.threads,
                                  beam_size=args.beam_size, vad=not args.no_vad,
                                  word_timestamps=not args.no_word_timestamps, threshold=args.speech_threshold,
                                  assume_english=args.assume_english)

    try:
        config = Config.from_env(model=args.model, max_checkable_claims=args.max_claims,
                                 search_credit_budget=args.search_budget, run_timeout_s=args.timeout,
                                 max_input_chars=args.max_text_chars)
        ar = verify_audio(args.audio, config, make_transcriber=make_transcriber, limits=limits,
                          speech_threshold=args.speech_threshold, max_text_chars=args.max_text_chars,
                          fact_check=not args.extract_only)
    except (AudioInputError, InputError) as exc:
        print(f"error: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_INVALID
    except (ConfigError, DependencyMissing, ModelUnavailable) as exc:
        print(f"error: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_INVALID
    except RunFailure as exc:
        print(f"run failed: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001
        print(f"run failed: unexpected {type(exc).__name__}: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_FAILURE

    if args.quiet:
        if ar.fact_check is None:
            print(f"NOT_FACT_CHECKED: {ar.not_checked_reason}")
        else:
            print(f"{ar.fact_check.summary_label.value}: {ar.fact_check.summary_text}")
    else:
        print(render_audio_report(ar, verbose=args.verbose))
    if args.json_out is not None:
        try:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(render_audio_json(ar), encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write JSON report: {exc}", file=sys.stderr)
            return EXIT_FAILURE
    if ar.fact_check is None:
        if ar.transcription.status == TranscriptionStatus.FAILED:
            return EXIT_FAILURE
        return EXIT_OK if args.extract_only and ar.transcription.segments else EXIT_PARTIAL
    return text_exit_code(ar.fact_check)


if __name__ == "__main__":
    sys.exit(main())
