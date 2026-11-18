"""``factcheck-video``: fact-check the speech and on-screen text of a video file.

Exit codes (same as ``factcheck`` / ``factcheck-ocr --fact-check``):
  0  every checked claim resolved
  1  partial result, nothing to check, or extraction only with nothing extracted
  2  invalid input, missing dependency/model, or configuration problem
  3  run failure
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional

from .models import Limits

EXIT_OK, EXIT_PARTIAL, EXIT_INVALID, EXIT_FAILURE = 0, 1, 2, 3


def _parser() -> argparse.ArgumentParser:
    from .transcribe import DEFAULT_COMPUTE_TYPE, DEFAULT_MODEL, DEFAULT_SPEECH_THRESHOLD, MODEL_SIZES_MB

    d = Limits()
    p = argparse.ArgumentParser(
        prog="factcheck-video",
        description="Fact-check the claims spoken in a video and shown as on-screen text. Speech is transcribed "
                    "locally (faster-whisper) and frames are read with the project's OCR; the extracted text then "
                    "goes through the same fact-checking pipeline as `factcheck`. Visual content other than text is "
                    "not interpreted.",
    )
    p.add_argument("video", nargs="?", type=Path, help="video file (mp4, mov, mkv, webm, avi, ...)")
    p.add_argument("--json-out", type=Path, help="also write the full report as JSON to this file")
    p.add_argument("-q", "--quiet", action="store_true", help="print only the overall verdict line")
    p.add_argument("-v", "--verbose", action="store_true", help="show every segment/text block and confidences")
    p.add_argument("--extract-only", action="store_true",
                   help="only extract speech and on-screen text; no fact-check (no API keys needed)")
    p.add_argument("--check", action="store_true", help="show dependency and model status, then exit")

    s = p.add_argument_group("speech (faster-whisper)")
    s.add_argument("--no-audio", action="store_true", help="skip speech transcription")
    s.add_argument("--whisper-model", default=DEFAULT_MODEL,
                   help=f"model size or local model directory (default {DEFAULT_MODEL}; multilingual sizes: "
                        + ", ".join(f"{k} ~{v} MB" for k, v in MODEL_SIZES_MB.items()) + ")")
    s.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE,
                   choices=["int8", "int8_float32", "int16", "float32"], help="CPU precision (default int8)")
    s.add_argument("--language", default=None, help="spoken language code, e.g. en, hi, fr (default: detect)")
    s.add_argument("--beam-size", type=int, default=5, help="decoding beam size (default 5; 1 is faster)")
    s.add_argument("--no-vad", action="store_true", help="disable voice-activity filtering of silence")
    s.add_argument("--no-word-timestamps", action="store_true", help="segment-level timestamps only (faster)")
    s.add_argument("--speech-threshold", type=float, default=DEFAULT_SPEECH_THRESHOLD,
                   help=f"segment confidence below which speech is not checked (default {DEFAULT_SPEECH_THRESHOLD})")
    s.add_argument("--threads", type=int, default=0, help="CPU threads for transcription (default: library default)")
    s.add_argument("--allow-download", action="store_true",
                   help="allow downloading the speech model (and medium OCR models) on first use")

    o = p.add_argument_group("on-screen text (OCR)")
    o.add_argument("--no-ocr", action="store_true", help="skip on-screen text")
    o.add_argument("--frame-interval", type=float, default=d.frame_interval_s,
                   help=f"seconds between sampled frames (default {d.frame_interval_s})")
    o.add_argument("--max-frames", type=int, default=d.max_frames, help=f"frame cap (default {d.max_frames})")
    o.add_argument("--ocr-engine", default="rapid", choices=["rapid", "paddle", "paddle-structure"])
    o.add_argument("--ocr-model-size", default="small", choices=["small", "medium"])
    o.add_argument("--ocr-threshold", type=float, default=0.80,
                   help="OCR confidence below which on-screen text is not checked (default 0.80)")
    o.add_argument("--min-score", type=float, default=0.5, help="discard OCR fragments below this (default 0.5)")
    o.add_argument("--keep-ocr-noise", action="store_true",
                   help="keep fragments the English noise filter would remove (stray letters, non-Latin symbols)")

    lim = p.add_argument_group("limits")
    lim.add_argument("--max-duration", type=float, default=d.max_duration_s,
                     help=f"maximum video length in seconds (default {d.max_duration_s:.0f})")
    lim.add_argument("--max-file-mb", type=int, default=d.max_file_mb, help=f"default {d.max_file_mb}")
    lim.add_argument("--extract-timeout", type=float, default=d.extract_timeout_s,
                     help=f"time limit for extraction in seconds (default {d.extract_timeout_s:.0f})")
    lim.add_argument("--max-text-chars", type=int, default=6000,
                     help="characters of extracted text sent to the fact-check (default 6000, max 20000)")

    f = p.add_argument_group("fact-check (same options as factcheck)")
    f.add_argument("--model", default=None, help="Gemini model (default from FACTCHECK_MODEL)")
    f.add_argument("--max-claims", type=int, default=None, help="checkable claims per run (default 6)")
    f.add_argument("--search-budget", type=int, default=None, help="Tavily credits per run (default 40)")
    f.add_argument("--timeout", type=float, default=None, help="fact-check time limit in seconds (default 180)")
    return p


def check_status(model: str) -> str:
    from .transcribe import model_status

    lines = []
    for mod, label in (("av", "PyAV (FFmpeg)"), ("faster_whisper", "faster-whisper"), ("ctranslate2", "CTranslate2"),
                       ("rapidocr", "RapidOCR"), ("onnxruntime", "onnxruntime")):
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
    if args.video is None:
        print("error: a video file is required (or use --check)", file=sys.stderr)
        return EXIT_INVALID
    if args.no_audio and args.no_ocr:
        print("error: --no-audio and --no-ocr together leave nothing to extract", file=sys.stderr)
        return EXIT_INVALID
    for name, value in (("--speech-threshold", args.speech_threshold), ("--ocr-threshold", args.ocr_threshold),
                        ("--min-score", args.min_score)):
        if not 0.0 <= value <= 1.0:
            print(f"error: {name} must be between 0 and 1", file=sys.stderr)
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
    from factcheck_ocr.engines import EngineFailure, EngineUnavailable, create_engine

    from .media import DependencyMissing, VideoInputError
    from .pipeline import verify_video
    from .report import render_video_json, render_video_report
    from .transcribe import ModelUnavailable, WhisperTranscriber

    try:
        limits = Limits(max_file_mb=args.max_file_mb, max_duration_s=args.max_duration,
                        frame_interval_s=args.frame_interval, max_frames=args.max_frames,
                        extract_timeout_s=args.extract_timeout)
    except Exception as exc:  # pydantic.ValidationError
        err = exc.errors()[0] if hasattr(exc, "errors") else {"loc": ("?",), "msg": str(exc)}
        print(f"error: invalid limit {err['loc'][0]}: {err['msg']}", file=sys.stderr)
        return EXIT_INVALID
    for warning in envfile.load_env_file():
        print(f"warning: {warning}", file=sys.stderr)

    def make_transcriber():
        return WhisperTranscriber(args.whisper_model, compute_type=args.compute_type,
                                  allow_download=args.allow_download, threads=args.threads, language=args.language,
                                  beam_size=args.beam_size, vad=not args.no_vad,
                                  word_timestamps=not args.no_word_timestamps, threshold=args.speech_threshold)

    def make_engine():
        return create_engine(args.ocr_engine, model_size=args.ocr_model_size, allow_download=args.allow_download)

    try:
        config = Config.from_env(model=args.model, max_checkable_claims=args.max_claims,
                                 search_credit_budget=args.search_budget, run_timeout_s=args.timeout,
                                 max_input_chars=args.max_text_chars)
        vr = verify_video(args.video, config, limits=limits,
                          make_transcriber=None if args.no_audio else make_transcriber,
                          make_engine=None if args.no_ocr else make_engine,
                          speech_threshold=args.speech_threshold, ocr_threshold=args.ocr_threshold,
                          min_score=args.min_score, max_text_chars=args.max_text_chars,
                          fact_check=not args.extract_only, noise_filter=not args.keep_ocr_noise)
    except (VideoInputError, InputError) as exc:
        print(f"error: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_INVALID
    except (ConfigError, DependencyMissing, ModelUnavailable, EngineUnavailable) as exc:
        print(f"error: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_INVALID
    except (RunFailure, EngineFailure) as exc:
        print(f"run failed: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001
        print(f"run failed: unexpected {type(exc).__name__}: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_FAILURE

    if args.quiet:
        if vr.fact_check is None:
            print(f"NOT_FACT_CHECKED: {vr.not_checked_reason}")
        else:
            print(f"{vr.fact_check.summary_label.value}: {vr.fact_check.summary_text}")
    else:
        print(render_video_report(vr, verbose=args.verbose))
    if args.json_out is not None:
        try:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(render_video_json(vr), encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write JSON report: {exc}", file=sys.stderr)
            return EXIT_FAILURE
    if vr.fact_check is None:
        return EXIT_OK if args.extract_only and vr.extraction.units else EXIT_PARTIAL
    return text_exit_code(vr.fact_check)


if __name__ == "__main__":
    sys.exit(main())
