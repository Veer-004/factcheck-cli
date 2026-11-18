"""Audio (factcheck-audio) evaluation: real speech recognition, no API calls by default.

    python eval/run_audio_eval.py                       # synthetic cases + real recordings, base model
    python eval/run_audio_eval.py --models base,small   # compare cached Whisper sizes
    python eval/run_audio_eval.py --real a.mp3 b.m4a    # choose the real recordings
    python eval/run_audio_eval.py --live                # also fact-check the first real English recording
                                                        # (spends Gemini/Tavily quota)

Synthetic cases: English sentences are synthesised offline with the Windows SAPI
voices (reference text and onset times are exact), placed in silence or noise
and encoded as mp3/wav/flac/m4a/ogg/opus. Measured: word error rate, number
recall, segment-onset error, language identification, and output on audio with
no speech (silence, noise, tones), where any segment is a hallucination.

Real recordings: the audio of real English videos (by default the user's sample
videos in Downloads, converted to mp3 in a temporary folder; nothing is copied
into the repository). There is no human reference transcript, so the reported
"agreement WER" is between two Whisper sizes. It is a consistency proxy, not an
accuracy figure. A Hindi recording checks that non-English audio is rejected.

Results are written to eval/results/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))  # audio_helpers.write_audio (encoder for every format)

from make_video_samples import tts  # noqa: E402
from run_video_eval import peak_rss_mb, wer, word_recall  # noqa: E402

OUT = Path(__file__).parent / "results"
RATE = 16000
DOWNLOADS = Path.home() / "Downloads"
DEFAULT_REAL = [("elon_bio", DOWNLOADS / "video.mp4", "en"), ("bolt_speed", DOWNLOADS / "4850012.mp4", "en"),
                ("agni_hindi", DOWNLOADS / "How Agni V Missile Works  NitishRajput.mp4", "hi")]


@dataclass
class Say:
    start: float
    text: str
    rate: int = 0


@dataclass
class Case:
    name: str
    ext: str
    duration: float
    says: list[Say] = field(default_factory=list)
    background: str = "silence"  # silence | noise | tones
    snr_db: float | None = None  # noise level relative to the speech (background == noise)


CASES = [
    Case("clean_facts", ".mp3", 10.0, [Say(0.5, "The Eiffel Tower was completed in 1889."),
                                       Say(5.0, "Mount Everest is 8849 metres tall.")]),
    Case("numbers_dates", ".wav", 13.0, [Say(0.5, "The United Nations was founded in 1945 with 51 member states."),
                                         Say(7.0, "Water boils at 100 degrees Celsius at sea level.")]),
    Case("negation", ".flac", 7.0, [Say(0.5, "The Great Wall of China is not visible from the Moon with the naked eye.")]),
    Case("long_pause", ".m4a", 26.0, [Say(0.5, "Light travels faster than sound."),
                                      Say(20.0, "The Pacific is the largest ocean on Earth.")]),
    Case("fast_speech", ".ogg", 7.0, [Say(0.5, "In 2002, eBay bought PayPal for 1.5 billion dollars.", rate=4)]),
    Case("noisy_10db", ".opus", 10.0, [Say(0.5, "The Eiffel Tower was completed in 1889."),
                                       Say(5.0, "Mount Everest is 8849 metres tall.")], "noise", 10.0),
    Case("silence", ".wav", 8.0),
    Case("noise_only", ".mp3", 8.0, background="noise", snr_db=None),
    Case("tones_only", ".flac", 8.0, background="tones"),
]


def onset(clip: np.ndarray) -> float:
    """Seconds of leading silence in a synthesised clip."""
    loud = np.flatnonzero(np.abs(clip) > 0.02)
    return float(loud[0]) / RATE if len(loud) else 0.0


def build(case: Case) -> tuple[np.ndarray, list[tuple[float, str]]]:
    n = int(case.duration * RATE)
    speech = np.zeros(n, dtype=np.float32)
    truth = []
    for s in case.says:
        clip = tts(s.text, rate=s.rate) * 0.8
        a = int(s.start * RATE)
        speech[a:a + len(clip)] += clip[: max(0, n - a)]
        truth.append((s.start + onset(clip), s.text))
    rng = np.random.default_rng(7)
    if case.background == "noise":
        if case.snr_db is not None and speech.any():
            p = float(np.mean(speech[np.abs(speech) > 0.02] ** 2))
            sigma = (p / 10 ** (case.snr_db / 10)) ** 0.5
        else:
            sigma = 0.1
        speech += rng.normal(0, sigma, n).astype(np.float32)
    elif case.background == "tones":
        t = np.arange(n) / RATE
        for k, f in enumerate((220.0, 277.2, 329.6, 440.0)):
            speech += (0.08 * np.sin(2 * np.pi * f * t) * (np.sin(2 * np.pi * (0.25 + k * 0.1) * t) > 0)).astype(np.float32)
    return np.clip(speech, -1, 1), truth


_SPELLING = {"metres": "meters", "metre": "meter", "kilometres": "kilometers", "colour": "color"}


def canonical(text: str) -> str:
    """Formatting-only normalisation for a fairer WER: 8,849 -> 8849, $1.5 billion -> 1.5 billion dollars,
    British/US spelling. Recognition errors are untouched."""
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    t = re.sub(r"\$(\d+(?:\.\d+)?)(\s+(?:million|billion|trillion|thousand))?", r"\1\2 dollars", t)
    return " ".join(_SPELLING.get(w.lower(), w) for w in t.split())


def numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))


def transcribe(path: Path, model: str, **kw):
    from factcheck_audio.pipeline import transcribe_audio, Timing
    from factcheck_audio.transcribe import EnglishTranscriber

    timing = Timing()
    t0 = time.perf_counter()
    tr_holder = {}

    def make():
        if model not in tr_holder:
            started = time.perf_counter()
            tr_holder[model] = EnglishTranscriber(model, **kw)
            tr_holder["load_s"] = round(time.perf_counter() - started, 2)
        return tr_holder[model]

    info, stage, warnings = transcribe_audio(path, make_transcriber=make, timing=timing)
    return info, stage, warnings, timing, round(time.perf_counter() - t0, 2), tr_holder.get("load_s")


def synthetic(model: str, tmp: Path) -> dict:
    from audio_helpers import write_audio

    rows = {}
    for case in CASES:
        samples, truth = build(case)
        path = write_audio(tmp / f"{case.name}{case.ext}", samples)
        info, stage, _, timing, wall, _ = transcribe(path, model)
        hyp = " ".join(s.text for s in stage.segments)
        ref = " ".join(t for _, t in truth)
        row = {"format": case.ext, "language": stage.language.detected, "accepted": stage.language.accepted,
               "status": stage.status.value, "segments": len(stage.segments),
               "rtf": timing.real_time_factor, "hypothesis": hyp}
        if truth:
            ref_nums = numbers(ref)
            hyp_nums = numbers(hyp)
            row.update(wer=round(wer(canonical(ref), canonical(hyp)), 3), raw_wer=round(wer(ref, hyp), 3),
                       word_recall=round(word_recall(canonical(ref), canonical(hyp)), 3),
                       numbers=f"{sum(1 for x in ref_nums if x in hyp_nums)}/{len(ref_nums)}")
            errs = []
            for t_on, text in truth:
                best = max(stage.segments, key=lambda s: word_recall(canonical(text), canonical(s.text)), default=None)
                if best is not None and word_recall(canonical(text), canonical(best.text)) >= 0.5:
                    errs.append(round(abs(best.start - t_on), 2))
            row["onset_errors_s"] = errs
            row["utterances_found"] = f"{len(errs)}/{len(truth)}"
        else:
            row["hallucinated_segments"] = len(stage.segments)
        rows[case.name] = row
    speech = [r for r in rows.values() if "wer" in r]
    errs = [e for r in speech for e in r["onset_errors_s"]]
    nums = [tuple(map(int, r["numbers"].split("/"))) for r in speech]
    return {
        "cases": rows,
        "summary": {
            "mean_wer": round(sum(r["wer"] for r in speech) / len(speech), 3),
            "max_wer": max(r["wer"] for r in speech),
            "mean_raw_wer": round(sum(r["raw_wer"] for r in speech) / len(speech), 3),
            "numbers": f"{sum(a for a, _ in nums)}/{sum(b for _, b in nums)}",
            "english_detected": f"{sum(r['language'] == 'en' for r in speech)}/{len(speech)}",
            "onset_error_mean_s": round(sum(errs) / len(errs), 2) if errs else None,
            "onset_error_max_s": max(errs) if errs else None,
            "no_speech_cases_clean": f"{sum(r['segments'] == 0 for r in rows.values() if 'wer' not in r)}/"
                                     f"{sum('wer' not in r for r in rows.values())}",
        },
    }


def to_mp3(src: Path, dst: Path) -> Path:
    import av

    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        stream = out.add_stream("libmp3lame", rate=44100)
        stream.layout = "mono"
        resampler = av.AudioResampler(format=stream.format.name, layout="mono", rate=44100)
        for frame in inp.decode(audio=0):
            for f in resampler.resample(frame):
                for p in stream.encode(f):
                    out.mux(p)
        for f in resampler.resample(None):
            for p in stream.encode(f):
                out.mux(p)
        for p in stream.encode(None):
            out.mux(p)
    return dst


def real(models: list[str], recordings: list[tuple[str, Path, str]], tmp: Path) -> dict:
    rows = {}
    for name, src, expected in recordings:
        if not src.exists():
            rows[name] = {"skipped": f"not found: {src}"}
            continue
        path = to_mp3(src, tmp / f"{name}.mp3")
        row = {"source": src.name, "expected_language": expected}
        texts = {}
        for model in models:
            info, stage, warnings, timing, wall, load_s = transcribe(path, model)
            segs = stage.segments
            texts[model] = " ".join(s.text for s in segs)
            row[model] = {
                "duration_s": info.duration_s, "language": stage.language.detected,
                "language_p": stage.language.probability, "accepted": stage.language.accepted,
                "status": stage.status.value, "segments": len(segs),
                "reliable_share": round(sum(s.reliable for s in segs) / len(segs), 3) if segs else None,
                "mean_confidence": round(sum(s.confidence for s in segs) / len(segs), 3) if segs else None,
                "words": len(texts[model].split()), "transcribe_s": timing.transcribe_s,
                "real_time_factor": timing.real_time_factor, "model_load_s": load_s,
                "correct_language_decision": (stage.language.accepted == (expected == "en")),
            }
        if len(models) > 1 and all(texts.values()):
            ref, other = models[-1], models[0]
            row["agreement_wer_vs_" + ref] = {m: round(wer(texts[ref], texts[m]), 3) for m in models if m != ref}
        row["first_words"] = texts[models[0]][:160]
        rows[name] = row
    return rows


def live(recording: Path, tmp: Path, budget: int) -> dict:
    from factcheck import envfile
    from factcheck.config import Config
    from factcheck_audio.pipeline import verify_audio
    from factcheck_audio.transcribe import EnglishTranscriber

    envfile.load_env_file()
    path = to_mp3(recording, tmp / "live.mp3")
    ar = verify_audio(path, Config.from_env(search_credit_budget=budget), make_transcriber=lambda: EnglishTranscriber())
    if ar.fact_check is None:
        return {"not_checked": ar.not_checked_reason}
    return {"summary": ar.fact_check.summary_label.value,
            "claims": [{"statement": t.statement, "verdict": t.verdict.value, "start": t.start, "end": t.end,
                        "segments": t.segment_ids, "confidence": t.transcription_confidence,
                        "supporting": len(t.supporting), "contradicting": len(t.contradicting),
                        "failure": t.failure_reason} for t in ar.traced_claims],
            "dropped": len(ar.fact_check.dropped_claims), "duplicates_merged": ar.duplicates_merged,
            "llm_calls": ar.fact_check.stats.llm_calls, "search_credits": ar.fact_check.stats.search_credits_used,
            "fact_check_s": ar.timing.fact_check_s}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="base", help="comma-separated cached faster-whisper sizes (default base)")
    ap.add_argument("--proxy-model", default="small", help="second size for the real-audio agreement proxy ('none' = off)")
    ap.add_argument("--real", nargs="*", type=Path, help="real English recordings (default: the sample videos)")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--live", action="store_true", help="also fact-check the first real English recording")
    ap.add_argument("--search-budget", type=int, default=20)
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    proxy = "" if args.proxy_model.lower() in ("", "none") else args.proxy_model
    real_models = models + ([proxy] if proxy and proxy not in models else [])
    recordings = ([(p.stem, p, "en") for p in args.real] if args.real else DEFAULT_REAL)
    results: dict = {"date": time.strftime("%Y-%m-%d %H:%M"), "models": models}
    with tempfile.TemporaryDirectory(prefix="audio-eval-") as tmp_s:
        tmp = Path(tmp_s)
        if not args.no_synthetic:
            for model in models:
                results[f"synthetic_{model}"] = syn = synthetic(model, tmp)
                print(f"\n== synthetic, {model}: {json.dumps(syn['summary'])}")
                for name, row in syn["cases"].items():
                    keep = {k: v for k, v in row.items() if k != "hypothesis"}
                    print(f"   {name:14} {json.dumps(keep)}")
        results["real"] = rows = real(real_models, recordings, tmp)
        print("\n== real recordings")
        for name, row in rows.items():
            print(f"   {name}: {json.dumps(row, ensure_ascii=False)}")
        if args.live:
            first = next((p for _, p, lang in recordings if lang == "en" and p.exists()), None)
            if first is not None:
                results["live"] = lv = live(first, tmp, args.search_budget)
                print(f"\n== live fact-check of {first.name}: {json.dumps(lv, ensure_ascii=False, indent=1)}")
    results["peak_rss_mb"] = peak_rss_mb()
    OUT.mkdir(exist_ok=True)
    path = OUT / f"audio-eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npeak RSS {results['peak_rss_mb']} MB\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
