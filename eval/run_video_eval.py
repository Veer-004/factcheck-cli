"""Video module evaluation on the synthetic set (eval/make_video_samples.py).

    python eval/make_video_samples.py                       # once (Windows: SAPI voices)
    python eval/run_video_eval.py                           # extraction/dedup/provenance, offline, model "base"
    python eval/run_video_eval.py --models tiny,base,small  # compare speech models (must be cached)
    python eval/run_video_eval.py --live                    # + real fact-check of the claim cases (API credits)

Scores are reported separately; none of them says anything about real-world
factual reliability:

* extraction  - speech word error rate (WER) against the synthesised script,
                on-screen word recall against the rendered captions;
* dedup       - (a) exact speech/screen unit merges and distinct on-screen
                observations per video vs the documented expectation,
                (b) precision/recall of the claim-level duplicate rule on the
                labelled pairs in eval/video_dedup_pairs.jsonl;
* provenance  - speech segment start within 0.75 s of the scripted start;
                on-screen first sighting within one sampling interval of the
                caption start;
* robustness  - corrupt/non-video files rejected, silence/noise produce no
                speech and no claims;
* fact-check  - (only with --live) verdicts for the claim cases.

Results are written to eval/results/ (git-ignored).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SAMPLES = Path(__file__).parent / "video_samples"
OUT = Path(__file__).parent / "results"
STOP = {"the", "is", "a", "of", "on", "in", "at", "was", "than", "and", "to"}


def peak_rss_mb() -> float | None:
    """Peak working set of this process (Windows), for a rough RAM figure."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return round(pmc.PeakWorkingSetSize / 2**20, 1)


def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def word_recall(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    pool = list(h)
    hit = 0
    for w in r:
        if w in pool:
            pool.remove(w)
            hit += 1
    return hit / max(1, len(r))


def dedup_pairs() -> dict:
    from factcheck_video.textmatch import claims_duplicate

    rows = [json.loads(l) for l in (Path(__file__).parent / "video_dedup_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tp = fp = fn = tn = 0
    misses, wrong_merges = [], []
    for r in rows:
        got = claims_duplicate((r["a"], r["a"]), (r["b"], r["b"]))
        if got and r["dup"]:
            tp += 1
        elif got and not r["dup"]:
            fp += 1
            wrong_merges.append(r["kind"])
        elif not got and r["dup"]:
            fn += 1
            misses.append(r["kind"])
        else:
            tn += 1
    return {"pairs": len(rows), "true_merges": tp, "wrong_merges": fp, "missed_merges": fn, "kept_apart_correctly": tn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "wrong_merge_kinds": wrong_merges, "missed_kinds": misses}


def run_model(model: str, cases: list[dict], args) -> dict:
    from factcheck_ocr.engines import create_engine
    from factcheck_video.media import VideoInputError
    from factcheck_video.models import Limits
    from factcheck_video.pipeline import extract_video
    from factcheck_video.transcribe import WhisperTranscriber

    t0 = time.perf_counter()
    transcriber = WhisperTranscriber(model, compute_type=args.compute_type)
    load_s = time.perf_counter() - t0
    engine = create_engine("rapid")
    limits = Limits(frame_interval_s=args.frame_interval)
    rows, totals = [], {"audio_s": 0.0, "transcribe_s": 0.0, "frames_ocr_s": 0.0, "media_s": 0.0}
    for case in cases:
        path = SAMPLES / case["file"]
        row = {"case": case["name"]}
        if case.get("corrupt"):
            try:
                extract_video(path, limits=limits, make_transcriber=lambda: transcriber, make_engine=lambda: engine)
                row["rejected"] = False
            except VideoInputError as exc:
                row["rejected"], row["error"] = True, str(exc)[:120]
            rows.append(row)
            continue
        from factcheck_video.pipeline import Timing

        timing = Timing()
        ex = extract_video(path, limits=limits, make_transcriber=lambda: transcriber, make_engine=lambda: engine,
                           timing=timing)
        totals["audio_s"] += case["duration"]
        totals["transcribe_s"] += timing.transcribe_s
        totals["frames_ocr_s"] += timing.frames_ocr_s
        ref_speech = " ".join(s["text"] for s in case["speech"])
        hyp_speech = " ".join(s.text for s in ex.audio.segments)
        row.update(audio=ex.audio.status.value, screen=ex.screen.status.value, segments=len(ex.audio.segments),
                   transcript=hyp_speech, observations=[o.text for o in ex.screen.observations],
                   exact_merges=ex.exact_merges)
        if ref_speech:
            row["wer"] = round(wer(ref_speech, hyp_speech), 3)
        else:
            row["spurious_speech"] = hyp_speech  # silence/noise: should be empty
        ref_text = " ".join(dict.fromkeys(c["text"] for c in case["captions"]))  # distinct captions
        if ref_text:
            row["screen_word_recall"] = round(word_recall(ref_text, " ".join(o.text for o in ex.screen.observations)), 3)
        if case.get("expect_distinct_screen") is not None:
            row["distinct_screen_ok"] = len(ex.screen.observations) == case["expect_distinct_screen"]
        row["exact_merges_ok"] = ex.exact_merges == case.get("expect_merged_units", 0)
        # provenance: each scripted utterance -> the first segment sharing a content word, start within 0.75 s
        prov = []
        for s in case["speech"]:
            words = set(norm_words(s["text"])) - STOP
            seg = next((g for g in ex.audio.segments if words & set(norm_words(g.text))), None)
            prov.append(seg is not None and abs(seg.start - s["start"]) <= 0.75)
        interval = ex.screen.interval_s or args.frame_interval
        for c in case["captions"]:  # each appearance must be sighted within one interval of its start
            obs = next((o for o in ex.screen.observations if word_recall(c["text"], o.text) >= 0.8), None)
            prov.append(obs is not None and any(c["start"] - 1e-6 <= t <= c["start"] + interval + 1e-6
                                                for t in obs.frame_times))
        row["provenance_detail"] = {
            "speech": [(s["start"], next((g.start for g in ex.audio.segments
                                          if (set(norm_words(s["text"])) - STOP) & set(norm_words(g.text))), None))
                       for s in case["speech"]],
            "screen": [(c["start"], next((o.frame_times for o in ex.screen.observations
                                          if word_recall(c["text"], o.text) >= 0.8), None)) for c in case["captions"]],
        }
        if prov:
            row["provenance_ok"] = f"{sum(prov)}/{len(prov)}"
        rows.append(row)
    rtf = totals["transcribe_s"] / totals["audio_s"] if totals["audio_s"] else None
    return {"model": model, "compute_type": args.compute_type, "model_load_s": round(load_s, 2),
            "transcribe_s_total": round(totals["transcribe_s"], 2), "audio_s_total": round(totals["audio_s"], 1),
            "real_time_factor": round(rtf, 3) if rtf else None, "frames_ocr_s_total": round(totals["frames_ocr_s"], 2),
            "peak_rss_mb_so_far": peak_rss_mb(), "cases": rows}


def summarize(res: dict) -> dict:
    rows = res["cases"]
    wers = [r["wer"] for r in rows if "wer" in r]
    rec = [r["screen_word_recall"] for r in rows if "screen_word_recall" in r]
    prov = [tuple(map(int, r["provenance_ok"].split("/"))) for r in rows if "provenance_ok" in r]
    return {
        "model": res["model"], "mean_wer": round(sum(wers) / len(wers), 3) if wers else None,
        "max_wer": max(wers) if wers else None,
        "screen_word_recall": round(sum(rec) / len(rec), 3) if rec else None,
        "distinct_screen_ok": f"{sum(r.get('distinct_screen_ok', False) for r in rows if 'distinct_screen_ok' in r)}"
                              f"/{sum('distinct_screen_ok' in r for r in rows)}",
        "exact_merges_ok": f"{sum(r.get('exact_merges_ok', False) for r in rows if 'exact_merges_ok' in r)}"
                           f"/{sum('exact_merges_ok' in r for r in rows)}",
        "provenance_ok": f"{sum(a for a, _ in prov)}/{sum(b for _, b in prov)}",
        "corrupt_rejected": f"{sum(r.get('rejected', False) for r in rows if 'rejected' in r)}"
                            f"/{sum('rejected' in r for r in rows)}",
        "no_spurious_speech": f"{sum(not r['spurious_speech'] for r in rows if 'spurious_speech' in r)}"
                              f"/{sum('spurious_speech' in r for r in rows)}",
        "real_time_factor": res["real_time_factor"], "model_load_s": res["model_load_s"],
        "peak_rss_mb": res["peak_rss_mb_so_far"],
    }


def live_factcheck(cases: list[dict], model: str, budget: int) -> list[dict]:
    from factcheck import envfile
    from factcheck.config import Config
    from factcheck_ocr.engines import create_engine
    from factcheck_video.pipeline import verify_video
    from factcheck_video.transcribe import WhisperTranscriber

    envfile.load_env_file()
    transcriber, engine = WhisperTranscriber(model), create_engine("rapid")
    out = []
    for case in cases:
        if not case.get("expect_claims"):
            continue
        config = Config.from_env(search_credit_budget=budget, max_checkable_claims=4)
        started = time.perf_counter()
        vr = verify_video(SAMPLES / case["file"], config, make_transcriber=lambda: transcriber,
                          make_engine=lambda: engine)
        claims = vr.fact_check.claims if vr.fact_check else []
        row = {"case": case["name"], "seconds": round(time.perf_counter() - started, 1),
               "duplicates_merged": vr.duplicates_merged,
               "claims": [{"statement": c.statement, "verdict": c.verdict.value,
                           "modalities": [m.value for m in s.modalities]} for c, s in zip(claims, vr.claim_sources)],
               "checks": []}
        for exp in case["expect_claims"]:
            hits = [c for c in claims if exp["needle"] in c.statement.lower()]
            ok = bool(hits) and all(c.verdict.value == exp["verdict"] for c in hits)
            if "count" in exp:
                ok = ok and len(hits) == exp["count"]
            row["checks"].append({**exp, "got": [c.verdict.value for c in hits], "ok": ok})
        print(f"  {case['name']}: " + ", ".join(f"{c['needle']}->{c['got']} {'ok' if c['ok'] else 'MISMATCH'}"
                                                for c in row["checks"]) + f"  (merged {vr.duplicates_merged})")
        out.append(row)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="base", help="comma-separated faster-whisper sizes (must be cached)")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--frame-interval", type=float, default=2.0)
    p.add_argument("--live", action="store_true", help="also run the real fact-check on the claim cases")
    p.add_argument("--search-budget", type=int, default=12)
    args = p.parse_args()
    manifest = SAMPLES / "manifest.json"
    if not manifest.is_file():
        print("run eval/make_video_samples.py first", file=sys.stderr)
        return 2
    cases = json.loads(manifest.read_text(encoding="utf-8"))
    report = {"dedup_pairs": dedup_pairs(), "models": [], "summary": []}
    dp = report["dedup_pairs"]
    print(f"claim dedup rule on {dp['pairs']} labelled pairs: precision {dp['precision']}, recall {dp['recall']} "
          f"(wrong merges {dp['wrong_merges']}, missed merges {dp['missed_merges']}: {dp['missed_kinds']})")
    for model in args.models.split(","):
        res = run_model(model.strip(), cases, args)
        report["models"].append(res)
        s = summarize(res)
        report["summary"].append(s)
        print(f"\n== {model}: " + ", ".join(f"{k} {v}" for k, v in s.items() if k != "model"))
        for r in res["cases"]:
            extra = {k: r[k] for k in ("wer", "screen_word_recall", "provenance_ok", "distinct_screen_ok",
                                       "exact_merges_ok", "rejected") if k in r}
            print(f"   {r['case']:<20} {extra}")
            if "provenance_ok" in r and r["provenance_ok"].split("/")[0] != r["provenance_ok"].split("/")[1]:
                print(f"   {'':<20} provenance detail (scripted start, found): {r['provenance_detail']}")
    if args.live:
        print("\nlive fact-check (model " + args.models.split(",")[0] + "):")
        report["live"] = live_factcheck(cases, args.models.split(",")[0].strip(), args.search_budget)
    OUT.mkdir(exist_ok=True)
    out = OUT / f"video-eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
