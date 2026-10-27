"""Live end-to-end evaluation of image fact-checking (real OCR + Gemini + Tavily).

    python eval/run_image_eval.py                  # all cases (uses API credits)
    python eval/run_image_eval.py --case meme_myth

Renders a few images with known text, runs ``verify_image`` with the configured
providers, and compares each claim's verdict with an expected outcome chosen by
hand before the run. "acceptable" lists verdicts that are defensible but weaker
(e.g. INSUFFICIENT_EVIDENCE when the web returns little). Results are written to
eval/results/ (git-ignored). This uses real API calls: each case is capped at
--search-budget credits (default 12).

This is NOT part of the offline test suite; the offline image tests live in
tests/test_image_factcheck.py and use fake providers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw  # noqa: E402

import make_ocr_samples as samples  # noqa: E402
from factcheck import envfile  # noqa: E402
from factcheck.config import Config  # noqa: E402
from factcheck.image import render_image_report, verify_image  # noqa: E402

OUT = Path(__file__).parent / "results"


def meme_myth():
    im = Image.new("RGB", (900, 700), (60, 90, 140))
    d = ImageDraw.Draw(im)
    f = samples.font("impact.ttf", 56)
    for text, y in (("THE GREAT WALL OF CHINA IS", 40), ("VISIBLE FROM THE MOON", 110), ("WITH THE NAKED EYE", 560)):
        w = d.textlength(text, font=f)
        d.text(((900 - w) / 2, y), text, font=f, fill="white", stroke_width=4, stroke_fill="black")
    return im


def tweet():
    im = Image.new("RGB", (900, 420), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), "Science Facts @sciencefacts", font=samples.font("segoeuib.ttf", 30), fill="black")
    body = ["The Eiffel Tower was completed in 1889.", "Water boils at 50 degrees Celsius at sea level."]
    samples.lines_at(d, body, 40, 110, samples.font("segoeui.ttf", 34), step=70)
    d.text((40, 330), "10:42 AM - 3 Mar 2026", font=samples.font("segoeui.ttf", 24), fill=(110, 110, 110))
    return im


def poster_opinion():
    im = Image.new("RGB", (900, 500), (250, 230, 200))
    d = ImageDraw.Draw(im)
    f = samples.font("arialbd.ttf", 60)
    for text, y in (("PINEAPPLE IS THE BEST", 120), ("PIZZA TOPPING!", 210)):
        d.text(((900 - d.textlength(text, font=f)) / 2, y), text, font=f, fill=(150, 30, 30))
    return im


# name -> (image factory, [(statement substring, expected verdict, acceptable alternatives)], expected summary)
CASES = {
    "meme_myth": (meme_myth, [("great wall", "REFUTED", ["MISLEADING", "INSUFFICIENT_EVIDENCE"])], None),
    "tweet_two_claims": (tweet, [("eiffel", "SUPPORTED", ["INSUFFICIENT_EVIDENCE"]),
                                 ("boils", "REFUTED", ["INSUFFICIENT_EVIDENCE"])], None),
    "poster_opinion": (poster_opinion, [], "NOTHING_TO_CHECK"),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=sorted(CASES), action="append")
    p.add_argument("--search-budget", type=int, default=12)
    p.add_argument("--show", action="store_true", help="print each full report")
    args = p.parse_args()
    envfile.load_env_file()
    OUT.mkdir(exist_ok=True)
    rows, strict, lenient, total = [], 0, 0, 0
    for name in args.case or list(CASES):
        make, expectations, expected_summary = CASES[name]
        path = OUT / f"image_case_{name}.png"
        make().save(path)
        config = Config.from_env(search_credit_budget=args.search_budget, max_checkable_claims=3)
        started = time.monotonic()
        try:
            ir = verify_image(path, config)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{name}: ERROR {type(exc).__name__}: {exc}")
            rows.append({"case": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        secs = time.monotonic() - started
        fc = ir.fact_check
        claims = fc.claims if fc else []
        print(f"\n== {name}  ({secs:.1f}s, OCR {ir.extraction.quality.status.value}, "
              f"summary {fc.summary_label.value if fc else 'NOT CHECKED'}, "
              f"credits {fc.stats.search_credits_used if fc else 0}, model calls {fc.stats.llm_calls if fc else 0})")
        print("   OCR text: " + ir.extraction.text.replace("\n", " / "))
        for c in claims:
            print(f"   {c.verdict.value:<22} score={c.score}  {c.statement}")
        result = {"case": name, "seconds": round(secs, 1), "ocr_status": ir.extraction.quality.status.value,
                  "ocr_text": ir.extraction.text, "summary": fc.summary_label.value if fc else None,
                  "claims": [{"statement": c.statement, "verdict": c.verdict.value, "score": c.score} for c in claims],
                  "checks": []}
        if expected_summary is not None:
            total += 1
            ok = fc is not None and fc.summary_label.value == expected_summary
            strict += ok
            lenient += ok
            result["checks"].append({"expected_summary": expected_summary, "ok": ok})
            print(f"   expected summary {expected_summary}: {'ok' if ok else 'MISMATCH'}")
        for needle, expected, acceptable in expectations:
            total += 1
            match = next((c for c in claims if needle in c.statement.lower()), None)
            got = match.verdict.value if match else "NO_MATCHING_CLAIM"
            s_ok, l_ok = got == expected, got == expected or got in acceptable
            strict += s_ok
            lenient += l_ok
            result["checks"].append({"claim": needle, "expected": expected, "got": got, "strict": s_ok, "lenient": l_ok})
            print(f"   expected {needle!r} -> {expected}: got {got} ({'ok' if s_ok else 'acceptable' if l_ok else 'MISMATCH'})")
        if args.show:
            print(render_image_report(ir))
        rows.append(result)
    print(f"\nstrict {strict}/{total}, lenient {lenient}/{total}")
    out = OUT / f"image-live-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"strict": strict, "lenient": lenient, "total": total, "cases": rows}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
