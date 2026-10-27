"""Evaluate an OCR engine on eval/ocr_samples (run make_ocr_samples.py first).

    python eval/run_ocr_eval.py --engine rapid
    python eval/run_ocr_eval.py --engine rapid --model-size medium --allow-download
    python eval/run_ocr_eval.py --engine paddle --allow-download   # separate Paddle venv

Metrics per image:
  CER          character error rate of the whole text, whitespace collapsed. Order
               sensitive: lines read in the wrong order count as errors.
  word recall  share of expected words found anywhere in the output (order-free), so
               recognition errors can be told apart from reading-order errors.
  order        share of consecutive expected-line pairs that appear in that order.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factcheck_ocr import create_engine, ocr_image  # noqa: E402

SAMPLES = Path(__file__).parent / "ocr_samples"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def cer(ref: str, hyp: str) -> float:
    a, b = norm(ref), norm(hyp)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


def word_recall(ref: str, hyp: str) -> float:
    words = lambda s: Counter(re.findall(r"[a-z0-9]+", s.lower()))  # noqa: E731
    r, h = words(ref), words(hyp)
    return sum(min(n, h[w]) for w, n in r.items()) / max(1, sum(r.values()))


def order_score(lines: list[str], hyp: str) -> float:
    flat = norm(hyp)
    pos = [flat.find(norm(l)) for l in lines]
    pairs = [(a, b) for a, b in zip(pos, pos[1:]) if a >= 0 and b >= 0]
    return sum(a < b for a, b in pairs) / len(pairs) if pairs else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="rapid")
    p.add_argument("--model-size", default="small", help="rapid engine: small or medium")
    p.add_argument("--layout", default="auto")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--show", action="store_true", help="print each output text")
    args = p.parse_args()
    rows = [json.loads(l) for l in (SAMPLES / "expected.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    t0 = time.perf_counter()
    engine = create_engine(args.engine, model_size=args.model_size, allow_download=args.allow_download)
    print(f"engine {engine.name} ready in {time.perf_counter() - t0:.1f}s (layout={args.layout})")
    print(f"{'image':<18} {'CER':>6} {'recall':>7} {'order':>6} {'secs':>6}")
    totals = []
    for row in rows:
        page = ocr_image(SAMPLES / row["image"], engine, layout=args.layout)
        ref = "\n".join(row["lines"])
        m = (cer(ref, page.text), word_recall(ref, page.text), order_score(row["lines"], page.text), page.stats.seconds)
        totals.append(m)
        print(f"{row['image'][:-4]:<18} {m[0]:6.3f} {m[1]:7.3f} {m[2]:6.2f} {m[3]:6.2f}")
        if args.show:
            print("    | " + page.text.replace("\n", "\n    | "))
    n = len(totals)
    print(f"{'mean':<18} {sum(t[0] for t in totals) / n:6.3f} {sum(t[1] for t in totals) / n:7.3f} "
          f"{sum(t[2] for t in totals) / n:6.2f} {sum(t[3] for t in totals) / n:6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
