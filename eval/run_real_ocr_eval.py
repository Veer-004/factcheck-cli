"""OCR evaluation on real frames with hand-written ground truth (eval/ocr_real/truth.json).

    python eval/run_real_ocr_eval.py                       # current defaults
    python eval/run_real_ocr_eval.py --show                # print each output
    python eval/run_real_ocr_eval.py --model-size medium   # compare settings

The frames come from the user's own videos and are not part of the repository
(eval/ocr_real/ is git-ignored). truth.json maps each frame to:
  required  the overlay text OCR should read
  optional  background, logos or partly hidden text that may be read (not counted
            as junk, not required)

Metrics (order-free, case-insensitive, tokens keep decimals and thousands: "1.5",
"1,000", "e-scooter"):
  recall   share of required tokens found
  numbers  share of required tokens containing a digit that were read exactly
           (the details a fact-check depends on)
  junk     share of output tokens that are neither required nor optional
  empty    frames with no text in the truth that still produced output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factcheck_ocr import create_engine, ocr_image  # noqa: E402

FRAMES = Path(__file__).parent / "ocr_real"
_TOKEN = re.compile(r"[^\W_]+(?:[.,'\-][^\W_]+)*")  # any script, so non-Latin junk ("难") counts too


def tokens(s: str) -> Counter:
    return Counter(_TOKEN.findall(unicodedata.normalize("NFKC", s).lower().replace("’", "'")))


def score(required: str, optional: str, hyp: str) -> dict:
    req, opt, out = tokens(required), tokens(optional), tokens(hyp)
    found = sum(min(n, out[w]) for w, n in req.items())
    nums = {w: n for w, n in req.items() if any(c.isdigit() for c in w)}
    nums_found = sum(min(n, out[w]) for w, n in nums.items())
    leftover = out - req
    junk = sum(n for w, n in leftover.items() if w not in opt)
    return {"req": sum(req.values()), "found": found, "nums": sum(nums.values()), "nums_found": nums_found,
            "out": sum(out.values()), "junk": junk, "junk_tokens": sorted((leftover - opt).elements())}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="rapid")
    p.add_argument("--model-size", default="small")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--json-out", type=Path)
    args, extra = p.parse_known_args()
    kwargs = json.loads(extra[0]) if extra else {}  # experimental engine/pipeline options as JSON
    truth = {k: v for k, v in json.loads((FRAMES / "truth.json").read_text(encoding="utf-8")).items()
             if not k.startswith("_")}
    t0 = time.perf_counter()
    engine = create_engine(args.engine, model_size=args.model_size, allow_download=args.allow_download,
                           **kwargs.get("engine", {}))
    print(f"engine {engine.name} ready in {time.perf_counter() - t0:.1f}s")
    rows, tot = [], Counter()
    for name, t in truth.items():
        page = ocr_image(FRAMES / name, engine, **kwargs.get("ocr", {}))
        s = score(t["required"], t["optional"], page.text)
        s.update(frame=name, seconds=page.stats.seconds, text=page.text)
        rows.append(s)
        for k in ("req", "found", "nums", "nums_found", "out", "junk"):
            tot[k] += s[k]
        tot["seconds"] += page.stats.seconds
        tot["empty_frames"] += not t["required"]
        tot["empty_with_output"] += (not t["required"] and not t["optional"] and s["out"] > 0)
        rec = s["found"] / s["req"] if s["req"] else 1.0
        print(f"{name:<14} recall {rec:5.2f}  numbers {s['nums_found']}/{s['nums']}  junk {s['junk']:>2}/{s['out']:<3} "
              f"{s['seconds']:5.2f}s  {' '.join(s['junk_tokens'])[:60]}")
        if args.show:
            print("    | " + page.text.replace("\n", "\n    | "))
    summary = {
        "recall": round(tot["found"] / tot["req"], 3),
        "numbers": f"{tot['nums_found']}/{tot['nums']}",
        "junk_rate": round(tot["junk"] / max(1, tot["out"]), 3),
        "junk_tokens": tot["junk"],
        "blank_frames_with_output": tot["empty_with_output"],
        "secs_per_frame": round(tot["seconds"] / len(rows), 2),
    }
    print("TOTAL", json.dumps(summary))
    if args.json_out:
        args.json_out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
