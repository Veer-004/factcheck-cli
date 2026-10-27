"""``factcheck-ocr``: extract text from images; ``--fact-check`` also fact-checks that text.

Plain extraction never imports ``factcheck``. ``--fact-check`` hands one image to
``factcheck.image.run_fact_check``, which uses the text fact-checker's exit codes
(0 resolved, 1 partial/unresolved or no usable text, 2 invalid input/config, 3 run failure).

Exit codes (extraction):
  0  text was extracted from every image
  1  completed, but at least one image produced no text
  2  invalid input or setup (missing file, unsupported image, engine or models not installed)
  3  the OCR engine failed while processing an image
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .engines import ENGINE_NAMES, MODEL_SIZES, EngineFailure, EngineUnavailable, create_engine, engine_status
from .image_io import MAX_SIDE, ImageInputError
from .models import PageResult
from .pipeline import DEFAULT_MIN_SCORE, LAYOUTS, ocr_image

EXIT_OK, EXIT_NO_TEXT, EXIT_INVALID, EXIT_FAILURE = 0, 1, 2, 3
MAX_IMAGES = 50


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="factcheck-ocr",
        description="Extract text from images (screenshots, memes, posters, documents) with local OCR. "
                    "Text only: images are not interpreted and the text is not fact-checked.",
        epilog="Exit codes: 0 text found, 1 some image had no text, 2 invalid input/setup, 3 engine failure.",
    )
    p.add_argument("images", nargs="*", type=Path, help="image files (PNG, JPEG, WEBP, BMP, TIFF, GIF)")
    p.add_argument("--engine", choices=ENGINE_NAMES, default=os.environ.get("FACTCHECK_OCR_ENGINE", "rapid"),
                   help="OCR engine (default rapid, or FACTCHECK_OCR_ENGINE)")
    p.add_argument("--layout", choices=LAYOUTS, default="auto",
                   help="reading order: auto (model order if the engine has one, else geometric), "
                        "geometric (columns/blocks from box positions), none (engine order)")
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                   help=f"drop recognised lines below this confidence, 0-1 (default {DEFAULT_MIN_SCORE})")
    p.add_argument("--max-side", type=int, default=MAX_SIDE,
                   help=f"downscale images whose longer side exceeds this many pixels (default {MAX_SIDE})")
    p.add_argument("--model-size", choices=MODEL_SIZES, default="small",
                   help="rapid engine: PP-OCRv6 small (default, bundled) or medium (better on memes and "
                        "stylised text, slower; needs a one-time --allow-download)")
    p.add_argument("--textline-orientation", action="store_true",
                   help="rapid engine: detect upside-down text lines (off by default: its classifier "
                        "misreads outlined meme text; the paddle engines always use their own classifier)")
    p.add_argument("--allow-download", action="store_true",
                   help="allow a one-time model download (rapid medium about 130 MB; paddle engines 140-400 MB)")
    p.add_argument("--json-out", type=Path, help="also write full results (boxes, scores, blocks) as JSON")
    fc = p.add_argument_group(
        "image fact-checking",
        "--fact-check runs the extracted text of ONE image through the fact-checking pipeline "
        "(Gemini + Tavily, same keys and limits as the factcheck command). Only the recognised text is "
        "checked; the image itself is not interpreted.")
    fc.add_argument("--fact-check", action="store_true", help="fact-check the text extracted from the image")
    fc.add_argument("--ocr-threshold", type=float, default=0.80,
                    help="claims whose image text has OCR confidence below this are reported but not checked "
                         "(0-1, default 0.80)")
    fc.add_argument("--model", help="Gemini model ID (default from FACTCHECK_MODEL)")
    fc.add_argument("--max-claims", type=int, help="maximum checkable claims to verify (1-12, default 6)")
    fc.add_argument("--search-budget", type=int, help="maximum Tavily credits for this run (0-40, default 40)")
    fc.add_argument("--timeout", type=float, help="run time limit in seconds (default 180)")
    p.add_argument("--check", action="store_true", help="show installed OCR packages and exit")
    p.add_argument("-q", "--quiet", action="store_true", help="print only the extracted text")
    p.add_argument("-v", "--verbose", action="store_true", help="also show each line's confidence and position")
    return p


def render_page(page: PageResult, *, verbose: bool, quiet: bool, header: bool) -> str:
    if quiet:
        return page.text
    out = []
    if header:
        out.append(f"=== {page.source} ===")
    if verbose:
        s = page.stats
        out.append(f"[{page.width}x{page.height}  engine={page.engine}  layout={page.layout}  "
                   f"lines kept {s.boxes_kept}/{s.boxes_detected}  {s.seconds:.2f}s]")
        for i, block in enumerate(page.blocks, 1):
            label = f" ({block.label})" if block.label else ""
            out.append(f"-- block {i}{label}")
            for line in block.lines:
                x0, y0, x1, y1 = (round(v) for v in line.bbox)
                out.append(f"  {line.score:.2f}  [{x0},{y0},{x1},{y1}]  {line.text}")
    else:
        out.append(page.text if page.text else "(no text found)")
    for w in page.warnings:
        out.append(f"note: {w}")
    if not verbose and page.stats.boxes_below_min_score:
        out.append(f"note: {page.stats.boxes_below_min_score} low-confidence line(s) omitted (see --min-score, -v)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    args = _parser().parse_args(argv)

    if args.check:
        print("OCR packages:")
        for name, value in engine_status().items():
            print(f"  {name}: {value}")
        return EXIT_OK
    if not args.images:
        print("error: no input: pass one or more image files", file=sys.stderr)
        return EXIT_INVALID
    if len(args.images) > MAX_IMAGES:
        print(f"error: at most {MAX_IMAGES} images per run", file=sys.stderr)
        return EXIT_INVALID
    if not 0.0 <= args.min_score <= 1.0:
        print("error: --min-score must be between 0 and 1", file=sys.stderr)
        return EXIT_INVALID
    if not 256 <= args.max_side <= 10000:
        print("error: --max-side must be between 256 and 10000", file=sys.stderr)
        return EXIT_INVALID
    if args.json_out is not None and args.json_out.is_dir():
        print(f"error: --json-out points to a directory: {args.json_out}", file=sys.stderr)
        return EXIT_INVALID
    missing = [p for p in args.images if not p.is_file()]
    if missing:
        print(f"error: image file not found: {missing[0]}", file=sys.stderr)
        return EXIT_INVALID

    if args.fact_check:
        try:  # imported only here: plain OCR never loads the fact-checking stack
            from factcheck.image import run_fact_check
        except ImportError as exc:
            print(f"error: fact-checking dependencies are not installed ({exc.name}); "
                  'run: pip install -e ".[ocr]"', file=sys.stderr)
            return EXIT_INVALID
        return run_fact_check(args)

    try:
        engine = create_engine(args.engine, model_size=args.model_size, allow_download=args.allow_download,
                               textline_orientation=args.textline_orientation)
    except EngineUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    pages: list[PageResult] = []
    errors: list[dict] = []
    code = EXIT_OK
    for path in args.images:
        try:
            page = ocr_image(path, engine, min_score=args.min_score, layout=args.layout, max_side=args.max_side)
        except ImageInputError as exc:
            print(f"error: {exc}", file=sys.stderr)
            errors.append({"source": path.name, "error": str(exc)})
            code = max(code, EXIT_INVALID)
            continue
        except EngineFailure as exc:
            print(f"error: {path.name}: {exc}", file=sys.stderr)
            errors.append({"source": path.name, "error": str(exc)})
            code = max(code, EXIT_FAILURE)
            continue
        pages.append(page)
        if not page.text:
            code = max(code, EXIT_NO_TEXT)  # precedence: failure 3 > invalid 2 > no text 1
        if pages[:-1]:
            print()
        print(render_page(page, verbose=args.verbose, quiet=args.quiet, header=len(args.images) > 1))

    if args.json_out is not None:
        payload = {"pages": [p.model_dump(mode="json") for p in pages], "errors": errors}
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
