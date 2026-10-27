"""Image fact-checking: OCR text -> the shared fact-checking pipeline.

    image -> OCR (factcheck_ocr) -> extraction-quality assessment
          -> claim extraction (with an OCR note) -> per-claim OCR gate
          -> retrieval -> evidence -> judge -> validation -> deterministic verdict -> report

Only the text recognised in the image is checked; the image itself is never
interpreted. Everything after OCR is the same code as the text workflow
(``graph.run_pipeline``), with two opt-in hooks:

* ``source_note`` tells the extraction model the text came from OCR (do not
  repair or complete it, classify fragments as VAGUE);
* ``claim_gate`` withholds a claim before retrieval when any OCR line it comes
  from is below the reliability threshold. Such a claim is reported as
  PROCESSING_FAILED with an ``ocr:`` reason: a technical outcome, never a truth
  verdict, and no search credits are spent on a possible misreading.

OCR confidence, evidence strength (score) and the verdict are kept separate:
OCR confidence only decides whether a claim's wording can be trusted enough to
check; it never enters the evidence score.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from factcheck_ocr.engines import EngineFailure, EngineUnavailable, OcrEngine, create_engine
from factcheck_ocr.image_io import ImageInputError
from factcheck_ocr.pipeline import DEFAULT_MIN_SCORE, ocr_image
from factcheck_ocr.quality import DEFAULT_RELIABLE_SCORE, ExtractionQuality, ExtractionStatus, OcrLine, assess

from . import envfile
from .cli import exit_code_for as text_exit_code
from .config import Config, ConfigError
from .graph import RunFailure, run_pipeline
from .llm import LLMClient
from .models import Claim, Report
from .prompts import OCR_SOURCE_NOTE
from .report import WIDTH, render_text
from .retrieval import SearchProvider
from .security import redact_secrets
from .textnorm import normalize_for_match
from .validation import InputError

EXIT_PARTIAL, EXIT_INVALID, EXIT_FAILURE = 1, 2, 3  # same codes as the text CLI
MAX_TEXT_LINES_SHOWN = 40


class ClaimTrace(BaseModel):
    """Where a claim's source excerpt sits in the OCR output."""

    claim_id: str
    line_numbers: list[int] = Field(default_factory=list)
    block_numbers: list[int] = Field(default_factory=list)
    min_ocr_score: Optional[float] = None
    ocr_reliable: bool = False
    located: bool = False


class ImageExtraction(BaseModel):
    image: str
    width: int
    height: int
    engine: str
    layout: str
    text: str
    quality: ExtractionQuality
    ocr_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class ImageReport(BaseModel):
    input_kind: str = "image"
    extraction: ImageExtraction
    fact_check: Optional[Report] = None  # None when the text was not good enough to check
    not_checked_reason: Optional[str] = None
    claim_traces: list[ClaimTrace] = Field(default_factory=list)
    ocr_confidence_note: str = ("OCR confidence measures how reliably characters were recognised. It is not "
                                "evidence that the text is true and is not part of the evidence-strength score.")


# ---------------------------------------------------------------- tracing


def locate(excerpt: str, lines: list[OcrLine]) -> list[OcrLine]:
    """OCR lines that the (normalised) excerpt spans; [] if it cannot be found."""
    target = normalize_for_match(excerpt)
    if not target:
        return []
    spans, parts, pos = [], [], 0
    for line in lines:
        norm = normalize_for_match(line.text)
        spans.append((pos, pos + len(norm)))
        parts.append(norm)
        pos += len(norm) + 1  # joined with single spaces, like normalize_for_match collapses whitespace
    start = " ".join(parts).find(target)
    if start < 0:
        return []
    end = start + len(target)
    return [line for line, (a, b) in zip(lines, spans) if a < end and b > start]


def trace_claim(claim_id: str, excerpt: str, quality: ExtractionQuality) -> ClaimTrace:
    hit = locate(excerpt, quality.lines)
    if not hit:
        return ClaimTrace(claim_id=claim_id)
    return ClaimTrace(
        claim_id=claim_id,
        line_numbers=[l.line_no for l in hit],
        block_numbers=sorted({l.block_no for l in hit}),
        min_ocr_score=round(min(l.score for l in hit), 4),
        ocr_reliable=all(l.reliable for l in hit),
        located=True,
    )


def ocr_gate(quality: ExtractionQuality):
    """Claim -> reason to withhold it (``ocr:`` prefix), or None to check it."""

    def gate(claim: Claim) -> Optional[str]:
        t = trace_claim(claim.claim_id, claim.source_excerpt, quality)
        if not t.located:
            return "ocr: the claim's source text could not be traced to the OCR lines"
        if not t.ocr_reliable:
            return (f"ocr: its source text (line(s) {_ranges(t.line_numbers)}) has OCR confidence "
                    f"{t.min_ocr_score:.2f}, below the {quality.threshold:.2f} reliability threshold, so the "
                    "wording may have been misread; it was not searched or judged")
        return None

    return gate


def _ranges(nums: list[int]) -> str:
    return f"{nums[0]}-{nums[-1]}" if len(nums) > 1 else str(nums[0]) if nums else "?"


# ---------------------------------------------------------------- workflow


def verify_image(
    path: Path,
    config: Config | None = None,
    llm: LLMClient | None = None,
    search: SearchProvider | None = None,
    *,
    engine: OcrEngine | None = None,
    engine_name: str = "rapid",
    ocr_threshold: float = DEFAULT_RELIABLE_SCORE,
    min_score: float = DEFAULT_MIN_SCORE,
    layout: str = "auto",
    model_size: str = "small",
    allow_download: bool = False,
) -> ImageReport:
    """OCR ``path`` and fact-check the recognised text.

    Raises ``InputError`` (missing/unsupported image, text over the input limit),
    ``ConfigError`` (OCR engine or credentials missing) or ``RunFailure`` (OCR
    engine or run-level failure). OCR runs before credentials are needed, so an
    image without usable text is reported without any model or search call.
    """
    if not 0.0 <= ocr_threshold <= 1.0:
        raise InputError("OCR threshold must be between 0 and 1")
    try:
        engine = engine or create_engine(engine_name, model_size=model_size, allow_download=allow_download)
        page = ocr_image(path, engine, min_score=min_score, layout=layout)
    except ImageInputError as exc:
        raise InputError(str(exc)) from None
    except EngineUnavailable as exc:
        raise ConfigError(str(exc)) from None
    except EngineFailure as exc:
        raise RunFailure(f"OCR failed: {exc}") from None

    quality = assess(page, ocr_threshold)
    extraction = ImageExtraction(
        image=str(path), width=page.width, height=page.height, engine=page.engine, layout=page.layout,
        text=page.text, quality=quality, ocr_seconds=page.stats.seconds, warnings=page.warnings,
    )
    if quality.status in (ExtractionStatus.NO_TEXT, ExtractionStatus.UNREADABLE):
        return ImageReport(extraction=extraction,
                           not_checked_reason="; ".join(quality.problems) + ". Try a clearer or higher-resolution image.")

    report = run_pipeline(page.text, config, llm, search, source_note=OCR_SOURCE_NOTE, claim_gate=ocr_gate(quality))
    if quality.status == ExtractionStatus.LOW_CONFIDENCE:
        report.warnings.extend(f"OCR: {p}" for p in quality.problems)
    traces = [trace_claim(r.claim_id, r.source_excerpt, quality) for r in report.claims]
    return ImageReport(extraction=extraction, fact_check=report, claim_traces=traces)


def exit_code_for(ir: ImageReport) -> int:
    if ir.fact_check is None:
        return EXIT_PARTIAL  # completed, but nothing could be checked (same as factcheck-ocr's "no text")
    return text_exit_code(ir.fact_check)  # withheld OCR claims are PROCESSING_FAILED -> partial


# ---------------------------------------------------------------- rendering


_STATUS_TEXT = {
    ExtractionStatus.OK: "OK",
    ExtractionStatus.LOW_CONFIDENCE: "LOW CONFIDENCE",
    ExtractionStatus.UNREADABLE: "UNREADABLE",
    ExtractionStatus.NO_TEXT: "NO TEXT FOUND",
}


def _trace_line(t: ClaimTrace, threshold: float) -> str:
    if not t.located:
        return "Image text: not traced to specific OCR lines."
    blocks = ", ".join(map(str, t.block_numbers))
    flag = "" if t.ocr_reliable else f" (below the {threshold:.2f} threshold)"
    return (f"Image text: line(s) {_ranges(t.line_numbers)} (block {blocks}), "
            f"OCR confidence {t.min_ocr_score:.2f}{flag}")


def render_image_report(ir: ImageReport, verbose: bool = False) -> str:
    ex, q = ir.extraction, ir.extraction.quality
    out = ["IMAGE FACT-CHECK", ""]
    out.append(f"Image: {ex.image} ({ex.width}x{ex.height}; OCR engine {ex.engine}, reading order {ex.layout})")
    status = f"Text extraction: {_STATUS_TEXT[q.status]}"
    if q.lines:
        status += f" ({len(q.lines)} line(s), OCR confidence {q.min_score:.2f}-{max(l.score for l in q.lines):.2f})"
    out.append(status)
    out += [f"   - {p}" for p in q.problems]
    out += [f"   - note: {w}" for w in ex.warnings]
    if q.lines:
        out.append("")
        out.append(f"Extracted text (OCR output, shown as recognised; '?' marks lines below {q.threshold:.2f}):")
        shown = q.lines if verbose else q.lines[:MAX_TEXT_LINES_SHOWN]
        prev_block = None
        for line in shown:
            if prev_block is not None and line.block_no != prev_block:
                out.append("   |")
            prev_block = line.block_no
            mark = "?" if not line.reliable else " "
            score = f"{line.score:.2f} " if verbose else ""
            out.append(f" {mark} | {score}{line.text}")
        if len(shown) < len(q.lines):
            out.append(f"   ... {len(q.lines) - len(shown)} more line(s) (use --verbose or --json-out)")
    out.append("")
    out.append(textwrap.fill(ir.ocr_confidence_note, WIDTH))
    out.append("")
    if ir.fact_check is None:
        out.append(textwrap.fill(f"NOT FACT-CHECKED: {ir.not_checked_reason}", WIDTH, subsequent_indent="  "))
        return "\n".join(out)
    extras = {t.claim_id: [_trace_line(t, q.threshold)] for t in ir.claim_traces}
    out.append(render_text(ir.fact_check, verbose=verbose, source_label="the image text", claim_extras=extras))
    return "\n".join(out)


def render_image_json(ir: ImageReport) -> str:
    return json.dumps(ir.model_dump(mode="json"), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- CLI (factcheck-ocr --fact-check)


def run_fact_check(args: argparse.Namespace) -> int:
    """Entry point used by ``factcheck-ocr --fact-check`` (arguments parsed by factcheck_ocr.cli)."""
    if len(args.images) != 1:
        print("error: --fact-check takes exactly one image", file=sys.stderr)
        return EXIT_INVALID
    for warning in envfile.load_env_file():
        print(f"warning: {warning}", file=sys.stderr)
    try:
        config = Config.from_env(model=args.model, max_checkable_claims=args.max_claims,
                                 search_credit_budget=args.search_budget, run_timeout_s=args.timeout)
        if args.json_out is not None and args.json_out.is_dir():
            raise InputError(f"--json-out points to a directory: {args.json_out}")
        ir = verify_image(args.images[0], config, engine_name=args.engine, ocr_threshold=args.ocr_threshold,
                          min_score=args.min_score, layout=args.layout, model_size=args.model_size,
                          allow_download=args.allow_download)
    except (InputError, ConfigError) as exc:
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
        if ir.fact_check is None:
            print(f"NOT_FACT_CHECKED: {ir.not_checked_reason}")
        else:
            print(f"{ir.fact_check.summary_label.value}: {ir.fact_check.summary_text}")
    else:
        print(render_image_report(ir, verbose=args.verbose))
    if args.json_out is not None:
        try:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(render_image_json(ir), encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write JSON report: {exc}", file=sys.stderr)
            return EXIT_FAILURE
    return exit_code_for(ir)
