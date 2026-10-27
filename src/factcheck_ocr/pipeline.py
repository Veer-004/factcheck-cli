"""Image -> PageResult: load, recognise, filter, order, render text.

Kept separate from the fact-checking pipeline: nothing here imports ``factcheck``.
"""

from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

from .engines import EngineOutput, OcrEngine
from .image_io import MAX_SIDE, bbox_from_polygon, load_image
from .layout import engine_order, reading_order
from .models import Block, PageResult, PageStats, TextBox

DEFAULT_MIN_SCORE = 0.5
LAYOUTS = ("auto", "geometric", "none")

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩]")


def clean_text(text: str) -> str:
    """NFC-normalise and strip control/bidi-override characters (OCR text is untrusted)."""
    return _CONTROL.sub("", unicodedata.normalize("NFC", text)).strip()


def to_text_boxes(raw: EngineOutput, scale: float) -> list[TextBox]:
    boxes = []
    for rb in raw.boxes:
        text = clean_text(rb.text)
        if not text or not rb.points:
            continue
        bbox, poly = bbox_from_polygon(rb.points, scale)
        boxes.append(TextBox(text=text, score=min(1.0, max(0.0, rb.score)), bbox=bbox, polygon=poly, label=rb.label))
    return boxes


def render_blocks(blocks: list[Block]) -> str:
    return "\n\n".join(b.text for b in blocks)


def process_boxes(
    raw: EngineOutput, *, scale: float, min_score: float, layout: str
) -> tuple[list[Block], str, PageStats]:
    """Engine output -> (blocks, layout used, stats). Pure function; used by the tests."""
    boxes = to_text_boxes(raw, scale)
    kept = [b for b in boxes if b.score >= min_score]
    stats = PageStats(boxes_detected=len(boxes), boxes_kept=len(kept), boxes_below_min_score=len(boxes) - len(kept))
    if layout == "none":
        return engine_order(kept), "none", stats
    if layout == "auto" and raw.ordered:
        return _model_blocks(kept), "model", stats
    return reading_order(kept), "geometric", stats


def _model_blocks(boxes: list[TextBox]) -> list[Block]:
    """Layout-model output: each region is already a block, in the model's reading order."""
    return engine_order(boxes)


def ocr_image(path: Path, engine: OcrEngine, *, min_score: float = DEFAULT_MIN_SCORE,
              layout: str = "auto", max_side: int = MAX_SIDE) -> PageResult:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    loaded = load_image(path, max_side=max_side)
    started = time.perf_counter()
    raw = engine.recognize(loaded.image)
    blocks, used, stats = process_boxes(raw, scale=loaded.scale, min_score=min_score, layout=layout)
    stats.seconds = round(time.perf_counter() - started, 3)
    if loaded.scale < 1.0:
        stats.downscaled_from = (loaded.width, loaded.height)
    return PageResult(
        source=path.name,
        width=loaded.width,
        height=loaded.height,
        engine=engine.name,
        layout=used,
        text=render_blocks(blocks),
        blocks=blocks,
        stats=stats,
        warnings=loaded.warnings + raw.notes,
    )
