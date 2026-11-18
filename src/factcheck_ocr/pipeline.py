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
# Symbol variants the recogniser emits for ordinary English text: "4000°℃" -> "4000°C",
# full-width ASCII ("２０２４") -> "2024". Deliberately not full NFKC, which would also
# turn "10³" into "103".
_SYMBOLS = str.maketrans({"℃": "°C", "℉": "°F", **{chr(c): chr(c - 0xFEE0) for c in range(0xFF01, 0xFF5F)}})
_DOUBLE_DEGREE = re.compile(r"°\s*°")
# A standalone "Il" is not English; it is the recogniser's reading of the Roman numeral
# "II" ("Pokhran Il", "Agni Il" in the real-frame evaluation).
_ROMAN_II = re.compile(r"(?<![A-Za-z])Il(?![A-Za-z])")


def clean_text(text: str) -> str:
    """NFC-normalise, fix symbol variants and strip control/bidi-override characters (OCR text is untrusted)."""
    text = unicodedata.normalize("NFC", text).translate(_SYMBOLS)
    text = _ROMAN_II.sub("II", _DOUBLE_DEGREE.sub("°", text))
    return _CONTROL.sub("", text).strip()


_LATIN_ALNUM = re.compile(r"[A-Za-z0-9]")
_REPEATED_RUN = re.compile(r"([A-Za-z])\1\1", re.IGNORECASE)
# Two-letter English words and common abbreviations that are real text even on their own.
_SHORT_WORDS = frozenset(
    """am an as at be by do go he hi if in is it me my no of oh ok on or so to up us we
    ad ai dr eu ex id mr ms pm st tv uk un vs""".split())


def is_noise(text: str) -> bool:
    """True for recognised fragments that are almost never real English text.

    Measured on real video frames (eval/run_real_ocr_eval.py), these come from
    logos, icons, textures and partly hidden letters:
      * no Latin letter or digit at all ("难", "三", "★", "??");
      * a single letter on its own ("X", "S", "Q"), or two letters that are not a common
        word or abbreviation ("um", "TE"; "Hi", "OK", "US", "BE" are kept); a box that is
        only digits ("1", "2", "9") is always kept, because figures matter;
      * every word is a short run of one repeated letter or contains three identical
        letters in a row ("IIII", "mwwwmm").
    """
    s = text.strip()
    latin = _LATIN_ALNUM.findall(s)
    if not latin:
        return True
    short = "".join(latin)
    if len(short) == 1 and not short.isdigit():
        return True
    if len(short) == 2 and not short.isdigit() and short.lower() not in _SHORT_WORDS:
        return True
    words = [re.sub(r"[^A-Za-z]", "", w) for w in s.split()]
    words = [w for w in words if w]
    if words and len(words) == len(s.split()) and all(len(w) <= 8 and _REPEATED_RUN.search(w) for w in words):
        return True
    return False


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


SMALL_TEXT_PX = 11  # text boxes shorter than this (in the pixels the recogniser saw) are tiny
SMALL_FIGURE_SCORE = 0.6  # confidence cap for figures in tiny text: below the 0.80 reliability threshold


def cap_small_figures(boxes: list[TextBox], scale: float) -> tuple[list[TextBox], int]:
    """Lower the confidence of boxes that contain digits but are only a few pixels tall.

    The recogniser reports high confidence for misread digits in tiny text: on the
    real-frame evaluation, a 10-px-tall "$1.5 billion" was read as "$1.8 billion" at
    0.92, the only misread figure among 45 (the next smallest, 11-12 px, were right).
    The text is kept as read; only its confidence is capped, so reports mark it with
    "?" and the fact-check gates withhold claims that depend on it.
    """
    out, capped = [], 0
    for b in boxes:
        height = (b.bbox[3] - b.bbox[1]) * scale
        if height < SMALL_TEXT_PX and any(c.isdigit() for c in b.text) and b.score > SMALL_FIGURE_SCORE:
            b = b.model_copy(update={"score": SMALL_FIGURE_SCORE})
            capped += 1
        out.append(b)
    return out, capped


def process_boxes(
    raw: EngineOutput, *, scale: float, min_score: float, layout: str, noise_filter: bool = True
) -> tuple[list[Block], str, PageStats]:
    """Engine output -> (blocks, layout used, stats). Pure function; used by the tests."""
    boxes = to_text_boxes(raw, scale)
    kept = [b for b in boxes if b.score >= min_score]
    below = len(boxes) - len(kept)
    noise = [b for b in kept if is_noise(b.text)] if noise_filter else []
    if noise:
        kept = [b for b in kept if not is_noise(b.text)]
    kept, small = cap_small_figures(kept, scale)
    stats = PageStats(boxes_detected=len(boxes), boxes_kept=len(kept), boxes_below_min_score=below,
                      boxes_filtered_noise=len(noise), boxes_small_figures=small)
    if layout == "none":
        return engine_order(kept), "none", stats
    if layout == "auto" and raw.ordered:
        return _model_blocks(kept), "model", stats
    return reading_order(kept), "geometric", stats


def _model_blocks(boxes: list[TextBox]) -> list[Block]:
    """Layout-model output: each region is already a block, in the model's reading order."""
    return engine_order(boxes)


def ocr_image(path: Path, engine: OcrEngine, *, min_score: float = DEFAULT_MIN_SCORE,
              layout: str = "auto", max_side: int = MAX_SIDE, noise_filter: bool = True) -> PageResult:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    loaded = load_image(path, max_side=max_side)
    started = time.perf_counter()
    raw = engine.recognize(loaded.image)
    blocks, used, stats = process_boxes(raw, scale=loaded.scale, min_score=min_score, layout=layout,
                                        noise_filter=noise_filter)
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
        warnings=loaded.warnings + raw.notes + filter_notes(stats),
    )


def filter_notes(stats: PageStats) -> list[str]:
    """Human-readable notes for what the English clean-up changed (never silent)."""
    notes = []
    if stats.boxes_filtered_noise:
        notes.append(f"{stats.boxes_filtered_noise} non-text fragment(s) (logos, icons, stray letters) removed as noise")
    if stats.boxes_small_figures:
        notes.append(f"{stats.boxes_small_figures} figure(s) in very small text marked uncertain (digits there are "
                     "often misread)")
    return notes
