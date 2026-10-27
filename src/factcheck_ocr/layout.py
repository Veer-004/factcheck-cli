"""Deterministic reading-order recovery from OCR text boxes.

OCR engines return text *regions* (usually one per visual line segment) in an
order that is not reliable for multi-column pages, posters or screenshots. This
module rebuilds a reading order from geometry alone:

1. **Recursive XY-cut.** The boxes are split recursively along empty
   horizontal bands (top before bottom) or empty vertical gutters (left before
   right). When both cuts exist, a vertical cut is taken only when the two sides
   really run side by side (most boxes of the smaller side share a row with a
   box on the other side). That keeps newspaper-style columns together while
   chat screenshots, whose left and right bubbles alternate, are read top to
   bottom. Columns are found before anything is joined, so a narrow gutter
   never merges two columns' rows.
2. **Lines.** Consecutive boxes in that order that share a row and are a word
   gap apart are joined left to right into a ``Line``.
3. **Blocks.** Consecutive lines in that order are grouped into a ``Block``
   unless there is a paragraph-sized vertical gap, a jump back up the page (a new
   column) or no horizontal overlap.

Limits (documented in the README): it assumes left-to-right, top-to-bottom
scripts and roughly horizontal text; it cannot separate columns that the
detector itself merged into one box; and text inside figures is ordered by
position like any other text.
"""

from __future__ import annotations

from statistics import median
from typing import Iterable, Optional, Sequence, TypeVar

from .models import BBox, Block, Line, TextBox

T = TypeVar("T", TextBox, Line)

# Fractions of the median line height.
ROW_OVERLAP = 0.5  # vertical overlap (of the shorter box) for two boxes to share a row
WORD_GAP = 1.6  # max horizontal gap between boxes joined into one line
EDGE_SHRINK = 0.15  # boxes are shrunk vertically by this much before projecting (detector boxes overlap slightly)
MIN_GUTTER = 0.6  # min width of an empty vertical band to cut columns
PARAGRAPH_GAP = 0.9  # vertical gap between lines that starts a new block
SIDE_BY_SIDE = 0.5  # share of the smaller side's lines that must share a row with the other side


def union(boxes: Iterable[BBox]) -> BBox:
    bs = list(boxes)
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def _v_overlap(a: BBox, b: BBox) -> float:
    """Vertical overlap as a fraction of the shorter box height."""
    inter = min(a[3], b[3]) - max(a[1], b[1])
    shorter = min(a[3] - a[1], b[3] - b[1])
    return inter / shorter if shorter > 0 else 0.0


def _h_overlap(a: BBox, b: BBox) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0])


def _median_height(boxes: list[TextBox]) -> float:
    heights = [b.height for b in boxes if b.height > 0]
    return median(heights) if heights else 1.0


def build_lines(ordered: Sequence[TextBox], unit: Optional[float] = None) -> list[Line]:
    """Join consecutive boxes (already in reading order) that share a row and are a word gap apart."""
    unit = unit or _median_height(list(ordered))
    rows: list[list[TextBox]] = []
    for box in ordered:
        if rows:
            last = rows[-1][-1]
            gap = box.bbox[0] - last.bbox[2]
            same_row = _v_overlap(box.bbox, last.bbox) >= ROW_OVERLAP
            if same_row and -0.5 * min(box.width, last.width) <= gap <= WORD_GAP * unit:
                rows[-1].append(box)
                continue
        rows.append([box])
    lines = []
    for row in rows:
        text = " ".join(b.text.strip() for b in row if b.text.strip())
        if not text:
            continue
        chars = sum(len(b.text) for b in row) or 1
        score = sum(b.score * len(b.text) for b in row) / chars  # character-weighted
        lines.append(Line(text=text, score=round(score, 4), bbox=union(b.bbox for b in row), box_count=len(row)))
    return lines


def _shrunk(b: BBox) -> tuple[float, float]:
    pad = (b[3] - b[1]) * EDGE_SHRINK
    return b[1] + pad, b[3] - pad


def _gaps(intervals: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Empty bands (start, end) of at least ``min_gap`` between the union of intervals."""
    out = []
    ivs = sorted(intervals)
    reach = ivs[0][1]
    for start, end in ivs[1:]:
        if start - reach >= min_gap:
            out.append((reach, start))
        reach = max(reach, end)
    return out


def _widest(gaps: list[tuple[float, float]]) -> Optional[float]:
    if not gaps:
        return None
    start, end = max(gaps, key=lambda g: g[1] - g[0])
    return (start + end) / 2


def _side_by_side(left: Sequence[T], right: Sequence[T]) -> bool:
    small, other = (left, right) if len(left) <= len(right) else (right, left)
    paired = sum(1 for a in small if any(_v_overlap(a.bbox, b.bbox) >= ROW_OVERLAP for b in other))
    return paired / len(small) >= SIDE_BY_SIDE


def xy_cut(items: Sequence[T], unit: float) -> list[T]:
    """Order boxes (or lines) by recursive XY-cut (see module docstring)."""
    if len(items) <= 1:
        return list(items)
    h_cut = _widest(_gaps([_shrunk(i.bbox) for i in items], min_gap=1e-6))
    v_cut = _widest(_gaps([(i.bbox[0], i.bbox[2]) for i in items], min_gap=MIN_GUTTER * unit))
    if v_cut is not None:
        left = [i for i in items if (i.bbox[0] + i.bbox[2]) / 2 < v_cut]
        right = [i for i in items if (i.bbox[0] + i.bbox[2]) / 2 >= v_cut]
        if h_cut is None or _side_by_side(left, right):
            return xy_cut(left, unit) + xy_cut(right, unit)
    if h_cut is not None:
        top = [i for i in items if sum(_shrunk(i.bbox)) / 2 < h_cut]
        bottom = [i for i in items if sum(_shrunk(i.bbox)) / 2 >= h_cut]
        if top and bottom:
            return xy_cut(top, unit) + xy_cut(bottom, unit)
    # No usable cut (boxes overlap in both directions): top-to-bottom, then left-to-right.
    return sorted(items, key=lambda i: (i.bbox[1], i.bbox[0]))


def group_blocks(ordered: list[Line], unit: float) -> list[Block]:
    blocks: list[list[Line]] = []
    for line in ordered:
        if blocks:
            prev = blocks[-1][-1]
            gap = line.bbox[1] - prev.bbox[3]
            new_column = line.bbox[1] < prev.bbox[1] - 0.5 * unit
            if gap <= PARAGRAPH_GAP * unit and not new_column and _h_overlap(line.bbox, prev.bbox):
                blocks[-1].append(line)
                continue
        blocks.append([line])
    return [
        Block(text="\n".join(l.text for l in group), bbox=union(l.bbox for l in group), lines=group)
        for group in blocks
    ]


def reading_order(boxes: list[TextBox]) -> list[Block]:
    """Boxes (any order) -> blocks in reading order. Empty input gives no blocks."""
    boxes = [b for b in boxes if b.text.strip()]
    if not boxes:
        return []
    unit = _median_height(boxes)
    return group_blocks(build_lines(xy_cut(boxes, unit), unit), unit)


def engine_order(boxes: list[TextBox]) -> list[Block]:
    """No layout analysis: one block per box, in the order the engine returned them."""
    return [
        Block(
            text=b.text.strip(),
            bbox=b.bbox,
            lines=[Line(text=b.text.strip(), score=b.score, bbox=b.bbox, box_count=1)],
            label=b.label,
        )
        for b in boxes
        if b.text.strip()
    ]
