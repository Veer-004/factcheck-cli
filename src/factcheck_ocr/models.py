"""Typed data models for OCR results.

``TextBox`` is what an engine returns for one detected text region. ``Line``,
``Block`` and ``PageResult`` are produced only by deterministic code in this
package (``layout.py`` / ``pipeline.py``). Recognised text is untrusted data: it
is never interpreted, only normalised for display.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# (x0, y0, x1, y1) in original-image pixel coordinates (after EXIF rotation).
BBox = tuple[float, float, float, float]


class TextBox(BaseModel):
    text: str
    score: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    polygon: list[tuple[float, float]] = Field(default_factory=list)
    # Set only by engines that classify layout regions (e.g. PP-StructureV3).
    label: Optional[str] = None

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class Line(BaseModel):
    """Boxes on the same visual row of one column, joined left to right."""

    text: str
    score: float
    bbox: BBox
    box_count: int


class Block(BaseModel):
    """Consecutive lines in reading order with no paragraph or column break between them."""

    text: str
    bbox: BBox
    lines: list[Line]
    label: Optional[str] = None


class PageStats(BaseModel):
    boxes_detected: int = 0
    boxes_kept: int = 0
    boxes_below_min_score: int = 0
    seconds: float = 0.0
    downscaled_from: Optional[tuple[int, int]] = None


class PageResult(BaseModel):
    source: str
    width: int
    height: int
    engine: str
    layout: str  # "geometric", "model" or "none"
    text: str
    blocks: list[Block]
    stats: PageStats
    warnings: list[str] = Field(default_factory=list)
