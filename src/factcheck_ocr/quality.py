"""Extraction-quality assessment of an OCR result (deterministic, no model).

OCR confidence says how sure the engine is that it read the *characters*
correctly. It says nothing about whether the text is true, and it is never used
as evidence or mixed into a fact-check score. It is also an imperfect signal: a
misread line can still score highly (see the README's meme finding), so "OK"
means "no warning signs", not "verified correct".

Statuses:
  NO_TEXT          nothing was recognised
  UNREADABLE       text was found but every line is below the reliability
                   threshold, or it contains no words
  LOW_CONFIDENCE   some lines are below the threshold, or low-confidence lines
                   were dropped by --min-score
  OK               every kept line meets the threshold
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

from .models import BBox, PageResult

DEFAULT_RELIABLE_SCORE = 0.80


class ExtractionStatus(str, Enum):
    OK = "OK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNREADABLE = "UNREADABLE"
    NO_TEXT = "NO_TEXT"


class OcrLine(BaseModel):
    """One line of the extracted text, in reading order."""

    line_no: int  # 1-based over the whole page
    block_no: int  # 1-based
    text: str
    score: float
    bbox: BBox
    reliable: bool


class ExtractionQuality(BaseModel):
    status: ExtractionStatus
    threshold: float
    lines: list[OcrLine] = Field(default_factory=list)
    unreliable_lines: int = 0
    dropped_low_confidence: int = 0
    min_score: float | None = None
    mean_score: float | None = None
    problems: list[str] = Field(default_factory=list)


def assess(page: PageResult, threshold: float = DEFAULT_RELIABLE_SCORE) -> ExtractionQuality:
    lines: list[OcrLine] = []
    for b_no, block in enumerate(page.blocks, 1):
        for line in block.lines:
            lines.append(OcrLine(line_no=len(lines) + 1, block_no=b_no, text=line.text, score=line.score,
                                 bbox=line.bbox, reliable=line.score >= threshold))
    dropped = page.stats.boxes_below_min_score
    q = ExtractionQuality(status=ExtractionStatus.OK, threshold=threshold, lines=lines, dropped_low_confidence=dropped)
    if not lines:
        q.status = ExtractionStatus.NO_TEXT
        q.problems.append("no text was recognised in the image"
                          + (f" ({dropped} low-confidence fragment(s) were discarded)" if dropped else ""))
        return q
    scores = [l.score for l in lines]
    q.min_score, q.mean_score = round(min(scores), 4), round(sum(scores) / len(scores), 4)
    q.unreliable_lines = sum(not l.reliable for l in lines)
    if not re.search(r"[^\W\d_]{2,}", page.text):
        q.status = ExtractionStatus.UNREADABLE
        q.problems.append("the recognised text contains no words")
    elif q.unreliable_lines == len(lines):
        q.status = ExtractionStatus.UNREADABLE
        q.problems.append(f"every line is below the OCR confidence threshold of {threshold:.2f}")
    elif q.unreliable_lines or dropped:
        q.status = ExtractionStatus.LOW_CONFIDENCE
        if q.unreliable_lines:
            q.problems.append(f"{q.unreliable_lines} of {len(lines)} line(s) are below the OCR confidence "
                              f"threshold of {threshold:.2f}")
        if dropped:
            q.problems.append(f"{dropped} low-confidence fragment(s) were discarded, so some words may be missing")
    return q
