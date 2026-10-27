"""OCR extraction-quality assessment (pure; no engine)."""

from __future__ import annotations

from factcheck_ocr.layout import reading_order
from factcheck_ocr.models import PageResult, PageStats, TextBox
from factcheck_ocr.pipeline import render_blocks
from factcheck_ocr.quality import ExtractionStatus, assess


def page(*lines, dropped=0) -> PageResult:
    boxes = [TextBox(text=t, score=s, bbox=(10, 10 + i * 30, 300, 34 + i * 30)) for i, (t, s) in enumerate(lines)]
    blocks = reading_order(boxes)
    return PageResult(source="x.png", width=400, height=400, engine="t", layout="geometric",
                      text=render_blocks(blocks), blocks=blocks, stats=PageStats(boxes_below_min_score=dropped))


def test_ok():
    q = assess(page(("Hello world", 0.95), ("second line", 0.9)))
    assert q.status == ExtractionStatus.OK and q.problems == []
    assert [l.line_no for l in q.lines] == [1, 2] and q.min_score == 0.9


def test_no_text():
    q = assess(page(dropped=3))
    assert q.status == ExtractionStatus.NO_TEXT and "3 low-confidence fragment" in q.problems[0]


def test_unreadable_when_all_lines_doubtful_or_no_words():
    assert assess(page(("hello", 0.5), ("world", 0.6))).status == ExtractionStatus.UNREADABLE
    assert assess(page(("12 34 %", 0.99))).status == ExtractionStatus.UNREADABLE


def test_low_confidence_for_some_lines_or_dropped_fragments():
    q = assess(page(("good line", 0.95), ("bad line", 0.7)))
    assert q.status == ExtractionStatus.LOW_CONFIDENCE and q.unreliable_lines == 1
    assert [l.reliable for l in q.lines] == [True, False]
    q = assess(page(("good line", 0.95), dropped=2))
    assert q.status == ExtractionStatus.LOW_CONFIDENCE and "may be missing" in q.problems[0]


def test_threshold_parameter():
    assert assess(page(("line", 0.75)), threshold=0.7).status == ExtractionStatus.OK
