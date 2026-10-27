"""Reading-order recovery (pure geometry; no OCR engine or image needed)."""

from __future__ import annotations

import random

from factcheck_ocr.layout import build_lines, engine_order, reading_order
from factcheck_ocr.models import TextBox

H = 20  # line height used throughout


def box(text: str, x0: float, y0: float, x1: float | None = None, h: float = H, score: float = 0.95) -> TextBox:
    x1 = x0 + 10 * len(text) if x1 is None else x1
    return TextBox(text=text, score=score, bbox=(x0, y0, x1, y0 + h))


def texts(blocks) -> list[str]:
    return [b.text for b in blocks]


def shuffled(boxes, seed=7):
    out = list(boxes)
    random.Random(seed).shuffle(out)
    return out


def column(lines: list[str], x0: float, y0: float, width: float = 200, step: float = 28) -> list[TextBox]:
    return [box(t, x0, y0 + i * step, x0 + width) for i, t in enumerate(lines)]


def test_empty_and_blank_input():
    assert reading_order([]) == []
    assert reading_order([box("   ", 0, 0)]) == []


def test_single_column_paragraph_is_one_block_in_order():
    lines = ["The quick brown fox", "jumps over the", "lazy dog."]
    blocks = reading_order(shuffled(column(lines, 10, 10)))
    assert texts(blocks) == ["The quick brown fox\njumps over the\nlazy dog."]


def test_word_boxes_on_one_row_are_joined_left_to_right():
    row = [box("world", 80, 10), box("Hello", 10, 12)]  # slight baseline jitter
    blocks = reading_order(row)
    assert texts(blocks) == ["Hello world"]
    assert blocks[0].lines[0].box_count == 2


def test_narrow_gutter_does_not_merge_column_rows():
    left = column(["alpha one", "alpha two"], 10, 10, width=200)
    right = column(["beta one", "beta two"], 222, 10, width=200)  # 12px gutter = 0.6 line heights
    assert texts(reading_order(shuffled(left + right))) == ["alpha one\nalpha two", "beta one\nbeta two"]


def test_two_columns_under_a_spanning_title():
    title = [box("CITY COUNCIL APPROVES BUDGET", 10, 10, 460, h=30)]
    left = column(["Left one", "Left two", "Left three"], 10, 70)
    right = column(["Right one", "Right two", "Right three"], 260, 70)
    blocks = reading_order(shuffled(title + left + right))
    assert texts(blocks) == [
        "CITY COUNCIL APPROVES BUDGET",
        "Left one\nLeft two\nLeft three",
        "Right one\nRight two\nRight three",
    ]


def test_three_columns():
    cols = [column([f"c{c} l{i}" for i in range(4)], 10 + c * 230, 10) for c in range(3)]
    blocks = reading_order(shuffled([b for col in cols for b in col]))
    assert texts(blocks) == ["\n".join(f"c{c} l{i}" for i in range(4)) for c in range(3)]


def test_chat_bubbles_alternating_sides_read_top_to_bottom():
    bubbles = [
        box("hi are you coming", 10, 10, 250),
        box("yes leaving now", 400, 60, 640),
        box("great see you", 10, 110, 250),
        box("ok", 560, 160, 640),
    ]
    blocks = reading_order(shuffled(bubbles))
    assert texts(blocks) == ["hi are you coming", "yes leaving now", "great see you", "ok"]


def test_meme_top_and_bottom_text_are_separate_blocks():
    top = [box("WHEN THE CODE WORKS", 40, 10, 460, h=40)]
    bottom = [box("ON THE FIRST TRY", 70, 400, 430, h=40)]
    assert texts(reading_order(bottom + top)) == ["WHEN THE CODE WORKS", "ON THE FIRST TRY"]


def test_paragraph_gap_splits_blocks_within_a_column():
    first = column(["Para one a", "Para one b"], 10, 10)
    second = column(["Para two a", "Para two b"], 10, 120)
    assert texts(reading_order(second + first)) == ["Para one a\nPara one b", "Para two a\nPara two b"]


def test_poster_details_side_by_side_follow_headline():
    boxes = [
        box("SUMMER FESTIVAL", 50, 10, 450, h=50),
        box("Saturday 12 July", 10, 120, 200),
        box("Central Park", 300, 120, 480),
    ]
    assert texts(reading_order(shuffled(boxes))) == ["SUMMER FESTIVAL", "Saturday 12 July", "Central Park"]


def test_line_score_is_character_weighted():
    lines = build_lines([box("aaaa", 0, 0, score=1.0), box("b", 50, 0, score=0.5)])
    assert lines[0].score == round((4 * 1.0 + 1 * 0.5) / 5, 4)


def test_engine_order_keeps_engine_sequence_and_labels():
    boxes = [box("second", 0, 100), box("first", 0, 0)]
    boxes[0].label = "text"
    blocks = engine_order(boxes)
    assert texts(blocks) == ["second", "first"]
    assert blocks[0].label == "text"
