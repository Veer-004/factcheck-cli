"""English OCR clean-up (factcheck_ocr.pipeline): noise filter, symbol fixes, tiny-figure confidence cap.

The cases come from real video frames (eval/run_real_ocr_eval.py): what the
recogniser produced for logos, icons, textures and tiny text.
"""

from __future__ import annotations

import pytest

from factcheck_ocr.engines import EngineOutput, RawBox
from factcheck_ocr.pipeline import SMALL_FIGURE_SCORE, clean_text, filter_notes, is_noise, process_boxes
from factcheck_ocr.quality import ExtractionStatus, assess


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@pytest.mark.parametrize("text", ["难", "三", "园", "★", "??", "X", "S", "Q", "T", "E", "um", "TE", "gm",
                                  "IIII", "mwwwmm", "mmm"])
def test_noise_fragments_are_detected(text):
    assert is_noise(text)


@pytest.mark.parametrize("text", [
    "1", "2", "9", "42",  # figures are always kept
    "Hi", "OK", "No", "US", "UK", "BE", "AI",  # real two-letter words / abbreviations
    "SPACEX", "JAMAICA", "$270bn", "4.43s", "100m", "69%", "e-scooter limit", "June 28, 1971",
    "Coffee", "Mississippi", "www.time.com", "Hello world",
])
def test_real_text_is_kept(text):
    assert not is_noise(text)


def test_symbol_variants_and_roman_numerals_are_fixed():
    assert clean_text("4000°℃") == "4000°C"
    assert clean_text("98.6℉") == "98.6°F"
    assert clean_text("２０２４ ＡＤ") == "2024 AD"
    assert clean_text("Pokhran Il tests") == "Pokhran II tests"
    assert clean_text("Agni Il Agni III") == "Agni II Agni III"
    # not touched: "Il" inside words, superscripts (full NFKC would turn 10³ into 103)
    assert clean_text("Illinois") == "Illinois"
    assert clean_text("10³ km²") == "10³ km²"


def test_noise_is_removed_and_counted():
    raw = EngineOutput(boxes=[RawBox("The fastest", 0.97, rect(10, 10, 200, 40)),
                              RawBox("难", 0.9, rect(10, 60, 30, 80)),
                              RawBox("mwwwmm", 0.6, rect(10, 100, 90, 130))])
    blocks, _, stats = process_boxes(raw, scale=1.0, min_score=0.5, layout="geometric")
    assert [l.text for b in blocks for l in b.lines] == ["The fastest"]
    assert stats.boxes_filtered_noise == 2 and stats.boxes_kept == 1
    assert "2 non-text fragment(s)" in filter_notes(stats)[0]


def test_noise_filter_can_be_disabled():
    raw = EngineOutput(boxes=[RawBox("难", 0.9, rect(10, 60, 30, 80))])
    blocks, _, stats = process_boxes(raw, scale=1.0, min_score=0.5, layout="geometric", noise_filter=False)
    assert blocks and stats.boxes_filtered_noise == 0


def test_figures_in_tiny_text_are_marked_uncertain_not_changed():
    raw = EngineOutput(boxes=[RawBox("$1.8 billion", 0.92, rect(190, 165, 260, 175)),  # 10 px tall
                              RawBox("PURCHASE PRICE:", 0.98, rect(190, 140, 300, 152)),  # tiny, but no digits
                              RawBox("$307 million", 0.99, rect(190, 180, 260, 192))])  # 12 px tall
    blocks, _, stats = process_boxes(raw, scale=1.0, min_score=0.5, layout="none")
    lines = {l.text: l.score for b in blocks for l in b.lines}
    assert lines["$1.8 billion"] == SMALL_FIGURE_SCORE  # text kept as read, confidence capped
    assert lines["PURCHASE PRICE:"] == 0.98 and lines["$307 million"] == 0.99
    assert stats.boxes_small_figures == 1
    assert "1 figure(s) in very small text marked uncertain" in filter_notes(stats)[0]


def test_tiny_text_size_is_measured_in_recogniser_pixels():
    # A 20-px box in an image downscaled by half was only 10 px for the recogniser.
    raw = EngineOutput(boxes=[RawBox("1998", 0.99, rect(10, 10, 60, 20))])  # engine pixels
    _, _, stats = process_boxes(raw, scale=0.5, min_score=0.5, layout="none")
    assert stats.boxes_small_figures == 1


def test_capped_figure_makes_quality_low_confidence():
    from factcheck_ocr.models import PageResult

    raw = EngineOutput(boxes=[RawBox("PURCHASE PRICE:", 0.98, rect(10, 10, 200, 40)),
                              RawBox("$1.8 billion", 0.92, rect(10, 50, 90, 60))])
    blocks, used, stats = process_boxes(raw, scale=1.0, min_score=0.5, layout="geometric")
    page = PageResult(source="x", width=400, height=300, engine="fake", layout=used,
                      text="\n\n".join(b.text for b in blocks), blocks=blocks, stats=stats)
    q = assess(page, 0.80)
    assert q.status == ExtractionStatus.LOW_CONFIDENCE
    assert [l.text for l in q.lines if not l.reliable] == ["$1.8 billion"]
