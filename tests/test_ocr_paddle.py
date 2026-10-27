"""PaddleOCR adapter: result conversion and the no-download guard (Paddle itself is not needed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from factcheck_ocr import paddle_engine
from factcheck_ocr.engines import EngineUnavailable


def test_text_output_converts_polygons():
    res = {"rec_texts": ["Hello"], "rec_scores": [0.9], "rec_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]]}
    out = paddle_engine.text_output(res)
    assert not out.ordered
    assert out.boxes[0].text == "Hello" and out.boxes[0].points[2] == (10.0, 5.0)


def test_structure_output_keeps_model_order_labels_and_line_scores():
    blocks = [
        SimpleNamespace(label="doc_title", content="Title", bbox=[0, 0, 100, 20]),
        SimpleNamespace(label="image", content="", bbox=[0, 30, 100, 90]),
        SimpleNamespace(label="text", content="Body line one\nBody line two", bbox=[0, 100, 100, 150]),
    ]
    res = {
        "parsing_res_list": blocks,
        "overall_ocr_res": {"rec_boxes": [[0, 0, 100, 20], [0, 100, 100, 120], [0, 125, 100, 145]],
                            "rec_scores": [0.9, 0.8, 0.6]},
    }
    out = paddle_engine.structure_output(res)
    assert out.ordered
    assert [(b.text, b.label) for b in out.boxes] == [("Title", "doc_title"), ("Body line one\nBody line two", "text")]
    assert out.boxes[1].score == pytest.approx(0.7)


def test_missing_models_uses_paddlex_cache_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path))
    (tmp_path / "official_models" / "PP-OCRv6_medium_det").mkdir(parents=True)
    assert paddle_engine.missing_models(paddle_engine.TEXT_MODELS) == [
        "PP-LCNet_x1_0_textline_ori", "PP-OCRv6_medium_rec"]


def test_engine_refuses_to_download_without_permission(tmp_path, monkeypatch):
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path))  # empty cache
    with pytest.raises(EngineUnavailable) as err:
        paddle_engine.PaddleEngine(structure=False)
    # Either Paddle is not installed in this venv, or the models are missing; never a download.
    assert "not installed" in str(err.value) or "--allow-download" in str(err.value)
    assert not (tmp_path / "official_models").exists()
