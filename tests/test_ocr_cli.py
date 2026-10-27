"""OCR pipeline, image loading and CLI with a fake engine (no OCR models, no network)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from factcheck_ocr import cli, engines
from factcheck_ocr.engines import EngineFailure, EngineOutput, EngineUnavailable, RawBox
from factcheck_ocr.pipeline import clean_text, process_boxes

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from factcheck_ocr.image_io import ImageInputError, load_image  # noqa: E402


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


class FakeEngine:
    name = "fake"

    def __init__(self, boxes=None, fail=False, ordered=False):
        self.boxes, self.fail, self.ordered, self.seen = boxes or [], fail, ordered, []

    def recognize(self, image):
        self.seen.append(image.size)
        if self.fail:
            raise EngineFailure("boom")
        return EngineOutput(boxes=list(self.boxes), ordered=self.ordered)


@pytest.fixture(autouse=True)
def _no_engine_env(monkeypatch):
    monkeypatch.delenv("FACTCHECK_OCR_ENGINE", raising=False)


@pytest.fixture
def png(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (400, 300), "white").save(path)
    return path


def use_engine(monkeypatch, engine):
    monkeypatch.setattr(cli, "create_engine", lambda name, **kw: engine)


# ------------------------------------------------------------- pipeline


def test_min_score_filters_and_counts():
    raw = EngineOutput(boxes=[RawBox("keep", 0.9, rect(0, 0, 40, 20)), RawBox("drop", 0.2, rect(0, 40, 40, 60))])
    blocks, layout, stats = process_boxes(raw, scale=1.0, min_score=0.5, layout="auto")
    assert [b.text for b in blocks] == ["keep"]
    assert (stats.boxes_detected, stats.boxes_kept, stats.boxes_below_min_score) == (2, 1, 1)
    assert layout == "geometric"


def test_coordinates_are_mapped_back_to_original_scale():
    raw = EngineOutput(boxes=[RawBox("x", 0.9, rect(10, 20, 30, 40))])
    blocks, _, _ = process_boxes(raw, scale=0.5, min_score=0.0, layout="none")
    assert blocks[0].bbox == (20.0, 40.0, 60.0, 80.0)


def test_model_order_used_when_engine_provides_it():
    raw = EngineOutput(boxes=[RawBox("below", 0.9, rect(0, 100, 50, 120)), RawBox("above", 0.9, rect(0, 0, 50, 20))],
                       ordered=True)
    blocks, layout, _ = process_boxes(raw, scale=1.0, min_score=0.0, layout="auto")
    assert (layout, [b.text for b in blocks]) == ("model", ["below", "above"])
    blocks, layout, _ = process_boxes(raw, scale=1.0, min_score=0.0, layout="geometric")
    assert (layout, [b.text for b in blocks]) == ("geometric", ["above", "below"])


def test_clean_text_strips_terminal_escapes_and_bidi_overrides():
    assert clean_text("\x1b[31mred\x1b[0m‮evil‬ ok\n") == "[31mred[0mevil ok"
    assert clean_text("café") == "café"  # NFC


# ------------------------------------------------------------- image loading


def test_exif_rotation_is_applied(tmp_path):
    path = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    Image.new("RGB", (40, 20), "white").save(path, exif=exif)
    loaded = load_image(path)
    assert (loaded.width, loaded.height) == (20, 40)


def test_transparency_is_flattened_to_white(tmp_path):
    path = tmp_path / "t.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(path)
    loaded = load_image(path)
    assert loaded.image.mode == "RGB" and loaded.image.getpixel((5, 5)) == (255, 255, 255)


def test_large_image_is_downscaled(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (5000, 100), "white").save(path)
    loaded = load_image(path, max_side=1000)
    assert loaded.image.size == (1000, 20) and loaded.scale == pytest.approx(0.2)
    assert (loaded.width, loaded.height) == (5000, 100)


@pytest.mark.parametrize("content", [b"", b"this is not an image"])
def test_bad_files_are_input_errors(tmp_path, content):
    path = tmp_path / "bad.png"
    path.write_bytes(content)
    with pytest.raises(ImageInputError):
        load_image(path)


def test_pixel_limit_checked_before_decode(tmp_path, monkeypatch):
    from factcheck_ocr import image_io

    path = tmp_path / "big.png"
    Image.new("RGB", (300, 300), "white").save(path)
    monkeypatch.setattr(image_io, "MAX_PIXELS", 10_000)
    with pytest.raises(ImageInputError, match="too large"):
        load_image(path)


def test_unsupported_format_rejected(tmp_path):
    path = tmp_path / "x.ico"
    Image.new("RGB", (16, 16), "white").save(path, format="ICO")
    with pytest.raises(ImageInputError, match="unsupported"):
        load_image(path)


# ------------------------------------------------------------- CLI


def test_cli_prints_text_and_writes_json(monkeypatch, png, tmp_path, capsys):
    use_engine(monkeypatch, FakeEngine([RawBox("Hello", 0.99, rect(10, 10, 60, 30)),
                                        RawBox("world", 0.98, rect(70, 10, 130, 30))]))
    out = tmp_path / "ocr.json"
    assert cli.main([str(png), "--json-out", str(out)]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == "Hello world"
    data = json.loads(out.read_text(encoding="utf-8"))
    page = data["pages"][0]
    assert page["text"] == "Hello world" and page["width"] == 400 and page["engine"] == "fake"
    assert page["blocks"][0]["lines"][0]["box_count"] == 2
    assert data["errors"] == []


def test_cli_no_text_exit_1(monkeypatch, png, capsys):
    use_engine(monkeypatch, FakeEngine([]))
    assert cli.main([str(png)]) == cli.EXIT_NO_TEXT
    assert "(no text found)" in capsys.readouterr().out


def test_cli_missing_file_exit_2(monkeypatch, tmp_path, capsys):
    use_engine(monkeypatch, FakeEngine())
    assert cli.main([str(tmp_path / "nope.png")]) == cli.EXIT_INVALID
    assert "not found" in capsys.readouterr().err


def test_cli_engine_unavailable_exit_2(monkeypatch, png, capsys):
    def unavailable(name, **kw):
        raise EngineUnavailable("RapidOCR is not installed")

    monkeypatch.setattr(cli, "create_engine", unavailable)
    assert cli.main([str(png)]) == cli.EXIT_INVALID
    assert "not installed" in capsys.readouterr().err


def test_cli_engine_failure_exit_3_and_continues(monkeypatch, png, tmp_path, capsys):
    use_engine(monkeypatch, FakeEngine(fail=True))
    assert cli.main([str(png), str(png)]) == cli.EXIT_FAILURE
    assert capsys.readouterr().err.count("boom") == 2


def test_cli_bad_image_among_good_exit_2(monkeypatch, png, tmp_path, capsys):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    use_engine(monkeypatch, FakeEngine([RawBox("Hi", 0.9, rect(0, 0, 20, 10))]))
    assert cli.main([str(png), str(bad)]) == cli.EXIT_INVALID
    captured = capsys.readouterr()
    assert "=== page.png ===" in captured.out and "Hi" in captured.out
    assert "cannot read image" in captured.err


def test_cli_verbose_shows_scores_and_low_confidence_note(monkeypatch, png, capsys):
    use_engine(monkeypatch, FakeEngine([RawBox("sure", 0.97, rect(0, 0, 40, 20)),
                                        RawBox("blur", 0.10, rect(0, 50, 40, 70))]))
    assert cli.main([str(png)]) == cli.EXIT_OK
    assert "1 low-confidence line(s) omitted" in capsys.readouterr().out
    cli.main([str(png), "-v"])
    out = capsys.readouterr().out
    assert "0.97  [0,0,40,20]  sure" in out and "lines kept 1/2" in out


@pytest.mark.parametrize("args", [["--min-score", "2"], ["--max-side", "10"], []])
def test_cli_rejects_bad_options(monkeypatch, png, args):
    use_engine(monkeypatch, FakeEngine())
    argv = args + ([str(png)] if args else [])
    assert cli.main(argv) == cli.EXIT_INVALID


def test_cli_check_does_not_load_an_engine(monkeypatch, capsys):
    monkeypatch.setattr(cli, "create_engine", lambda *a, **k: pytest.fail("engine loaded"))
    assert cli.main(["--check"]) == cli.EXIT_OK
    assert "rapidocr:" in capsys.readouterr().out


needs_rapidocr = pytest.mark.skipif(
    not (engines.importlib.util.find_spec("rapidocr") and engines.importlib.util.find_spec("onnxruntime")),
    reason="rapidocr/onnxruntime not installed")


@needs_rapidocr
def test_rapid_medium_refuses_to_download_without_permission(monkeypatch, tmp_path):
    monkeypatch.setattr(engines, "rapid_models", lambda size="small": [tmp_path / f"PP-OCRv6_det_{size}.onnx"])
    with pytest.raises(EngineUnavailable, match="--allow-download"):
        engines.RapidEngine(model_size="medium")
    with pytest.raises(EngineUnavailable, match="reinstall rapidocr"):  # bundled models are never fetched
        engines.RapidEngine(model_size="small", allow_download=True)


@needs_rapidocr
def test_rapid_orientation_classifier_off_by_default(monkeypatch):
    seen = {}

    class FakeRapid:
        def __init__(self, params):
            seen.update(params)

    import rapidocr

    monkeypatch.setattr(rapidocr, "RapidOCR", FakeRapid, raising=False)
    monkeypatch.setattr(engines, "rapid_models", lambda size="small": [])
    engines.RapidEngine()
    assert seen["Global.use_cls"] is False and seen["Global.text_score"] == 0.0
    engines.RapidEngine(textline_orientation=True)
    assert seen["Global.use_cls"] is True


def test_unknown_engine_name():
    with pytest.raises(EngineUnavailable, match="unknown OCR engine"):
        engines.create_engine("tesseract")


def test_ocr_package_is_independent_of_factcheck():
    code = "import sys, factcheck_ocr.cli; print(sorted(m for m in ('factcheck', 'langgraph', 'google.genai', 'tavily') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
