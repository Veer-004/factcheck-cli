"""Live OCR smoke test: runs the real default engine on generated sample images.

Skipped unless FACTCHECK_OCR_LIVE_TESTS=1 (needs ``.[ocr]`` installed; no network:
the default RapidOCR models ship inside the wheel).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.ocr_live

LIVE = os.environ.get("FACTCHECK_OCR_LIVE_TESTS") == "1"


@pytest.mark.skipif(not LIVE, reason="set FACTCHECK_OCR_LIVE_TESTS=1")
def test_live_rapid_reads_two_columns_and_meme(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    import make_ocr_samples as samples

    from factcheck_ocr import create_engine, ocr_image

    engine = create_engine("rapid")
    for make, name in ((samples.two_column, "two_column"), (samples.meme, "meme")):
        image, expected = make()
        path = tmp_path / f"{name}.png"
        image.save(path)
        page = ocr_image(path, engine)
        got = " ".join(page.text.split()).lower()
        assert got == " ".join(" ".join(expected).split()).lower(), page.text
