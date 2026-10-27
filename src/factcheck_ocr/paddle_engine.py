"""PaddleOCR 3.x engines (optional; install ``.[ocr-paddle]`` in a separate venv).

* ``paddle``: the PaddleOCR text pipeline with PP-OCRv6 medium detection and
  recognition plus the text-line orientation classifier. Document orientation
  and unwarping are off: unwarping distorts screenshots and memes. Reading order
  is then recovered geometrically by ``layout.py``.
* ``paddle-structure``: PP-StructureV3 with layout detection (PP-DocLayout_plus-L),
  region detection (PP-DocBlockLayout) and PP-OCRv5 server OCR. Table, formula,
  seal and chart recognition are off. Its ``parsing_res_list`` is already sorted
  in reading order, so the layout model's order is used as is.

Models are downloaded on first use into the PaddleX cache
(``PADDLE_PDX_CACHE_HOME``, default ``~/.paddlex``) from Hugging Face by default
(``PADDLE_PDX_MODEL_SOURCE``). This adapter checks the cache first and refuses to
download unless ``allow_download`` is set. When everything is cached it also
sets ``PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`` so start-up makes no network
request.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from .engines import EngineFailure, EngineOutput, EngineUnavailable, RawBox

TEXT_MODELS = ("PP-OCRv6_medium_det", "PP-LCNet_x1_0_textline_ori", "PP-OCRv6_medium_rec")
STRUCTURE_MODELS = (
    "PP-DocLayout_plus-L", "PP-DocBlockLayout",
    "PP-OCRv5_server_det", "PP-LCNet_x1_0_textline_ori", "PP-OCRv5_server_rec",
)


def cache_dir() -> Path:
    return Path(os.environ.get("PADDLE_PDX_CACHE_HOME", str(Path.home() / ".paddlex"))) / "official_models"


def missing_models(names: tuple[str, ...]) -> list[str]:
    root = cache_dir()
    return [n for n in names if not (root / n).is_dir()]


def _rect(b: Any) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = (float(v) for v in b)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _inside(point: tuple[float, float], b: Any) -> bool:
    return b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]


@contextlib.contextmanager
def _stdout_to_stderr():
    """Paddle's C++ code prints diagnostics (e.g. ``ReduceMeanCheckIfOneDNNSupport``) straight to
    file descriptor 1, which would mix them into the extracted text. Point fd 1 at stderr meanwhile."""
    sys.stdout.flush()
    try:
        saved = os.dup(1)
    except OSError:  # no real fd 1 (e.g. captured by a test runner)
        yield
        return
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


class PaddleEngine:
    def __init__(self, *, structure: bool, allow_download: bool = False):
        self.name = "paddle-structure" if structure else "paddle"
        if importlib.util.find_spec("paddleocr") is None or importlib.util.find_spec("paddle") is None:
            raise EngineUnavailable(
                'PaddleOCR is not installed; in a separate venv run: pip install -e ".[ocr-paddle]"')
        needed = STRUCTURE_MODELS if structure else TEXT_MODELS
        missing = missing_models(needed)
        if missing and not allow_download:
            raise EngineUnavailable(
                f"PaddleOCR models not in {cache_dir()}: {', '.join(missing)}. "
                "Re-run with --allow-download to fetch them once (from Hugging Face).")
        if not missing:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        self._structure = structure
        try:
            with _stdout_to_stderr():
                self._pipe = self._create(structure)
        except Exception as exc:  # noqa: BLE001 - initialisation problems are setup problems
            raise EngineUnavailable(f"PaddleOCR failed to initialise: {type(exc).__name__}: {exc}") from None

    @staticmethod
    def _create(structure: bool) -> Any:
        if structure:
            from paddleocr import PPStructureV3

            return PPStructureV3(
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=True, use_table_recognition=False,
                use_formula_recognition=False, use_seal_recognition=False,
                use_chart_recognition=False, use_region_detection=True, device="cpu",
            )
        from paddleocr import PaddleOCR

        return PaddleOCR(
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=True, device="cpu",
        )

    def recognize(self, image: Any) -> EngineOutput:
        import numpy as np

        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])  # Paddle expects OpenCV (BGR) arrays
        try:
            with _stdout_to_stderr():
                results = list(self._pipe.predict(bgr))
        except Exception as exc:  # noqa: BLE001
            raise EngineFailure(f"PaddleOCR failed: {type(exc).__name__}: {exc}") from None
        if not results:
            return EngineOutput(boxes=[])
        res = results[0]
        return structure_output(res) if self._structure else text_output(res)


def text_output(res: Any) -> EngineOutput:
    boxes = [
        RawBox(text=str(t), score=float(s), points=[(float(x), float(y)) for x, y in poly])
        for t, s, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"])
    ]
    return EngineOutput(boxes=boxes)


def structure_output(res: Any) -> EngineOutput:
    """PP-StructureV3 blocks, in model reading order. A block's score is the mean confidence of
    the OCR lines whose centre falls inside it (the layout model gives no text confidence)."""
    ocr = res["overall_ocr_res"]
    lines = [
        ((float(b[0] + b[2]) / 2, float(b[1] + b[3]) / 2), float(s))
        for b, s in zip(ocr["rec_boxes"], ocr["rec_scores"])
    ]
    boxes = []
    for block in res["parsing_res_list"]:
        content = (getattr(block, "content", "") or "").strip()
        if not content:
            continue  # image/chart regions without recognised text
        scores = [s for centre, s in lines if _inside(centre, block.bbox)]
        score = sum(scores) / len(scores) if scores else 0.0
        boxes.append(RawBox(text=content, score=score, points=_rect(block.bbox), label=str(block.label)))
    return EngineOutput(boxes=boxes, ordered=True)
