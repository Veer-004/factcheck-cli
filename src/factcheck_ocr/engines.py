"""OCR engine adapters.

Every engine takes an RGB ``PIL.Image`` and returns ``EngineOutput``: text boxes in
*engine* pixel coordinates, plus whether the engine already put them in reading
order (layout model) or the geometric layout in ``layout.py`` should do it.

* ``rapid`` (default): RapidOCR on ONNX Runtime with the PP-OCRv6 small models that
  ship inside the ``rapidocr`` wheel (no download). ``model_size="medium"`` uses the
  PP-OCRv6 medium models instead (more accurate on stylised text such as memes,
  slower), downloaded once into the rapidocr package folder.
* ``paddle``: PaddleOCR 3.x text pipeline (PP-OCRv6 medium by default).
* ``paddle-structure``: PaddleOCR PP-StructureV3 (layout detection + model
  reading order; table/formula/chart/seal modules are switched off).

The Paddle engines download models on first use (from Hugging Face by default) to
the PaddleX cache. Every engine refuses to run with models that are not on disk
unless ``allow_download`` is set (``--allow-download`` on the CLI).

Engine libraries are imported lazily, so this module (and the tests) work without
any OCR package installed.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

ENGINE_NAMES = ("rapid", "paddle", "paddle-structure")
MODEL_SIZES = ("small", "medium")  # rapid only: PP-OCRv6 small (bundled) or medium (downloaded once)


class EngineUnavailable(RuntimeError):
    """The engine's package or models are missing (a setup problem, not a run failure)."""


class EngineFailure(RuntimeError):
    """The engine raised while processing an image."""


@dataclass
class RawBox:
    text: str
    score: float
    points: list[tuple[float, float]]  # polygon in engine pixels
    label: Optional[str] = None


@dataclass
class EngineOutput:
    boxes: list[RawBox]
    ordered: bool = False  # True when a layout model already produced reading order
    notes: list[str] = field(default_factory=list)


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: Any) -> EngineOutput: ...


# ---------------------------------------------------------------- RapidOCR


class RapidEngine:
    name = "rapid"

    def __init__(self, model_size: str = "small", allow_download: bool = False, threads: Optional[int] = None,
                 textline_orientation: bool = False):
        if importlib.util.find_spec("rapidocr") is None or importlib.util.find_spec("onnxruntime") is None:
            raise EngineUnavailable('RapidOCR is not installed; run: pip install -e ".[ocr]"')
        if model_size not in MODEL_SIZES:
            raise EngineUnavailable(f"unknown model size {model_size!r}; choose one of: {', '.join(MODEL_SIZES)}")
        from rapidocr import ModelType, RapidOCR

        missing = [p.name for p in rapid_models(model_size) if not p.is_file()]
        if missing and model_size == "small":  # shipped in the wheel; never fetch them silently
            raise EngineUnavailable(f"RapidOCR bundled models are missing ({', '.join(missing)}); reinstall rapidocr")
        if missing and not allow_download:
            raise EngineUnavailable(
                f"RapidOCR {model_size} models are not downloaded ({', '.join(missing)}). "
                "Re-run with --allow-download to fetch them once (about 130 MB, from RapidAI's ModelScope repository).")
        self.name = "rapid" if model_size == "small" else f"rapid-{model_size}"
        params: dict[str, Any] = {
            "Global.log_level": "error",
            "Global.text_score": 0.0,  # keep every result; the pipeline applies --min-score and counts drops
            "Det.model_type": ModelType(model_size),
            "Rec.model_type": ModelType(model_size),
            # RapidOCR's text-direction classifier (PP-OCR v2.0 mobile) wrongly flips outlined meme
            # text by 180 degrees, producing confident garbage (measured: eval/run_ocr_eval.py,
            # meme.png). Off unless the caller expects upside-down text.
            "Global.use_cls": textline_orientation,
        }
        if threads:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
        try:
            self._ocr = RapidOCR(params=params)
        except Exception as exc:  # noqa: BLE001 - surfaced as a setup problem
            raise EngineUnavailable(f"RapidOCR failed to initialise: {type(exc).__name__}: {exc}") from None

    def recognize(self, image: Any) -> EngineOutput:
        try:
            out = self._ocr(image)
        except Exception as exc:  # noqa: BLE001
            raise EngineFailure(f"RapidOCR failed: {type(exc).__name__}: {exc}") from None
        if out is None or getattr(out, "txts", None) is None or getattr(out, "boxes", None) is None:
            return EngineOutput(boxes=[])
        boxes = [
            RawBox(text=str(t), score=float(s), points=[(float(x), float(y)) for x, y in poly])
            for poly, t, s in zip(out.boxes, out.txts, out.scores)
        ]
        return EngineOutput(boxes=boxes)


def rapid_models(model_size: str = "small") -> list[Path]:
    """ONNX files RapidOCR resolves to (its package ``models`` folder). The small ones ship in the
    wheel; medium ones are downloaded there on first use."""
    spec = importlib.util.find_spec("rapidocr")
    if spec is None or spec.origin is None:
        return []
    root = Path(spec.origin).parent / "models"
    return [root / f"PP-OCRv6_det_{model_size}.onnx", root / f"PP-OCRv6_rec_{model_size}.onnx",
            root / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"]


# ---------------------------------------------------------------- factory


def create_engine(name: str, *, model_size: str = "small", allow_download: bool = False,
                  threads: Optional[int] = None, textline_orientation: bool = False) -> OcrEngine:
    if name == "rapid":
        return RapidEngine(model_size=model_size, allow_download=allow_download, threads=threads,
                           textline_orientation=textline_orientation)
    if name in ("paddle", "paddle-structure"):
        from .paddle_engine import PaddleEngine

        return PaddleEngine(structure=(name == "paddle-structure"), allow_download=allow_download)
    raise EngineUnavailable(f"unknown OCR engine {name!r}; choose one of: {', '.join(ENGINE_NAMES)}")


def engine_status() -> dict[str, str]:
    """Installed versions only (no model loading, no network)."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("rapidocr", "onnxruntime", "paddleocr", "paddlex", "paddlepaddle", "pillow", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    if out["rapidocr"] != "not installed":
        for size in MODEL_SIZES:
            missing = [p.name for p in rapid_models(size) if not p.is_file()]
            out[f"rapidocr {size} models"] = "present" if not missing else f"not downloaded ({', '.join(missing)})"
    if out["paddleocr"] != "not installed":
        from .paddle_engine import cache_dir

        out["paddle model cache"] = str(cache_dir())
    return out
