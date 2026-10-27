"""Safe loading of untrusted image files.

Images are decoded with Pillow under explicit limits: file size, decoded pixel
count (decompression bombs), and an allow-list of formats. EXIF orientation is
applied so phone photos are upright, transparency is flattened onto white, and
very large images are downscaled before OCR (engine coordinates are mapped back
to the original image by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 50_000_000  # checked from the header (Image.open is lazy), before any decode
MAX_SIDE = 4096  # longer side passed to the engine
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF"}


class ImageInputError(ValueError):
    """The file is missing, too large, or not a supported image."""


@dataclass
class LoadedImage:
    image: Any  # PIL.Image.Image in RGB mode
    width: int  # original (EXIF-rotated) size
    height: int
    scale: float  # engine pixels / original pixels (<= 1)
    format: str
    warnings: list[str]


def _pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImageInputError('Pillow is not installed; run: pip install -e ".[ocr]"') from None
    return Image, ImageOps


def load_image(path: Path, max_side: int = MAX_SIDE) -> LoadedImage:
    Image, ImageOps = _pillow()
    if not path.is_file():
        raise ImageInputError(f"image file not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ImageInputError(f"image file is empty: {path}")
    if size > MAX_FILE_BYTES:
        raise ImageInputError(f"image file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB: {path}")

    warnings: list[str] = []
    try:
        with Image.open(path) as im:
            fmt = im.format or "?"
            if fmt not in ALLOWED_FORMATS:
                raise ImageInputError(f"unsupported image format {fmt}: {path}")
            if im.width * im.height > MAX_PIXELS:
                raise ImageInputError(f"image is too large ({im.width}x{im.height} pixels): {path}")
            if getattr(im, "n_frames", 1) > 1:
                warnings.append("multi-frame image: only the first frame was read")
            im.seek(0)
            im = ImageOps.exif_transpose(im)
            rgb = _to_rgb(im, Image)
    except ImageInputError:
        raise
    except Image.DecompressionBombError:
        raise ImageInputError(f"image is too large to decode safely: {path}") from None
    except (OSError, ValueError, SyntaxError) as exc:  # Pillow raises these for corrupt/unknown files
        raise ImageInputError(f"cannot read image {path.name}: {type(exc).__name__}") from None

    width, height = rgb.size
    scale = 1.0
    if max(width, height) > max_side:
        scale = max_side / max(width, height)
        rgb = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
        warnings.append(f"downscaled from {width}x{height} for OCR")
    return LoadedImage(image=rgb, width=width, height=height, scale=scale, format=fmt, warnings=warnings)


def _to_rgb(im: Any, Image: Any) -> Any:
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        background = Image.new("RGBA", im.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, im).convert("RGB")
    return im.convert("RGB")


def bbox_from_polygon(points: Any, scale: float = 1.0) -> tuple[tuple[float, float, float, float], list[tuple[float, float]]]:
    """Engine polygon (in engine pixels) -> (bbox, polygon) in original-image pixels."""
    poly = [(float(x) / scale, float(y) / scale) for x, y in points]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys)), poly
