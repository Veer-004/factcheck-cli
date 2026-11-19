"""Safe handling of uploaded files.

* The client's filename is used for display only. It is never part of a
  filesystem path: each upload is written to ``upload<ext>`` inside a new,
  uniquely named temporary directory, where ``<ext>`` comes from an allow-list.
* Extension and declared content type are both checked (415 when unsupported).
* Size is checked from the multipart part size when known, and again while
  copying in 1 MB chunks, so the copy stops at the limit (413).
* The temporary directory is removed when processing ends, whether it succeeded
  or raised.

The module's own validation (decoding, corrupt files, duration limits) still
runs on the stored file; this layer only enforces what is cheap to check first.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import anyio
from fastapi import UploadFile

from factcheck_audio.audio_io import AUDIO_EXTENSIONS
from factcheck_video.media import VIDEO_EXTENSIONS

from .config import MB, Settings
from .errors import APIError

log = logging.getLogger("factcheck_api")

CHUNK = 1 * MB
IMAGE_EXTENSIONS = frozenset(".png .jpg .jpeg .webp .bmp .tif .tiff .gif".split())  # factcheck_ocr.image_io formats

EXTENSIONS = {"image": IMAGE_EXTENSIONS, "video": VIDEO_EXTENSIONS, "audio": AUDIO_EXTENSIONS}
# Declared types accepted besides the kind's own "<kind>/*" family. Browsers and tools often send
# application/octet-stream for files they do not recognise.
EXTRA_TYPES = {
    "image": {"application/octet-stream"},
    "video": {"application/octet-stream", "application/mp4", "application/x-matroska"},
    "audio": {"application/octet-stream", "application/ogg", "video/webm", "video/ogg", "video/mp4"},
}


@dataclass
class StoredUpload:
    path: Path  # temporary file to hand to the module
    directory: Path  # its private temporary directory
    display_name: str  # sanitised client filename, for reports and messages only
    size: int


def display_name(filename: Optional[str]) -> str:
    """Basename of the client filename without control characters (display only)."""
    name = re.split(r"[\\/]", filename or "")[-1]
    name = "".join(ch for ch in unicodedata.normalize("NFC", name) if unicodedata.category(ch)[0] != "C").strip()
    return (name or "upload")[:120]


def check_type(kind: str, filename: Optional[str], content_type: Optional[str]) -> str:
    """Validate extension and declared content type; returns the lower-case extension."""
    name = display_name(filename)
    ext = os.path.splitext(name)[1].lower()
    allowed = EXTENSIONS[kind]
    if ext not in allowed:
        raise APIError(415, "unsupported_media_type",
                       f"Unsupported {kind} file extension {ext or '(none)'!r}. Supported: "
                       + ", ".join(sorted(allowed)))
    declared = (content_type or "").split(";")[0].strip().lower()
    if declared and not (declared.startswith(f"{kind}/") or declared in EXTRA_TYPES[kind]):
        raise APIError(415, "unsupported_media_type",
                       f"Unsupported content type {declared!r} for a {kind} upload.")
    return ext


def _too_large(kind: str, limit: int) -> APIError:
    return APIError(413, "file_too_large", f"The {kind} file exceeds the {limit // MB} MB upload limit.")


@asynccontextmanager
async def stored_upload(file: UploadFile, kind: str, settings: Settings) -> AsyncIterator[StoredUpload]:
    """Copy ``file`` to a private temporary directory; always removed on exit."""
    ext = check_type(kind, file.filename, file.content_type)
    limit = settings.max_bytes(kind)
    if file.size is not None and file.size > limit:
        raise _too_large(kind, limit)
    root = settings.temp_dir
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="factcheck-api-", dir=root))
    try:
        path = directory / f"upload{ext}"
        size = 0
        with open(path, "wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    raise _too_large(kind, limit)
                await anyio.to_thread.run_sync(out.write, chunk)
        if size == 0:
            raise APIError(400, "empty_file", f"The uploaded {kind} file is empty.")
        yield StoredUpload(path=path, directory=directory, display_name=display_name(file.filename), size=size)
    finally:
        await file.close()
        shutil.rmtree(directory, ignore_errors=True)
        if directory.exists():  # e.g. a file still open on Windows
            log.warning("temporary upload directory could not be removed: %s", directory)
