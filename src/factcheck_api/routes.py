"""API layer: HTTP endpoints, request validation and threadpool offloading."""

from __future__ import annotations

import functools
import importlib.util
import os
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Callable, Optional

import anyio
from fastapi import APIRouter, File, Form, Request, UploadFile

from factcheck.config import credential_status

from . import responses, services
from .errors import ErrorResponse
from .schemas import FactCheckResponse, HealthResponse, ModuleStatus, TextRequest
from .uploads import EXTENSIONS, stored_upload

router = APIRouter()

MODULE_PACKAGES = {
    "text": [],
    "image": ["rapidocr", "onnxruntime", "PIL", "numpy"],
    "video": ["av", "faster_whisper", "rapidocr", "onnxruntime", "PIL", "numpy"],
    "audio": ["av", "faster_whisper", "numpy"],
}


def _errors(*codes: int) -> dict:
    text = {400: "Invalid input (empty, corrupted or unreadable).", 413: "File larger than the upload limit.",
            415: "Unsupported file type.", 422: "Request validation error (missing or malformed fields).",
            500: "Processing failed or unexpected internal error.",
            503: "Fact-checking credentials, an external service or a local model is unavailable."}
    return {c: {"model": ErrorResponse, "description": text[c]} for c in codes}


@router.get("/health", response_model=HealthResponse, tags=["health"], summary="Backend health")
async def health() -> HealthResponse:
    """Liveness plus which modules have their optional dependencies installed, and whether fact-checking
    credentials are configured (presence only; values are never returned). No external call is made."""
    try:
        ver = version("factcheck")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
        ver = "unknown"
    modules = {}
    for name, pkgs in MODULE_PACKAGES.items():
        missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
        modules[name] = ModuleStatus(available=not missing, missing=missing)
    creds = credential_status()
    provider = os.environ.get("FACTCHECK_PROVIDER", "gemini").strip().lower()
    configured = creds.get("TAVILY_API_KEY", False) and (provider == "ollama" or creds.get("GEMINI_API_KEY", False))
    return HealthResponse(version=ver, modules=modules, fact_check_configured=bool(configured))


async def _offload(request: Request, group: str, fn: Callable, *args):
    """Run a synchronous module call in a worker thread, bounded by the group's limiter.

    The await is not cancelled while the thread runs (anyio's default), so the temporary upload is never
    deleted under a running job, even when the client disconnects.
    """
    return await anyio.to_thread.run_sync(functools.partial(fn, *args), limiter=request.app.state.limiters[group])


@router.post("/api/v1/fact-check/text", response_model=FactCheckResponse, tags=["fact-check"],
             summary="Fact-check text", responses=_errors(400, 422, 500, 503))
async def fact_check_text(request: Request, body: TextRequest) -> FactCheckResponse:
    """Extract the checkable claims from `text`, retrieve web evidence and return a verdict per claim
    (the `factcheck` command). Typically 10-60 s. Any verdict, including `INSUFFICIENT_EVIDENCE`, is a
    successful (200) result."""
    settings = request.app.state.settings
    if len(body.text) > settings.max_text_chars:
        from .errors import APIError

        raise APIError(413, "text_too_large", f"The text exceeds {settings.max_text_chars} characters.")
    started = time.perf_counter()
    opts = services.Options(body.max_claims, body.search_budget)
    report = await _offload(request, "text", services.run_text, request.app.state.backend, body.text, opts)
    return responses.from_text(report, time.perf_counter() - started)


async def _media(request: Request, kind: str, group: str, file: UploadFile, max_claims: Optional[int],
                 search_budget: Optional[int], runner: Callable, convert: Callable) -> FactCheckResponse:
    started = time.perf_counter()
    opts = services.Options(max_claims, search_budget)
    async with stored_upload(file, kind, request.app.state.settings) as up:
        # Temporary names never reach the client: the modules mention the file by path or by name.
        replacements = {str(up.path): up.display_name, str(up.directory): "", up.path.name: up.display_name}
        report = await _offload(request, group, runner, request.app.state.backend, up.path, opts, replacements)
        return convert(report, up.display_name, time.perf_counter() - started, replacements)


def _formats(kind: str) -> str:
    return ", ".join(sorted(EXTENSIONS[kind]))


MaxClaims = Annotated[Optional[int], Form(ge=1, le=12, description="Checkable claims to verify (default 6).")]
SearchBudget = Annotated[Optional[int], Form(ge=0, le=40, description="Web-search credits for this run (default 40).")]


@router.post("/api/v1/fact-check/image", response_model=FactCheckResponse, tags=["fact-check"],
             summary="Fact-check the text in an image", responses=_errors(400, 413, 415, 422, 500, 503))
async def fact_check_image(request: Request,
                           file: UploadFile = File(..., description=f"Image ({_formats('image')})."),
                           max_claims: MaxClaims = None,
                           search_budget: SearchBudget = None) -> FactCheckResponse:
    """OCR the image (RapidOCR) and fact-check the recognised text (`factcheck-ocr --fact-check`). Each claim's
    `provenance` gives its OCR lines, blocks and OCR confidence. Images without readable text return
    `status: not_checked`."""
    return await _media(request, "image", "image", file, max_claims, search_budget, services.run_image,
                        responses.from_image)


@router.post("/api/v1/fact-check/video", response_model=FactCheckResponse, tags=["fact-check"],
             summary="Fact-check the speech and on-screen text of a video",
             responses=_errors(400, 413, 415, 422, 500, 503))
async def fact_check_video(request: Request,
                           file: UploadFile = File(..., description=f"Video ({_formats('video')})."),
                           max_claims: MaxClaims = None,
                           search_budget: SearchBudget = None) -> FactCheckResponse:
    """Transcribe the speech (faster-whisper) and read on-screen text (OCR of sampled frames), then fact-check
    both (`factcheck-video`). Each claim's `timestamps` list every place it was spoken or shown (modality, time,
    segment / text-block ids, confidence); `provenance` holds merged duplicates. The full transcript and OCR
    observations are in `report.extraction`. Takes roughly 0.5-1x the video length on a laptop CPU; video and
    audio jobs run one at a time."""
    return await _media(request, "video", "media", file, max_claims, search_budget, services.run_video,
                        responses.from_video)


@router.post("/api/v1/fact-check/audio", response_model=FactCheckResponse, tags=["fact-check"],
             summary="Fact-check the speech in an English audio file",
             responses=_errors(400, 413, 415, 422, 500, 503))
async def fact_check_audio(request: Request,
                           file: UploadFile = File(..., description=f"Audio ({_formats('audio')})."),
                           max_claims: MaxClaims = None,
                           search_budget: SearchBudget = None) -> FactCheckResponse:
    """Transcribe English speech (faster-whisper, word timestamps) and fact-check it (`factcheck-audio`).
    Each claim's `timestamps` give where it was spoken, its segment ids and transcription confidence. The
    full timed transcript is in `report.transcription.segments`. Non-English audio returns
    `status: not_checked`. Transcription runs at about 0.06x the audio length; video and audio jobs run
    one at a time."""
    return await _media(request, "audio", "media", file, max_claims, search_budget, services.run_audio,
                        responses.from_audio)
