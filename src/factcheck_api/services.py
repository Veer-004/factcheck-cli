"""Integration layer: calls the existing module functions with the same defaults as their CLIs.

Nothing here judges claims. Each ``run_*`` function is synchronous (the modules
are), and the routes run it in a worker thread. ``Backend`` holds the provider
and model factories; production uses the defaults (keys from the environment /
``.env``, faster-whisper, RapidOCR). Tests swap in fakes.

Module exceptions are translated into ``APIError`` with a sanitised message:

=========================================  ======  ===========================
module exception                           status  code
=========================================  ======  ===========================
InputError / AudioInputError /             400     invalid_input
VideoInputError (corrupt, empty, too long)
... "unsupported" format                   415     unsupported_media_type
ConfigError (missing API keys)             503     service_not_configured
DependencyMissing / ModelUnavailable /     503     dependency_unavailable
EngineUnavailable
RunFailure: provider auth / LLM failure    503     external_service_error
RunFailure: other / OCR or transcription   500     processing_failed
failure
=========================================  ======  ===========================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from factcheck.config import Config, ConfigError
from factcheck.graph import RunFailure

from .errors import APIError, clean_message

log = logging.getLogger("factcheck_api")

# Same as the factcheck-video / factcheck-audio CLI default (--max-text-chars 6000).
MEDIA_MAX_TEXT_CHARS = 6000


def _default_ocr_engine():
    from factcheck_ocr.engines import create_engine

    return create_engine("rapid", model_size="small", allow_download=False)


def _default_video_transcriber():
    from factcheck_video.transcribe import WhisperTranscriber

    return WhisperTranscriber()


def _default_audio_transcriber():
    from factcheck_audio.transcribe import EnglishTranscriber

    return EnglishTranscriber()


def _default_credits():
    from .planner import TavilyCredits

    return TavilyCredits()


def _default_planner():
    from .planner import PlanningAgent

    return PlanningAgent()


def _default_config(**overrides: Any) -> Config:
    return Config.from_env(**overrides)


@dataclass
class Backend:
    llm: Any = None  # None: the shared engine builds Gemini (+ Ollama fallback) from the environment
    search: Any = None  # None: Tavily from the environment
    make_config: Callable[..., Config] = _default_config
    make_ocr_engine: Callable[[], Any] = _default_ocr_engine
    make_video_transcriber: Callable[[], Any] = _default_video_transcriber
    make_audio_transcriber: Callable[[], Any] = _default_audio_transcriber
    # Remaining Tavily credits for the run planner (planner.TavilyCredits); call .invalidate() after a run.
    credits: Any = field(default_factory=lambda: _default_credits())
    # The AI planning agent (planner.PlanningAgent): (mode, content, duration_s=, credits=) -> PlanResponse.
    planner: Any = field(default_factory=lambda: _default_planner())
    extra: dict = field(default_factory=dict)


@dataclass
class Options:
    max_claims: Optional[int] = None  # None: module default (6), or the planning agent's choice
    search_budget: Optional[int] = None  # None: module default (40), or the planning agent's choice
    auto_plan: bool = False  # let the AI planning agent fill the values that are None
    chosen_plan: Any = None  # set when the agent planned this run (schemas.PlanResponse)

    def overrides(self) -> dict[str, Any]:
        return {"max_checkable_claims": self.max_claims, "search_credit_budget": self.search_budget}

    @property
    def wants_plan(self) -> bool:
        return self.auto_plan and (self.max_claims is None or self.search_budget is None)


def _with_planning(backend: Backend, mode: str, options: Options, call: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap a module call so the planning agent sizes the run once the text to check is known (for files:
    after OCR / transcription, inside ``run_pipeline``). Values the client gave are kept."""
    if not options.wants_plan:
        return call

    def planner(text: str, _config: Config) -> dict[str, Any]:
        plan = backend.planner(mode, text, credits=backend.credits())
        options.chosen_plan = plan
        return {"max_checkable_claims": plan.max_claims if options.max_claims is None else None,
                "search_credit_budget": plan.search_budget if options.search_budget is None else None}

    def planned():
        from factcheck.graph import use_run_planner

        with use_run_planner(planner):
            return call()

    return planned


# ---------------------------------------------------------------- error translation


def translate(exc: BaseException, replacements: Optional[dict[str, str]] = None) -> APIError:
    """Module exception -> APIError (see the table in the module docstring)."""
    from factcheck.validation import InputError
    from factcheck_audio.audio_io import AudioInputError
    from factcheck_video.media import DependencyMissing, VideoInputError
    from factcheck_video.transcribe import ModelUnavailable

    msg = clean_message(str(exc), replacements)
    if isinstance(exc, (InputError, AudioInputError, VideoInputError)):
        if "unsupported" in msg.lower():
            return APIError(415, "unsupported_media_type", msg)
        return APIError(400, "invalid_input", msg)
    if isinstance(exc, ConfigError):
        log.error("fact-check configuration problem: %s", clean_message(str(exc)))
        return APIError(503, "service_not_configured",
                        "The fact-checking service is not configured on the server (missing or invalid "
                        "credentials or settings).")
    if isinstance(exc, (DependencyMissing, ModelUnavailable)) or type(exc).__name__ == "EngineUnavailable":
        log.error("dependency unavailable: %s", clean_message(str(exc)))
        return APIError(503, "dependency_unavailable",
                        "A required component (speech or OCR model) is not available on the server.")
    if isinstance(exc, RunFailure):
        low = msg.lower()
        log.error("run failure: %s", msg)
        if "authentication" in low or "access error" in low:
            return APIError(503, "external_service_error",
                            "An external fact-checking service rejected the request (authentication/access).")
        if "claim extraction" in low:
            return APIError(503, "external_service_error",
                            "The language-model service failed or returned invalid output. Try again later.")
        return APIError(500, "processing_failed", "Processing failed: " + msg)
    if type(exc).__name__ == "EngineFailure":
        log.error("OCR engine failure: %s", msg)
        return APIError(500, "processing_failed", "OCR processing failed.")
    raise exc  # unexpected: handled (logged, generic 500) by the app


def _guarded(fn: Callable[[], Any], replacements: Optional[dict[str, str]] = None) -> Any:
    try:
        return fn()
    except APIError:
        raise
    except Exception as exc:  # noqa: BLE001 - translated or re-raised
        raise translate(exc, replacements) from None


# ---------------------------------------------------------------- extraction only (for AI planning)


def extract_for_plan(backend: Backend, kind: str, path: Path,
                     replacements: dict[str, str]) -> tuple[str, Optional[float]]:
    """Read a file's text the way its fact-check would (OCR / transcription), without fact-checking it.
    Returns (text, media duration in seconds). Used by "Calculate with AI" so the planning agent sees
    the real content."""

    def call():
        if kind == "image":
            from factcheck.image import ocr_image

            return ocr_image(path, backend.make_ocr_engine()).text, None
        config = backend.make_config(max_input_chars=MEDIA_MAX_TEXT_CHARS)
        if kind == "video":
            from factcheck_video.pipeline import verify_video

            vr = verify_video(path, config, make_transcriber=backend.make_video_transcriber,
                              make_engine=backend.make_ocr_engine, max_text_chars=MEDIA_MAX_TEXT_CHARS,
                              fact_check=False)
            return vr.checked_text or "", vr.extraction.media.duration_s
        from factcheck_audio.pipeline import verify_audio

        ar = verify_audio(path, config, make_transcriber=backend.make_audio_transcriber,
                          max_text_chars=MEDIA_MAX_TEXT_CHARS, fact_check=False)
        if ar.transcription.error:
            raise RunFailure(f"transcription failed: {ar.transcription.error}")
        return ar.checked_text or "", ar.media.duration_s

    return _guarded(call, replacements)


# ---------------------------------------------------------------- the four modules


def run_text(backend: Backend, text: str, options: Options):
    from factcheck.graph import verify_text

    def call():
        config = backend.make_config(**options.overrides())
        return verify_text(text, config, backend.llm, backend.search)

    return _guarded(_with_planning(backend, "text", options, call))


def run_image(backend: Backend, path: Path, options: Options, replacements: dict[str, str]):
    from factcheck.image import verify_image

    def call():
        config = backend.make_config(**options.overrides())
        return verify_image(path, config, backend.llm, backend.search, engine=backend.make_ocr_engine())

    return _guarded(_with_planning(backend, "image", options, call), replacements)


def run_video(backend: Backend, path: Path, options: Options, replacements: dict[str, str]):
    from factcheck_video.pipeline import verify_video

    def call():
        config = backend.make_config(**options.overrides(), max_input_chars=MEDIA_MAX_TEXT_CHARS)
        return verify_video(path, config, backend.llm, backend.search,
                            make_transcriber=backend.make_video_transcriber, make_engine=backend.make_ocr_engine,
                            max_text_chars=MEDIA_MAX_TEXT_CHARS)

    return _guarded(_with_planning(backend, "video", options, call), replacements)


def run_audio(backend: Backend, path: Path, options: Options, replacements: dict[str, str]):
    from factcheck_audio.models import TranscriptionStatus
    from factcheck_audio.pipeline import verify_audio

    def call():
        config = backend.make_config(**options.overrides(), max_input_chars=MEDIA_MAX_TEXT_CHARS)
        return verify_audio(path, config, backend.llm, backend.search,
                            make_transcriber=backend.make_audio_transcriber, max_text_chars=MEDIA_MAX_TEXT_CHARS)

    report = _guarded(_with_planning(backend, "audio", options, call), replacements)
    if report.transcription.status == TranscriptionStatus.FAILED:  # the CLI's exit code 3
        log.error("transcription failed: %s", clean_message(report.transcription.error or "", replacements))
        raise APIError(500, "processing_failed",
                       "Transcription failed: " + clean_message(report.transcription.error or "unknown error",
                                                                replacements))
    return report
