"""Data models for audio fact-checking (deterministic code only; no model output).

Times are seconds from the start of the audio. They come from the speech
recogniser and are approximate (Whisper boundaries can be off by a few hundred
milliseconds). Transcript segments and words reuse the video module's models so
both modules describe speech the same way.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from factcheck_video.models import TranscriptSegment  # noqa: F401  (re-exported)

SUPPORTED_LANGUAGE = "en"


class AudioLimits(BaseModel):
    """Resource limits for one audio file. All are caps, not guarantees."""

    max_file_mb: int = Field(500, ge=1, le=4096)
    max_duration_s: float = Field(3600.0, gt=0, le=4 * 3600)
    extract_timeout_s: float = Field(1800.0, gt=0, le=4 * 3600)


class AudioInfo(BaseModel):
    path: str
    container: str = ""
    size_bytes: int = 0
    duration_s: Optional[float] = None
    codec: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bit_rate: Optional[int] = None
    has_cover_art: bool = False  # e.g. an embedded picture in an MP3/M4A (ignored)


class TranscriptionStatus(str, Enum):
    OK = "OK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # some segments below the reliability threshold
    UNRELIABLE = "UNRELIABLE"  # speech recognised, but none of it reliably
    NO_SPEECH = "NO_SPEECH"  # silence, music or noise
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"  # detected language is not English
    FAILED = "FAILED"  # decoding/transcription error; see ``error``


class LanguageCheck(BaseModel):
    """Whisper's language identification and the English-only decision."""

    detected: Optional[str] = None
    probability: Optional[float] = None
    english_probability: Optional[float] = None
    accepted: bool = False
    method: str = "detected"  # detected | assumed (--assume-english) | english-only model
    note: Optional[str] = None


class Transcription(BaseModel):
    status: TranscriptionStatus
    model: Optional[str] = None
    compute_type: Optional[str] = None
    threshold: float = 0.0
    language: LanguageCheck = Field(default_factory=LanguageCheck)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    audio_seconds: Optional[float] = None  # seconds of audio actually decoded and transcribed
    error: Optional[str] = None
    notes: list[str] = Field(default_factory=list)
