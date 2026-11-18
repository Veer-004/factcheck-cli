"""Data models for video extraction (deterministic code only; no model output).

Times are seconds from the start of the video. Speech times come from the
speech recogniser and are approximate (Whisper segment boundaries can be off by
a few hundred milliseconds). On-screen times are the timestamps of *sampled*
frames: text first seen at ``first_seen`` may have appeared up to one sampling
interval earlier, and may stay up to one interval after ``last_seen``.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Modality(str, Enum):
    SPEECH = "speech"
    SCREEN = "screen"


class Limits(BaseModel):
    """Resource limits for one video. All are caps, not guarantees."""

    max_file_mb: int = Field(500, ge=1, le=4096)
    max_duration_s: float = Field(600.0, gt=0, le=3600)
    frame_interval_s: float = Field(2.0, ge=0.25, le=60)
    max_frames: int = Field(150, ge=1, le=2000)
    frame_max_side: int = Field(1920, ge=320, le=4096)
    extract_timeout_s: float = Field(900.0, gt=0, le=7200)


class MediaInfo(BaseModel):
    path: str
    container: str = ""
    size_bytes: int = 0
    duration_s: Optional[float] = None
    has_video: bool = False
    has_audio: bool = False
    video_codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    rotation: int = 0
    audio_codec: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None


class StageStatus(str, Enum):
    OK = "OK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # some output below the reliability threshold
    UNRELIABLE = "UNRELIABLE"  # output exists but none of it is reliable
    NO_CONTENT = "NO_CONTENT"  # ran, found no speech / no text
    NO_STREAM = "NO_STREAM"  # the file has no audio (or video) stream
    SKIPPED = "SKIPPED"  # disabled by the user
    FAILED = "FAILED"  # error; see the stage's error


class Word(BaseModel):
    text: str
    start: float
    end: float
    probability: float


class TranscriptSegment(BaseModel):
    segment_id: int
    start: float
    end: float
    text: str
    confidence: float  # exp(avg_logprob): the recogniser's own estimate, not accuracy
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0
    reliable: bool = True
    words: list[Word] = Field(default_factory=list)


class AudioStage(BaseModel):
    status: StageStatus
    model: Optional[str] = None
    compute_type: Optional[str] = None
    language: Optional[str] = None
    language_probability: Optional[float] = None
    threshold: float = 0.0
    segments: list[TranscriptSegment] = Field(default_factory=list)
    audio_seconds: Optional[float] = None
    seconds: float = 0.0
    error: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


class ScreenLine(BaseModel):
    text: str
    score: float


class ScreenText(BaseModel):
    """One piece of on-screen text (an OCR block), merged across the frames it was seen in."""

    observation_id: int
    text: str  # best-scoring reading (lines joined with newlines)
    lines: list[ScreenLine] = Field(default_factory=list)
    min_score: float
    reliable: bool
    frame_times: list[float] = Field(default_factory=list)  # sampled frames where it was seen
    readings: int = 1  # how many frames contributed (incl. near-identical OCR variants)
    variants: list[str] = Field(default_factory=list)  # other near-identical OCR readings

    @property
    def first_seen(self) -> float:
        return self.frame_times[0]

    @property
    def last_seen(self) -> float:
        return self.frame_times[-1]


class ScreenStage(BaseModel):
    status: StageStatus
    engine: Optional[str] = None
    threshold: float = 0.0
    interval_s: float = 0.0
    frames_sampled: int = 0
    frames_ocr: int = 0  # frames actually sent to OCR (unchanged frames reuse the previous result)
    observations: list[ScreenText] = Field(default_factory=list)
    seconds: float = 0.0
    error: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


class SourceRef(BaseModel):
    """One place a unit's text came from."""

    modality: Modality
    start: float
    end: float
    text: str
    confidence: float
    reliable: bool
    segment_id: Optional[int] = None
    observation_id: Optional[int] = None
    frame_times: list[float] = Field(default_factory=list)


class Unit(BaseModel):
    """A piece of text placed in the fact-check input: a speech segment, an on-screen block,
    or both when their text is identical after normalisation."""

    unit_id: str
    text: str
    start: float
    modalities: list[Modality]
    refs: list[SourceRef]

    @property
    def reliable(self) -> bool:
        return any(r.reliable for r in self.refs)


class Extraction(BaseModel):
    media: MediaInfo
    audio: AudioStage
    screen: ScreenStage
    units: list[Unit] = Field(default_factory=list)
    exact_merges: int = 0  # speech segments whose text was also shown on screen
    subtitle_merges: int = 0  # on-screen captions attached to the speech they transcribe (merge.attach_subtitles)
    seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
