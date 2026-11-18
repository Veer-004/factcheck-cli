"""Video fact-checking workflow.

    video -> probe (PyAV/FFmpeg) -> audio track -> 16 kHz WAV (temp) -> faster-whisper -> transcript segments
                                 -> sampled frames -> factcheck_ocr (RapidOCR)   -> on-screen text observations
          -> alignment + exact merge (merge.py) -> labelled, time-ordered text
          -> factcheck.graph.run_pipeline (claim extraction with a video note, validation with the
             conservative duplicate check, extraction-reliability gate, retrieval, judge, verdict)
          -> claim provenance (modality, timestamps, segment/observation ids) -> report

Speech and screen text are both extracted before anything is compared;
de-duplication saves downstream verification, not extraction. Nothing in the
frames other than text is interpreted.
"""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from factcheck.config import Config
from factcheck.graph import run_pipeline
from factcheck.llm import LLMClient
from factcheck.models import Claim, Report
from factcheck.retrieval import SearchProvider
from factcheck.textnorm import original_span
from factcheck_ocr.engines import OcrEngine
from factcheck_ocr.quality import DEFAULT_RELIABLE_SCORE

from .frames import read_screen_text
from .media import (
    Deadline,
    ExtractionTimeout,
    PartialAudio,
    VideoInputError,
    display_rotation,
    extract_audio,
    iter_frames,
    probe,
    sample_times,
)
from .merge import Document, attach_subtitles, build_document, build_units
from .models import (
    AudioStage,
    Extraction,
    Limits,
    Modality,
    ScreenStage,
    StageStatus,
    TranscriptSegment,
    Unit,
)
from .textmatch import claims_duplicate
from .transcribe import DEFAULT_SPEECH_THRESHOLD, Transcriber, TranscriptionFailed

DEFAULT_MAX_TEXT_CHARS = 6000

VIDEO_SOURCE_NOTE = """About this text: it was extracted automatically from a video. Lines under [speech] are an
automatic speech transcript; lines under [on-screen text] were read by OCR from sampled video frames; lines under
[speech, also shown on screen] were spoken and also shown on screen (with identical wording or as subtitles). The
labels are not part of the content: never include a label in a source_excerpt.
- Transcripts and OCR can contain recognition errors. Do not correct, complete or reinterpret any word, name or
  number, and do not guess missing words. Copy source_excerpt exactly as the characters appear below.
- The same claim may be both spoken and shown on screen. Extract it once for EACH place it appears, each with its
  own source_excerpt, and when two places make the same claim use exactly the same statement wording for both.
  Duplicates are merged afterwards by code. If the spoken and on-screen versions differ in any detail (a number,
  date, name, unit, negation), write each statement exactly as its own source says.
- Speech fillers, greetings, questions, calls to subscribe or like, and opinions are not factual claims.
- Classify garbled or fragmentary text, and slogans or headlines that are not complete verifiable assertions, as
  VAGUE. Do not add context from outside the text to complete them.
- The video is the material being fact-checked. It is not evidence that its own claims are true."""

TIMESTAMP_NOTE = ("Speech times come from the speech recogniser and are approximate (typically within about half a "
                  "second). On-screen times are the sampled frames in which the text was seen: it may have appeared up "
                  "to one sampling interval earlier and stayed up to one interval later.")
CONFIDENCE_NOTE = ("Speech and OCR confidence describe how reliably words were recognised. They are not evidence that "
                   "anything said or shown is true, and they are not part of the evidence-strength score.")

_DUP_REASON = re.compile(r"^near-duplicate of (C\d+)$")


class SourceSpan(BaseModel):
    """Where a claim's source text was spoken or shown."""

    modality: Modality
    unit_id: str
    start: float
    end: float
    text: str
    confidence: float
    reliable: bool
    segment_id: Optional[int] = None
    observation_id: Optional[int] = None
    frame_times: list[float] = Field(default_factory=list)
    word_level: bool = False  # True when start/end were narrowed to the excerpt's words


class MergedClaim(BaseModel):
    statement: str
    source_excerpt: str


class ClaimSource(BaseModel):
    claim_id: str
    located: bool
    modalities: list[Modality] = Field(default_factory=list)
    spans: list[SourceSpan] = Field(default_factory=list)
    reliable: bool = False
    merged: list[MergedClaim] = Field(default_factory=list)  # duplicates folded into this claim


class Timing(BaseModel):
    probe_s: float = 0.0
    audio_extract_s: float = 0.0
    transcribe_s: float = 0.0
    frames_ocr_s: float = 0.0
    extraction_total_s: float = 0.0
    fact_check_s: float = 0.0
    total_s: float = 0.0


class VideoReport(BaseModel):
    input_kind: str = "video"
    extraction: Extraction
    checked_text: Optional[str] = None  # exactly what was given to claim extraction
    omitted_units: list[str] = Field(default_factory=list)  # unit ids over the text limit (not checked)
    fact_check: Optional[Report] = None
    not_checked_reason: Optional[str] = None
    claim_sources: list[ClaimSource] = Field(default_factory=list)
    duplicates_merged: int = 0
    timing: Timing = Field(default_factory=Timing)
    timestamp_note: str = TIMESTAMP_NOTE
    confidence_note: str = CONFIDENCE_NOTE


# ---------------------------------------------------------------- extraction


def _audio_status(segments: list[TranscriptSegment]) -> StageStatus:
    if not segments:
        return StageStatus.NO_CONTENT
    reliable = sum(s.reliable for s in segments)
    return StageStatus.OK if reliable == len(segments) else StageStatus.UNRELIABLE if reliable == 0 else StageStatus.LOW_CONFIDENCE


def run_audio(path: Path, has_audio: bool, make_transcriber: Optional[Callable[[], Transcriber]],
              limits: Limits, deadline: Deadline, timing: Timing, threshold: float) -> AudioStage:
    if make_transcriber is None:
        return AudioStage(status=StageStatus.SKIPPED, threshold=threshold)
    if not has_audio:
        return AudioStage(status=StageStatus.NO_STREAM, threshold=threshold, notes=["the video has no audio track"])
    transcriber = make_transcriber()  # model problems raise here (configuration error, exit 2)
    stage = AudioStage(status=StageStatus.NO_CONTENT, model=transcriber.name, compute_type=transcriber.compute_type,
                       threshold=threshold)
    with tempfile.TemporaryDirectory(prefix="factcheck-video-") as tmp:
        wav = Path(tmp) / "audio.wav"
        started = time.perf_counter()
        try:
            stage.audio_seconds = round(extract_audio(path, wav, limits, deadline), 2)
        except PartialAudio as exc:
            stage.audio_seconds = round(exc.seconds, 2)
            stage.notes.append(f"{exc}; only the first {exc.seconds:.0f} s of audio were transcribed")
        except ExtractionTimeout as exc:
            stage.status, stage.error = StageStatus.FAILED, str(exc)
            return stage
        except VideoInputError as exc:
            stage.status, stage.error = StageStatus.FAILED, str(exc)
            return stage
        timing.audio_extract_s = round(time.perf_counter() - started, 2)
        if not stage.audio_seconds:
            stage.notes.append("the audio track is empty")
            return stage
        started = time.perf_counter()
        try:
            tr = transcriber.transcribe(wav, deadline)
        except TranscriptionFailed as exc:
            stage.status, stage.error = StageStatus.FAILED, f"transcription failed: {exc}"
            return stage
        finally:
            timing.transcribe_s = round(time.perf_counter() - started, 2)
    stage.seconds = round(timing.audio_extract_s + timing.transcribe_s, 2)
    stage.segments = [s for s in tr.segments if s.text.strip()]
    stage.language, stage.language_probability = tr.language, tr.language_probability
    stage.notes += tr.notes
    stage.status = _audio_status(stage.segments)
    if stage.status == StageStatus.NO_CONTENT:
        stage.notes.append("no speech was recognised (silence, music or noise)")
    return stage


def run_screen(path: Path, has_video: bool, duration: Optional[float], rotation: int,
               make_engine: Optional[Callable[[], OcrEngine]], limits: Limits, deadline: Deadline,
               threshold: float, min_score: float, noise_filter: bool = True) -> ScreenStage:
    if make_engine is None:
        return ScreenStage(status=StageStatus.SKIPPED, threshold=threshold)
    if not has_video:
        return ScreenStage(status=StageStatus.NO_STREAM, threshold=threshold, notes=["the file has no video track"])
    engine = make_engine()
    times = sample_times(duration, limits.frame_interval_s, limits.max_frames)
    interval = limits.frame_interval_s if len(times) < 2 else round(times[1] - times[0], 3)
    stage = read_screen_text(iter_frames(path, times, limits, deadline, rotation), engine, threshold=threshold,
                             min_score=min_score, interval=interval, planned=len(times), deadline=deadline,
                             noise_filter=noise_filter)
    if interval > limits.frame_interval_s + 1e-6:
        stage.notes.append(f"sampling interval widened to {interval:.2f} s to stay within {limits.max_frames} frames")
    if stage.status == StageStatus.NO_CONTENT and stage.frames_sampled:
        stage.notes.append("no text was recognised in the sampled frames")
    return stage


def extract_video(path: Path, *, limits: Optional[Limits] = None,
                  make_transcriber: Optional[Callable[[], Transcriber]] = None,
                  make_engine: Optional[Callable[[], OcrEngine]] = None,
                  speech_threshold: float = DEFAULT_SPEECH_THRESHOLD,
                  ocr_threshold: float = DEFAULT_RELIABLE_SCORE, min_score: float = 0.5,
                  timing: Optional[Timing] = None, noise_filter: bool = True) -> Extraction:
    """Probe the file, transcribe its audio and read its on-screen text.

    ``make_transcriber`` / ``make_engine`` = None skips that modality. They are
    factories so a model is only loaded when the file actually has that stream.
    Raises ``VideoInputError`` for unusable files; model/dependency problems
    propagate from the factories.
    """
    limits = limits or Limits()
    timing = timing or Timing()
    started = time.perf_counter()
    deadline = Deadline(limits.extract_timeout_s)
    media = probe(path, limits)
    timing.probe_s = round(time.perf_counter() - started, 2)
    warnings: list[str] = []
    if media.duration_s is None:
        warnings.append("the container does not state a duration; frame sampling starts at 0 s only")

    audio = run_audio(path, media.has_audio, make_transcriber, limits, deadline, timing, speech_threshold)
    rotation = display_rotation(path) if media.has_video and make_engine is not None else 0
    media.rotation = rotation
    screen_started = time.perf_counter()
    try:
        screen = run_screen(path, media.has_video, media.duration_s, rotation, make_engine, limits, deadline,
                            ocr_threshold, min_score, noise_filter)
    except ExtractionTimeout as exc:
        screen = ScreenStage(status=StageStatus.FAILED, threshold=ocr_threshold, error=str(exc))
    timing.frames_ocr_s = round(time.perf_counter() - screen_started, 2)

    interval = screen.interval_s or limits.frame_interval_s
    units, merges = build_units(audio.segments, screen.observations, interval)
    units, subtitles = attach_subtitles(units, interval)
    timing.extraction_total_s = round(time.perf_counter() - started, 2)
    for stage, name in ((audio, "audio"), (screen, "on-screen text")):
        if stage.status == StageStatus.FAILED:
            warnings.append(f"{name} extraction failed: {stage.error}")
    return Extraction(media=media, audio=audio, screen=screen, units=units, exact_merges=merges,
                      subtitle_merges=subtitles, seconds=timing.extraction_total_s, warnings=warnings)


# ---------------------------------------------------------------- provenance


def _narrow_to_words(seg: TranscriptSegment, excerpt: str) -> Optional[tuple[float, float]]:
    """Start/end of the excerpt's words inside one segment, when word timestamps exist."""
    if not seg.words:
        return None
    joined, offsets = "", []
    for w in seg.words:
        piece = w.text.strip()
        if not piece:
            continue
        if joined and w.text[:1].isspace():  # no leading space: continues the previous word (" 50" + "%")
            joined += " "
        offsets.append((len(joined), len(joined) + len(piece), w))
        joined += piece
    span = original_span(joined, excerpt)
    if span is None:
        return None
    a = joined.find(span)
    b = a + len(span)
    hit = [w for (s, e, w) in offsets if s < b and e > a]
    return (hit[0].start, hit[-1].end) if hit else None


def _spans(units: list[Unit], excerpt: str, segments: dict[int, TranscriptSegment]) -> list[SourceSpan]:
    out = []
    for u in units:
        for ref in u.refs:
            span = SourceSpan(modality=ref.modality, unit_id=u.unit_id, start=ref.start, end=ref.end, text=ref.text,
                              confidence=ref.confidence, reliable=ref.reliable, segment_id=ref.segment_id,
                              observation_id=ref.observation_id, frame_times=ref.frame_times)
            if ref.segment_id is not None and len(units) == 1 and ref.segment_id in segments:
                narrowed = _narrow_to_words(segments[ref.segment_id], excerpt)
                if narrowed:
                    span.start, span.end, span.word_level = narrowed[0], narrowed[1], True
            out.append(span)
    return out


def trace(claim_id: str, excerpt: str, doc: Document, segments: dict[int, TranscriptSegment]) -> ClaimSource:
    units = doc.locate(excerpt)
    if not units:
        return ClaimSource(claim_id=claim_id, located=False)
    spans = _spans(units, excerpt, segments)
    return ClaimSource(claim_id=claim_id, located=True,
                       modalities=sorted({s.modality for s in spans}, key=lambda m: m.value),
                       spans=spans, reliable=any(u.reliable for u in units))


def _describe(units: list[Unit]) -> str:
    parts = []
    for u in units:
        for r in u.refs:
            kind = "speech" if r.modality == Modality.SPEECH else "on-screen text"
            parts.append(f"{kind} at {fmt_time(r.start)} (confidence {r.confidence:.2f})")
    return "; ".join(parts[:3])


def reliability_gate(doc: Document):
    """Withhold a claim whose every source is below its confidence threshold (``extraction:`` reason)."""

    def gate(claim: Claim) -> Optional[str]:
        units = doc.locate(claim.source_excerpt)
        if not units:
            return "extraction: the claim's source text could not be traced to a speech segment or on-screen text"
        if not any(u.reliable for u in units):
            return (f"extraction: its source ({_describe(units)}) is below the recognition confidence threshold, so "
                    "the words may have been misheard or misread; it was not searched or judged")
        return None

    return gate


def fmt_time(seconds: float) -> str:
    s = max(0.0, seconds)
    m, sec = divmod(s, 60)
    return f"{int(m):02d}:{sec:04.1f}"


# ---------------------------------------------------------------- workflow


def verify_video(path: Path, config: Config | None = None, llm: LLMClient | None = None,
                 search: SearchProvider | None = None, *, limits: Optional[Limits] = None,
                 make_transcriber: Optional[Callable[[], Transcriber]] = None,
                 make_engine: Optional[Callable[[], OcrEngine]] = None,
                 speech_threshold: float = DEFAULT_SPEECH_THRESHOLD, ocr_threshold: float = DEFAULT_RELIABLE_SCORE,
                 min_score: float = 0.5, max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
                 fact_check: bool = True, noise_filter: bool = True) -> VideoReport:
    """Extract speech and on-screen text from ``path`` and fact-check it.

    Extraction runs before any credential is needed: a video with nothing to
    check is reported without any model or search call. ``fact_check=False``
    stops after extraction (no API keys needed).
    """
    started = time.perf_counter()
    timing = Timing()
    extraction = extract_video(path, limits=limits, make_transcriber=make_transcriber, make_engine=make_engine,
                               speech_threshold=speech_threshold, ocr_threshold=ocr_threshold, min_score=min_score,
                               timing=timing, noise_filter=noise_filter)
    vr = VideoReport(extraction=extraction, timing=timing)
    if not extraction.units:
        vr.not_checked_reason = _nothing_reason(extraction)
        timing.total_s = round(time.perf_counter() - started, 2)
        return vr
    reliable_units = [u for u in extraction.units if u.reliable]
    if not reliable_units:
        vr.not_checked_reason = ("all extracted speech and on-screen text is below the recognition confidence "
                                 "thresholds, so nothing could be checked reliably. Try a clearer recording.")
    config = config or Config.from_env()
    limit = min(max_text_chars, config.max_input_chars)
    doc = build_document(extraction.units, limit)
    vr.checked_text = doc.text
    vr.omitted_units = [u.unit_id for u in doc.omitted]
    if doc.omitted:
        extraction.warnings.append(
            f"{len(doc.omitted)} later segment(s)/text block(s) from {fmt_time(doc.omitted[0].start)} onward were not "
            f"checked: the extracted text exceeds the {limit}-character limit (--max-text-chars)")
    if not fact_check or vr.not_checked_reason:
        if not fact_check:
            vr.not_checked_reason = "extraction only (--extract-only); no fact-check was run"
        timing.total_s = round(time.perf_counter() - started, 2)
        return vr

    fc_started = time.perf_counter()
    report = run_pipeline(doc.text, config, llm, search, source_note=VIDEO_SOURCE_NOTE,
                          claim_gate=reliability_gate(doc), duplicate_check=claims_duplicate)
    timing.fact_check_s = round(time.perf_counter() - fc_started, 2)
    segments = {s.segment_id: s for s in extraction.audio.segments}
    sources = {r.claim_id: trace(r.claim_id, r.source_excerpt, doc, segments) for r in report.claims}
    vr.duplicates_merged = _fold_duplicates(report, sources, doc, segments)
    for stage, name in ((extraction.audio, "speech"), (extraction.screen, "on-screen text")):
        if stage.status in (StageStatus.LOW_CONFIDENCE, StageStatus.UNRELIABLE):
            report.warnings.append(f"some {name} is below its recognition confidence threshold")
    vr.fact_check = report
    vr.claim_sources = [sources[r.claim_id] for r in report.claims]
    timing.total_s = round(time.perf_counter() - started, 2)
    return vr


def _fold_duplicates(report: Report, sources: dict[str, ClaimSource], doc: Document,
                     segments: dict[int, TranscriptSegment]) -> int:
    """Turn validation's "near-duplicate of Cn" drops into provenance of Cn."""
    kept, merged = [], 0
    for d in report.dropped_claims:
        m = _DUP_REASON.match(d.reason)
        if not m or m.group(1) not in sources:
            kept.append(d)
            continue
        src = sources[m.group(1)]
        src.merged.append(MergedClaim(statement=d.statement, source_excerpt=d.source_excerpt))
        extra = trace(src.claim_id, d.source_excerpt, doc, segments)
        known = {(s.unit_id, s.modality, s.segment_id, s.observation_id) for s in src.spans}
        for s in extra.spans:
            if (s.unit_id, s.modality, s.segment_id, s.observation_id) not in known:
                src.spans.append(s)
        src.spans.sort(key=lambda s: s.start)
        src.modalities = sorted({s.modality for s in src.spans}, key=lambda m: m.value)
        src.located = src.located or extra.located
        merged += 1
    if merged:
        report.dropped_claims = kept
        report.warnings = [w for w in report.warnings if not w.endswith("dropped by validation")]
        if kept:
            report.warnings.insert(0, f"{len(kept)} extracted claim(s) dropped by validation")
        report.warnings.append(f"{merged} duplicate claim(s) (same claim spoken and/or shown again) were merged and "
                               "checked once")
    return merged


def _nothing_reason(ex: Extraction) -> str:
    parts = []
    for stage, name in ((ex.audio, "speech"), (ex.screen, "on-screen text")):
        if stage.status == StageStatus.SKIPPED:
            parts.append(f"{name}: skipped")
        elif stage.status == StageStatus.NO_STREAM:
            parts.append(f"{name}: no {'audio' if name == 'speech' else 'video'} track")
        elif stage.status == StageStatus.FAILED:
            parts.append(f"{name}: failed ({stage.error})")
        else:
            parts.append(f"{name}: none found")
    return "no speech or on-screen text was extracted (" + "; ".join(parts) + ")"

