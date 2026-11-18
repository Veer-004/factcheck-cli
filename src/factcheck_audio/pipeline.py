"""Audio fact-checking workflow.

    audio file -> validation + probe (PyAV/FFmpeg) -> 16 kHz mono WAV (temp)
               -> faster-whisper: language identification (English only) -> timed transcript segments
               -> transcript document (plain prose, no timestamps)
               -> factcheck.graph.run_pipeline (claim extraction with an audio note, validation with the
                  conservative duplicate check, transcription-reliability gate, retrieval, judge, citation
                  validation, verdict)
               -> claim traceability (timestamps, segments, transcript, confidence) -> report

The fact-checking engine is the shared one; nothing here judges claims.
Transcription confidence only decides whether a claim's words are reliable
enough to search for; it never enters the verdict or the evidence-strength score.
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
from factcheck.models import Citation, Claim, Report, Verdict
from factcheck.retrieval import SearchProvider
from factcheck_video.textmatch import claims_duplicate  # conservative duplicate rule (negations, numbers)

from .audio_io import AudioInputError, Deadline, ExtractionTimeout, PartialAudio, decode_to_wav, probe_audio
from .document import Occurrence, TranscriptDocument, build_document
from .models import (
    SUPPORTED_LANGUAGE,
    AudioInfo,
    AudioLimits,
    LanguageCheck,
    Transcription,
    TranscriptionStatus,
    TranscriptSegment,
)
from .transcribe import (
    DEFAULT_SPEECH_THRESHOLD,
    UNCERTAIN_LANGUAGE_P,
    Transcriber,
    Transcript,
    TranscriptionFailed,
    mark_non_latin,
)

DEFAULT_MAX_TEXT_CHARS = 6000

AUDIO_SOURCE_NOTE = """About this text: it is an automatic speech-to-text transcript of an English audio recording
(for example a podcast, interview, lecture or voice note). It has no punctuation guarantees and no speaker labels.
- Transcripts can contain recognition errors. Do not correct, complete or reinterpret any word, name or number, and
  do not guess missing or unclear words. Copy source_excerpt exactly as the characters appear below.
- Extract only complete, independently verifiable factual assertions (history, science, politics and geopolitics,
  health, technology, numbers, dates, statistics, measurements). Keep negations, qualifiers and numbers exactly as
  spoken.
- Greetings, filler words, questions, opinions, predictions, jokes, promotional calls to action (subscribe, like,
  buy, visit, use a code) and sponsor messages are not factual claims.
- Classify fragments, sentences that trail off, garbled passages and statements that depend on unstated context as
  VAGUE. Do not add context from outside the transcript to complete them.
- A speaker may repeat a claim. Extract it once for EACH place it is said, each with its own source_excerpt, and use
  exactly the same statement wording each time; repeats are merged afterwards by code. If a repeat differs in any
  detail (a number, date, name, unit or negation), write each statement exactly as it was said.
- The recording is the material being fact-checked. It is not evidence that its own claims are true."""

TIMESTAMP_NOTE = ("Times come from the speech recogniser and are approximate (typically within about half a second). "
                  "'(word timing)' means the range was narrowed to the claim's own words.")
CONFIDENCE_NOTE = ("Transcription confidence describes how reliably the words were recognised. It is not evidence "
                   "that anything said is true, and it is not part of the evidence-strength score.")

_DUP_REASON = re.compile(r"^near-duplicate of (C\d+)$")


# ---------------------------------------------------------------- report models


class SpokenSpan(BaseModel):
    """One place a claim was spoken."""

    start: float
    end: float
    word_level: bool = False  # True when start/end were narrowed to the excerpt's own words
    segment_ids: list[int]
    transcript: str  # the full text of the segment(s), as transcribed
    confidence: float  # lowest segment confidence in the span (recognition, not truth)
    reliable: bool


class MergedClaim(BaseModel):
    statement: str
    source_excerpt: str


class ClaimSource(BaseModel):
    claim_id: str
    located: bool
    spans: list[SpokenSpan] = Field(default_factory=list)
    reliable: bool = False
    merged: list[MergedClaim] = Field(default_factory=list)  # repeats folded into this claim


class CitedSource(BaseModel):
    url: str
    title: str = ""
    quote: str


class TracedClaim(BaseModel):
    """Everything about one claim in one place: what was said, where, how reliably, and the verdict."""

    claim_id: str
    statement: str
    transcript_excerpt: str
    start: Optional[float] = None
    end: Optional[float] = None
    segment_ids: list[int] = Field(default_factory=list)
    transcript: str = ""
    transcription_confidence: Optional[float] = None
    times_spoken: int = 0
    verdict: Verdict
    evidence_strength: Optional[int] = None
    supporting: list[CitedSource] = Field(default_factory=list)
    contradicting: list[CitedSource] = Field(default_factory=list)
    citations_rejected_by_validation: int = 0
    failure_reason: Optional[str] = None


class Timing(BaseModel):
    probe_s: float = 0.0
    decode_s: float = 0.0
    transcribe_s: float = 0.0
    extraction_total_s: float = 0.0
    fact_check_s: float = 0.0
    total_s: float = 0.0
    real_time_factor: Optional[float] = None  # transcription seconds per second of audio


class AudioReport(BaseModel):
    input_kind: str = "audio"
    media: AudioInfo
    transcription: Transcription
    warnings: list[str] = Field(default_factory=list)
    checked_text: Optional[str] = None  # exactly what was given to claim extraction
    omitted_segment_ids: list[int] = Field(default_factory=list)  # over the text limit (not checked)
    fact_check: Optional[Report] = None
    not_checked_reason: Optional[str] = None
    claim_sources: list[ClaimSource] = Field(default_factory=list)
    traced_claims: list[TracedClaim] = Field(default_factory=list)
    duplicates_merged: int = 0
    timing: Timing = Field(default_factory=Timing)
    timestamp_note: str = TIMESTAMP_NOTE
    confidence_note: str = CONFIDENCE_NOTE


# ---------------------------------------------------------------- transcription


def _status(segments: list[TranscriptSegment]) -> TranscriptionStatus:
    if not segments:
        return TranscriptionStatus.NO_SPEECH
    reliable = sum(s.reliable for s in segments)
    if reliable == len(segments):
        return TranscriptionStatus.OK
    return TranscriptionStatus.UNRELIABLE if reliable == 0 else TranscriptionStatus.LOW_CONFIDENCE


def language_check(tr: Transcript) -> LanguageCheck:
    method = getattr(tr, "method", "detected")
    check = LanguageCheck(detected=tr.language, probability=tr.language_probability,
                          english_probability=getattr(tr, "english_probability", None), method=method)
    if method != "detected":
        check.accepted = True
        check.note = ("language identification skipped (--assume-english)" if method == "assumed"
                      else "English-only speech model: language identification not available")
        return check
    check.accepted = tr.language == SUPPORTED_LANGUAGE
    if not check.accepted:
        p = f" (p={tr.language_probability:.2f})" if tr.language_probability is not None else ""
        check.note = (f"the detected language is {tr.language or 'unknown'}{p}; only English is supported, so the "
                      "audio was not transcribed. If it is English (e.g. strongly accented), re-run with "
                      "--assume-english")
    elif tr.language_probability is not None and tr.language_probability < UNCERTAIN_LANGUAGE_P:
        check.note = (f"English was identified with low certainty (p={tr.language_probability:.2f}); if parts of the "
                      "recording are in another language, their transcript is unreliable")
    return check


def transcribe_audio(path: Path, *, limits: Optional[AudioLimits] = None,
                     make_transcriber: Callable[[], Transcriber],
                     speech_threshold: float = DEFAULT_SPEECH_THRESHOLD,
                     timing: Optional[Timing] = None) -> tuple[AudioInfo, Transcription, list[str]]:
    """Validate, decode and transcribe ``path``. Returns (metadata, transcription, warnings).

    Raises ``AudioInputError`` for unusable files (before any model is loaded);
    model/dependency problems propagate from ``make_transcriber``.
    """
    limits = limits or AudioLimits()
    timing = timing or Timing()
    started = time.perf_counter()
    deadline = Deadline(limits.extract_timeout_s)
    info, warnings = probe_audio(path, limits)
    timing.probe_s = round(time.perf_counter() - started, 2)
    stage = Transcription(status=TranscriptionStatus.NO_SPEECH, threshold=speech_threshold)
    with tempfile.TemporaryDirectory(prefix="factcheck-audio-") as tmp:
        wav = Path(tmp) / "audio.wav"
        t0 = time.perf_counter()
        try:
            stage.audio_seconds = round(decode_to_wav(path, wav, limits, deadline), 2)
        except PartialAudio as exc:
            stage.audio_seconds = round(exc.seconds, 2)
            warnings.append(f"{exc}: the file looks damaged; only the first {exc.seconds:.0f} s were transcribed")
        except ExtractionTimeout as exc:
            stage.status, stage.error = TranscriptionStatus.FAILED, str(exc)
            return info, stage, warnings
        timing.decode_s = round(time.perf_counter() - t0, 2)
        if not stage.audio_seconds:
            raise AudioInputError(f"{path.name} contains no decodable audio samples (empty audio)")
        if info.duration_s is None and stage.audio_seconds >= limits.max_duration_s - 0.05:
            warnings.append(f"only the first {limits.max_duration_s:.0f} s were transcribed (--max-duration)")
        transcriber = make_transcriber()  # model problems raise here (configuration error, exit 2)
        stage.model, stage.compute_type = transcriber.name, transcriber.compute_type
        t0 = time.perf_counter()
        try:
            tr = transcriber.transcribe(wav, deadline)
        except TranscriptionFailed as exc:
            stage.status, stage.error = TranscriptionStatus.FAILED, f"transcription failed: {exc}"
            return info, stage, warnings
        finally:
            timing.transcribe_s = round(time.perf_counter() - t0, 2)
    if stage.audio_seconds:
        timing.real_time_factor = round(timing.transcribe_s / stage.audio_seconds, 3)
    stage.notes += tr.notes
    stage.language = language_check(tr)
    if not stage.language.accepted:
        stage.status = TranscriptionStatus.UNSUPPORTED_LANGUAGE
        return info, stage, warnings
    if stage.language.note:
        stage.notes.append(stage.language.note)
    segments, non_latin = mark_non_latin([s for s in tr.segments if s.text.strip()])
    if non_latin:
        stage.notes.append(f"{non_latin} segment(s) are not in Latin script (probably not English) and will not be "
                           "checked")
    stage.segments = segments
    stage.status = _status(segments)
    if stage.status == TranscriptionStatus.NO_SPEECH:
        stage.notes.append("no speech was recognised (silence, music or noise)")
    return info, stage, warnings


# ---------------------------------------------------------------- traceability


def fmt_time(seconds: float) -> str:
    s = max(0.0, seconds)
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return f"{int(h)}:{int(m):02d}:{sec:04.1f}" if h >= 1 else f"{int(m):02d}:{sec:04.1f}"


def _span(o: Occurrence) -> SpokenSpan:
    return SpokenSpan(start=o.start, end=o.end, word_level=o.word_level, segment_ids=[s.segment_id for s in o.segments],
                      transcript=" ".join(s.text for s in o.segments),
                      confidence=min(s.confidence for s in o.segments),
                      reliable=any(s.reliable for s in o.segments))


def trace(claim_id: str, excerpt: str, doc: TranscriptDocument) -> ClaimSource:
    spans = [_span(o) for o in doc.locate(excerpt)]
    return ClaimSource(claim_id=claim_id, located=bool(spans), spans=spans, reliable=any(s.reliable for s in spans))


def reliability_gate(doc: TranscriptDocument):
    """Withhold a claim whose every spoken source is below the confidence threshold (``transcription:`` reason)."""

    def gate(claim: Claim) -> Optional[str]:
        found = doc.locate(claim.source_excerpt)
        if not found:
            return "transcription: the claim's source text could not be traced to a transcript segment"
        if not any(s.reliable for o in found for s in o.segments):
            where = "; ".join(f"{fmt_time(o.start)} (confidence {min(s.confidence for s in o.segments):.2f})"
                              for o in found[:3])
            return (f"transcription: the speech it comes from ({where}) is below the recognition confidence "
                    "threshold, so the words may have been misheard; it was not searched or judged")
        return None

    return gate


def _fold_duplicates(report: Report, sources: dict[str, ClaimSource], doc: TranscriptDocument) -> int:
    """Turn validation's "near-duplicate of Cn" drops into extra spoken places of Cn."""
    kept, merged = [], 0
    for d in report.dropped_claims:
        m = _DUP_REASON.match(d.reason)
        if not m or m.group(1) not in sources:
            kept.append(d)
            continue
        src = sources[m.group(1)]
        src.merged.append(MergedClaim(statement=d.statement, source_excerpt=d.source_excerpt))
        known = {(tuple(s.segment_ids), s.start) for s in src.spans}
        for s in trace(src.claim_id, d.source_excerpt, doc).spans:
            if (tuple(s.segment_ids), s.start) not in known:
                src.spans.append(s)
        src.spans.sort(key=lambda s: s.start)
        src.located = bool(src.spans)
        src.reliable = any(s.reliable for s in src.spans)
        merged += 1
    if merged:
        report.dropped_claims = kept
        report.warnings = [w for w in report.warnings if not w.endswith("dropped by validation")]
        if kept:
            report.warnings.insert(0, f"{len(kept)} extracted claim(s) dropped by validation")
        report.warnings.append(f"{merged} repeated claim(s) (the same claim said again) were merged and checked once")
    return merged


def _cited(cites: list[Citation]) -> list[CitedSource]:
    return [CitedSource(url=c.url, title=c.title, quote=c.quote) for c in cites]


def traced_claims(report: Report, sources: dict[str, ClaimSource]) -> list[TracedClaim]:
    out = []
    for r in report.claims:
        src = sources.get(r.claim_id)
        first = src.spans[0] if src and src.spans else None
        out.append(TracedClaim(
            claim_id=r.claim_id, statement=r.statement, transcript_excerpt=r.source_excerpt,
            start=first.start if first else None, end=first.end if first else None,
            segment_ids=first.segment_ids if first else [], transcript=first.transcript if first else "",
            transcription_confidence=first.confidence if first else None,
            times_spoken=len(src.spans) if src else 0, verdict=r.verdict, evidence_strength=r.score,
            supporting=_cited(r.supporting), contradicting=_cited(r.refuting),
            citations_rejected_by_validation=r.rejected_citations, failure_reason=r.failure_reason))
    return out


# ---------------------------------------------------------------- workflow


def verify_audio(path: Path, config: Config | None = None, llm: LLMClient | None = None,
                 search: SearchProvider | None = None, *, make_transcriber: Callable[[], Transcriber],
                 limits: Optional[AudioLimits] = None, speech_threshold: float = DEFAULT_SPEECH_THRESHOLD,
                 max_text_chars: int = DEFAULT_MAX_TEXT_CHARS, fact_check: bool = True) -> AudioReport:
    """Transcribe ``path`` (English only) and fact-check what was said.

    Transcription runs before any credential is needed: audio with nothing to
    check is reported without any model or search call. ``fact_check=False``
    stops after transcription (no API keys needed).
    """
    started = time.perf_counter()
    timing = Timing()
    info, stage, warnings = transcribe_audio(path, limits=limits, make_transcriber=make_transcriber,
                                             speech_threshold=speech_threshold, timing=timing)
    timing.extraction_total_s = round(time.perf_counter() - started, 2)
    ar = AudioReport(media=info, transcription=stage, warnings=warnings, timing=timing)

    def done(reason: Optional[str]) -> AudioReport:
        ar.not_checked_reason = reason
        timing.total_s = round(time.perf_counter() - started, 2)
        return ar

    if stage.status == TranscriptionStatus.FAILED:
        return done(f"transcription failed: {stage.error}")
    if stage.status == TranscriptionStatus.UNSUPPORTED_LANGUAGE:
        return done(stage.language.note)
    if not stage.segments:
        return done("no speech was recognised in the audio (silence, music or noise)")

    config = config or Config.from_env()
    limit = min(max_text_chars, config.max_input_chars)
    doc = build_document(stage.segments, limit)
    ar.checked_text = doc.text
    ar.omitted_segment_ids = [s.segment_id for s in doc.omitted]
    if doc.omitted:
        ar.warnings.append(f"{len(doc.omitted)} later segment(s) from {fmt_time(doc.omitted[0].start)} onward were not "
                           f"checked: the transcript exceeds the {limit}-character limit (--max-text-chars)")
    if not any(s.reliable for s in doc.segments):
        return done("all recognised speech is below the transcription confidence threshold, so nothing could be "
                    "checked reliably. Try a clearer recording.")
    if not fact_check:
        return done("transcription only (--extract-only); no fact-check was run")

    fc_started = time.perf_counter()
    report = run_pipeline(doc.text, config, llm, search, source_note=AUDIO_SOURCE_NOTE,
                          claim_gate=reliability_gate(doc), duplicate_check=claims_duplicate)
    timing.fact_check_s = round(time.perf_counter() - fc_started, 2)
    sources = {r.claim_id: trace(r.claim_id, r.source_excerpt, doc) for r in report.claims}
    ar.duplicates_merged = _fold_duplicates(report, sources, doc)
    if stage.status in (TranscriptionStatus.LOW_CONFIDENCE, TranscriptionStatus.UNRELIABLE):
        report.warnings.append("some speech is below the transcription confidence threshold")
    ar.fact_check = report
    ar.claim_sources = [sources[r.claim_id] for r in report.claims]
    ar.traced_claims = traced_claims(report, sources)
    return done(None)
