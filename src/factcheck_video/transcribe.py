"""Local speech recognition with faster-whisper (CTranslate2 Whisper).

Models are fetched from the Hugging Face Hub (Systran/faster-whisper-*) only
when the caller allows downloads; otherwise only the local cache
(``~/.cache/huggingface/hub``) is used, so a run never starts a large download
by surprise. Downloaded models are reused on later runs.

A transcript is a machine estimate of what was said. Segment confidence is
Whisper's own average token log-probability, mapped to 0-1; it is a weak
signal (a fluent hallucination can score well), never evidence of truth.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Optional, Protocol

from .media import Deadline, DependencyMissing
from .models import TranscriptSegment, Word

DEFAULT_MODEL = "base"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_SPEECH_THRESHOLD = 0.40
# Approximate download sizes of the multilingual CTranslate2 conversions (float16 weights;
# int8 quantisation happens at load time, so the download is the same for every compute type).
MODEL_SIZES_MB = {"tiny": 75, "base": 145, "small": 484, "medium": 1530, "large-v3": 3090, "turbo": 1620}
# Whisper's own heuristics (openai/whisper transcribe.py): repetitive output compresses
# too well; high no-speech probability with low log-probability is silence.
MAX_COMPRESSION_RATIO = 2.4
NO_SPEECH_PROB = 0.6


class ModelUnavailable(RuntimeError):
    """The speech model is not available locally and downloads are not allowed (exit code 2)."""


class TranscriptionFailed(RuntimeError):
    pass


class Transcript:
    def __init__(self, segments: list[TranscriptSegment], language: Optional[str],
                 language_probability: Optional[float], audio_seconds: Optional[float], notes: list[str]):
        self.segments = segments
        self.language = language
        self.language_probability = language_probability
        self.audio_seconds = audio_seconds
        self.notes = notes


class Transcriber(Protocol):
    name: str
    compute_type: str

    def transcribe(self, wav: Path, deadline: Optional[Deadline] = None) -> Transcript: ...


def segment_reliable(confidence: float, no_speech_prob: float, compression_ratio: float, threshold: float) -> bool:
    if compression_ratio > MAX_COMPRESSION_RATIO:
        return False
    if no_speech_prob > NO_SPEECH_PROB and confidence < 0.5:
        return False
    return confidence >= threshold


def model_status(model: str) -> str:
    """'cached', 'not downloaded' or 'faster-whisper not installed' (never downloads)."""
    if importlib.util.find_spec("faster_whisper") is None:
        return "faster-whisper not installed"
    if Path(model).is_dir():
        return "local directory"
    from faster_whisper.utils import download_model

    try:
        download_model(model, local_files_only=True)
        return "cached"
    except Exception:  # noqa: BLE001 - huggingface_hub raises several types for "not in cache"
        return "not downloaded"


class WhisperTranscriber:
    """faster-whisper on CPU. ``model`` is a size name (tiny/base/small/...) or a local model directory."""

    def __init__(self, model: str = DEFAULT_MODEL, *, compute_type: str = DEFAULT_COMPUTE_TYPE,
                 allow_download: bool = False, threads: int = 0, language: Optional[str] = None,
                 beam_size: int = 5, vad: bool = True, word_timestamps: bool = True,
                 threshold: float = DEFAULT_SPEECH_THRESHOLD):
        if importlib.util.find_spec("faster_whisper") is None:
            raise DependencyMissing('faster-whisper is not installed; run: pip install -e ".[video]"')
        from faster_whisper import WhisperModel

        self.name = model
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad = vad
        self.word_timestamps = word_timestamps
        self.threshold = threshold
        local = Path(model).is_dir()
        try:
            self._model = WhisperModel(model, device="cpu", compute_type=compute_type, cpu_threads=threads,
                                       local_files_only=not (allow_download or local))
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            if not allow_download and not local and _looks_missing(exc):
                size = MODEL_SIZES_MB.get(model)
                hint = f" (about {size} MB" + ", from huggingface.co/Systran)" if size else ""
                raise ModelUnavailable(
                    f"speech model {model!r} is not downloaded. Re-run with --allow-download to fetch it once{hint}; "
                    "it is cached under ~/.cache/huggingface/hub and reused.") from None
            raise ModelUnavailable(f"speech model {model!r} failed to load: {msg[:300]}") from None

    def transcribe(self, wav: Path, deadline: Optional[Deadline] = None) -> Transcript:
        notes: list[str] = []
        try:
            segments, info = self._model.transcribe(
                str(wav), language=self.language, beam_size=self.beam_size, vad_filter=self.vad,
                word_timestamps=self.word_timestamps, condition_on_previous_text=False)
            out: list[TranscriptSegment] = []
            for seg in segments:  # a generator: decoding happens while iterating
                if deadline is not None and deadline.remaining() <= 0:
                    notes.append(f"transcription stopped at {seg.end:.0f} s: extraction time limit reached")
                    break
                out.append(to_segment(len(out) + 1, seg, self.threshold))
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionFailed(f"{type(exc).__name__}: {str(exc)[:300]}") from None
        return Transcript(split_sentences(out), info.language, round(info.language_probability, 3),
                          round(info.duration, 2), notes)


def to_segment(segment_id: int, seg: Any, threshold: float) -> TranscriptSegment:
    confidence = round(math.exp(min(0.0, float(seg.avg_logprob))), 3)
    words = [Word(text=w.word, start=round(w.start, 2), end=round(w.end, 2), probability=round(w.probability, 3))
             for w in (seg.words or [])]
    return TranscriptSegment(
        segment_id=segment_id, start=round(float(seg.start), 2), end=round(float(seg.end), 2),
        text=" ".join(str(seg.text).split()), confidence=confidence,
        no_speech_prob=round(float(seg.no_speech_prob), 3), compression_ratio=round(float(seg.compression_ratio), 2),
        reliable=segment_reliable(confidence, float(seg.no_speech_prob), float(seg.compression_ratio), threshold),
        words=words,
    )


_SENTENCE_END = (".", "?", "!", "。", "？", "！")


def split_sentences(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Split segments with word timestamps at sentence ends, renumbering ids from 1.

    Whisper can put several sentences, even ones spoken seconds apart, into one
    segment. Sentence-sized pieces get accurate start/end times (from the words) and
    line up with subtitles. Confidence values are the parent segment's (Whisper
    reports them per segment). Segments without word timestamps are kept whole.
    """
    out: list[TranscriptSegment] = []
    for s in segments:
        pieces: list[list[Word]] = [[]]
        for i, w in enumerate(s.words):
            pieces[-1].append(w)
            nxt = s.words[i + 1].text.strip() if i + 1 < len(s.words) else ""
            # "U.S. has" / "Dr. Smith": only split when the next word starts a new sentence
            if w.text.strip().endswith(_SENTENCE_END) and nxt and not nxt[0].islower():
                pieces.append([])
        pieces = [p for p in pieces if p]
        if len(pieces) <= 1:
            out.append(s.model_copy(update={"segment_id": len(out) + 1}))
            continue
        for p in pieces:
            text = " ".join("".join(w.text for w in p).split())
            out.append(s.model_copy(update={"segment_id": len(out) + 1, "start": p[0].start, "end": p[-1].end,
                                            "text": text, "words": p}))
    return out


def _looks_missing(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("localentrynotfound", "local_files_only", "cannot find", "not found",
                                   "snapshot", "offline", "does not appear to have"))

