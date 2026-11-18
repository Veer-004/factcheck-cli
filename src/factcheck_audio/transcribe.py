"""English-only speech recognition, built on the video module's faster-whisper wrapper.

Model loading (local cache only unless downloads are allowed), segment
confidence and sentence splitting are reused unchanged from
``factcheck_video.transcribe``. What is added here is the English-only policy:

* Whisper identifies the spoken language first (over up to ~90 s of speech).
  When it is not English, decoding is skipped: no transcript is produced, so a
  non-English recording is never "translated" or force-decoded into English
  words that were not said.
* ``assume_english`` (or an English-only ``*.en`` model) skips identification
  and decodes as English; it is meant for accented English that Whisper
  misidentifies.
* Segments that come out in a non-Latin script are marked unreliable, so they
  are never fact-checked.

Transcription confidence is Whisper's own estimate of how well it recognised
the words. It is never evidence that what was said is true.
"""

from __future__ import annotations

import importlib.util
import unicodedata
from pathlib import Path
from typing import Optional

from factcheck_video.transcribe import (  # shared, unchanged
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_MODEL,
    DEFAULT_SPEECH_THRESHOLD,
    MODEL_SIZES_MB,
    ModelUnavailable,
    Transcriber,
    Transcript,
    TranscriptionFailed,
    WhisperTranscriber,
    model_status,
    split_sentences,
    to_segment,
)

from .audio_io import Deadline, DependencyMissing
from .models import SUPPORTED_LANGUAGE, TranscriptSegment

__all__ = ["DEFAULT_COMPUTE_TYPE", "DEFAULT_MODEL", "DEFAULT_SPEECH_THRESHOLD", "DEFAULT_THREADS", "MODEL_SIZES_MB",
           "AudioTranscript", "EnglishTranscriber", "ModelUnavailable", "Transcriber", "Transcript",
           "TranscriptionFailed", "mark_non_latin", "model_status"]

# 0 = CTranslate2's own default. Measured on an i5-12500H (163 s English MP3, base int8): 0 -> 9.0 s,
# 4 -> 9.3 s, 8 -> 10.5 s, 12 -> 11.4 s; more threads only added contention.
DEFAULT_THREADS = 0
LANGUAGE_DETECTION_SEGMENTS = 3  # 30-s windows of speech used to identify the language
UNCERTAIN_LANGUAGE_P = 0.6
MIN_LATIN_SHARE = 0.6  # a segment with fewer Latin letters than this is not English text


class AudioTranscript(Transcript):
    def __init__(self, segments: list[TranscriptSegment], language: Optional[str],
                 language_probability: Optional[float], audio_seconds: Optional[float], notes: list[str], *,
                 english_probability: Optional[float] = None, method: str = "detected", decoded: bool = True):
        super().__init__(segments, language, language_probability, audio_seconds, notes)
        self.english_probability = english_probability
        self.method = method
        self.decoded = decoded


def latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters)


def mark_non_latin(segments: list[TranscriptSegment]) -> tuple[list[TranscriptSegment], int]:
    """Mark segments written mostly in a non-Latin script as unreliable. Returns (segments, count)."""
    out, n = [], 0
    for s in segments:
        if s.reliable and latin_share(s.text) < MIN_LATIN_SHARE:
            s = s.model_copy(update={"reliable": False})
            n += 1
        out.append(s)
    return out, n


class EnglishTranscriber(WhisperTranscriber):
    """faster-whisper on CPU with the English-only policy described in the module docstring."""

    def __init__(self, model: str = DEFAULT_MODEL, *, compute_type: str = DEFAULT_COMPUTE_TYPE,
                 allow_download: bool = False, threads: int = DEFAULT_THREADS, beam_size: int = 5, vad: bool = True,
                 word_timestamps: bool = True, threshold: float = DEFAULT_SPEECH_THRESHOLD,
                 assume_english: bool = False):
        if importlib.util.find_spec("faster_whisper") is None:
            raise DependencyMissing('faster-whisper is not installed; run: pip install -e ".[audio]"')
        english_only_model = Path(model).name.endswith(".en")
        super().__init__(model, compute_type=compute_type, allow_download=allow_download, threads=threads,
                         language=SUPPORTED_LANGUAGE if (assume_english or english_only_model) else None,
                         beam_size=beam_size, vad=vad, word_timestamps=word_timestamps, threshold=threshold)
        self.method = "english-only model" if english_only_model else "assumed" if assume_english else "detected"

    def transcribe(self, wav: Path, deadline: Optional[Deadline] = None) -> AudioTranscript:
        notes: list[str] = []
        try:
            segments, info = self._model.transcribe(
                str(wav), language=self.language, beam_size=self.beam_size, vad_filter=self.vad,
                word_timestamps=self.word_timestamps, condition_on_previous_text=False,
                language_detection_segments=LANGUAGE_DETECTION_SEGMENTS)
            probs = dict(info.all_language_probs or [])
            en_p = probs.get(SUPPORTED_LANGUAGE)
            common = dict(english_probability=round(en_p, 3) if en_p is not None else None, method=self.method)
            if self.language is None and info.language != SUPPORTED_LANGUAGE:
                # Not English: stop before decoding anything (the segment generator is lazy).
                return AudioTranscript([], info.language, round(info.language_probability, 3), round(info.duration, 2),
                                       notes, decoded=False, **common)
            out: list[TranscriptSegment] = []
            for seg in segments:  # decoding happens while iterating
                if deadline is not None and deadline.remaining() <= 0:
                    notes.append(f"transcription stopped at {seg.end:.0f} s: extraction time limit reached")
                    break
                out.append(to_segment(len(out) + 1, seg, self.threshold))
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionFailed(f"{type(exc).__name__}: {str(exc)[:300]}") from None
        return AudioTranscript(split_sentences(out), info.language, round(info.language_probability, 3),
                               round(info.duration, 2), notes, **common)
