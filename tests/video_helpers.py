"""Shared helpers for the video tests: tiny generated videos, a fake transcriber and a colour-keyed fake OCR engine."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from factcheck_ocr.engines import EngineOutput, RawBox
from factcheck_video.models import TranscriptSegment, Word
from factcheck_video.transcribe import Transcript, segment_reliable

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("make_video_samples", ROOT / "eval" / "make_video_samples.py")
samples = importlib.util.module_from_spec(_spec)
sys.modules["make_video_samples"] = samples
_spec.loader.exec_module(samples)

write_video = samples.write_video
Caption = samples.Caption
RATE = samples.RATE

RED, GREEN, BLUE, GREY = (200, 30, 30), (30, 160, 60), (30, 60, 200), (90, 90, 90)


def tone(seconds: float, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def seg(i: int, start: float, end: float, text: str, conf: float = 0.9, words: bool = True,
        threshold: float = 0.4) -> TranscriptSegment:
    ws = []
    if words:
        toks = text.split()
        step = (end - start) / max(1, len(toks))
        ws = [Word(text=" " + w, start=round(start + k * step, 2), end=round(start + (k + 1) * step, 2), probability=0.9)
              for k, w in enumerate(toks)]
    return TranscriptSegment(segment_id=i, start=start, end=end, text=text, confidence=conf,
                             reliable=segment_reliable(conf, 0.0, 1.5, threshold), words=ws)


class FakeTranscriber:
    """Returns scripted segments; records the WAV it was given (to check temp-file cleanup)."""

    name = "fake-whisper"
    compute_type = "int8"

    def __init__(self, segments: list[TranscriptSegment], language: str = "en", error: Exception | None = None):
        self.segments = segments
        self.language = language
        self.error = error
        self.wav_paths: list[Path] = []
        self.wav_sizes: list[int] = []

    def transcribe(self, wav: Path, deadline=None) -> Transcript:
        self.wav_paths.append(wav)
        self.wav_sizes.append(wav.stat().st_size)
        if self.error:
            raise self.error
        return Transcript(list(self.segments), self.language, 0.99, 1.0, [])


def box(text: str, score: float, y: float = 20.0) -> RawBox:
    return RawBox(text, score, [(20, y), (20 + 11 * len(text), y), (20 + 11 * len(text), y + 22), (20, y + 22)])


class ColorEngine:
    """Fake OCR: the frame's background colour (pixel 2,2) selects the scripted lines."""

    name = "color-fake"

    def __init__(self, by_color: dict[tuple, list[tuple[str, float]]]):
        self.by_color = by_color
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        px = image.convert("RGB").getpixel((2, 2))
        best = min(self.by_color, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, px)), default=None)
        if best is None or sum((a - b) ** 2 for a, b in zip(best, px)) > 3 * 40 ** 2:
            return EngineOutput(boxes=[])
        return EngineOutput(boxes=[box(t, s, 20 + 40 * k) for k, (t, s) in enumerate(self.by_color[best])])
