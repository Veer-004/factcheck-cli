"""Shared helpers for the audio tests: real encoded audio files (via PyAV) and fake transcribers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from factcheck_audio.transcribe import AudioTranscript
from factcheck_video.models import TranscriptSegment
from video_helpers import seg  # noqa: F401  (re-exported: scripted transcript segments with word times)

RATE = 16000

# extension -> (container format, codec, encoder sample rate)
FORMATS = {
    ".wav": ("wav", "pcm_s16le", 16000),
    ".mp3": ("mp3", "libmp3lame", 44100),
    ".m4a": ("ipod", "aac", 44100),
    ".flac": ("flac", "flac", 16000),
    ".ogg": ("ogg", "libopus", 48000),
    ".opus": ("opus", "libopus", 48000),
}


def tone(seconds: float, freq: float = 440.0, amp: float = 0.3, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, rate: int = RATE) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.float32)


def noise(seconds: float, amp: float = 0.2, seed: int = 0, rate: int = RATE) -> np.ndarray:
    return (amp * np.random.default_rng(seed).standard_normal(int(seconds * rate))).astype(np.float32)


def write_audio(path: Path, samples: np.ndarray, rate: int = RATE, *, stereo: bool = False) -> Path:
    """Encode mono float32 ``samples`` into the format given by ``path``'s extension."""
    import av

    fmt, codec, enc_rate = FORMATS[path.suffix.lower()]
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    layout = "stereo" if stereo else "mono"
    if stereo:
        pcm = np.repeat(pcm, 2)  # interleaved L/R
    with av.open(str(path), "w", format=fmt) as out:
        stream = out.add_stream(codec, rate=enc_rate)
        stream.layout = layout
        resampler = av.AudioResampler(format=stream.format.name, layout=layout, rate=enc_rate)
        step = 1024
        channels = 2 if stereo else 1
        for i in range(0, len(pcm) // channels, step):
            chunk = pcm[i * channels:(i + step) * channels]
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout=layout)
            frame.sample_rate = rate
            for f in resampler.resample(frame):
                for packet in stream.encode(f):
                    out.mux(packet)
        for f in resampler.resample(None):
            for packet in stream.encode(f):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return path


class FakeTranscriber:
    """Scripted transcript; records the WAV it was given (to check decoding and temp-file cleanup)."""

    name = "fake-whisper"
    compute_type = "int8"

    def __init__(self, segments: list[TranscriptSegment], language: str = "en", probability: float = 0.99,
                 error: Exception | None = None, method: str = "detected"):
        self.segments = segments
        self.language = language
        self.probability = probability
        self.error = error
        self.method = method
        self.wav_paths: list[Path] = []
        self.wav_sizes: list[int] = []

    def transcribe(self, wav: Path, deadline=None) -> AudioTranscript:
        self.wav_paths.append(wav)
        self.wav_sizes.append(wav.stat().st_size)
        if self.error:
            raise self.error
        english = self.language == "en" or self.method != "detected"
        return AudioTranscript(list(self.segments) if english else [], self.language, self.probability, 1.0, [],
                               english_probability=self.probability if self.language == "en" else 0.01,
                               method=self.method, decoded=english)
