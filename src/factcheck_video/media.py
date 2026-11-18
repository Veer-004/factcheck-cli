"""Media access through PyAV (Python bindings to the FFmpeg libraries).

PyAV's wheels bundle FFmpeg, so no separate ``ffmpeg`` executable is needed.
Everything streams: audio is decoded packet by packet into a temporary 16 kHz
mono WAV, and frames are decoded only around the sampled timestamps, so a large
video is never loaded into memory.
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from .models import Limits, MediaInfo

SAMPLE_RATE = 16000  # what Whisper expects
VIDEO_EXTENSIONS = frozenset(".mp4 .m4v .mov .mkv .webm .avi .mpg .mpeg .ts .3gp .wmv .flv".split())
SEEK_AHEAD_S = 4.0  # seek instead of decoding forward when the next sample is further away than this


class VideoInputError(ValueError):
    """Unusable input file (CLI exit code 2)."""


class DependencyMissing(RuntimeError):
    """An optional dependency is not installed (CLI exit code 2)."""


class ExtractionTimeout(RuntimeError):
    """The extraction time limit was reached."""


class PartialAudio(RuntimeError):
    """Audio decoding failed part-way; ``seconds`` of audio were written before the error."""

    def __init__(self, seconds: float, message: str):
        super().__init__(message)
        self.seconds = seconds


class Deadline:
    def __init__(self, seconds: float):
        self.limit = seconds
        self.end = time.monotonic() + seconds

    def remaining(self) -> float:
        return self.end - time.monotonic()

    def check(self, what: str) -> None:
        if time.monotonic() > self.end:
            raise ExtractionTimeout(f"extraction time limit of {self.limit:.0f} s reached during {what}")


def _av() -> Any:
    try:
        import av
    except ImportError:
        raise DependencyMissing('PyAV is not installed; run: pip install -e ".[video]"') from None
    return av


def _open(path: Path) -> Any:
    av = _av()
    try:
        return av.open(str(path), mode="r")
    except av.error.FFmpegError as exc:
        raise VideoInputError(f"cannot read {path.name}: not a readable video/audio file ({type(exc).__name__})") from None
    except OSError as exc:
        raise VideoInputError(f"cannot open {path.name}: {exc.strerror or type(exc).__name__}") from None


def validate_path(path: Path, limits: Limits) -> int:
    """Existence, type and size checks before any decoding. Returns the size in bytes."""
    if not path.exists():
        raise VideoInputError(f"video file not found: {path}")
    if not path.is_file():
        raise VideoInputError(f"not a file: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise VideoInputError(f"unsupported file extension {path.suffix or '(none)'!r}; expected one of: "
                              + ", ".join(sorted(VIDEO_EXTENSIONS)))
    size = path.stat().st_size
    if size == 0:
        raise VideoInputError(f"video file is empty: {path}")
    if size > limits.max_file_mb * 1024 * 1024:
        raise VideoInputError(f"video file is {size / 2**20:.0f} MB; the limit is {limits.max_file_mb} MB (--max-file-mb)")
    return size


def probe(path: Path, limits: Limits) -> MediaInfo:
    size = validate_path(path, limits)
    av = _av()
    with _open(path) as c:
        info = MediaInfo(path=str(path), container=c.format.name if c.format else "", size_bytes=size)
        if c.duration is not None:
            info.duration_s = round(c.duration / av.time_base, 3)
        if c.streams.video:
            vs = c.streams.video[0]
            info.has_video = True
            info.video_codec = vs.codec_context.name
            info.width, info.height = vs.codec_context.width or None, vs.codec_context.height or None
            if vs.average_rate:
                info.fps = round(float(vs.average_rate), 3)
            if info.duration_s is None and vs.duration is not None and vs.time_base:
                info.duration_s = round(float(vs.duration * vs.time_base), 3)
        if c.streams.audio:
            a = c.streams.audio[0]
            info.has_audio = True
            info.audio_codec = a.codec_context.name
            info.sample_rate = a.codec_context.sample_rate or None
            info.channels = a.codec_context.channels or None
            if info.duration_s is None and a.duration is not None and a.time_base:
                info.duration_s = round(float(a.duration * a.time_base), 3)
    if not info.has_video and not info.has_audio:
        raise VideoInputError(f"{path.name} contains no video or audio stream")
    if info.duration_s is not None and info.duration_s > limits.max_duration_s:
        raise VideoInputError(f"video is {info.duration_s:.0f} s long; the limit is {limits.max_duration_s:.0f} s "
                              "(--max-duration)")
    return info


def extract_audio(path: Path, out_wav: Path, limits: Limits, deadline: Optional[Deadline] = None) -> float:
    """Decode the first audio stream into a 16 kHz mono 16-bit WAV. Returns seconds written.

    Stops at ``limits.max_duration_s`` (the duration in the container header can be
    missing or wrong, so the limit is enforced on the decoded samples too).
    """
    av = _av()
    max_samples = int(limits.max_duration_s * SAMPLE_RATE)
    written = 0
    with _open(path) as c:
        if not c.streams.audio:
            raise VideoInputError(f"{path.name} has no audio stream")
        stream = c.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)

            def write(frames: list[Any]) -> None:
                nonlocal written
                for f in frames:
                    data = f.to_ndarray().reshape(-1)[: max_samples - written]
                    w.writeframes(data.tobytes())
                    written += len(data)

            try:
                for i, frame in enumerate(c.decode(stream)):
                    if deadline is not None and i % 200 == 0:
                        deadline.check("audio extraction")
                    write(resampler.resample(frame))
                    if written >= max_samples:
                        break
                write(resampler.resample(None))  # flush
            except av.error.FFmpegError as exc:
                if written == 0:
                    raise VideoInputError(f"cannot decode the audio of {path.name} ({type(exc).__name__})") from None
                # keep what was decoded before the corruption; the caller reports a warning
                raise PartialAudio(written / SAMPLE_RATE, f"audio decoding stopped early ({type(exc).__name__})") from None
    return written / SAMPLE_RATE


def sample_times(duration: Optional[float], interval: float, max_frames: int) -> list[float]:
    """Sampling plan: every ``interval`` seconds starting at 0, capped at ``max_frames``.

    When the cap would cut the video short, the interval is widened so the samples
    still cover the whole duration (reported as a note by the caller).
    """
    if not duration or duration <= 0:
        return [0.0]
    n = int(duration // interval) + 1
    if n > max_frames:
        interval = duration / max_frames
        n = max_frames
    return [round(i * interval, 3) for i in range(n) if i * interval < duration] or [0.0]


@dataclass
class SampledFrame:
    requested: float
    time: float  # timestamp of the decoded frame
    image: Any  # PIL.Image (RGB), downscaled to frame_max_side


def iter_frames(path: Path, times: list[float], limits: Limits, deadline: Optional[Deadline] = None,
                rotation: int = 0) -> Iterator[SampledFrame]:
    """Yield the first decoded frame at or after each requested time (sorted ascending)."""
    av = _av()
    with _open(path) as c:
        if not c.streams.video:
            return
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        tb = float(vs.time_base) if vs.time_base else None
        pending = sorted(times)
        idx = 0
        last_time = -1.0
        while idx < len(pending):
            target = pending[idx]
            if deadline is not None:
                deadline.check("frame sampling")
            if tb and target - last_time > SEEK_AHEAD_S:
                try:
                    c.seek(int(target / tb), stream=vs, backward=True, any_frame=False)
                except av.error.FFmpegError:
                    pass  # unseekable stream: fall back to decoding forward
            found = False
            try:
                for frame in c.decode(vs):
                    t = frame.time
                    if t is None:
                        continue
                    last_time = t
                    if t + 1e-3 < target:
                        continue
                    # one decoded frame can satisfy several requested times when the video is sparse
                    image = _to_image(frame, limits.frame_max_side, rotation)
                    while idx < len(pending) and pending[idx] <= t + 1e-3:
                        idx += 1
                    yield SampledFrame(requested=target, time=round(t, 3), image=image)
                    found = True
                    break
            except av.error.FFmpegError:
                return  # corrupt tail: stop sampling; the caller notes fewer frames than planned
            if not found:
                return  # end of stream


def _to_image(frame: Any, max_side: int, rotation: int) -> Any:
    from PIL import Image

    im = frame.to_image()
    if rotation % 360:
        im = im.rotate(rotation, expand=True)  # display-matrix rotation (phone videos)
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return im


def display_rotation(path: Path) -> int:
    """Rotation from the video stream's display matrix (degrees counter-clockwise), or 0."""
    try:
        with _open(path) as c:
            if not c.streams.video:
                return 0
            for frame in c.decode(c.streams.video[0]):
                return int(round(getattr(frame, "rotation", 0) or 0))
    except Exception:  # noqa: BLE001 - best effort; rotation only affects OCR quality
        return 0
    return 0
