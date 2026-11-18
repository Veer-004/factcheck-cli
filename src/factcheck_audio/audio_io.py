"""Audio file validation, probing and decoding (PyAV / bundled FFmpeg libraries).

Decoding to the 16 kHz mono WAV that Whisper expects reuses the video module's
streaming decoder (``factcheck_video.media.extract_audio``), so both modules
hand Whisper exactly the same kind of input. Nothing is loaded into memory as a
whole; the WAV lives in a temporary directory that is always removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from factcheck_video.media import (  # shared, unchanged
    SAMPLE_RATE,
    VIDEO_EXTENSIONS,
    Deadline,
    DependencyMissing,
    ExtractionTimeout,
    PartialAudio,
    VideoInputError,
)
from factcheck_video.media import extract_audio as _decode_to_wav

from .models import AudioInfo, AudioLimits

__all__ = ["AUDIO_EXTENSIONS", "AudioInputError", "DependencyMissing", "ExtractionTimeout", "Deadline",
           "PartialAudio", "decode_to_wav", "probe_audio", "validate_path", "SAMPLE_RATE"]

# Formats checked in the tests are mp3, wav, m4a, flac, ogg and opus; the rest are other
# audio containers FFmpeg reads.
AUDIO_EXTENSIONS = frozenset(
    ".mp3 .wav .wave .m4a .m4b .aac .flac .ogg .oga .opus .wma .aif .aiff .aifc .amr .mka .weba .caf .ac3 .mp2"
    .split())


class AudioInputError(ValueError):
    """Unusable input file (CLI exit code 2)."""


def _av() -> Any:
    try:
        import av
    except ImportError:
        raise DependencyMissing('PyAV is not installed; run: pip install -e ".[audio]"') from None
    return av


def validate_path(path: Path, limits: AudioLimits) -> int:
    """Existence, extension and size checks before any decoding. Returns the size in bytes."""
    if not path.exists():
        raise AudioInputError(f"audio file not found: {path}")
    if not path.is_file():
        raise AudioInputError(f"not a file: {path}")
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS and ext not in AUDIO_EXTENSIONS:
        raise AudioInputError(f"{path.name} looks like a video file; use factcheck-video for videos")
    if ext not in AUDIO_EXTENSIONS:
        raise AudioInputError(f"unsupported file extension {path.suffix or '(none)'!r}; expected one of: "
                              + ", ".join(sorted(AUDIO_EXTENSIONS)))
    size = path.stat().st_size
    if size == 0:
        raise AudioInputError(f"audio file is empty (0 bytes): {path}")
    if size > limits.max_file_mb * 1024 * 1024:
        raise AudioInputError(f"audio file is {size / 2**20:.0f} MB; the limit is {limits.max_file_mb} MB "
                              "(--max-file-mb)")
    return size


def _is_cover_art(stream: Any) -> bool:
    disposition = getattr(stream, "disposition", None)
    try:
        if disposition is not None and "attached_pic" in str(disposition).lower():
            return True
    except Exception:  # noqa: BLE001 - best effort
        pass
    return stream.codec_context.name in ("mjpeg", "png", "bmp", "gif", "webp")


def probe_audio(path: Path, limits: AudioLimits) -> tuple[AudioInfo, list[str]]:
    """Validate and read the container header. Returns the metadata and any notes."""
    size = validate_path(path, limits)
    av = _av()
    notes: list[str] = []
    try:
        container = av.open(str(path), mode="r")
    except av.error.FFmpegError as exc:
        raise AudioInputError(f"cannot read {path.name}: not a readable audio file or it is corrupted "
                              f"({type(exc).__name__})") from None
    except OSError as exc:
        raise AudioInputError(f"cannot open {path.name}: {exc.strerror or type(exc).__name__}") from None
    with container as c:
        info = AudioInfo(path=str(path), container=c.format.name if c.format else "", size_bytes=size)
        if not c.streams.audio:
            raise AudioInputError(f"{path.name} contains no audio stream")
        a = c.streams.audio[0]
        info.codec = a.codec_context.name
        info.sample_rate = a.codec_context.sample_rate or None
        info.channels = a.codec_context.channels or None
        info.bit_rate = a.codec_context.bit_rate or c.bit_rate or None
        if c.duration is not None:
            info.duration_s = round(c.duration / av.time_base, 3)
        elif a.duration is not None and a.time_base:
            info.duration_s = round(float(a.duration * a.time_base), 3)
        for vs in c.streams.video:
            if _is_cover_art(vs):
                info.has_cover_art = True
            else:
                notes.append("the file also has a video track; only its audio is used (factcheck-video reads "
                             "on-screen text)")
                break
        if len(c.streams.audio) > 1:
            notes.append(f"the file has {len(c.streams.audio)} audio streams; only the first is transcribed")
    if info.duration_s is not None and info.duration_s > limits.max_duration_s:
        raise AudioInputError(f"audio is {info.duration_s:.0f} s long; the limit is {limits.max_duration_s:.0f} s "
                              "(--max-duration)")
    if info.duration_s is None:
        notes.append("the file does not state its duration; the length limit is enforced while decoding")
    return info, notes


def decode_to_wav(path: Path, out_wav: Path, limits: AudioLimits, deadline: Optional[Deadline] = None) -> float:
    """16 kHz mono 16-bit WAV of the first audio stream; returns the seconds written.

    Raises ``AudioInputError`` when nothing can be decoded, ``PartialAudio`` when
    decoding failed part-way (the WAV holds what was decoded before the error) and
    ``ExtractionTimeout`` at the time limit.
    """
    try:
        return _decode_to_wav(path, out_wav, limits, deadline)  # type: ignore[arg-type]  (uses max_duration_s only)
    except VideoInputError as exc:
        raise AudioInputError(str(exc).replace("video/audio", "audio")) from None
