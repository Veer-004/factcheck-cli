"""Media layer (PyAV/FFmpeg) on real, tiny generated videos: validation, probing, audio, frame sampling."""

from __future__ import annotations

import time
import wave

import pytest

pytest.importorskip("av")
pytest.importorskip("PIL")

from video_helpers import BLUE, GREEN, RED, Caption, silence, tone, write_video  # noqa: E402

from factcheck_video.media import (  # noqa: E402
    Deadline,
    ExtractionTimeout,
    VideoInputError,
    extract_audio,
    iter_frames,
    probe,
    sample_times,
    validate_path,
)
from factcheck_video.models import Limits  # noqa: E402


@pytest.fixture
def av_video(tmp_path):
    return write_video(tmp_path / "clip.mp4", duration=3.0, audio=tone(3.0),
                       backgrounds=[(0, 1, RED), (1, 2, GREEN), (2, 3, BLUE)])


def test_probe_reports_streams_and_metadata(av_video):
    info = probe(av_video, Limits())
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (640, 360)
    assert info.video_codec == "h264" and info.audio_codec == "aac"
    assert info.fps == pytest.approx(10, abs=0.1)
    assert 2.8 <= info.duration_s <= 3.2


def test_video_without_audio_stream(tmp_path):
    path = write_video(tmp_path / "mute.mp4", duration=2.0, audio=None)
    info = probe(path, Limits())
    assert info.has_video and not info.has_audio
    with pytest.raises(VideoInputError, match="no audio stream"):
        extract_audio(path, tmp_path / "a.wav", Limits())


def test_mov_container_is_supported(tmp_path):
    path = write_video(tmp_path / "clip.mov", duration=1.0, audio=tone(1.0))
    info = probe(path, Limits())
    assert info.has_video and info.has_audio and "mov" in info.container


def test_extract_audio_writes_16k_mono_wav(av_video, tmp_path):
    out = tmp_path / "a.wav"
    seconds = extract_audio(av_video, out, Limits())
    assert 2.8 <= seconds <= 3.3
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert abs(w.getnframes() / 16000 - seconds) < 0.01


def test_extract_audio_respects_duration_cap(tmp_path):
    path = write_video(tmp_path / "long.mp4", duration=4.0, audio=tone(4.0))
    seconds = extract_audio(path, tmp_path / "a.wav", Limits(max_duration_s=1.5))
    assert seconds == pytest.approx(1.5, abs=0.01)


def test_silent_audio_track_extracts_fine(tmp_path):
    path = write_video(tmp_path / "quiet.mp4", duration=2.0, audio=silence(2.0))
    assert extract_audio(path, tmp_path / "a.wav", Limits()) > 1.5


@pytest.mark.parametrize("name,content,match", [
    ("missing.mp4", None, "not found"),
    ("empty.mp4", b"", "empty"),
    ("notes.txt", b"hello", "unsupported file extension"),
    ("fake.mp4", b"this is not a video at all" * 10, "not a readable video"),
])
def test_invalid_inputs(tmp_path, name, content, match):
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(VideoInputError, match=match):
        probe(path, Limits())


def test_directory_is_rejected(tmp_path):
    d = tmp_path / "dir.mp4"
    d.mkdir()
    with pytest.raises(VideoInputError, match="not a file"):
        validate_path(d, Limits())


def test_truncated_corrupt_video(av_video, tmp_path):
    bad = tmp_path / "bad.mp4"
    data = av_video.read_bytes()
    bad.write_bytes(data[: len(data) // 4])
    with pytest.raises(VideoInputError):
        probe(bad, Limits())


def test_size_and_duration_limits(av_video):
    with pytest.raises(VideoInputError, match="--max-duration"):
        probe(av_video, Limits(max_duration_s=1.0))
    big = Limits(max_file_mb=1)
    validate_path(av_video, big)  # tiny file passes
    av_video.write_bytes(av_video.read_bytes() + b"\0" * (1024 * 1024))
    with pytest.raises(VideoInputError, match="--max-file-mb"):
        validate_path(av_video, big)


def test_sample_times_plan():
    assert sample_times(10.0, 2.0, 100) == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert sample_times(None, 2.0, 100) == [0.0]
    capped = sample_times(100.0, 1.0, 10)
    assert len(capped) == 10 and capped[-1] < 100 and capped[1] == pytest.approx(10.0)


def test_iter_frames_returns_timestamped_frames(av_video):
    frames = list(iter_frames(av_video, [0.0, 1.0, 2.0], Limits()))
    assert [round(f.time, 1) for f in frames] == [0.0, 1.0, 2.0]
    colours = [f.image.getpixel((2, 2)) for f in frames]
    for got, want in zip(colours, (RED, GREEN, BLUE)):
        assert sum(abs(a - b) for a, b in zip(got, want)) < 60


def test_iter_frames_seeks_far_targets(tmp_path):
    path = write_video(tmp_path / "long.mp4", duration=12.0, backgrounds=[(0, 10, RED), (10, 12, BLUE)])
    [first, late] = list(iter_frames(path, [0.0, 10.5], Limits()))
    assert late.time == pytest.approx(10.5, abs=0.11)
    assert sum(abs(a - b) for a, b in zip(late.image.getpixel((2, 2)), BLUE)) < 60


def test_frames_are_downscaled(tmp_path):
    path = write_video(tmp_path / "big.mp4", duration=1.0, size=(1280, 720))
    [f] = list(iter_frames(path, [0.0], Limits(frame_max_side=640)))
    assert max(f.image.size) == 640


def test_frame_sampling_stops_at_deadline(av_video):
    d = Deadline(0.001)
    time.sleep(0.01)
    with pytest.raises(ExtractionTimeout):
        list(iter_frames(av_video, [0.0, 1.0], Limits(), deadline=d))


def test_captions_render_in_video(tmp_path):
    # sanity check for the generator used by the evaluation set
    path = write_video(tmp_path / "cap.mp4", duration=1.0, captions=[Caption(0, 1, "HELLO", "center")])
    [f] = list(iter_frames(path, [0.0], Limits()))
    centre = f.image.crop((250, 150, 390, 210)).convert("L")
    assert centre.getextrema()[1] > 200  # white text is present
