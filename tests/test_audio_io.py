"""Audio file validation, probing and decoding on real encoded files (PyAV; no speech model)."""

from __future__ import annotations

import wave

import pytest

pytest.importorskip("av")

from audio_helpers import FORMATS, noise, silence, tone, write_audio  # noqa: E402

from factcheck_audio.audio_io import AUDIO_EXTENSIONS, AudioInputError, decode_to_wav, probe_audio  # noqa: E402
from factcheck_audio.models import AudioLimits  # noqa: E402


@pytest.mark.parametrize("ext", sorted(FORMATS))
def test_supported_formats_probe_and_decode_to_16k_mono(tmp_path, ext):
    path = write_audio(tmp_path / f"clip{ext}", tone(3.0))
    info, notes = probe_audio(path, AudioLimits())
    assert info.duration_s == pytest.approx(3.0, abs=0.06) and info.codec and info.sample_rate
    assert info.channels == 1 and info.size_bytes == path.stat().st_size and notes == []
    out = tmp_path / "out.wav"
    seconds = decode_to_wav(path, out, AudioLimits())
    assert seconds == pytest.approx(3.0, abs=0.06)
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.getnframes() == round(seconds * 16000)


def test_stereo_is_downmixed(tmp_path):
    path = write_audio(tmp_path / "stereo.mp3", tone(2.0), stereo=True)
    info, _ = probe_audio(path, AudioLimits())
    assert info.channels == 2
    out = tmp_path / "o.wav"
    decode_to_wav(path, out, AudioLimits())
    with wave.open(str(out)) as w:
        assert w.getnchannels() == 1


def test_common_extensions_are_accepted():
    assert {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"} <= AUDIO_EXTENSIONS


@pytest.mark.parametrize("name,content,needle", [
    ("empty.mp3", b"", "empty"),
    ("garbage.mp3", b"this is not audio at all" * 50, "corrupted"),
    ("garbage.wav", b"RIFF\x00\x00\x00\x00WAVEjunkjunk" * 10, "corrupted"),
    ("notes.txt", b"hello", "unsupported file extension"),
    ("clip.mp4", b"\0" * 100, "use factcheck-video"),
])
def test_invalid_files_are_rejected_with_a_clear_message(tmp_path, name, content, needle):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(AudioInputError, match=needle):
        probe_audio(path, AudioLimits())


def test_missing_file_and_directory(tmp_path):
    with pytest.raises(AudioInputError, match="not found"):
        probe_audio(tmp_path / "nope.mp3", AudioLimits())
    (tmp_path / "dir.mp3").mkdir()
    with pytest.raises(AudioInputError, match="not a file"):
        probe_audio(tmp_path / "dir.mp3", AudioLimits())


def test_size_and_duration_limits(tmp_path):
    path = write_audio(tmp_path / "long.wav", tone(5.0))
    with pytest.raises(AudioInputError, match="--max-duration"):
        probe_audio(path, AudioLimits(max_duration_s=2.0))
    big = tmp_path / "big.wav"
    big.write_bytes(b"\0" * (2 * 1024 * 1024))
    with pytest.raises(AudioInputError, match="--max-file-mb"):
        probe_audio(big, AudioLimits(max_file_mb=1))


def test_wav_with_no_samples_decodes_to_zero_seconds(tmp_path):
    path = tmp_path / "header_only.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
    probe_audio(path, AudioLimits())  # a valid (if empty) file
    assert decode_to_wav(path, tmp_path / "o.wav", AudioLimits()) == 0.0


def test_truncated_file_yields_the_decodable_part(tmp_path):
    path = write_audio(tmp_path / "cut.mp3", tone(6.0))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])  # e.g. an interrupted download
    seconds = decode_to_wav(path, tmp_path / "o.wav", AudioLimits())
    assert 1.0 < seconds < 6.0


def test_decoding_stops_at_the_duration_limit(tmp_path):
    path = write_audio(tmp_path / "a.flac", tone(4.0))
    assert decode_to_wav(path, tmp_path / "o.wav", AudioLimits(max_duration_s=1.5)) == pytest.approx(1.5)


@pytest.mark.parametrize("samples", [silence(2.0), noise(2.0)], ids=["silence", "noise"])
def test_silence_and_noise_are_valid_audio(tmp_path, samples):
    path = write_audio(tmp_path / "x.ogg", samples)
    info, _ = probe_audio(path, AudioLimits())
    assert decode_to_wav(path, tmp_path / "o.wav", AudioLimits()) == pytest.approx(2.0, abs=0.05)
    assert info.codec == "opus"
