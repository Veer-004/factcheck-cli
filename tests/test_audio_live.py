"""Opt-in: real faster-whisper on synthesised English speech (no API calls).

    set FACTCHECK_AUDIO_LIVE_TESTS=1 && python -m pytest tests/test_audio_live.py

Needs Windows (SAPI voices for the speech) and the cached ``base`` model.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytestmark = [
    pytest.mark.audio_live,
    pytest.mark.skipif(os.environ.get("FACTCHECK_AUDIO_LIVE_TESTS") != "1", reason="set FACTCHECK_AUDIO_LIVE_TESTS=1"),
    pytest.mark.skipif(sys.platform != "win32", reason="speech synthesis uses Windows SAPI"),
]

RATE = 16000


@pytest.fixture(scope="module")
def transcriber():
    from factcheck_audio.transcribe import EnglishTranscriber, model_status

    if model_status("base") != "cached":
        pytest.skip("faster-whisper base is not downloaded")
    return EnglishTranscriber("base")


def speech_track(tmp_path, ext, parts, duration):
    from audio_helpers import write_audio
    from video_helpers import samples

    track = np.zeros(int(duration * RATE), dtype=np.float32)
    onsets = []
    for start, text in parts:
        clip = samples.tts(text) * 0.8
        a = int(start * RATE)
        track[a:a + len(clip)] += clip[: len(track) - a]
        loud = np.flatnonzero(np.abs(clip) > 0.02)
        onsets.append(start + loud[0] / RATE)
    return write_audio(tmp_path / f"speech{ext}", track), onsets


@pytest.mark.parametrize("ext", [".mp3", ".m4a"])
def test_english_speech_is_transcribed_with_accurate_times(tmp_path, transcriber, config, ext):
    from factcheck_audio.pipeline import transcribe_audio

    path, onsets = speech_track(tmp_path, ext, [(0.5, "The Eiffel Tower was completed in 1889."),
                                                (7.0, "The Great Wall of China is not visible from the Moon.")], 12.0)
    info, stage, _ = transcribe_audio(path, make_transcriber=lambda: transcriber)
    assert stage.language.detected == "en" and stage.language.accepted
    text = " ".join(s.text for s in stage.segments)
    assert "1889" in text and "not visible" in text
    starts = [s.start for s in stage.segments]
    for t in onsets:
        assert min(abs(t - s) for s in starts) < 0.8, (t, starts)
    assert all(s.words for s in stage.segments)  # word-level timestamps


def test_silence_produces_no_transcript(tmp_path, transcriber):
    from audio_helpers import write_audio

    from factcheck_audio.models import TranscriptionStatus
    from factcheck_audio.pipeline import transcribe_audio

    path = write_audio(tmp_path / "quiet.wav", np.zeros(5 * RATE, dtype=np.float32))
    _, stage, _ = transcribe_audio(path, make_transcriber=lambda: transcriber)
    assert stage.status == TranscriptionStatus.NO_SPEECH and stage.segments == []
