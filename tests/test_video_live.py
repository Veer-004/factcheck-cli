"""Real speech recognition + real OCR on a generated video (no web calls).

Opt-in: set FACTCHECK_VIDEO_LIVE_TESTS=1. Needs Windows (SAPI speech synthesis for
the test audio), faster-whisper with the model already cached (FACTCHECK_VIDEO_MODEL,
default "base"; no download is attempted) and RapidOCR.
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = [
    pytest.mark.video_live,
    pytest.mark.skipif(os.environ.get("FACTCHECK_VIDEO_LIVE_TESTS") != "1", reason="set FACTCHECK_VIDEO_LIVE_TESTS=1"),
    pytest.mark.skipif(sys.platform != "win32", reason="test speech is synthesised with Windows SAPI"),
]


def test_real_extraction_merges_identical_speech_and_subtitle(tmp_path):
    from video_helpers import Caption, samples, write_video

    from factcheck_ocr.engines import create_engine
    from factcheck_video.models import Modality, StageStatus
    from factcheck_video.pipeline import extract_video
    from factcheck_video.transcribe import WhisperTranscriber

    sentence = "Water boils at 100 degrees Celsius at sea level."
    audio = samples.build_audio(samples.Case("t", 6.0, "", speech=[samples.Speech(0.5, sentence)]))
    path = write_video(tmp_path / "live.mp4", duration=6.0, audio=audio, captions=[Caption(0.3, 5.0, sentence)])
    model = os.environ.get("FACTCHECK_VIDEO_MODEL", "base")
    ex = extract_video(path, make_transcriber=lambda: WhisperTranscriber(model),
                       make_engine=lambda: create_engine("rapid"))
    assert ex.audio.status == StageStatus.OK and ex.audio.language == "en"
    [segment] = ex.audio.segments
    assert "boils at 100 degrees" in segment.text.lower()
    assert 0.0 <= segment.start <= 1.5 and 3.0 <= segment.end <= 6.0
    assert ex.screen.status == StageStatus.OK and len(ex.screen.observations) == 1
    assert ex.exact_merges == 1
    [unit] = ex.units
    assert set(unit.modalities) == {Modality.SPEECH, Modality.SCREEN}
