"""Regression guards: adding the audio module must not change the other three modules."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from factcheck.report import _failure_summary

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("reason,summary", [
    ("ocr: the image text is unreadable", "the image text it comes from could not be read reliably"),
    ("extraction: its source is below the threshold", "the speech or on-screen text it comes from may have been "
                                                      "misheard or misread"),
    ("judgment remained invalid after a repair attempt", "the AI model's answer failed validation twice"),
    ("LLM unavailable: 503", "the AI model did not respond in time or was unavailable"),
    ("search budget exhausted", "the per-run search or model budget was used up"),
    ("something odd", "an internal processing error occurred"),
    ("transcription: below the threshold", "the speech it comes from may have been misheard"),
])
def test_failure_summaries_existing_wording_is_unchanged(reason, summary):
    assert _failure_summary(reason).startswith(summary)


def test_four_independent_entry_points():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"] == {
        "factcheck": "factcheck.cli:main",
        "factcheck-ocr": "factcheck_ocr.cli:main",
        "factcheck-video": "factcheck_video.cli:main",
        "factcheck-audio": "factcheck_audio.cli:main",
    }
    extras = data["project"]["optional-dependencies"]
    assert set(extras["audio"]) <= set(extras["video"])  # nothing new to install; no OCR needed for audio
    assert not any("ocr" in d or "pillow" in d for d in extras["audio"])
    assert data["project"]["dependencies"] == ["langgraph>=1.0,<2", "pydantic>=2.7,<3", "google-genai>=1.30",
                                               "tavily-python>=0.7"]  # no new API dependency


@pytest.mark.parametrize("module,prog,options", [
    ("factcheck.cli", "factcheck", ["--json", "--verbose"]),
    ("factcheck_ocr.cli", "factcheck-ocr", ["--fact-check", "--keep-noise"]),
    ("factcheck_video.cli", "factcheck-video", ["--extract-only", "--whisper-model", "--keep-ocr-noise", "--model"]),
    ("factcheck_audio.cli", "factcheck-audio", ["--extract-only", "--whisper-model", "--assume-english", "--model"]),
])
def test_cli_help_of_every_module(module, prog, options, capsys, monkeypatch):
    import importlib

    mod = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [prog, "--help"])
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"]) if "argv" in mod.main.__code__.co_varnames else mod.main()
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert f"usage: {prog}" in text
    for opt in options:
        assert opt in text, (prog, opt)


def test_video_module_does_not_depend_on_audio():
    for path in (ROOT / "src" / "factcheck_video").glob("*.py"):
        assert "factcheck_audio" not in path.read_text(encoding="utf-8"), path.name
    for pkg in ("factcheck", "factcheck_ocr"):
        for path in (ROOT / "src" / pkg).glob("*.py"):
            assert "factcheck_audio" not in path.read_text(encoding="utf-8"), path.name
