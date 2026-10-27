"""Tests for the minimal .env loader (temporary files only; never the real .env)."""

from __future__ import annotations

import os

import pytest

from factcheck import cli, envfile

FAKE_TV = "tvly-FAKEKEYFORTESTSONLY0000000000000000"


@pytest.fixture
def dotenv_enabled(monkeypatch):
    monkeypatch.delenv("FACTCHECK_NO_DOTENV", raising=False)
    monkeypatch.delenv("FACTCHECK_ENV_FILE", raising=False)
    for name in ("GEMINI_API_KEY", "TAVILY_API_KEY", "FACTCHECK_PROVIDER", "FACTCHECK_OLLAMA_MODEL"):
        monkeypatch.delenv(name, raising=False)
    envfile.loaded_from_file.clear()
    yield
    envfile.loaded_from_file.clear()


def test_parse_formats():
    text = (
        "# comment\n\n"
        "FACTCHECK_OLLAMA_MODEL=qwen3:4b\n"
        "export TAVILY_API_KEY = tvly-xyz # inline comment\n"
        'GEMINI_API_KEY="AIza with # hash kept"\n'
        "FACTCHECK_PROVIDER='ollama'\r\n"
        "EMPTY=\n"
        "this line is broken\n"
    )
    values, problems = envfile.parse_env(text)
    assert values == {
        "FACTCHECK_OLLAMA_MODEL": "qwen3:4b",
        "TAVILY_API_KEY": "tvly-xyz",
        "GEMINI_API_KEY": "AIza with # hash kept",
        "FACTCHECK_PROVIDER": "ollama",
        "EMPTY": "",
    }
    assert problems == ["line 8: not in KEY=VALUE form (ignored)"]


def test_non_printable_value_rejected_without_printing_it():
    values, problems = envfile.parse_env("TAVILY_API_KEY=tvly-​secret\n")
    assert values == {} and problems == ["line 1: TAVILY_API_KEY contains non-printable characters (ignored)"]
    assert "secret" not in problems[0]


def test_load_sets_variables_and_skips_placeholders(tmp_path, dotenv_enabled):
    f = tmp_path / ".env"
    f.write_text(f"TAVILY_API_KEY={FAKE_TV}\nGEMINI_API_KEY=\nFACTCHECK_PROVIDER=ollama\n", encoding="utf-8")
    assert envfile.load_env_file(f) == []
    assert os.environ["TAVILY_API_KEY"] == FAKE_TV and os.environ["FACTCHECK_PROVIDER"] == "ollama"
    assert "GEMINI_API_KEY" not in os.environ
    assert envfile.loaded_from_file == {"TAVILY_API_KEY", "FACTCHECK_PROVIDER"}


def test_bom_and_crlf_files(tmp_path, dotenv_enabled):
    f = tmp_path / ".env"
    f.write_bytes(("TAVILY_API_KEY=" + FAKE_TV + "\r\n").encode("utf-8-sig"))
    envfile.load_env_file(f)
    assert os.environ["TAVILY_API_KEY"] == FAKE_TV


def test_usable_environment_variable_wins(tmp_path, monkeypatch, dotenv_enabled):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-FROM-ENVIRONMENT-123")
    f = tmp_path / ".env"
    f.write_text(f"TAVILY_API_KEY={FAKE_TV}\n", encoding="utf-8")
    envfile.load_env_file(f)
    assert os.environ["TAVILY_API_KEY"] == "tvly-FROM-ENVIRONMENT-123"
    assert "TAVILY_API_KEY" not in envfile.loaded_from_file


@pytest.mark.parametrize("broken", ["​", "\x01", "   "])
def test_broken_environment_variable_is_replaced(tmp_path, monkeypatch, dotenv_enabled, broken):
    # e.g. the 1-character non-printable value left by a failed clipboard paste
    monkeypatch.setenv("TAVILY_API_KEY", broken)
    f = tmp_path / ".env"
    f.write_text(f"TAVILY_API_KEY={FAKE_TV}\n", encoding="utf-8")
    envfile.load_env_file(f)
    assert os.environ["TAVILY_API_KEY"] == FAKE_TV


def test_disabled_flag_and_missing_file(tmp_path, monkeypatch, dotenv_enabled):
    f = tmp_path / ".env"
    f.write_text(f"TAVILY_API_KEY={FAKE_TV}\n", encoding="utf-8")
    monkeypatch.setenv("FACTCHECK_NO_DOTENV", "1")
    assert envfile.load_env_file(f) == [] and "TAVILY_API_KEY" not in os.environ
    monkeypatch.delenv("FACTCHECK_NO_DOTENV")
    assert envfile.load_env_file(tmp_path / "missing.env") == []


def test_env_file_path_override(tmp_path, monkeypatch, dotenv_enabled):
    f = tmp_path / "keys.env"
    f.write_text(f"TAVILY_API_KEY={FAKE_TV}\n", encoding="utf-8")
    monkeypatch.setenv("FACTCHECK_ENV_FILE", str(f))
    envfile.load_env_file()
    assert os.environ["TAVILY_API_KEY"] == FAKE_TV


def test_cli_reads_dotenv_from_cwd_and_never_prints_values(tmp_path, monkeypatch, capsys, dotenv_enabled):
    (tmp_path / ".env").write_text(f"TAVILY_API_KEY={FAKE_TV}\nBROKEN LINE\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--check-config"]) == 0
    out = capsys.readouterr()
    assert "TAVILY_API_KEY: set (from .env)" in out.out and "GEMINI_API_KEY: NOT SET" in out.out
    assert "line 2: not in KEY=VALUE form" in out.err
    assert FAKE_TV not in out.out + out.err


def test_test_suite_never_reads_dotenv():
    # conftest's autouse fixture sets this for every test.
    assert os.environ.get("FACTCHECK_NO_DOTENV") == "1"
