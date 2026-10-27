from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from factcheck.config import Config  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
AS_OF = date(2026, 10, 26)


def load_page(name: str) -> str:
    return (FIXTURES / "pages" / name).read_text(encoding="utf-8")


@pytest.fixture
def config() -> Config:
    return Config(as_of=AS_OF)


@pytest.fixture(autouse=True)
def _no_real_credentials(request, monkeypatch):
    """Offline tests must never reach a real provider, even if keys are set locally.

    Only tests explicitly marked ``@pytest.mark.live`` keep the environment.
    """
    if request.node.get_closest_marker("live") is None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("FACTCHECK_PROVIDER", raising=False)
        for name in ("FACTCHECK_OLLAMA_FALLBACK", "FACTCHECK_OLLAMA_MODEL", "FACTCHECK_OLLAMA_TIMEOUT"):
            monkeypatch.delenv(name, raising=False)
        # A real Ollama server may be running locally: point any accidental request at a
        # closed port (discard service) so offline tests can never reach it.
        monkeypatch.setenv("FACTCHECK_OLLAMA_URL", "http://127.0.0.1:9")
    # Never read a developer's real .env during tests (it may hold live keys).
    monkeypatch.setenv("FACTCHECK_NO_DOTENV", "1")
