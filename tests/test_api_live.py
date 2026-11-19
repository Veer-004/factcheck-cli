"""Opt-in live API smoke test: the real FastAPI app with real Gemini + Tavily.

    set FACTCHECK_LIVE_TESTS=1 && python -m pytest tests/test_api_live.py

Skipped unless FACTCHECK_LIVE_TESTS=1 and both API keys are set (``.env`` is
read only in that case). One claim, at most 6 search credits.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")

pytestmark = pytest.mark.live

LIVE = os.environ.get("FACTCHECK_LIVE_TESTS") == "1"
if LIVE:
    from factcheck.envfile import load_env_file

    load_env_file()
KEYS = bool(os.environ.get("GEMINI_API_KEY", "").strip()) and bool(os.environ.get("TAVILY_API_KEY", "").strip())


@pytest.mark.skipif(not (LIVE and KEYS), reason="set FACTCHECK_LIVE_TESTS=1 and GEMINI_API_KEY/TAVILY_API_KEY")
def test_live_text_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from factcheck_api.app import create_app
    from factcheck_api.config import Settings

    with TestClient(create_app(Settings(temp_dir=str(tmp_path)))) as c:
        assert c.get("/health").json()["fact_check_configured"] is True
        r = c.post("/api/v1/fact-check/text", json={"text": "The Eiffel Tower in Paris was completed in 1889.",
                                                    "max_claims": 1, "search_budget": 6})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed" and body["report"]["stats"]["search_credits_used"] <= 6
    [claim] = body["claims"]
    assert claim["verdict"] in {"SUPPORTED", "INSUFFICIENT_EVIDENCE", "CONFLICTING"}
    assert all(c["url"].startswith("http") for c in claim["citations"])
