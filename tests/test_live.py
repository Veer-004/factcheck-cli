"""Live smoke test (real Gemini + Tavily).

Skipped unless FACTCHECK_LIVE_TESTS=1 and both API keys are set. Bounded to a
single claim and a 6-credit search budget (at most 2 searches + 1 extract per
round, one fallback round).
"""

from __future__ import annotations

import os

import pytest

from factcheck import Config, Verdict, verify_text

pytestmark = pytest.mark.live

LIVE = os.environ.get("FACTCHECK_LIVE_TESTS") == "1"
if LIVE:  # opt-in only: allow keys from ./.env for the live smoke test
    from factcheck.envfile import load_env_file

    load_env_file()
KEYS =bool(os.environ.get("GEMINI_API_KEY", "").strip()) and bool(os.environ.get("TAVILY_API_KEY", "").strip())


@pytest.mark.skipif(not (LIVE and KEYS), reason="set FACTCHECK_LIVE_TESTS=1 and GEMINI_API_KEY/TAVILY_API_KEY")
def test_live_single_claim_smoke():
    config = Config.from_env(max_checkable_claims=1, search_credit_budget=6, run_timeout_s=120)
    report = verify_text("The Eiffel Tower in Paris was completed in 1889.", config)
    assert report.stats.search_credits_used <= 6
    assert report.stats.llm_calls <= 3
    [claim] = report.claims
    assert claim.verdict in {Verdict.SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE, Verdict.CONFLICTING}
    for c in claim.supporting + claim.refuting:
        assert c.url.startswith("http")
