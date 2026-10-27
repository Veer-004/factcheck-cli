"""Offline tests of the real provider adapters (SDK calls are mocked; no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from google.genai import errors as gerr

from factcheck import llm as llm_mod
from factcheck.limits import CallBudget, CreditBudget, Deadline
from factcheck.llm import (
    BudgetedLLM,
    GeminiClient,
    LLMAuthError,
    LLMOutputError,
    LLMUnavailableError,
    parse_structured,
    to_response_schema,
)
from factcheck.models import LLMExtraction, LLMJudgment, SearchResult
from factcheck.retrieval import RetrievalService, SearchError, SearchRateLimitError, TavilyProvider, extract_credits

GOOD = json.dumps({"claims": []})


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


@pytest.mark.parametrize("schema", [LLMExtraction, LLMJudgment])
def test_response_schema_is_self_contained(schema):
    s = to_response_schema(schema)
    nodes = list(_walk(s))
    assert not any("$ref" in n or "$defs" in n or "default" in n or "title" in n for n in nodes)
    assert s["type"] == "object"


def test_parse_structured_handles_fences_and_rejects_garbage():
    assert parse_structured("```json\n" + GOOD + "\n```", LLMExtraction).claims == []
    with pytest.raises(LLMOutputError):
        parse_structured("not json", LLMExtraction)
    with pytest.raises(LLMOutputError):
        parse_structured(json.dumps({"claims": [{"statement": 1}]}), LLMExtraction)
    with pytest.raises(LLMOutputError):
        parse_structured("", LLMExtraction)


class _Models:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return SimpleNamespace(text=out)


def _client(outcomes, monkeypatch, **kw):
    sleeps = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: sleeps.append(s))
    c = GeminiClient("test-key-not-real", "gemini-test-model", **kw)
    c._client = SimpleNamespace(models=_Models(outcomes))
    return c, sleeps


def _call(c):
    return c.generate_structured(system="sys", user="usr", schema=LLMExtraction, purpose="extract_claims")


def _api_error(code, retry=None, message="err"):
    details = {"error": {"code": code, "message": message, "status": "X"}}
    if retry:
        details["error"]["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry}]
    return gerr.ClientError(code, details)


def test_gemini_config_requests_json_without_tools(monkeypatch):
    c, _ = _client([GOOD], monkeypatch)
    assert _call(c).claims == []
    cfg = c._client.models.calls[0]["config"]
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema
    # Gemini 3 guidance: no sub-1.0 temperature (model default) and an explicit thinking level.
    assert not cfg.tools and cfg.temperature is None
    assert cfg.thinking_config.thinking_level.value == "LOW"
    assert cfg.system_instruction == "sys"


def test_gemini_429_respects_retry_after_then_succeeds(monkeypatch):
    c, sleeps = _client([_api_error(429, "3s"), _api_error(429), GOOD], monkeypatch)
    assert _call(c).claims == []
    assert sleeps == [3.0, 4.0]  # server delay, then exponential backoff


def test_gemini_429_gives_up_after_two_retries(monkeypatch):
    c, sleeps = _client([_api_error(429)] * 3, monkeypatch)
    with pytest.raises(LLMUnavailableError):
        _call(c)
    assert len(sleeps) == 2


def test_gemini_retry_after_above_cap_not_waited(monkeypatch):
    c, sleeps = _client([_api_error(429, "120s")], monkeypatch, max_retry_wait_s=30)
    with pytest.raises(LLMUnavailableError, match="cap"):
        _call(c)
    assert sleeps == []


def test_gemini_retry_never_exceeds_deadline(monkeypatch):
    c, sleeps = _client([_api_error(429, "5s")], monkeypatch, deadline=Deadline(1.0))
    with pytest.raises(LLMUnavailableError, match="run time"):
        _call(c)
    assert sleeps == []


@pytest.mark.parametrize("err", [_api_error(401), _api_error(403), _api_error(400, message="API key not valid."), _api_error(404)])
def test_gemini_auth_errors_fail_fast(monkeypatch, err):
    c, sleeps = _client([err, GOOD], monkeypatch)
    with pytest.raises(LLMAuthError):
        _call(c)
    assert sleeps == [] and len(c._client.models.calls) == 1


def test_gemini_malformed_output_not_retried_internally(monkeypatch):
    c, _ = _client(["{oops", GOOD], monkeypatch)
    with pytest.raises(LLMOutputError):
        _call(c)
    assert len(c._client.models.calls) == 1


def test_gemini_error_messages_are_redacted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTHISISAFAKEKEYFORTESTING1234")
    c, _ = _client([_api_error(500, message="key AIzaSyTHISISAFAKEKEYFORTESTING1234 leaked")] * 3, monkeypatch)
    with pytest.raises(LLMUnavailableError) as exc:
        _call(c)
    assert "AIzaSyTHISISAFAKE" not in str(exc.value)


def test_budgeted_llm_enforces_call_limit():
    inner = SimpleNamespace(generate_structured=lambda **kw: "ok")
    b = BudgetedLLM(inner, CallBudget(2), Deadline(60))
    assert b.generate_structured(system="", user="", schema=None, purpose="x") == "ok"
    assert b.generate_structured(system="", user="", schema=None, purpose="x") == "ok"
    with pytest.raises(LLMUnavailableError, match="budget"):
        b.generate_structured(system="", user="", schema=None, purpose="x")


def test_missing_keys_rejected_before_any_request():
    with pytest.raises(LLMAuthError):
        GeminiClient("", "m")
    from factcheck.retrieval import SearchAuthError

    with pytest.raises(SearchAuthError):
        TavilyProvider("")


# --------------------------------------------------------------------------- Tavily adapter


class _Tavily:
    def __init__(self, search_out=None, extract_out=None, raise_exc=None):
        self.search_out, self.extract_out, self.raise_exc = search_out, extract_out, raise_exc
        self.kwargs = []

    def search(self, **kw):
        self.kwargs.append(kw)
        if self.raise_exc:
            raise self.raise_exc
        return self.search_out

    def extract(self, **kw):
        self.kwargs.append(kw)
        return self.extract_out


def _tavily(fake):
    t = TavilyProvider("tvly-test-not-real")
    t._client = fake
    return t


def test_tavily_search_maps_and_filters_results():
    fake = _Tavily({"results": [
        {"url": "https://a.example/x", "title": "A", "content": "c", "published_date": "2025-01-01", "score": 0.9},
        {"url": "javascript:alert(1)", "title": "bad"},
        "not-a-dict",
    ]})
    out = _tavily(fake).search("q", 5)
    assert [r.url for r in out] == ["https://a.example/x"]
    kw = fake.kwargs[0]
    assert kw["search_depth"] == "basic" and kw["include_raw_content"] is False and kw["max_results"] == 5


def test_tavily_errors_mapped():
    from tavily import errors as terr

    from factcheck.retrieval import SearchAuthError

    with pytest.raises(SearchAuthError):
        _tavily(_Tavily(raise_exc=terr.InvalidAPIKeyError("bad"))).search("q", 5)
    with pytest.raises(SearchRateLimitError):
        _tavily(_Tavily(raise_exc=terr.UsageLimitExceededError("429"))).search("q", 5)
    with pytest.raises(SearchError):
        _tavily(_Tavily(raise_exc=ConnectionError("down"))).search("q", 5)


def test_extract_credit_math():
    assert [extract_credits(n) for n in (0, 1, 5, 6, 10)] == [0, 1, 1, 2, 2]


def test_retrieval_service_retries_429_then_succeeds(config):
    calls = []

    class Flaky:
        def search(self, q, n):
            calls.append(q)
            if len(calls) < 3:
                raise SearchRateLimitError("429")
            return [SearchResult(url="https://a.example/x", content="c")]

        def extract(self, urls):
            return []

    sleeps = []
    svc = RetrievalService(Flaky(), CreditBudget(40), Deadline(60), config, sleep=sleeps.append)
    rnd = svc.run_round(["q"])
    assert rnd.searches_succeeded == 1 and sleeps == [2.0, 4.0]


def test_retrieval_service_cache_avoids_repeat_spend(config):
    class Counting:
        n = 0

        def search(self, q, n):
            Counting.n += 1
            return [SearchResult(url="https://a.example/x", content="c")]

        def extract(self, urls):
            return []

    budget = CreditBudget(40)
    svc = RetrievalService(Counting(), budget, Deadline(60), config)
    svc.run_round(["same query"])
    used = budget.used
    svc.run_round(["Same  Query"])
    assert Counting.n == 1 and budget.used == used + 1  # only the (uncached) extract is re-attempted
