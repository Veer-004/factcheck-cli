"""Regression tests for request timeouts, bounded retries and deadline handling.

The Gemini tests drive the real google-genai request path into an
``httpx.MockTransport``: no network, but the timeout values asserted are the
ones httpx would actually enforce.
"""

from __future__ import annotations

import json

import httpx
import pytest
import requests
from tavily import errors as terr

from factcheck import diagnostics
from factcheck import llm as llm_mod
from factcheck.limits import MIN_REQUEST_S, CreditBudget, Deadline, split_timeout
from factcheck.llm import (
    GeminiClient,
    LLMAuthError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMTimeoutError,
    PhaseTimeoutClient,
)
from factcheck.models import LLMExtraction, SearchResult
from factcheck.retrieval import (
    PhaseTimeoutSession,
    RetrievalService,
    SearchAuthError,
    SearchError,
    SearchRateLimitError,
    TavilyProvider,
)

FAKE_KEY = "AIzaSyFAKEKEYFORTESTSONLY000000000000"
OK_BODY = {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps({"claims": []})}]}}]}


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _gemini(monkeypatch, handler, **kw):
    """GeminiClient whose SDK sends through a MockTransport; returns (client, seen requests, sleeps)."""
    sleeps: list[float] = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: sleeps.append(s))
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request, len(seen))

    c = GeminiClient(FAKE_KEY, "gemini-test-model", **kw)
    c._http = PhaseTimeoutClient(transport=httpx.MockTransport(record))
    c._client._api_client._httpx_client = c._http
    return c, seen, sleeps


def _call(c):
    return c.generate_structured(system="s", user="u", schema=LLMExtraction, purpose="extract_claims")


def _ok(request, n):
    return httpx.Response(200, json=OK_BODY)


# --------------------------------------------------------------------------- timeout units


def test_split_timeout_sums_to_total_and_caps_connect():
    assert split_timeout(60) == (10.0, 50.0)
    assert split_timeout(15) == (3.75, 11.25)
    for total in (4, 15, 60, 120):
        c, r = split_timeout(total)
        assert c <= 10 and c + r == pytest.approx(total)


def test_gemini_timeout_reaches_httpx_in_seconds_split_by_phase(monkeypatch):
    c, seen, _ = _gemini(monkeypatch, _ok, timeout_s=60)
    assert _call(c).claims == []
    # HttpOptions.timeout is milliseconds; the SDK converts it to 60 s and
    # PhaseTimeoutClient turns it into a total rather than a per-phase limit.
    assert c._client._api_client._http_options.timeout == 60_000
    assert seen[0].extensions["timeout"] == {"connect": 10.0, "read": 50.0, "write": 10.0, "pool": 10.0}
    assert seen[0].headers["x-server-timeout"] == "60"


def test_per_request_timeout_clipped_to_remaining_deadline(monkeypatch):
    clock = FakeClock()
    c, seen, _ = _gemini(monkeypatch, _ok, timeout_s=60, deadline=Deadline(20, clock=clock))
    _call(c)
    assert seen[0].extensions["timeout"]["connect"] + seen[0].extensions["timeout"]["read"] == pytest.approx(20)


def test_phase_client_leaves_explicit_timeout_objects_alone():
    client = PhaseTimeoutClient()
    req = client.build_request("POST", "https://example.invalid/x", timeout=httpx.Timeout(7.0))
    assert req.extensions["timeout"] == {"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0}


# --------------------------------------------------------------------------- bounded retries


def test_sdk_does_not_retry_and_read_timeout_is_not_retried(monkeypatch):
    def stall(request, n):
        raise httpx.ReadTimeout("timed out", request=request)

    c, seen, sleeps = _gemini(monkeypatch, stall, timeout_s=60, max_retries=2)
    with pytest.raises(LLMTimeoutError, match="no response within 50s"):
        _call(c)
    assert len(seen) == 1 and sleeps == []  # one HTTP attempt total: no SDK or app retry


def test_connect_failures_retry_a_bounded_number_of_times(monkeypatch):
    def refuse(request, n):
        raise httpx.ConnectTimeout("connect timed out", request=request)

    c, seen, sleeps = _gemini(monkeypatch, refuse, timeout_s=60, max_retries=2)
    with pytest.raises(LLMNetworkError, match="could not connect to the Gemini API within 10s"):
        _call(c)
    assert len(seen) == 3 and sleeps == [2.0, 4.0]


def test_429_retries_bounded_then_rate_limit_error_with_status(monkeypatch):
    def limited(request, n):
        return httpx.Response(429, json={"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    c, seen, sleeps = _gemini(monkeypatch, limited, timeout_s=60, max_retries=2)
    with pytest.raises(LLMRateLimitError) as exc:
        _call(c)
    assert exc.value.status_code == 429 and len(seen) == 3 and len(sleeps) == 2


def test_auth_failure_not_retried_and_keeps_status(monkeypatch):
    def denied(request, n):
        return httpx.Response(403, json={"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}})

    c, seen, sleeps = _gemini(monkeypatch, denied)
    with pytest.raises(LLMAuthError) as exc:
        _call(c)
    assert exc.value.status_code == 403 and len(seen) == 1 and sleeps == []


# --------------------------------------------------------------------------- deadline


def test_no_request_started_when_deadline_nearly_exhausted(monkeypatch):
    c, seen, sleeps = _gemini(monkeypatch, _ok, deadline=Deadline(MIN_REQUEST_S - 0.1, clock=FakeClock()))
    with pytest.raises(LLMTimeoutError, match="not enough run time left to start"):
        _call(c)
    assert seen == [] and sleeps == []


def test_no_retry_sleep_when_retry_would_not_fit(monkeypatch):
    def limited(request, n):
        body = {"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED",
                          "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}]}}
        return httpx.Response(429, json=body)

    # 7 s left: sleeping 3 s would leave 4 s < MIN_REQUEST_S for the retry itself.
    c, seen, sleeps = _gemini(monkeypatch, limited, deadline=Deadline(7.0, clock=FakeClock()))
    with pytest.raises(LLMRateLimitError, match="not enough run time left to retry"):
        _call(c)
    assert len(seen) == 1 and sleeps == []


def test_timeout_scenario_from_live_failure_fails_fast_with_cause(monkeypatch):
    """Previously: 60 s timeout + retry consumed the whole 120 s run, ending in a
    generic 'not enough run time left'. Now: one attempt, error names the cause."""
    clock = FakeClock()

    def stall(request, n):
        clock.t += request.extensions["timeout"]["connect"] + request.extensions["timeout"]["read"]
        raise httpx.ReadTimeout("timed out", request=request)

    deadline = Deadline(120, clock=clock)
    c, seen, sleeps = _gemini(monkeypatch, stall, timeout_s=60, deadline=deadline)
    with pytest.raises(LLMTimeoutError, match="ReadTimeout"):
        _call(c)
    assert len(seen) == 1 and sleeps == [] and deadline.remaining() == pytest.approx(60)


# --------------------------------------------------------------------------- sanitised, actionable errors


def test_network_error_is_sanitised_and_actionable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    def dns(request, n):
        raise httpx.ConnectError(f"getaddrinfo failed for key={FAKE_KEY}", request=request)

    c, _, _ = _gemini(monkeypatch, dns, max_retries=0)
    with pytest.raises(LLMNetworkError) as exc:
        _call(c)
    msg = str(exc.value)
    assert FAKE_KEY not in msg and "AIzaSy" not in msg
    assert "ConnectError" in msg and "network error" in msg


def test_certificate_failure_not_retried_and_points_to_ssl_cert_file(monkeypatch):
    def tls(request, n):
        raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed", request=request)

    c, seen, sleeps = _gemini(monkeypatch, tls, max_retries=2)
    with pytest.raises(LLMNetworkError, match="SSL_CERT_FILE"):
        _call(c)
    assert len(seen) == 1 and sleeps == []


# --------------------------------------------------------------------------- Tavily


def test_tavily_session_splits_scalar_timeout(monkeypatch):
    captured = {}
    monkeypatch.setattr(requests.Session, "request", lambda self, m, u, **kw: captured.update(kw) or "resp")
    assert PhaseTimeoutSession().request("POST", "https://api.tavily.com/search", timeout=15) == "resp"
    assert captured["timeout"] == (3.75, 11.25)
    provider = TavilyProvider("tvly-test-not-real", timeout_s=15)
    assert isinstance(provider._client.session, PhaseTimeoutSession)


@pytest.mark.parametrize(
    "exc,cls,status",
    [
        (terr.InvalidAPIKeyError("bad key"), SearchAuthError, 401),
        (terr.ForbiddenError("no"), SearchAuthError, 403),
        (terr.UsageLimitExceededError("429"), SearchRateLimitError, 429),
        (terr.BadRequestError("bad"), SearchError, 400),
        (terr.TimeoutError(15), SearchError, None),
        (requests.ConnectionError("Max retries exceeded tvly-abcdefghijklmnop"), SearchError, None),
    ],
)
def test_tavily_errors_classified_with_status_and_redacted(exc, cls, status):
    provider = TavilyProvider("tvly-test-not-real")

    def boom(**kw):
        raise exc

    provider._client.search = boom
    with pytest.raises(cls) as err:
        provider.search("q", 1)
    assert type(err.value) is cls and err.value.status_code == status
    assert "tvly-abcdefghijklmnop" not in str(err.value)


def test_retrieval_does_not_start_search_without_time_for_it(config):
    calls = []

    class P:
        def search(self, q, n):
            calls.append(q)
            return []

        def extract(self, urls):
            calls.append(urls)
            return []

    clock = FakeClock()
    svc = RetrievalService(P(), CreditBudget(40), Deadline(config.search_timeout_s - 1, clock=clock), config)
    rnd = svc.run_round(["q1", "q2"])
    assert calls == [] and "not enough run time" in rnd.errors[0]


def test_retrieval_retry_sleep_requires_time_for_the_retry(config):
    sleeps = []

    class Limited:
        def search(self, q, n):
            raise SearchRateLimitError("429")

        def extract(self, urls):
            return []

    # Enough for one search, but not for a 2 s sleep plus another full request.
    svc = RetrievalService(Limited(), CreditBudget(40), Deadline(config.search_timeout_s + 1, clock=FakeClock()),
                           config, sleep=sleeps.append)
    rnd = svc.run_round(["q"])
    assert sleeps == [] and rnd.searches_succeeded == 0 and rnd.errors


# --------------------------------------------------------------------------- diagnostics are opt-in


def test_diagnostics_require_explicit_provider():
    with pytest.raises(SystemExit) as exc:
        diagnostics.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("provider", ["gemini", "tavily"])
def test_diagnostics_make_no_request_without_keys(provider, monkeypatch, capsys):
    def explode(*a, **kw):
        raise AssertionError("provider must not be constructed")

    monkeypatch.setattr(diagnostics, "GeminiClient", explode)
    monkeypatch.setattr(diagnostics, "TavilyProvider", explode)
    assert diagnostics.main([provider]) == 2
    assert "no request was made" in capsys.readouterr().err


@pytest.fixture
def no_dns(monkeypatch):
    """Diagnostics measure DNS separately; never resolve real hosts in unit tests."""
    monkeypatch.setattr(diagnostics.socket, "getaddrinfo",
                        lambda *a, **kw: [(diagnostics.socket.AF_INET, None, None, "", ("192.0.2.1", 443))])


def test_gemini_diagnostic_uses_15s_and_no_retries(no_dns):
    seen = {}

    class Client:
        def __init__(self, key, model, **kw):
            seen.update(kw)

        def generate_structured(self, **kw):
            raise LLMNetworkError(f"could not connect ({FAKE_KEY})")

    r = diagnostics.diagnose_gemini(FAKE_KEY, "m", client_factory=Client)
    assert seen["timeout_s"] == 15.0 and seen["max_retries"] == 0
    assert isinstance(seen["deadline"], Deadline) and seen["deadline"].seconds == 15.0
    assert seen["thinking_level"] == "low" and seen["temperature"] is None
    assert not r.ok and r.error_type == "LLMNetworkError"
    assert FAKE_KEY not in r.render() and "dns" in r.stages


def test_tavily_diagnostic_reports_status_and_success():
    class Denied:
        def __init__(self, key, timeout_s):
            assert timeout_s == 15.0

        def search(self, q, n):
            raise SearchAuthError("Tavily rejected the credentials (Unauthorized)", status_code=401)

    r = diagnostics.diagnose_tavily("tvly-x", provider_factory=Denied)
    assert not r.ok and r.status_code == 401 and "http_status: 401" in r.render()

    class Fine:
        def __init__(self, key, timeout_s):
            pass

        def search(self, q, n):
            assert n == 1
            return [SearchResult(url="https://a.example/x")]

    ok = diagnostics.diagnose_tavily("tvly-x", provider_factory=Fine)
    assert ok.ok and "1 result" in ok.detail


# --------------------------------------------------------------------------- Gemini 3 request settings


def test_request_payload_uses_model_default_temperature_and_low_thinking(monkeypatch):
    bodies = []

    def capture(request, n):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    c, _, _ = _gemini(monkeypatch, capture)
    _call(c)
    gen = bodies[0]["generationConfig"]
    assert "temperature" not in gen
    # The SDK serialises the field name as thinking_level; the API accepts either spelling.
    assert list(gen["thinkingConfig"].values()) == ["LOW"]
    assert gen["responseMimeType"] == "application/json"


def test_thinking_and_temperature_are_configurable(monkeypatch):
    bodies = []

    def capture(request, n):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    c, _, _ = _gemini(monkeypatch, capture, thinking_level=None, temperature=1.0)
    _call(c)
    gen = bodies[0]["generationConfig"]
    assert "thinkingConfig" not in gen and gen["temperature"] == 1.0


def test_config_defaults_follow_gemini3_guidance(monkeypatch):
    from factcheck.config import Config

    cfg = Config()
    assert cfg.temperature is None and cfg.thinking_level == "low"
    monkeypatch.setenv("FACTCHECK_THINKING_LEVEL", "high")
    assert Config.from_env().thinking_level == "high"


# --------------------------------------------------------------------------- stage timings


def test_trace_extension_records_names_and_times_only():
    client = PhaseTimeoutClient()
    events = client.begin_trace()
    req = client.build_request("POST", "https://example.invalid/x", headers={"x-goog-api-key": FAKE_KEY}, timeout=15)
    req.extensions["trace"]("connection.connect_tcp.started", {"request": req})
    client.end_trace()
    assert [name for name, _ in events] == ["request.start", "connection.connect_tcp.started"]
    assert all(isinstance(t, float) for _, t in events)
    assert FAKE_KEY not in repr(events)
    assert "trace" not in client.build_request("GET", "https://example.invalid/y").extensions


def test_summarize_trace_reports_stages_and_stalled_stage():
    from factcheck.llm import format_timings, summarize_trace

    ev = [("request.start", 0.0),
          ("connection.connect_tcp.started", 0.1), ("connection.connect_tcp.complete", 0.3),
          ("connection.start_tls.started", 0.3), ("connection.start_tls.complete", 0.5),
          ("http11.send_request_headers.started", 0.5), ("http11.send_request_headers.complete", 0.51),
          ("http11.receive_response_headers.started", 0.52), ("http11.receive_response_headers.failed", 11.77)]
    t = summarize_trace(ev)
    assert t["dns+tcp"] == pytest.approx(0.2) and t["tls"] == pytest.approx(0.2)
    assert t["wait_for_response"] == pytest.approx(11.25)
    assert t["failed_stage"] == "wait_for_response" and t["last_stage"] == "wait_for_response"
    assert "failed during wait_for_response" in format_timings(t)
    # Abandoned mid-stage: measured up to the last event.
    open_ev = ev[:3] + [("connection.start_tls.started", 0.3), ("http11.x.y", 4.3)]
    assert summarize_trace(open_ev)["tls"] == pytest.approx(4.0)


def test_read_timeout_error_includes_stage_timings(monkeypatch):
    def stall(request, n):
        # MockTransport emits no httpcore events; simulate what the trace would contain.
        request.extensions["trace"]("http11.receive_response_headers.started", {})
        raise httpx.ReadTimeout("timed out", request=request)

    c, _, _ = _gemini(monkeypatch, stall, max_retries=0)
    with pytest.raises(LLMTimeoutError, match=r"stage timings: .*wait_for_response"):
        _call(c)
    assert c.last_timings["last_stage"] == "wait_for_response"


# --------------------------------------------------------------------------- hard wall-clock deadline


def test_run_with_deadline_returns_on_time_and_aborts_blocked_worker():
    import threading
    import time as _t

    release = threading.Event()
    aborted = []

    def blocked():
        release.wait(30)
        return "late"

    start = _t.monotonic()
    out = diagnostics.run_with_deadline(blocked, 1.0, lambda: (aborted.append(True), release.set()))
    assert not out.finished and aborted
    assert _t.monotonic() - start < 1.2  # genuinely bounded, including the abort grace


def test_gemini_diagnostic_enforces_total_deadline_including_client_creation(no_dns):
    import time as _t

    class SlowToCreate:
        def __init__(self, *a, **kw):
            _t.sleep(5)  # e.g. a stalled DNS lookup or slow client setup

    start = _t.monotonic()
    r = diagnostics.diagnose_gemini(FAKE_KEY, "m", timeout_s=2.0, client_factory=SlowToCreate)
    elapsed = _t.monotonic() - start
    assert elapsed < 2.3 and r.elapsed_s < 2.3
    assert not r.ok and r.error_type == "WallClockDeadlineExceeded" and "2s wall-clock deadline" in r.detail


def test_gemini_diagnostic_abort_closes_http_client_and_reports_last_stage(no_dns):
    import threading

    closed = threading.Event()

    class Http:
        def close(self):
            closed.set()

    class Stalled:
        def __init__(self, *a, **kw):
            self._http = Http()
            self.current_trace = [("request.start", 0.0), ("http11.receive_response_headers.started", 0.5)]

        def generate_structured(self, **kw):
            closed.wait(10)  # a real socket read fails once the client is closed
            raise httpx.ReadError("connection closed")

    r = diagnostics.diagnose_gemini(FAKE_KEY, "m", timeout_s=2.0, client_factory=Stalled)
    assert closed.is_set() and r.error_type == "WallClockDeadlineExceeded"
    assert "last stage reached: wait_for_response" in r.detail


def test_gemini_model_diagnostic_reports_access_and_generate_support(no_dns):
    class Models:
        def get(self, model, config=None):
            assert config["http_options"]["timeout"] == 15000
            return type("M", (), {"name": f"models/{model}", "supported_actions": ["generateContent"],
                                  "input_token_limit": 1048576, "output_token_limit": 65536, "thinking": True})()

    class Client:
        def __init__(self, key, model, **kw):
            self._http = PhaseTimeoutClient()
            self._client = type("C", (), {"models": Models()})()

    r = diagnostics.diagnose_gemini_model(FAKE_KEY, "gemini-3.8-flash", client_factory=Client)
    assert r.ok and "supports generateContent=True" in r.detail and "cleanup" in r.stages
    assert r.info["request"].startswith("models.get")


def test_diagnostic_output_never_contains_key_or_headers(no_dns, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    class Leaky:
        def __init__(self, *a, **kw):
            raise RuntimeError(f"boom x-goog-api-key: {FAKE_KEY}")

    text = diagnostics.diagnose_gemini(FAKE_KEY, "m", client_factory=Leaky).render()
    assert FAKE_KEY not in text and "AIzaSy" not in text
