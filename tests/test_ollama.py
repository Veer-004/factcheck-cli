"""Offline tests for the Ollama local adapter (httpx.MockTransport; no Ollama needed)."""

from __future__ import annotations

import json

import httpx
import pytest

from factcheck.limits import MIN_REQUEST_S, Deadline
from factcheck.llm import LLMNetworkError, LLMOutputError, LLMTimeoutError, LLMUnavailableError, PhaseTimeoutClient
from factcheck.models import LLMExtraction
from factcheck.ollama import OllamaClient


class FakeClock:
    def __init__(self):
        self.t = 50.0

    def __call__(self):
        return self.t


def ok_body(content=json.dumps({"claims": []}), **extra):
    return {"model": "qwen3:4b", "message": {"role": "assistant", "content": content, **extra},
            "done": True, "done_reason": "stop"}


def _client(handler, **kw):
    seen: list[httpx.Request] = []

    def record(request):
        seen.append(request)
        return handler(request, len(seen))

    c = OllamaClient("qwen3:4b", http_client=PhaseTimeoutClient(transport=httpx.MockTransport(record)), **kw)
    return c, seen


def _call(c):
    return c.generate_structured(system="SYS", user="USER", schema=LLMExtraction, purpose="judge")


def ok(request, n):
    return httpx.Response(200, json=ok_body())


def test_request_uses_documented_chat_api_with_schema_format_and_no_thinking():
    c, seen = _client(ok)
    assert _call(c).claims == []
    req = seen[0]
    assert req.method == "POST" and str(req.url) == "http://localhost:11434/api/chat"
    body = json.loads(req.content)
    assert body["model"] == "qwen3:4b" and body["stream"] is False and body["think"] is False
    assert body["format"]["type"] == "object" and "claims" in body["format"]["properties"]  # JSON schema, not "json"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == "USER"  # untrusted content stays in the user turn
    assert body["options"]["num_ctx"] >= 8192 and body["options"]["num_predict"] > 0
    assert "tools" not in body and "authorization" not in req.headers


def test_custom_base_url_and_timeout_split():
    c, seen = _client(ok, base_url="http://gpu-box:11434/", timeout_s=40)
    _call(c)
    assert str(seen[0].url) == "http://gpu-box:11434/api/chat"
    t = seen[0].extensions["timeout"]
    assert t["connect"] + t["read"] == pytest.approx(40)


def test_timeout_clipped_to_run_deadline_and_refused_when_exhausted():
    clock = FakeClock()
    c, seen = _client(ok, timeout_s=90, deadline=Deadline(30, clock=clock))
    _call(c)
    t = seen[0].extensions["timeout"]
    assert t["connect"] + t["read"] == pytest.approx(30)
    c2, seen2 = _client(ok, deadline=Deadline(MIN_REQUEST_S - 0.1, clock=FakeClock()))
    with pytest.raises(LLMTimeoutError, match="not enough run time left to start the Ollama request"):
        _call(c2)
    assert seen2 == []


def test_thinking_trace_is_discarded_and_never_returned():
    body = ok_body(content="<think>secret chain of thought</think>" + json.dumps({"claims": []}),
                   thinking="more hidden reasoning")
    c, _ = _client(lambda r, n: httpx.Response(200, json=body))
    result = _call(c)
    assert result.claims == [] and c.thinking_returned is True
    assert "secret chain" not in repr(result) and "hidden reasoning" not in repr(result)
    c2, _ = _client(ok)
    _call(c2)
    assert c2.thinking_returned is False


@pytest.mark.parametrize("content,match", [
    ("not json at all", "schema"),                                   # malformed JSON
    (json.dumps({"claims": [{"statement": 5}]}), "schema"),          # schema-invalid
    (json.dumps({"claims": "nope"}), "schema"),
])
def test_malformed_or_schema_invalid_output_is_llm_output_error(content, match):
    c, seen = _client(lambda r, n: httpx.Response(200, json=ok_body(content=content)))
    with pytest.raises(LLMOutputError, match=match):
        _call(c)
    assert len(seen) == 1  # no internal retry


@pytest.mark.parametrize("body,match", [
    ({"message": {"role": "assistant"}}, "message.content"),
    ({"unexpected": True}, "message.content"),
    ({"message": {"content": None}, "done": True}, "no text content"),
    ({"message": {"content": '{"claims": ['}, "done": True, "done_reason": "length"}, "truncated"),
])
def test_unexpected_responses_are_llm_output_errors(body, match):
    c, _ = _client(lambda r, n: httpx.Response(200, json=body))
    with pytest.raises(LLMOutputError, match=match):
        _call(c)


def test_non_json_http_body_is_output_error():
    c, _ = _client(lambda r, n: httpx.Response(200, text="<html>proxy page</html>"))
    with pytest.raises(LLMOutputError):
        _call(c)


def test_server_not_running_is_actionable_network_error():
    def refused(request, n):
        raise httpx.ConnectError("[WinError 10061] No connection could be made", request=request)

    c, seen = _client(refused)
    with pytest.raises(LLMNetworkError, match=r"Is Ollama running\? .*ollama serve"):
        _call(c)
    assert len(seen) == 1


def test_missing_model_tells_user_to_pull():
    c, _ = _client(lambda r, n: httpx.Response(404, json={"error": "model 'qwen3:4b' not found"}))
    with pytest.raises(LLMUnavailableError, match=r"not installed .*ollama pull qwen3:4b") as exc:
        _call(c)
    assert exc.value.status_code == 404


def test_think_rejection_is_reported_as_compatibility_issue():
    c, _ = _client(lambda r, n: httpx.Response(400, json={"error": "\"qwen3:4b\" does not support think"}))
    with pytest.raises(LLMUnavailableError, match="may not support disabling thinking") as exc:
        _call(c)
    assert exc.value.status_code == 400


def test_server_error_and_read_timeout():
    c, _ = _client(lambda r, n: httpx.Response(500, json={"error": "llama runner process has terminated"}))
    with pytest.raises(LLMUnavailableError, match="HTTP 500") as exc:
        _call(c)
    assert exc.value.status_code == 500

    def stall(request, n):
        raise httpx.ReadTimeout("timed out", request=request)

    c2, seen = _client(stall, timeout_s=40)
    with pytest.raises(LLMTimeoutError, match="Ollama sent no response"):
        _call(c2)
    assert len(seen) == 1


def test_error_bodies_are_bounded_and_redacted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyFAKEKEYFORTESTSONLY000000000000")
    body = {"error": "echo AIzaSyFAKEKEYFORTESTSONLY000000000000 " + "x" * 2000}
    c, _ = _client(lambda r, n: httpx.Response(500, json=body))
    with pytest.raises(LLMUnavailableError) as exc:
        _call(c)
    assert "AIzaSyFAKE" not in str(exc.value) and len(str(exc.value)) < 400


def test_requests_are_serialised_one_at_a_time():
    import threading
    import time

    active = peak = 0
    lock = threading.Lock()

    def slow(request, n):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return httpx.Response(200, json=ok_body())

    c, _ = _client(slow)
    threads = [threading.Thread(target=_call, args=(c,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


def test_invalid_base_url_rejected():
    with pytest.raises(ValueError):
        OllamaClient("qwen3:4b", base_url="file:///etc/passwd")
