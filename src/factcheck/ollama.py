"""Ollama (local) adapter for the ``LLMClient`` protocol - used as the local fallback.

Uses the documented HTTP API only (docs.ollama.com/api/chat, checked 2026-10-27):
``POST {base}/api/chat`` with ``stream: false``, ``format: <JSON schema>`` (structured
output) and ``think: false`` (no thinking output, "if the model permits it"). It never
shells out to the ``ollama`` CLI, never pulls models and never starts the server.

* Thinking traces are discarded: ``message.thinking`` is ignored and any
  ``<think>...</think>`` block in the content is removed before parsing. They are
  never printed, logged or stored in reports.
* The output is parsed by the shared ``parse_structured`` (Pydantic) and then goes
  through the same citation/quote/verdict validation as Gemini's output.
* Local inference is slow and usually single-model, so requests are serialised
  (one at a time per client) and bounded by the run deadline. No retries: local
  failures (server down, model missing) do not fix themselves within a run.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from .limits import MIN_REQUEST_S, Deadline
from .llm import (
    LLMError,
    LLMNetworkError,
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
    PhaseTimeoutClient,
    _classify_transport_error,
    format_timings,
    parse_structured,
    summarize_trace,
    to_response_schema,
)
from .security import redact_secrets

SERVICE = "Ollama"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
ERROR_MAX = 200


class OllamaClient:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 90.0,
        deadline: Deadline | None = None,
        num_ctx: int = 8192,
        max_output_tokens: int = 2048,
        http_client: httpx.Client | None = None,
    ):
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("Ollama base URL must be http(s)://host[:port]")
        created = time.monotonic()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.host = parts.hostname
        self.timeout_s = timeout_s
        self.deadline = deadline
        # Prompts here (claim + up to 8 passages of 1200 chars) need more than the
        # small default context window of many local models; output is a short JSON.
        self.options = {"num_ctx": num_ctx, "num_predict": max_output_tokens}
        self._http = http_client or PhaseTimeoutClient()
        self._one_at_a_time = threading.Lock()
        self.client_init_s = round(time.monotonic() - created, 3)
        self.last_timings: dict[str, Any] = {}
        self.current_trace: list[tuple[str, float]] = []
        self.thinking_returned = False

    def build_payload(self, system: str, user: str, schema_json: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema_json,
            "think": False,
            "options": dict(self.options),
        }

    def _remaining_budget(self) -> float:
        if self.deadline is None:
            return self.timeout_s
        remaining = self.deadline.remaining()
        if remaining < MIN_REQUEST_S:
            raise LLMTimeoutError(f"not enough run time left to start the Ollama request ({remaining:.1f}s remaining)")
        return min(self.timeout_s, remaining)

    def _error_text(self, response: httpx.Response) -> str:
        try:
            data = response.json()
            text = data.get("error") if isinstance(data, dict) else None
        except ValueError:
            text = None
        text = text if isinstance(text, str) else (response.text if response.content else "")
        return redact_secrets(" ".join(str(text).split()))[:ERROR_MAX]

    def _http_error(self, response: httpx.Response) -> LLMError:
        code = response.status_code
        detail = self._error_text(response)
        low = detail.lower()
        if code == 404 or ("not found" in low and "model" in low):
            return LLMUnavailableError(
                f"Ollama model '{self.model}' is not installed (HTTP {code}: {detail}); "
                f"run `ollama pull {self.model}`", status_code=code)
        if "think" in low:
            return LLMUnavailableError(
                f"Ollama rejected think=false for '{self.model}' (HTTP {code}: {detail}); "
                "this Ollama version or model may not support disabling thinking", status_code=code)
        return LLMUnavailableError(f"Ollama request failed (HTTP {code}: {detail})", status_code=code)

    def _parse(self, response: httpx.Response, schema):
        try:
            data = response.json()
            message = data["message"]
            content = message["content"]
        except (ValueError, KeyError, TypeError):
            raise LLMOutputError("Ollama response did not contain message.content") from None
        if isinstance(data, dict) and data.get("done_reason") == "length":
            raise LLMOutputError(f"Ollama output was truncated at num_predict={self.options['num_predict']}")
        if not isinstance(content, str):
            raise LLMOutputError("Ollama returned no text content")
        # Record (never store) whether a reasoning trace came back despite think=false,
        # then discard it; only the answer is parsed.
        self.thinking_returned = bool(message.get("thinking")) or bool(_THINK_BLOCK.search(content))
        content = _THINK_BLOCK.sub("", content).strip()
        return parse_structured(content, schema)

    def generate_structured(self, *, system, user, schema, purpose, metadata=None):
        payload = self.build_payload(system, user, to_response_schema(schema))
        timeout_s = self._remaining_budget()
        # Serialise local inference; waiting for the lock counts against the same budget.
        if not self._one_at_a_time.acquire(timeout=timeout_s):
            raise LLMTimeoutError(f"Ollama was busy with another request for {timeout_s:.0f}s")
        try:
            timeout_s = self._remaining_budget()
            events = self._http.begin_trace() if isinstance(self._http, PhaseTimeoutClient) else []
            self.current_trace = events
            try:
                response = self._http.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout_s)
            except Exception as exc:  # transport: refused, timeout, TLS
                self.last_timings = summarize_trace(events)
                error, _ = _classify_transport_error(exc, timeout_s, SERVICE, self.host)
                if isinstance(error, LLMNetworkError) and type(exc).__name__ in ("ConnectError", "ConnectTimeout"):
                    error = LLMNetworkError(
                        f"{error}. Is Ollama running? Start it (open the Ollama app or run `ollama serve`) "
                        f"and check FACTCHECK_OLLAMA_URL={self.base_url}")
                elif self.last_timings:
                    error = type(error)(f"{error} [stage timings: {format_timings(self.last_timings)}]")
                raise error from None
            finally:
                if isinstance(self._http, PhaseTimeoutClient):
                    self._http.end_trace()
            self.last_timings = summarize_trace(events)
            if response.status_code != 200:
                raise self._http_error(response)
            return self._parse(response, schema)
        finally:
            self._one_at_a_time.release()
