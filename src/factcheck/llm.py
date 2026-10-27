"""LLM interface, budget wrapper and the Gemini implementation.

The model is only ever asked for structured JSON. It gets no tools, no search
and no network access: its entire world is the prompt we build.
"""

from __future__ import annotations

import functools
import json
import os
import re
import ssl
import threading
import time
from typing import Any, Mapping, Protocol, TypeVar

import certifi
import httpx
from pydantic import BaseModel, ValidationError

from .limits import MIN_REQUEST_S, BudgetExceeded, CallBudget, Deadline, split_timeout
from .security import redact_secrets

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class for LLM failures. ``status_code`` is the HTTP status, when there was one."""

    def __init__(self, message: str = "", *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LLMAuthError(LLMError):
    """Credentials missing/invalid or model not accessible: fail the run fast."""


class LLMOutputError(LLMError):
    """The model returned output that does not parse into the requested schema."""


class LLMUnavailableError(LLMError):
    """Transient failure that persisted after bounded retries, a timeout, or an exhausted budget."""


class LLMRateLimitError(LLMUnavailableError):
    """HTTP 429 persisted after bounded retries."""


class LLMTimeoutError(LLMUnavailableError):
    """The API did not respond within the per-request time budget."""


class LLMNetworkError(LLMUnavailableError):
    """Could not reach the API (DNS, connection, TLS, dropped connection)."""


class LLMClient(Protocol):
    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        purpose: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> T:
        """Return an instance of ``schema``.

        ``metadata`` is for tracing and test fakes only; implementations must
        never send it to a model.
        """
        ...


class BudgetedLLM:
    """Enforces the per-run LLM call budget and deadline around any client."""

    def __init__(self, inner: LLMClient, budget: CallBudget, deadline: Deadline):
        self.inner = inner
        self.budget = budget
        self.deadline = deadline

    @property
    def calls(self) -> int:
        return self.budget.used

    def generate_structured(self, *, system, user, schema, purpose, metadata=None):
        if self.deadline.expired():
            raise LLMUnavailableError("run time limit reached before the model call")
        try:
            self.budget.acquire()
        except BudgetExceeded as exc:
            raise LLMUnavailableError(str(exc)) from None
        return self.inner.generate_structured(
            system=system, user=user, schema=schema, purpose=purpose, metadata=metadata
        )


def is_recoverable(exc: LLMError) -> bool:
    """Operational failures that a different provider may not share.

    Recoverable: rate limits, timeouts, network/TLS failures and HTTP 5xx (or no
    status). NOT recoverable: authentication/model-access errors (LLMAuthError),
    malformed output (LLMOutputError - handled by the graph's bounded repair) and
    other HTTP 4xx, which mean the request or configuration is wrong.
    """
    if isinstance(exc, (LLMRateLimitError, LLMTimeoutError, LLMNetworkError)):
        return True
    if type(exc) is LLMUnavailableError:
        return exc.status_code is None or exc.status_code >= 500
    return False


class FallbackLLM:
    """Primary provider with a fallback provider for recoverable failures.

    Policy:
    * The primary handles every call while it works; the fallback is never called
      when the primary succeeds.
    * After a recoverable primary failure the *same* request goes to the fallback, and
      the primary is marked unavailable for the rest of the run (so later calls do not
      wait for it to time out again).
    * Authentication/configuration errors are raised immediately - the run stops
      (exit code 3) rather than hiding a bad key behind a slower model.
    * Every fallback attempt that follows a failed primary attempt consumes one unit
      of the shared LLM call budget, and none starts with less than MIN_REQUEST_S of
      run time left; both providers enforce the same run deadline.
    * The fallback only ever sees the same prompt: claims and retrieved passages. It
      gets no tools and no search, and its output goes through the same validation.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient, budget: CallBudget, deadline: Deadline,
                 primary_label: str = "primary", fallback_label: str = "fallback"):
        self.primary = primary
        self.fallback = fallback
        self.budget = budget
        self.deadline = deadline
        self.primary_label = primary_label
        self.fallback_label = fallback_label
        self.primary_down_reason: str | None = None
        self.fallback_calls = 0
        self._lock = threading.Lock()

    def generate_structured(self, *, system, user, schema, purpose, metadata=None):
        kwargs = dict(system=system, user=user, schema=schema, purpose=purpose, metadata=metadata)
        with self._lock:
            down = self.primary_down_reason
        if down is None:
            try:
                return self.primary.generate_structured(**kwargs)
            except LLMError as exc:
                if not is_recoverable(exc):
                    raise
                primary_error = exc
                with self._lock:
                    self.primary_down_reason = self.primary_down_reason or redact_secrets(str(exc))[:200]
            if self.deadline.remaining() < MIN_REQUEST_S:
                raise type(primary_error)(
                    f"{primary_error}; no run time left for the {self.fallback_label} fallback",
                    status_code=primary_error.status_code) from None
            try:
                self.budget.acquire()  # the fallback attempt is an extra model call
            except BudgetExceeded:
                raise type(primary_error)(
                    f"{primary_error}; LLM call budget exhausted, {self.fallback_label} fallback not attempted",
                    status_code=primary_error.status_code) from None
        else:
            primary_error = None
        try:
            result = self.fallback.generate_structured(**kwargs)
        except LLMOutputError:
            with self._lock:
                self.fallback_calls += 1
            raise
        except LLMError as exc:
            cause = f"{self.primary_label} failed ({redact_secrets(str(primary_error or down))[:200]})"
            raise type(exc)(f"{cause}; {self.fallback_label} fallback also failed: {exc}",
                            status_code=exc.status_code) from None
        with self._lock:
            self.fallback_calls += 1
        return result


# --------------------------------------------------------------------------
# JSON schema conversion
# --------------------------------------------------------------------------

_ALLOWED_KEYS = {
    "type", "properties", "required", "items", "enum", "anyOf", "description",
    "format", "minItems", "maxItems", "minimum", "maximum",
}


def to_response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema -> self-contained schema in the documented subset.

    ``$ref``/``$defs`` are inlined and keys outside a conservative allow-list
    are dropped. Constraints the API does not enforce are re-checked by
    Pydantic when the response is parsed.
    """
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 20:
            raise ValueError("schema too deep")
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                merged = {**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged, depth + 1)
            out = {}
            for key, value in node.items():
                if key == "properties":
                    out[key] = {p: resolve(v, depth + 1) for p, v in value.items()}
                elif key in _ALLOWED_KEYS:
                    out[key] = resolve(value, depth + 1)
            return out
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        return node

    return resolve({k: v for k, v in raw.items() if k != "$defs"})


def parse_structured(text: str | None, schema: type[T]) -> T:
    if not text or not text.strip():
        raise LLMOutputError("model returned an empty response")
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.S)
    if fence:
        cleaned = fence.group(1)
    try:
        return schema.model_validate_json(cleaned)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )
        raise LLMOutputError(f"response did not match the schema: {problems}") from None


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------


def _retry_delay_seconds(details: Any) -> float | None:
    """Find a google.rpc.RetryInfo ``retryDelay`` (e.g. "17s") anywhere in error details."""
    text = json.dumps(details, default=str) if not isinstance(details, str) else details
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', text)
    return float(m.group(1)) if m else None


@functools.lru_cache(maxsize=1)
def _default_ssl_context() -> ssl.SSLContext:
    """Same trust store as the SDK's default client; built once (loading CAs is slow)."""
    return ssl.create_default_context(
        cafile=os.environ.get("SSL_CERT_FILE", certifi.where()), capath=os.environ.get("SSL_CERT_DIR")
    )


class PhaseTimeoutClient(httpx.Client):
    """httpx client that turns the SDK's per-request scalar timeout into phase limits.

    The google-genai SDK passes ``HttpOptions.timeout`` to httpx as one scalar,
    which httpx applies to connect, write, read and pool *separately*. Here the
    scalar is split: connect/write/pool get ``split_timeout``'s connect share
    (<= 10 s) and the response read gets the rest, so a stalled TCP/TLS handshake
    fails in seconds instead of minutes. Other settings match the SDK's default
    client (redirects, certifi/SSL_CERT_FILE).

    These are per-phase limits, not a hard wall-clock cap: DNS resolution is not
    covered by them, and the read limit restarts whenever bytes arrive.

    When a trace is active on the calling thread (``begin_trace``), httpcore stage
    events (connect_tcp, start_tls, send, receive headers ...) are timestamped.
    Only event names and times are recorded - never request data or headers.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("follow_redirects", True)
        kwargs.setdefault("verify", _default_ssl_context())
        super().__init__(**kwargs)
        self._trace_local = threading.local()

    def begin_trace(self) -> list[tuple[str, float]]:
        events: list[tuple[str, float]] = [("request.start", time.monotonic())]
        self._trace_local.events = events
        return events

    def end_trace(self) -> None:
        self._trace_local.events = None

    def build_request(self, *args: Any, timeout: Any = httpx.USE_CLIENT_DEFAULT, **kwargs: Any) -> httpx.Request:
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
            connect, read = split_timeout(float(timeout))
            timeout = httpx.Timeout(connect=connect, read=read, write=connect, pool=connect)
        events = getattr(self._trace_local, "events", None)
        if events is not None:
            extensions = dict(kwargs.pop("extensions", None) or {})
            # Record the event name and time only; ``info`` carries the request (headers).
            extensions["trace"] = lambda name, info: events.append((name, time.monotonic()))
            kwargs["extensions"] = extensions
        return super().build_request(*args, timeout=timeout, **kwargs)


_STAGES = {
    "connect_tcp": "dns+tcp",
    "start_tls": "tls",
    "send_request_headers": "send",
    "receive_response_headers": "wait_for_response",
    "receive_response_body": "read_body",
}


def summarize_trace(events: list[tuple[str, float]]) -> dict[str, Any]:
    """Seconds spent per network stage, from trace events, plus the last stage reached.

    A stage that started but never completed (the request was abandoned or is
    still blocked) is measured up to the last recorded event.
    """
    starts: dict[str, float] = {}
    out: dict[str, Any] = {}
    last = None
    for name, t in events:
        stage, _, phase = name.rpartition(".")
        stage = stage.split(".")[-1]
        label = _STAGES.get(stage)
        if label is None:
            continue
        if phase == "started":
            starts[stage] = t
            last = label
        elif phase in ("complete", "failed") and stage in starts:
            out[label] = round(t - starts[stage], 3)
            if phase == "failed":
                out["failed_stage"] = label
    if last is not None:
        out["last_stage"] = last
        if last not in out:
            out[last] = round(events[-1][1] - starts[next(k for k, v in _STAGES.items() if v == last)], 3)
    if events and "connect_tcp" not in starts and "send_request_headers" in starts:
        out["connection"] = "reused"
    return out


def format_timings(t: dict[str, Any]) -> str:
    parts = [f"{label} {t[label]:.2f}s" for label in _STAGES.values() if label in t]
    if t.get("failed_stage"):
        parts.append(f"failed during {t['failed_stage']}")
    if t.get("connection") == "reused":
        parts.insert(0, "connection reused")
    return ", ".join(parts)


_NETWORK_ERRORS = {"ConnectError", "ReadError", "WriteError", "RemoteProtocolError", "NetworkError", "ProxyError"}


def _classify_transport_error(
    exc: Exception, timeout_s: float, service: str = "Gemini", host: str = "generativelanguage.googleapis.com"
) -> tuple[LLMError, bool]:
    """Map a non-HTTP exception to (error, retryable) with an actionable message."""
    name = type(exc).__name__
    detail = f"{name}: {redact_secrets(str(exc))[:200]}"
    connect_s, read_s = split_timeout(timeout_s)
    if name == "ConnectTimeout":
        return LLMNetworkError(
            f"could not connect to the {service} API within {connect_s:.0f}s ({name}); check network, proxy "
            f"or firewall access to {host}"
        ), True
    if name == "ReadTimeout":
        # Not retried: another attempt would spend the same budget waiting again.
        return LLMTimeoutError(
            f"{service} sent no response within {read_s:.0f}s ({name}); the model or network is too slow for the "
            "configured llm_timeout_s"
        ), False
    if name in ("WriteTimeout", "PoolTimeout"):
        return LLMTimeoutError(f"{service} request timed out ({name})"), False
    if "CERTIFICATE_VERIFY_FAILED" in str(exc):
        return LLMNetworkError(
            f"TLS certificate verification failed for the {service} API ({detail}); if a proxy or antivirus "
            "intercepts HTTPS, point SSL_CERT_FILE at its CA bundle"
        ), False
    if name in _NETWORK_ERRORS or isinstance(exc, OSError):
        return LLMNetworkError(f"network error talking to the {service} API ({detail})"), True
    return LLMUnavailableError(f"{service} request failed ({detail})"), False


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        max_retry_wait_s: float = 30.0,
        deadline: Deadline | None = None,
        max_output_tokens: int = 8192,
        temperature: float | None = None,
        thinking_level: str | None = "low",
    ):
        from google import genai
        from google.genai import types

        if not api_key:
            raise LLMAuthError("GEMINI_API_KEY is not set")
        created = time.monotonic()
        self._types = types
        self._http = PhaseTimeoutClient()
        # SDK-level retries are disabled so that our bounded policy is the only one.
        # HttpOptions.timeout is in milliseconds; PhaseTimeoutClient makes it a total.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
                httpx_client=self._http,
            ),
        )
        self.client_init_s = round(time.monotonic() - created, 3)
        self.temperature = temperature
        self.thinking_level = thinking_level
        self.last_timings: dict[str, Any] = {}
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_retry_wait_s = max_retry_wait_s
        self.deadline = deadline
        self.max_output_tokens = max_output_tokens

    def generate_structured(self, *, system, user, schema, purpose, metadata=None):
        from google.genai import errors

        schema_json = to_response_schema(schema)
        attempt = 0
        while True:
            timeout_s = self.timeout_s
            if self.deadline is not None:
                remaining = self.deadline.remaining()
                if remaining < MIN_REQUEST_S:
                    raise LLMTimeoutError(
                        f"not enough run time left to start the Gemini request ({remaining:.1f}s remaining)"
                    )
                timeout_s = min(timeout_s, remaining)
            config = self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,  # None = model default (recommended for Gemini 3)
                thinking_config=(
                    self._types.ThinkingConfig(thinking_level=self.thinking_level.upper())
                    if self.thinking_level else None
                ),
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=schema_json,
                tools=None,
                http_options=self._types.HttpOptions(
                    timeout=int(timeout_s * 1000), retry_options=self._types.HttpRetryOptions(attempts=1)
                ),
            )
            events = self._http.begin_trace()
            self.current_trace = events  # readable by a watchdog if this request is abandoned
            try:
                response = self._client.models.generate_content(model=self.model, contents=user, config=config)
                self.last_timings = summarize_trace(events)
                return parse_structured(response.text, schema)
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                message = redact_secrets(f"{code} {getattr(exc, 'status', '')} {getattr(exc, 'message', '')}")
                if code in (401, 403) or (code == 400 and "api key" in message.lower()):
                    raise LLMAuthError(
                        f"Gemini rejected the credentials or model access ({message})", status_code=code
                    ) from None
                if code == 404:
                    raise LLMAuthError(
                        f"Gemini model '{self.model}' is not available to this key ({message})", status_code=code
                    ) from None
                if code == 429:
                    error, retryable = LLMRateLimitError(f"Gemini rate limit ({message})", status_code=code), True
                else:
                    retryable = isinstance(code, int) and code >= 500
                    error = LLMUnavailableError(f"Gemini request failed ({message})", status_code=code)
                wait = _retry_delay_seconds(getattr(exc, "details", None)) or 2.0 * (2**attempt)
            except LLMOutputError:
                raise
            except Exception as exc:  # transport: timeouts, connection/TLS errors
                self.last_timings = summarize_trace(events)
                error, retryable = _classify_transport_error(exc, timeout_s)
                if self.last_timings:
                    error = type(error)(f"{error} [stage timings: {format_timings(self.last_timings)}]")
                wait = 2.0 * (2**attempt)
            finally:
                self._http.end_trace()
            if not retryable or attempt >= self.max_retries:
                raise error from None
            if wait > self.max_retry_wait_s:
                raise type(error)(
                    f"{error}; asked to wait {wait:.0f}s before retrying, above the {self.max_retry_wait_s:.0f}s cap",
                    status_code=error.status_code,
                ) from None
            # Sleep only if the retry itself would still have time to run afterwards.
            if self.deadline is not None and self.deadline.remaining() - wait < MIN_REQUEST_S:
                raise type(error)(
                    f"{error}; not enough run time left to retry", status_code=error.status_code
                ) from None
            time.sleep(wait)
            attempt += 1
