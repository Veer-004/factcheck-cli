"""Opt-in live provider diagnostics: exactly one minimal request each.

    python -m factcheck.diagnostics gemini-model   # GET models/<model>: access + metadata, no generation
    python -m factcheck.diagnostics gemini         # 1 small generateContent call (app settings)
    python -m factcheck.diagnostics tavily         # 1 basic search (1 Tavily credit)
    python -m factcheck.diagnostics ollama         # local: server up? model installed? 1 small chat

Every diagnostic has a hard wall-clock deadline (default 15 s; 60 s for the
local Ollama check, which may have to load the model first) covering DNS,
client creation, connection/TLS setup and the response wait. The request runs
in a daemon worker thread; when the deadline passes, the HTTP client is closed
to abort the socket and the diagnostic returns. A worker still blocked (e.g.
in DNS, which has no timeout) cannot keep the process alive, and its own
per-phase HTTP timeouts bound it anyway. No retries at any level.

Output: elapsed time, per-stage timings, HTTP status and sanitised error text.
Never API keys, headers or request bodies. Nothing runs unless invoked.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from . import envfile
from .config import GEMINI_KEY_ENV, TAVILY_KEY_ENV, Config, ConfigError
from .limits import Deadline
from .llm import GeminiClient, LLMOutputError, format_timings, summarize_trace
from .retrieval import TavilyProvider
from .security import redact_secrets

DIAG_TIMEOUT_S = 15.0
ABORT_GRACE_S = 0.5
GEMINI_HOST = "generativelanguage.googleapis.com"


class _Ping(BaseModel):
    ok: bool


@dataclass
class DiagnosticResult:
    provider: str
    ok: bool
    elapsed_s: float
    detail: str = ""
    error_type: str | None = None
    status_code: int | None = None
    stages: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"provider:    {self.provider}"]
        lines += [f"{k + ':':<12} {v}" for k, v in self.info.items()]
        lines += [f"result:      {'OK' if self.ok else 'FAILED'}", f"elapsed:     {self.elapsed_s:.2f}s (wall clock)"]
        if self.stages:
            lines.append("stages:")
            lines += [f"  {k:<18} {v:.2f}s" if isinstance(v, float) else f"  {k:<18} {v}" for k, v in self.stages.items()]
        if self.error_type:
            lines.append(f"error_type:  {self.error_type}")
        if self.status_code is not None:
            lines.append(f"http_status: {self.status_code}")
        if self.detail:
            lines.append(f"detail:      {redact_secrets(self.detail)}")
        return "\n".join(lines)


@dataclass
class _Outcome:
    finished: bool
    value: Any = None
    error: BaseException | None = None


def run_with_deadline(work: Callable[[], Any], total_s: float, abort: Callable[[], None]) -> _Outcome:
    """Run ``work`` in a daemon thread; give up after ``total_s`` wall-clock seconds.

    On expiry ``abort`` is called (it should close the HTTP client so the blocked
    socket fails) and the worker gets ABORT_GRACE_S to unwind. The call returns
    within ``total_s`` either way.
    """
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = work()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            box["error"] = exc

    worker = threading.Thread(target=run, name="factcheck-diagnostic", daemon=True)
    worker.start()
    worker.join(max(0.0, total_s - ABORT_GRACE_S))
    if worker.is_alive():
        try:
            abort()
        except Exception:  # noqa: BLE001 - best effort
            pass
        worker.join(ABORT_GRACE_S)
        return _Outcome(finished=False)
    return _Outcome(finished=True, value=box.get("value"), error=box.get("error"))


def _fail(provider: str, start: float, exc: BaseException, **kw: Any) -> DiagnosticResult:
    return DiagnosticResult(provider, False, time.monotonic() - start, detail=redact_secrets(str(exc)),
                            error_type=type(exc).__name__, status_code=getattr(exc, "status_code", None), **kw)


def _dns(host: str, stages: dict[str, Any]) -> None:
    """Separate DNS measurement (httpx's connect timeout does not cover name resolution)."""
    t = time.monotonic()
    try:
        addrs = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        stages["dns"] = round(time.monotonic() - t, 3)
        stages["dns_error"] = type(exc).__name__
        return
    stages["dns"] = round(time.monotonic() - t, 3)
    stages["dns_addresses"] = (f"{sum(a[0] == socket.AF_INET for a in addrs)} IPv4, "
                               f"{sum(a[0] == socket.AF_INET6 for a in addrs)} IPv6")


def _llm_diagnostic(api_key: str, model: str, total_s: float, client_factory, request: Callable[[Any], str],
                       settings: dict[str, Any], label: str, provider: str = "gemini", host: str = GEMINI_HOST,
                       endpoint: str = f"https://{GEMINI_HOST}/v1beta") -> DiagnosticResult:
    """One bounded LLM request (shared by the Gemini and Ollama diagnostics)."""
    start = time.monotonic()
    deadline = Deadline(total_s)
    stages: dict[str, Any] = {}
    holder: dict[str, Any] = {}
    info = {"model": model, "endpoint": endpoint, "request": label, **settings}

    def work() -> str:
        _dns(host, stages)
        t = time.monotonic()
        client = client_factory(api_key, model, timeout_s=total_s, max_retries=0, deadline=deadline,
                                max_output_tokens=1024, **settings)
        holder["client"] = client
        stages["client_init"] = round(time.monotonic() - t, 3)
        try:
            return request(client)
        finally:
            stages.update(summarize_trace(getattr(client, "current_trace", None) or []))
            t = time.monotonic()
            http = getattr(client, "_http", None)
            if http is not None:
                http.close()
            stages["cleanup"] = round(time.monotonic() - t, 3)

    def abort() -> None:
        client = holder.get("client")
        if client is not None:
            stages.update(summarize_trace(list(getattr(client, "current_trace", None) or [])))
            if getattr(client, "_http", None) is not None:
                client._http.close()

    outcome = run_with_deadline(work, total_s, abort)
    if not outcome.finished:
        reached = stages.get("last_stage") or ("client_init" if "dns" in stages else "dns")
        return DiagnosticResult(
            provider, False, time.monotonic() - start, error_type="WallClockDeadlineExceeded",
            detail=(f"no result within the {total_s:.0f}s wall-clock deadline; last stage reached: {reached}. "
                    + (f"Timings: {format_timings(stages)}" if format_timings(stages) else "")),
            stages=stages, info=info)
    if isinstance(outcome.error, LLMOutputError):
        return DiagnosticResult(provider, True, time.monotonic() - start, stages=stages, info=info,
                                detail=f"HTTP 200: authenticated and model reachable, but output did not parse: {outcome.error}")
    if outcome.error is not None:
        return _fail(provider, start, outcome.error, stages=stages, info=info)
    return DiagnosticResult(provider, True, time.monotonic() - start, detail=outcome.value, stages=stages, info=info)


def diagnose_gemini(api_key: str, model: str, timeout_s: float = DIAG_TIMEOUT_S, client_factory=GeminiClient,
                    temperature: float | None = None, thinking_level: str | None = "low") -> DiagnosticResult:
    """One small structured generateContent call with the app's request settings."""

    def request(client) -> str:
        client.generate_structured(system="This is an automated connectivity check.",
                                   user="Respond with ok set to true.", schema=_Ping, purpose="diagnostic")
        return "authenticated; model returned valid structured output"

    settings = {"temperature": temperature, "thinking_level": thinking_level}
    return _llm_diagnostic(api_key, model, timeout_s, client_factory, request, settings, "generateContent")


def diagnose_gemini_model(api_key: str, model: str, timeout_s: float = DIAG_TIMEOUT_S,
                          client_factory=GeminiClient) -> DiagnosticResult:
    """One metadata request (models.get): is this model ID valid and accessible to this key?"""

    def request(client) -> str:
        client.current_trace = client._http.begin_trace()
        try:
            m = client._client.models.get(
                model=model, config={"http_options": {"timeout": int(timeout_s * 1000)}}
            )
        finally:
            client._http.end_trace()
        actions = getattr(m, "supported_actions", None) or []
        return (f"model exists and is accessible: name={getattr(m, 'name', '?')}, "
                f"supports generateContent={'generateContent' in actions}, "
                f"input_token_limit={getattr(m, 'input_token_limit', '?')}, "
                f"output_token_limit={getattr(m, 'output_token_limit', '?')}, "
                f"thinking={getattr(m, 'thinking', '?')}")

    return _llm_diagnostic(api_key, model, timeout_s, client_factory, request, {}, "models.get (no generation)")


OLLAMA_DIAG_TIMEOUT_S = 60.0


def diagnose_ollama(model: str, base_url: str, timeout_s: float = OLLAMA_DIAG_TIMEOUT_S,
                    client_factory=None) -> DiagnosticResult:
    """Local check: server reachable, model installed, then one small structured chat (think=false)."""
    from urllib.parse import urlsplit

    import httpx

    from .llm import LLMNetworkError, LLMUnavailableError
    from .ollama import OllamaClient

    def factory(_key, m, *, timeout_s, max_retries, deadline, max_output_tokens):
        return OllamaClient(m, base_url=base_url, timeout_s=timeout_s, deadline=deadline,
                            max_output_tokens=max_output_tokens)

    def request(client) -> str:
        try:
            version = client._http.get(f"{client.base_url}/api/version", timeout=5).json().get("version", "?")
            tags = client._http.get(f"{client.base_url}/api/tags", timeout=5).json().get("models", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMNetworkError(
                f"Ollama is not reachable at {client.base_url} ({type(exc).__name__}); open the Ollama app or "
                "run `ollama serve`") from None
        names = {t.get("name") for t in tags if isinstance(t, dict)} | {t.get("model") for t in tags if isinstance(t, dict)}
        if model not in names and f"{model}:latest" not in names:
            raise LLMUnavailableError(
                f"Ollama {version} is running but model '{model}' is not installed; run `ollama pull {model}`")
        client.generate_structured(system="This is an automated connectivity check.",
                                   user="Respond with ok set to true.", schema=_Ping, purpose="diagnostic")
        thinking = "yes (discarded)" if getattr(client, "thinking_returned", False) else "no"
        return (f"Ollama {version}: model {model} returned valid structured output; "
                f"thinking output returned despite think=false: {thinking}")

    return _llm_diagnostic("", model, timeout_s, client_factory or factory, request, {}, "api/chat",
                           provider="ollama", host=urlsplit(base_url).hostname or "localhost", endpoint=base_url)


def diagnose_tavily(api_key: str, timeout_s: float = DIAG_TIMEOUT_S, provider_factory=TavilyProvider) -> DiagnosticResult:
    """One basic search with max_results=1 (1 credit)."""
    start = time.monotonic()
    holder: dict[str, Any] = {}

    def work():
        provider = provider_factory(api_key, timeout_s=timeout_s)
        holder["provider"] = provider
        return provider.search("Eiffel Tower completion year", 1)

    def abort():
        client = getattr(holder.get("provider"), "_client", None)
        if client is not None and hasattr(client, "session"):
            client.session.close()

    outcome = run_with_deadline(work, timeout_s, abort)
    if not outcome.finished:
        return DiagnosticResult("tavily", False, time.monotonic() - start, error_type="WallClockDeadlineExceeded",
                                detail=f"no result within the {timeout_s:.0f}s wall-clock deadline")
    if outcome.error is not None:
        return _fail("tavily", start, outcome.error)
    return DiagnosticResult("tavily", True, time.monotonic() - start,
                            detail=f"authenticated; search returned {len(outcome.value)} result(s)")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="python -m factcheck.diagnostics", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("provider", choices=["gemini-model", "gemini", "tavily", "ollama"])
    ap.add_argument("--model", help="model ID (default: FACTCHECK_MODEL, or FACTCHECK_OLLAMA_MODEL for ollama)")
    ap.add_argument("--timeout", type=float, default=None,
                    help="wall-clock deadline in seconds (default 15, ollama 60; max 30, ollama 120)")
    ap.add_argument("--thinking-level", choices=["low", "medium", "high", "default"],
                    help="gemini only: override the configured thinking level ('default' = send none)")
    ap.add_argument("--temperature", type=float, help="gemini only: send this temperature (default: none sent)")
    args = ap.parse_args(argv)
    for warning in envfile.load_env_file():
        print(f"warning: {warning}", file=sys.stderr)
    local = args.provider == "ollama"
    if args.timeout is None:
        args.timeout = OLLAMA_DIAG_TIMEOUT_S if local else DIAG_TIMEOUT_S
    limit = 120.0 if local else 30.0
    if not 2.0 <= args.timeout <= limit:
        print(f"error: --timeout must be between 2 and {limit:.0f} seconds", file=sys.stderr)
        return 2
    if local:
        try:
            cfg = Config.from_env()
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        result = diagnose_ollama(args.model or cfg.ollama_model, cfg.ollama_url, args.timeout)
        print(result.render())
        return 0 if result.ok else 1

    env = TAVILY_KEY_ENV if args.provider == "tavily" else GEMINI_KEY_ENV
    key = os.environ.get(env, "").strip()
    if not key:
        print(f"error: {env} is not set; no request was made", file=sys.stderr)
        return 2
    if args.provider == "tavily":
        result = diagnose_tavily(key, args.timeout)
    else:
        try:
            cfg = Config.from_env(model=args.model)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.provider == "gemini-model":
            result = diagnose_gemini_model(key, cfg.model, args.timeout)
        else:
            level = cfg.thinking_level if args.thinking_level is None else (
                None if args.thinking_level == "default" else args.thinking_level)
            temp = cfg.temperature if args.temperature is None else args.temperature
            result = diagnose_gemini(key, cfg.model, args.timeout, temperature=temp, thinking_level=level)
    print(result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
