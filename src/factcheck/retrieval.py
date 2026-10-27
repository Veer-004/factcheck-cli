"""Web retrieval: provider interface, Tavily implementation, budgeted rounds.

Tavily credit costs (verified 2026-10-26, docs.tavily.com/documentation/api-credits):
basic search = 1 credit; basic extract = 1 credit per 5 *successful* URLs.
Credits are reserved conservatively before each request and failed requests
are not refunded, so the configured budget is an upper bound on spend.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import requests

from .config import Config
from .evidence import dedupe_results, normalize_url
from .limits import MIN_REQUEST_S, CreditBudget, Deadline, split_timeout
from .models import ExtractedPage, SearchResult
from .security import is_safe_url, redact_secrets
from .textnorm import normalize_for_match

SEARCH_CREDITS = 1
URLS_PER_EXTRACT_CREDIT = 5


def extract_credits(n_urls: int) -> int:
    return math.ceil(n_urls / URLS_PER_EXTRACT_CREDIT) if n_urls > 0 else 0


class SearchError(RuntimeError):
    """Technical retrieval failure. ``status_code`` is the HTTP status, when known."""

    def __init__(self, message: str = "", *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class SearchAuthError(SearchError):
    """Missing/invalid credentials: fail the run fast."""


class SearchRateLimitError(SearchError):
    """HTTP 429 / usage limit. Retried a bounded number of times."""


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchResult]: ...

    def extract(self, urls: list[str]) -> list[ExtractedPage]: ...


class PhaseTimeoutSession(requests.Session):
    """requests session that turns a scalar timeout into (connect, read) limits.

    requests applies a scalar timeout to the connection *and* to each read, so
    ``timeout=20`` can take ~40 s. ``split_timeout`` makes the pair sum to the
    scalar and keeps connection stalls short.
    """

    def request(self, method, url, **kwargs):  # type: ignore[override]
        timeout = kwargs.get("timeout")
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
            kwargs["timeout"] = split_timeout(float(timeout))
        return super().request(method, url, **kwargs)


class TavilyProvider:
    """Thin adapter over ``tavily-python``. Always requires an explicit API key
    (the SDK's keyless mode is deliberately not used)."""

    def __init__(self, api_key: str, timeout_s: float = 20.0):
        from tavily import TavilyClient

        if not api_key:
            raise SearchAuthError("TAVILY_API_KEY is not set")
        self._client = TavilyClient(api_key=api_key, session=PhaseTimeoutSession())
        self.timeout_s = timeout_s

    def _call(self, fn, *args, **kwargs):
        from tavily import errors as terr

        try:
            return fn(*args, **kwargs)
        except (terr.InvalidAPIKeyError, terr.MissingAPIKeyError, terr.ForbiddenError) as exc:
            status = 401 if isinstance(exc, terr.InvalidAPIKeyError) else 403 if isinstance(exc, terr.ForbiddenError) else None
            raise SearchAuthError(
                f"Tavily rejected the credentials ({redact_secrets(str(exc))})", status_code=status
            ) from None
        except terr.UsageLimitExceededError as exc:
            raise SearchRateLimitError(
                f"Tavily usage/rate limit ({redact_secrets(str(exc))})", status_code=429
            ) from None
        except terr.TimeoutError as exc:
            raise SearchError(
                f"Tavily did not respond in time ({redact_secrets(str(exc))}); check network access to api.tavily.com"
            ) from None
        except Exception as exc:
            status = 400 if isinstance(exc, terr.BadRequestError) else None
            raise SearchError(
                f"Tavily request failed: {type(exc).__name__}: {redact_secrets(str(exc))[:300]}", status_code=status
            ) from None

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        data = self._call(
            self._client.search,
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            # Without this Tavily omits published_date for topic="general", so every
            # passage was undated and the time-sensitive recency rules could never
            # see a recent source (docs.tavily.com search reference, 2026-10-27).
            include_published_date=True,
            timeout=self.timeout_s,
        )
        out = []
        for r in (data or {}).get("results", [])[:max_results]:
            if not isinstance(r, dict) or not is_safe_url(r.get("url", "")):
                continue
            out.append(
                SearchResult(
                    url=r["url"],
                    title=str(r.get("title") or "")[:300],
                    content=str(r.get("content") or "")[:4000],
                    published_date=r.get("published_date") if isinstance(r.get("published_date"), str) else None,
                    score=r.get("score") if isinstance(r.get("score"), (int, float)) else None,
                )
            )
        return out

    def extract(self, urls: list[str]) -> list[ExtractedPage]:
        data = self._call(
            self._client.extract, urls=urls, extract_depth="basic", format="text", timeout=self.timeout_s
        )
        out = []
        for r in (data or {}).get("results", []):
            if isinstance(r, dict) and is_safe_url(r.get("url", "")):
                out.append(ExtractedPage(url=r["url"], raw_content=str(r.get("raw_content") or "")[:100_000]))
        return out


@dataclass
class RetrievalRound:
    results: list[SearchResult] = field(default_factory=list)
    pages: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    searches_attempted: int = 0
    searches_succeeded: int = 0
    credits: int = 0
    budget_exhausted: bool = False


class RetrievalService:
    """Runs bounded retrieval rounds with a shared budget and an in-memory, per-run cache."""

    def __init__(
        self,
        provider: SearchProvider,
        budget: CreditBudget,
        deadline: Deadline,
        config: Config,
        sleep=time.sleep,
    ):
        self.provider = provider
        self.budget = budget
        self.deadline = deadline
        self.config = config
        self._sleep = sleep
        self._lock = threading.Lock()
        self._search_cache: dict[str, list[SearchResult]] = {}
        self._page_cache: dict[str, str] = {}

    def _request_fits(self, extra_wait: float = 0.0) -> bool:
        """True if a provider request (bounded by search_timeout_s) still fits in the run."""
        needed = max(MIN_REQUEST_S, self.config.search_timeout_s)
        return self.deadline.remaining() - extra_wait >= needed

    def _with_retries(self, fn, *args):
        attempt = 0
        while True:
            try:
                return fn(*args)
            except SearchRateLimitError:
                if attempt >= self.config.max_429_retries:
                    raise
                wait = 2.0 * (2**attempt)
                if not self._request_fits(extra_wait=wait):
                    raise
                self._sleep(wait)
                attempt += 1

    def run_round(self, queries: list[str]) -> RetrievalRound:
        rnd = RetrievalRound()
        for query in queries[: self.config.max_queries_per_round]:
            if not self._request_fits():
                rnd.errors.append("not enough run time left to start a web search")
                break
            key = normalize_for_match(query)
            with self._lock:
                cached = self._search_cache.get(key)
            if cached is not None:
                rnd.results.extend(cached)
                rnd.searches_attempted += 1
                rnd.searches_succeeded += 1
                continue
            if not self.budget.reserve(SEARCH_CREDITS):
                rnd.budget_exhausted = True
                rnd.notes.append("search-credit budget reached; remaining queries skipped")
                break
            rnd.credits += SEARCH_CREDITS
            rnd.searches_attempted += 1
            try:
                found = self._with_retries(self.provider.search, query, self.config.max_results_per_query)
            except SearchAuthError:
                raise
            except SearchError as exc:
                rnd.errors.append(redact_secrets(str(exc)))
                continue
            found = list(found)[: self.config.max_results_per_query]
            with self._lock:
                self._search_cache[key] = found
            rnd.results.extend(found)
            rnd.searches_succeeded += 1

        unique = dedupe_results(rnd.results, self.config.max_urls_per_claim)
        wanted = [r.url for r in unique][: self.config.max_extract_urls]
        with self._lock:
            for url in wanted:
                if normalize_url(url) in self._page_cache:
                    rnd.pages[normalize_url(url)] = self._page_cache[normalize_url(url)]
        to_fetch = [u for u in wanted if normalize_url(u) not in rnd.pages]
        if to_fetch and not self._request_fits():
            rnd.notes.append("not enough run time left for page extraction; snippets used at reduced weight")
        elif to_fetch:
            cost = extract_credits(len(to_fetch))
            if not self.budget.reserve(cost):
                rnd.budget_exhausted = True
                rnd.notes.append("search-credit budget reached; using search snippets instead of full pages")
            else:
                rnd.credits += cost
                try:
                    pages = self._with_retries(self.provider.extract, to_fetch)
                    with self._lock:
                        for page in pages:
                            if page.raw_content.strip():
                                self._page_cache[normalize_url(page.url)] = page.raw_content
                                rnd.pages[normalize_url(page.url)] = page.raw_content
                    missing = len(to_fetch) - sum(1 for u in to_fetch if normalize_url(u) in rnd.pages)
                    if missing:
                        rnd.notes.append(f"page extraction failed for {missing} URL(s); snippets used at reduced weight")
                except SearchAuthError:
                    raise
                except SearchError as exc:
                    rnd.notes.append(f"page extraction failed ({redact_secrets(str(exc))}); snippets used at reduced weight")
        return rnd
