"""Deterministic fake providers for offline tests and offline evaluation.

Nothing here touches the network. Fake LLM responses may be pydantic objects,
dicts, raw JSON strings (parsed by the real parser, so malformed output is
exercised end-to-end) or exceptions.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Callable

from factcheck.llm import LLMOutputError, parse_structured
from factcheck.models import Claim, EvidencePassage, ExtractedPage, LLMExtraction, LLMJudgment, SearchResult
from factcheck.retrieval import SearchAuthError, SearchError

MALFORMED = "this is { not valid json"

JudgeFn = Callable[[Claim, list[EvidencePassage], int, str], Any]


class FakeLLM:
    def __init__(self, extraction: Any = None, judge: JudgeFn | None = None):
        self.extraction = extraction
        self.judge = judge
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._extract_attempt = 0

    def _materialize(self, resp: Any, schema):
        if callable(resp) and not isinstance(resp, type):
            resp = resp()
        if isinstance(resp, BaseException):
            raise resp
        if isinstance(resp, str):
            return parse_structured(resp, schema)
        if isinstance(resp, dict):
            return parse_structured(json.dumps(resp), schema)
        if resp is None:
            raise LLMOutputError("fake: no scripted response")
        return resp

    def generate_structured(self, *, system, user, schema, purpose, metadata=None):
        with self._lock:
            self.calls.append({"purpose": purpose, "system": system, "user": user, "metadata": metadata})
        if purpose == "extract_claims":
            resp = self.extraction
            if isinstance(resp, list):
                with self._lock:
                    idx = min(self._extract_attempt, len(resp) - 1)
                    self._extract_attempt += 1
                resp = resp[idx]
            return self._materialize(resp, schema)
        if purpose == "judge":
            assert self.judge is not None, "no fake judge configured"
            resp = self.judge(metadata["claim"], metadata["passages"], metadata["attempt"], user)
            return self._materialize(resp, schema)
        raise AssertionError(f"unexpected purpose {purpose}")

    def calls_for(self, purpose: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["purpose"] == purpose]


class FakeSearch:
    """Query -> results lookup by case-insensitive substring of the scripted keys."""

    def __init__(
        self,
        corpus: dict[str, list[dict]] | None = None,
        pages: dict[str, str] | None = None,
        fail_queries: Callable[[str], bool] | None = None,
        fail_extract: bool = False,
        auth_fail: bool = False,
    ):
        self.corpus = corpus or {}
        self.pages = pages or {}
        self.fail_queries = fail_queries or (lambda q: False)
        self.fail_extract = fail_extract
        self.auth_fail = auth_fail
        self.search_calls: list[str] = []
        self.extract_calls: list[list[str]] = []
        self._lock = threading.Lock()

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        with self._lock:
            self.search_calls.append(query)
        if self.auth_fail:
            raise SearchAuthError("fake: invalid API key")
        if self.fail_queries(query):
            raise SearchError("fake: connection reset")
        q = query.lower()
        for key, results in self.corpus.items():
            if key.lower() in q:
                return [SearchResult(**r) for r in results][:max_results]
        return []

    def extract(self, urls: list[str]) -> list[ExtractedPage]:
        with self._lock:
            self.extract_calls.append(list(urls))
        if self.fail_extract:
            raise SearchError("fake: extraction service down")
        return [ExtractedPage(url=u, raw_content=self.pages[u]) for u in urls if u in self.pages]


# --------------------------------------------------------------------------
# Helpers for building scripted responses
# --------------------------------------------------------------------------


def claim_dict(
    statement: str,
    excerpt: str | None = None,
    *,
    checkability: str = "CHECKABLE",
    claim_type: str = "FACTUAL",
    time_sensitive: bool = False,
    queries: list[str] | None = None,
    fallback: list[str] | None = None,
) -> dict:
    return {
        "statement": statement,
        "source_excerpt": excerpt if excerpt is not None else statement,
        "checkability": checkability,
        "claim_type": claim_type,
        "time_sensitive": time_sensitive,
        "time_anchor": None,
        "primary_queries": queries if queries is not None else [statement],
        "fallback_queries": fallback or [],
    }


def extraction(*claims: dict) -> dict:
    return {"claims": list(claims)}


def sentence_with(text: str, needle: str) -> str:
    """The sentence of ``text`` containing ``needle`` (verbatim), for building valid quotes."""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if needle.lower() in sent.lower():
            return sent.strip()
    raise AssertionError(f"needle {needle!r} not in passage")


def rule_judge(rules: dict[str, tuple[str, str, str]], verdict: str = "SUPPORTED", correction: dict | None = None,
               numeric: dict[str, dict] | None = None) -> JudgeFn:
    """Build a judge: ``rules`` maps a URL substring to (stance, directness, quote needle).

    Passages whose URL matches no rule are marked IRRELEVANT. ``numeric`` maps a
    URL substring to a numeric_comparison dict.
    """

    def judge(claim: Claim, passages: list[EvidencePassage], attempt: int, user: str) -> dict:
        assessments = []
        for p in passages:
            for frag, (stance, directness, needle) in rules.items():
                if frag in p.url and needle.lower() in p.text.lower():
                    a = {
                        "evidence_id": p.evidence_id,
                        "stance": stance,
                        "directness": directness,
                        "quote": sentence_with(p.text, needle),
                        "explanation": "scripted",
                    }
                    if numeric and frag in numeric:
                        a["numeric_comparison"] = numeric[frag]
                    assessments.append(a)
                    break
            else:
                assessments.append(
                    {"evidence_id": p.evidence_id, "stance": "IRRELEVANT", "directness": "INFERENTIAL",
                     "quote": p.text[:40], "explanation": ""}
                )
        out = {"assessments": assessments, "proposed_verdict": verdict, "rationale": "scripted"}
        if correction is not None:
            ids = [p.evidence_id for p in passages if any(f in p.url for f in correction.get("urls", []))]
            out["correction"] = {"text": correction["text"], "evidence_ids": ids}
        return out

    return judge


def result(url: str, title: str, content: str, published_date: str | None = None) -> dict:
    return {"url": url, "title": title, "content": content, "published_date": published_date}


def judgment(assessments: list[dict], verdict: str = "SUPPORTED", correction: dict | None = None) -> LLMJudgment:
    return LLMJudgment.model_validate(
        {"assessments": assessments, "proposed_verdict": verdict, "rationale": "", "correction": correction}
    )


def ext(*claims: dict) -> LLMExtraction:
    return LLMExtraction.model_validate({"claims": list(claims)})
