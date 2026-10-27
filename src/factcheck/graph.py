"""LangGraph orchestration.

Main graph:
    validate_input -> extract_claims -> validate_claims
        -> (Send fan-out, one per checkable claim) claim_worker
        -> summarize -> render_report

Per-claim subgraph (invoked by ``claim_worker``):
    retrieve -> process_evidence -> sufficiency_gate
        -> retrieve (at most one fallback round) | judge | finalize
    judge -> validate_judgment -> judge (at most one repair) | finalize

Parallel claim results are merged with an ``operator.add`` reducer, so workers
never overwrite one another; results are re-sorted by claim index afterwards.
"""

from __future__ import annotations

import operator
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Any, Optional, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import GEMINI_KEY_ENV, TAVILY_KEY_ENV, Config, ConfigError
from .evidence import build_evidence
from .judging import SUMMARY_TEXT, select_verdict, summary_label
from .limits import CallBudget, CreditBudget, Deadline
from .llm import (
    BudgetedLLM,
    FallbackLLM,
    GeminiClient,
    LLMAuthError,
    LLMClient,
    LLMOutputError,
    LLMUnavailableError,
)
from .models import (
    Checkability,
    Claim,
    ClaimResult,
    DroppedClaim,
    EvidencePassage,
    LLMExtraction,
    LLMJudgment,
    Report,
    RunStats,
    SearchResult,
    Strengths,
    Verdict,
)
from .prompts import EXTRACTION_SYSTEM, JUDGE_SYSTEM, extraction_user_prompt, judge_user_prompt
from .retrieval import RetrievalService, SearchAuthError, SearchProvider, TavilyProvider
from .scoring import compute_strengths, score_claim
from .security import make_boundary, redact_secrets
from .validation import (
    JudgmentValidation,
    validate_claims,
    validate_correction,
    validate_input,
    validate_judgment,
)


class RunFailure(RuntimeError):
    """Run-level failure (CLI exit code 3)."""


@dataclass
class Deps:
    llm: BudgetedLLM
    retrieval: RetrievalService
    config: Config
    deadline: Deadline
    boundary: str


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


class MainState(TypedDict, total=False):
    text: str
    extraction: LLMExtraction
    checkable: list[Claim]
    not_checkable: list[Claim]
    dropped: list[DroppedClaim]
    over_limit: list[DroppedClaim]
    results: Annotated[list[ClaimResult], operator.add]
    warnings: Annotated[list[str], operator.add]
    final_results: list[ClaimResult]
    report: Report


class WorkerInput(TypedDict):
    claim: Claim


class ClaimState(TypedDict, total=False):
    claim: Claim
    rounds_done: int
    search_results: list[SearchResult]
    pages: dict[str, str]
    retrieval_errors: list[str]
    notes: list[str]
    searches_attempted: int
    searches_succeeded: int
    credits: int
    budget_exhausted: bool
    passages: list[EvidencePassage]
    sources_considered: int
    gate: str
    judge_attempts: int
    raw_judgment: Optional[LLMJudgment]
    problems: list[str]
    validation: Optional[JudgmentValidation]
    failure: Optional[str]
    result: ClaimResult


# --------------------------------------------------------------------------
# Per-claim subgraph
# --------------------------------------------------------------------------

SUFFICIENT_RELEVANCE = 0.5


def _fallback_queries(claim: Claim, used: set[str]) -> list[str]:
    queries = [q for q in claim.fallback_queries if q not in used]
    if not queries and claim.statement[:200] not in used:
        queries = [claim.statement[:200]]
    return queries


def build_claim_graph(deps: Deps):
    cfg = deps.config

    def retrieve(state: ClaimState) -> dict[str, Any]:
        claim = state["claim"]
        rounds_done = state.get("rounds_done", 0)
        if deps.deadline.expired():
            return {"failure": "run time limit reached before retrieval finished", "rounds_done": rounds_done + 1}
        if rounds_done == 0:
            queries = claim.primary_queries
        else:
            queries = _fallback_queries(claim, set(claim.primary_queries))
        rnd = deps.retrieval.run_round(queries)
        pages = dict(state.get("pages", {}))
        pages.update(rnd.pages)
        return {
            "rounds_done": rounds_done + 1,
            "search_results": state.get("search_results", []) + rnd.results,
            "pages": pages,
            "retrieval_errors": state.get("retrieval_errors", []) + rnd.errors,
            "notes": state.get("notes", []) + rnd.notes,
            "searches_attempted": state.get("searches_attempted", 0) + rnd.searches_attempted,
            "searches_succeeded": state.get("searches_succeeded", 0) + rnd.searches_succeeded,
            "credits": state.get("credits", 0) + rnd.credits,
            "budget_exhausted": state.get("budget_exhausted", False) or rnd.budget_exhausted,
        }

    def process_evidence(state: ClaimState) -> dict[str, Any]:
        bundle = build_evidence(state["claim"], state.get("search_results", []), state.get("pages", {}), cfg)
        notes = list(state.get("notes", []))
        if bundle.injection_sources:
            notes.append(
                f"{bundle.injection_sources} source(s) contained instruction-like text and were down-weighted"
            )
        return {"passages": bundle.passages, "sources_considered": bundle.sources_considered, "notes": notes}

    def sufficiency_gate(state: ClaimState) -> dict[str, Any]:
        if state.get("failure"):
            return {"gate": "finalize"}
        claim = state["claim"]
        passages = state.get("passages", [])
        rounds_done = state.get("rounds_done", 0)
        can_fallback = (
            rounds_done <= cfg.max_fallback_rounds
            and not state.get("budget_exhausted")
            and bool(_fallback_queries(claim, set(claim.primary_queries)))
            and not deps.deadline.expired()
        )
        if state.get("searches_attempted", 0) == 0:
            if state.get("budget_exhausted"):
                return {"gate": "finalize", "failure": "search-credit budget exhausted before this claim was searched"}
            reason = "; ".join(dict.fromkeys(state.get("retrieval_errors", []))) or "no searches could be run"
            return {"gate": "finalize", "failure": f"{reason} (claim not searched)"}
        if state.get("searches_succeeded", 0) == 0:
            return {"gate": "fallback" if can_fallback else "retrieval_failed"}
        strong = [p for p in passages if p.relevance >= SUFFICIENT_RELEVANCE]
        enough = len(strong) >= cfg.min_passages_for_sufficiency and any(not p.snippet_only for p in strong)
        if not enough and can_fallback:
            return {"gate": "fallback"}
        if not passages:
            return {"gate": "insufficient"}
        return {"gate": "judge"}

    def judge(state: ClaimState) -> dict[str, Any]:
        claim = state["claim"]
        attempts = state.get("judge_attempts", 0)
        problems = state.get("problems") or None
        user = judge_user_prompt(claim, state["passages"], deps.boundary, problems if attempts else None)
        try:
            raw = deps.llm.generate_structured(
                system=JUDGE_SYSTEM,
                user=user,
                schema=LLMJudgment,
                purpose="judge",
                metadata={"claim": claim, "passages": state["passages"], "attempt": attempts + 1},
            )
        except LLMOutputError as exc:
            return {"judge_attempts": attempts + 1, "raw_judgment": None, "problems": [str(exc)]}
        except LLMUnavailableError as exc:
            return {"judge_attempts": attempts + 1, "failure": f"judge model unavailable: {exc}"}
        return {"judge_attempts": attempts + 1, "raw_judgment": raw, "problems": []}

    def validate_judgment_node(state: ClaimState) -> dict[str, Any]:
        if state.get("failure"):
            return {}
        raw = state.get("raw_judgment")
        if raw is None:
            problems = state.get("problems") or ["the response could not be parsed"]
            validation = None
        else:
            validation = validate_judgment(raw, state["claim"], state["passages"], cfg)
            problems = validation.fatal + validation.rejected[:6] if validation.fatal else []
        if not problems:
            return {"validation": validation, "problems": []}
        if state.get("judge_attempts", 0) <= cfg.max_judge_repairs:
            return {"validation": None, "problems": problems}
        return {
            "validation": None,
            "problems": problems,
            "failure": "judgment remained invalid after a repair attempt: " + "; ".join(problems[:3]),
        }

    def finalize(state: ClaimState) -> dict[str, Any]:
        return {"result": finalize_claim(state, cfg)}

    def route_gate(state: ClaimState) -> str:
        gate = state.get("gate")
        if gate == "fallback":
            return "retrieve"
        if gate == "judge":
            return "judge"
        return "finalize"

    def route_validation(state: ClaimState) -> str:
        if state.get("failure") or state.get("validation") is not None:
            return "finalize"
        return "judge"

    g = StateGraph(ClaimState)
    g.add_node("retrieve", retrieve)
    g.add_node("process_evidence", process_evidence)
    g.add_node("sufficiency_gate", sufficiency_gate)
    g.add_node("judge", judge)
    g.add_node("validate_judgment", validate_judgment_node)
    g.add_node("finalize", finalize)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "process_evidence")
    g.add_edge("process_evidence", "sufficiency_gate")
    g.add_conditional_edges("sufficiency_gate", route_gate, ["retrieve", "judge", "finalize"])
    g.add_edge("judge", "validate_judgment")
    g.add_conditional_edges("validate_judgment", route_validation, ["judge", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


def _base_result(claim: Claim, verdict: Verdict, **kw: Any) -> ClaimResult:
    return ClaimResult(
        claim_id=claim.claim_id,
        index=claim.index,
        statement=claim.statement,
        source_excerpt=claim.source_excerpt,
        claim_type=claim.claim_type,
        checkability=claim.checkability,
        time_sensitive=claim.time_sensitive,
        verdict=verdict,
        **kw,
    )


def finalize_claim(state: ClaimState, cfg: Config) -> ClaimResult:
    claim = state["claim"]
    common = dict(
        retrieval_rounds=state.get("rounds_done", 0),
        search_credits=state.get("credits", 0),
        sources_considered=state.get("sources_considered", 0),
        notes=list(dict.fromkeys(state.get("notes", []))),
    )
    if state.get("failure"):
        return _base_result(claim, Verdict.PROCESSING_FAILED, failure_reason=state["failure"], **common)
    gate = state.get("gate")
    if gate == "retrieval_failed":
        errors = list(dict.fromkeys(state.get("retrieval_errors", [])))
        return _base_result(
            claim,
            Verdict.RETRIEVAL_FAILED,
            failure_reason="every web retrieval attempt failed: " + "; ".join(errors[:3]),
            **common,
        )
    if gate == "insufficient":
        return _base_result(
            claim,
            Verdict.INSUFFICIENT_EVIDENCE,
            rationale="retrieval succeeded but returned no passages relevant to the claim",
            score=0,
            strengths=Strengths(),
            **common,
        )

    v: JudgmentValidation = state["validation"]
    strengths = compute_strengths(v.supporting, v.refuting, v.context)
    verdict, reason = select_verdict(strengths)
    score, s_score, r_score, caps = score_claim(
        verdict, v.supporting, v.refuting, v.context, strengths, claim.time_sensitive
    )
    raw = state.get("raw_judgment")
    correction, correction_note = validate_correction(
        raw.correction if raw else None, claim, verdict, v, state.get("passages", [])
    )
    notes = common.pop("notes")
    if v.proposed_verdict and v.proposed_verdict != verdict:
        notes.append(
            f"model proposed {v.proposed_verdict.value}; deterministic rules selected {verdict.value}"
        )
    if v.rejected:
        notes.append(f"{len(v.rejected)} model citation(s) rejected by validation: " + "; ".join(v.rejected[:4]))
    if state.get("judge_attempts", 0) > 1:
        notes.append("judgment needed one repair attempt")
    order = lambda c: (-c.weight, c.evidence_id)  # noqa: E731
    return _base_result(
        claim,
        verdict,
        rationale=reason,
        model_proposed_verdict=v.proposed_verdict,
        supporting=sorted(v.supporting, key=order),
        refuting=sorted(v.refuting, key=order),
        context=sorted(v.context, key=order),
        correction=correction,
        correction_note=correction_note,
        score=score,
        support_score=s_score,
        refute_score=r_score,
        score_caps=caps,
        strengths=strengths,
        rejected_citations=len(v.rejected),
        evidence=state.get("passages", []),
        notes=notes,
        **common,
    )


# --------------------------------------------------------------------------
# Main graph
# --------------------------------------------------------------------------


def build_graph(deps: Deps):
    cfg = deps.config
    claim_graph = build_claim_graph(deps)

    def validate_input_node(state: MainState) -> dict[str, Any]:
        return {"text": validate_input(state["text"], cfg)}

    def extract_claims(state: MainState) -> dict[str, Any]:
        user = extraction_user_prompt(state["text"], deps.boundary)
        last_error = ""
        for attempt in range(2):  # one bounded retry for malformed output
            try:
                extraction = deps.llm.generate_structured(
                    system=EXTRACTION_SYSTEM,
                    user=user if attempt == 0 else user + f"\n\nYour previous answer was invalid ({last_error}). Return valid JSON matching the schema.",
                    schema=LLMExtraction,
                    purpose="extract_claims",
                    metadata={"text": state["text"], "attempt": attempt + 1},
                )
                return {"extraction": extraction}
            except LLMOutputError as exc:
                last_error = str(exc)[:300]
            except LLMUnavailableError as exc:
                raise RunFailure(f"claim extraction failed: {exc}") from None
        raise RunFailure(f"claim extraction returned malformed output twice: {last_error}")

    def validate_claims_node(state: MainState) -> dict[str, Any]:
        cv = validate_claims(state["extraction"], state["text"], cfg)
        warnings = []
        if cv.dropped:
            warnings.append(f"{len(cv.dropped)} extracted claim(s) dropped by validation")
        if cv.over_limit:
            warnings.append(f"{len(cv.over_limit)} checkable claim(s) not checked (over the per-run limit)")
        return {
            "checkable": cv.checkable,
            "not_checkable": cv.not_checkable,
            "dropped": cv.dropped,
            "over_limit": cv.over_limit,
            "warnings": warnings,
        }

    def route_claims(state: MainState):
        claims = state.get("checkable", [])
        if not claims:
            return "summarize"
        return [Send("claim_worker", {"claim": c}) for c in claims]

    def claim_worker(state: WorkerInput) -> dict[str, Any]:
        claim = state["claim"]
        try:
            out = claim_graph.invoke({"claim": claim}, config={"recursion_limit": cfg.recursion_limit})
            result = out["result"]
        except (LLMAuthError, SearchAuthError):
            raise
        except GraphRecursionError:
            result = _base_result(claim, Verdict.PROCESSING_FAILED, failure_reason="per-claim step limit reached")
        except Exception as exc:  # noqa: BLE001 - one claim's bug must not sink the run
            result = _base_result(
                claim,
                Verdict.PROCESSING_FAILED,
                failure_reason=f"unexpected error: {type(exc).__name__}: {redact_secrets(str(exc))[:200]}",
            )
        return {"results": [result]}

    def summarize(state: MainState) -> dict[str, Any]:
        not_checkable = [
            _base_result(
                c,
                Verdict.NOT_CHECKABLE,
                rationale=f"classified as {c.checkability.value.replace('_', ' ').lower()}; not suitable for factual verification",
            )
            for c in state.get("not_checkable", [])
        ]
        by_id: dict[str, ClaimResult] = {}
        for r in state.get("results", []) + not_checkable:
            by_id.setdefault(r.claim_id, r)
        return {"final_results": sorted(by_id.values(), key=lambda r: r.index)}

    def render_report(state: MainState) -> dict[str, Any]:
        results = state.get("final_results", [])
        label = summary_label(results)
        counts = Counter(r.verdict.value for r in results)
        breakdown = ", ".join(f"{n} {v.lower().replace('_', ' ')}" for v, n in sorted(counts.items()))
        text = SUMMARY_TEXT[label] + (f" ({len(results)} claim(s): {breakdown}.)" if results else "")
        warnings = list(dict.fromkeys(state.get("warnings", [])))
        if any(r.verdict in (Verdict.PROCESSING_FAILED,) and "budget" in (r.failure_reason or "") for r in results):
            warnings.append("a per-run budget was reached; affected claims are marked PROCESSING_FAILED")
        report = Report(
            input_chars=len(state["text"]),
            summary_label=label,
            summary_text=text,
            claims=results,
            dropped_claims=state.get("dropped", []),
            over_limit_claims=state.get("over_limit", []),
            warnings=warnings,
            as_of=cfg.as_of.isoformat(),
        )
        return {"report": report}

    g = StateGraph(MainState)
    g.add_node("validate_input", validate_input_node)
    g.add_node("extract_claims", extract_claims)
    g.add_node("validate_claims", validate_claims_node)
    g.add_node("claim_worker", claim_worker)
    g.add_node("summarize", summarize)
    g.add_node("render_report", render_report)
    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "extract_claims")
    g.add_edge("extract_claims", "validate_claims")
    g.add_conditional_edges("validate_claims", route_claims, ["claim_worker", "summarize"])
    g.add_edge("claim_worker", "summarize")
    g.add_edge("summarize", "render_report")
    g.add_edge("render_report", END)
    return g.compile()


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def build_llm(config: Config, deadline: Deadline, call_budget: CallBudget) -> tuple[LLMClient, str]:
    """Build the configured provider chain: Gemini (+ optional Ollama fallback), or Ollama only."""
    from .ollama import OllamaClient  # local adapter; no network activity until a request is made

    def ollama() -> OllamaClient:
        return OllamaClient(config.ollama_model, base_url=config.ollama_url,
                            timeout_s=config.ollama_timeout_s, deadline=deadline)

    if config.provider == "ollama":
        return ollama(), f"ollama:{config.ollama_model}"
    key = os.environ.get(GEMINI_KEY_ENV, "").strip()
    if not key:
        # A missing key is a configuration error, not an outage: never silently go local.
        raise ConfigError(f"{GEMINI_KEY_ENV} is not set (or set FACTCHECK_PROVIDER=ollama to run locally)")
    primary = GeminiClient(
        key,
        config.model,
        timeout_s=config.llm_timeout_s,
        max_retries=config.max_429_retries,
        max_retry_wait_s=config.max_retry_wait_s,
        deadline=deadline,
        temperature=config.temperature,
        thinking_level=config.thinking_level,
    )
    if not config.ollama_fallback:
        return primary, config.model
    chain = FallbackLLM(primary, ollama(), call_budget, deadline,
                        primary_label=f"gemini:{config.model}", fallback_label=f"ollama:{config.ollama_model}")
    return chain, config.model


def verify_text(
    text: str,
    config: Config | None = None,
    llm: LLMClient | None = None,
    search: SearchProvider | None = None,
) -> Report:
    """Fact-check ``text`` and return a :class:`Report`.

    Raises ``InputError`` (invalid input), ``ConfigError`` (missing credentials
    or bad configuration) or ``RunFailure`` (run-level failure).
    """
    config = config or Config.from_env()
    text = validate_input(text, config)
    deadline = Deadline(config.run_timeout_s)
    call_budget = CallBudget(config.max_llm_calls)
    model_label = None
    if llm is None:
        llm, model_label = build_llm(config, deadline, call_budget)
    if search is None:
        key = os.environ.get(TAVILY_KEY_ENV, "").strip()
        if not key:
            raise ConfigError(f"{TAVILY_KEY_ENV} is not set")
        search = TavilyProvider(key, timeout_s=config.search_timeout_s)

    credits = CreditBudget(config.search_credit_budget)
    deps = Deps(
        llm=BudgetedLLM(llm, call_budget, deadline),
        retrieval=RetrievalService(search, credits, deadline, config),
        config=config,
        deadline=deadline,
        boundary=make_boundary(),
    )
    graph = build_graph(deps)
    started = time.monotonic()
    try:
        out = graph.invoke(
            {"text": text},
            config={"recursion_limit": config.recursion_limit, "max_concurrency": config.max_concurrency},
        )
    except (LLMAuthError, SearchAuthError) as exc:
        raise RunFailure(f"authentication/access error: {redact_secrets(str(exc))}") from None
    except GraphRecursionError:
        raise RunFailure("graph step limit reached") from None
    report: Report = out["report"]
    report.stats = RunStats(
        llm_calls=deps.llm.calls,
        search_credits_used=credits.used,
        search_credit_budget=credits.limit,
        elapsed_s=round(time.monotonic() - started, 2),
        model=model_label or type(llm).__name__,
    )
    if isinstance(llm, FallbackLLM):
        report.stats.fallback_model = llm.fallback_label
        report.stats.fallback_calls = llm.fallback_calls
        if llm.fallback_calls or llm.primary_down_reason:
            report.warnings.append(
                f"{llm.primary_label} was unavailable ({llm.primary_down_reason}); the local fallback model "
                f"{llm.fallback_label} handled {llm.fallback_calls} model call(s)")
    if deadline.expired():
        report.warnings.append(f"run time limit of {config.run_timeout_s:.0f}s was reached")
    return report
