"""Deterministic, experimental evidence-strength scoring (0-100).

The score summarises characteristics of *validated* evidence (authority,
directness, recency, independence). It is not a probability, is not calibrated,
and never uses the judge model's self-reported confidence.

Passage weight  w = authority x directness x recency x penalties
  authority:  primary 1.0, established secondary 0.75, other 0.4
  directness: explicit 1.0, partial 0.6, inferential 0.3
  recency (time-sensitive claims only): <=1 year 1.0, undated 0.8, older 0.6
  penalties:  snippet-only x0.5, injection-flagged x0.5 (applied independently,
              so a snippet that is also flagged gets x0.25 - the conservative reading)

Side strength   S = 1 - prod(1 - cluster_weight) over independent clusters,
                where cluster_weight is the max passage weight in that cluster.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from .models import Authority, Citation, Directness, Strengths, Verdict

AUTHORITY_WEIGHT = {Authority.PRIMARY: 1.0, Authority.SECONDARY: 0.75, Authority.OTHER: 0.4}
DIRECTNESS_WEIGHT = {Directness.EXPLICIT: 1.0, Directness.PARTIAL: 0.6, Directness.INFERENTIAL: 0.3}
RECENCY_WEIGHT = {True: 1.0, None: 0.8, False: 0.6}
SNIPPET_PENALTY = 0.5
INJECTION_PENALTY = 0.5

CAP_SINGLE_CLUSTER = 70
CAP_NO_PRIMARY = 85
CAP_ONLY_FACT_CHECK = 60
CAP_INJECTION = 60
CAP_NOT_RECENT = 60
CAP_MISLEADING = 75


def passage_weight(c: Citation, time_sensitive: bool) -> float:
    w = AUTHORITY_WEIGHT[c.authority] * DIRECTNESS_WEIGHT[c.directness]
    if time_sensitive:
        w *= RECENCY_WEIGHT[c.recent]
    if c.snippet_only:
        w *= SNIPPET_PENALTY
    if c.injection_flagged:
        w *= INJECTION_PENALTY
    return round(w, 6)


def side_strength(citations: Iterable[Citation]) -> float:
    """Noisy-OR over independent clusters, each represented by its strongest passage."""
    best: dict[str, float] = {}
    for c in citations:
        best[c.cluster_id] = max(best.get(c.cluster_id, 0.0), c.weight)
    product = 1.0
    for w in best.values():
        product *= 1.0 - min(max(w, 0.0), 1.0)
    return round(1.0 - product, 6)


def compute_strengths(supporting: list[Citation], refuting: list[Citation], context: list[Citation]) -> Strengths:
    return Strengths(
        support=side_strength(supporting),
        refute=side_strength(refuting),
        context=side_strength(context),
    )


def round5(x: float) -> int:
    return int(math.floor(x / 5.0 + 0.5)) * 5


def _caps(contributing: list[Citation], time_sensitive: bool) -> list[tuple[int, str]]:
    caps: list[tuple[int, str]] = []
    contributing = [c for c in contributing if c.weight > 0]
    if not contributing:
        return caps
    if len({c.cluster_id for c in contributing}) <= 1:
        caps.append((CAP_SINGLE_CLUSTER, "only one independent source cluster"))
    if not any(c.authority == Authority.PRIMARY for c in contributing):
        caps.append((CAP_NO_PRIMARY, "no primary-authority source"))
    if all(c.fact_check_site for c in contributing):
        caps.append((CAP_ONLY_FACT_CHECK, "only fact-check sites"))
    if any(c.injection_flagged for c in contributing):
        caps.append((CAP_INJECTION, "a contributing passage was flagged for possible prompt injection"))
    if time_sensitive and not any(c.recent for c in contributing):
        caps.append((CAP_NOT_RECENT, "time-sensitive claim with no passage dated within one year"))
    return caps


def _apply(base: float, caps: list[tuple[int, str]]) -> tuple[int, list[str]]:
    score = min(100, max(0, round5(100.0 * base)))
    applied = []
    for cap, reason in sorted(caps):
        if score > cap:
            score = cap
            applied.append(f"capped at {cap}: {reason}")
    return score, applied


def score_claim(
    verdict: Verdict,
    supporting: list[Citation],
    refuting: list[Citation],
    context: list[Citation],
    strengths: Strengths,
    time_sensitive: bool,
) -> tuple[int | None, int | None, int | None, list[str]]:
    """Return (score, support_score, refute_score, cap_notes).

    * SUPPORTED / REFUTED / MISLEADING / INSUFFICIENT_EVIDENCE -> single score.
    * CONFLICTING -> no single score; each side's strength is scored separately.
    * NOT_CHECKABLE and technical failures -> no score.
    """
    S, R, M = strengths.support, strengths.refute, strengths.context
    if verdict == Verdict.SUPPORTED:
        score, notes = _apply(S * (1.0 - R), _caps(supporting, time_sensitive))
        return score, None, None, notes
    if verdict == Verdict.REFUTED:
        score, notes = _apply(R * (1.0 - S), _caps(refuting, time_sensitive))
        return score, None, None, notes
    if verdict == Verdict.MISLEADING:
        caps = _caps(supporting + context, time_sensitive)
        caps.append((CAP_MISLEADING, "misleading verdict"))
        score, notes = _apply(math.sqrt(S * M) * (1.0 - R), caps)
        return score, None, None, notes
    if verdict == Verdict.CONFLICTING:
        s_score, s_notes = _apply(S, _caps(supporting, time_sensitive))
        r_score, r_notes = _apply(R, _caps(refuting, time_sensitive))
        notes = [f"support side {n}" for n in s_notes] + [f"refute side {n}" for n in r_notes]
        return None, s_score, r_score, notes
    if verdict == Verdict.INSUFFICIENT_EVIDENCE:
        score, notes = _apply(max(S, R, M), _caps(supporting + refuting + context, time_sensitive))
        return score, None, None, notes
    return None, None, None, []
