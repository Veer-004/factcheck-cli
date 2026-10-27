"""Deterministic verdict and summary rules.

The judge model only *proposes* stances for individual passages and a verdict.
The final verdict is computed here from separately calculated support,
refutation and material-context strengths of validated citations.
"""

from __future__ import annotations

from collections.abc import Iterable

from .models import ClaimResult, Strengths, SummaryLabel, Verdict

# A side "meets the evidence threshold" when its noisy-OR strength reaches this
# value. One explicit established-secondary passage (0.75) or one partial
# primary passage (0.6) meets it; a single "other" source (0.4) or any single
# snippet from a secondary source (0.375) does not.
EVIDENCE_THRESHOLD = 0.5


def select_verdict(strengths: Strengths, threshold: float = EVIDENCE_THRESHOLD) -> tuple[Verdict, str]:
    S, R, M = strengths.support, strengths.refute, strengths.context
    meets = lambda x: x >= threshold  # noqa: E731
    if meets(S) and meets(R):
        return Verdict.CONFLICTING, "credible evidence on both sides meets the evidence threshold"
    if meets(R):
        return Verdict.REFUTED, "sufficient evidence directly contradicts the claim"
    if meets(S) and meets(M):
        return Verdict.MISLEADING, "the claim has support but omits or distorts material context"
    if meets(S):
        return Verdict.SUPPORTED, "sufficient evidence supports the claim"
    return Verdict.INSUFFICIENT_EVIDENCE, "retrieved evidence is not strong enough for a reliable verdict"


def summary_label(results: Iterable[ClaimResult]) -> SummaryLabel:
    results = list(results)
    verdicts = [r.verdict for r in results]
    checkable = [v for v in verdicts if v != Verdict.NOT_CHECKABLE]
    if Verdict.REFUTED in verdicts:
        return SummaryLabel.CONTAINS_FALSE_CLAIMS
    if Verdict.MISLEADING in verdicts:
        return SummaryLabel.CONTAINS_MISLEADING_CLAIMS
    if not checkable:
        return SummaryLabel.NOTHING_TO_CHECK
    if all(v == Verdict.SUPPORTED for v in checkable):
        return SummaryLabel.ALL_CHECKED_CLAIMS_SUPPORTED
    return SummaryLabel.PARTIALLY_RESOLVED


SUMMARY_TEXT = {
    SummaryLabel.CONTAINS_FALSE_CLAIMS: "At least one claim is contradicted by the retrieved evidence.",
    SummaryLabel.CONTAINS_MISLEADING_CLAIMS: "No claim was refuted, but at least one omits or distorts material context.",
    SummaryLabel.ALL_CHECKED_CLAIMS_SUPPORTED: "Every checked claim is supported by the retrieved evidence.",
    SummaryLabel.PARTIALLY_RESOLVED: "Some claims could not be resolved (conflicting or insufficient evidence, or a technical failure).",
    SummaryLabel.NOTHING_TO_CHECK: "The text contains no checkable factual claims.",
}
