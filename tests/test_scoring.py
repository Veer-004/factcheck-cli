from __future__ import annotations

import itertools
import random

import pytest

from factcheck.judging import EVIDENCE_THRESHOLD, select_verdict, summary_label
from factcheck.models import (
    Authority,
    Checkability,
    Citation,
    ClaimResult,
    ClaimType,
    Directness,
    Stance,
    Strengths,
    SummaryLabel,
    Verdict,
)
from factcheck.scoring import compute_strengths, passage_weight, round5, score_claim, side_strength


def cite(eid="E1", cluster="K1", authority=Authority.PRIMARY, directness=Directness.EXPLICIT, stance=Stance.SUPPORTS,
         snippet=False, injection=False, fact_check=False, recent=None, time_sensitive=False) -> Citation:
    c = Citation(evidence_id=eid, url=f"https://{cluster}.example/{eid}", quote="q", stance=stance,
                 directness=directness, authority=authority, cluster_id=cluster, snippet_only=snippet,
                 injection_flagged=injection, fact_check_site=fact_check, recent=recent)
    c.weight = passage_weight(c, time_sensitive)
    return c


@pytest.mark.parametrize(
    "kw,ts,expected",
    [
        ({}, False, 1.0),
        ({"authority": Authority.SECONDARY}, False, 0.75),
        ({"authority": Authority.OTHER}, False, 0.4),
        ({"directness": Directness.PARTIAL}, False, 0.6),
        ({"directness": Directness.INFERENTIAL}, False, 0.3),
        ({"recent": True}, True, 1.0),
        ({"recent": None}, True, 0.8),
        ({"recent": False}, True, 0.6),
        ({"recent": False}, False, 1.0),  # recency ignored when not time-sensitive
        ({"snippet": True}, False, 0.5),
        ({"injection": True}, False, 0.5),
        ({"snippet": True, "injection": True}, False, 0.25),
    ],
)
def test_passage_weight_table(kw, ts, expected):
    assert cite(time_sensitive=ts, **kw).weight == pytest.approx(expected)


def test_noisy_or_combines_independent_clusters_only():
    a = cite("E1", "K1", Authority.SECONDARY)  # 0.75
    b = cite("E2", "K2", Authority.OTHER)  # 0.4
    assert side_strength([a, b]) == pytest.approx(1 - 0.25 * 0.6)
    # Same cluster: max, not combination - repeating a source adds nothing.
    same = [cite(f"E{i}", "K1", Authority.SECONDARY) for i in range(5)]
    assert side_strength(same) == pytest.approx(0.75)
    assert side_strength([]) == 0.0


def test_round5():
    assert [round5(x) for x in (0, 2.4, 2.5, 7.49, 7.5, 97.5, 100)] == [0, 0, 5, 5, 10, 100, 100]


def test_supported_score_and_single_cluster_cap():
    sup = [cite("E1", "K1")]
    score, _, _, caps = score_claim(Verdict.SUPPORTED, sup, [], [], compute_strengths(sup, [], []), False)
    assert score == 70 and any("one independent source cluster" in c for c in caps)


def test_two_primary_clusters_uncapped():
    sup = [cite("E1", "K1"), cite("E2", "K2")]
    score, *_ = score_claim(Verdict.SUPPORTED, sup, [], [], compute_strengths(sup, [], []), False)
    assert score == 100


def test_no_primary_cap():
    sup = [cite("E1", "K1", Authority.SECONDARY), cite("E2", "K2", Authority.SECONDARY), cite("E3", "K3", Authority.SECONDARY)]
    score, _, _, caps = score_claim(Verdict.SUPPORTED, sup, [], [], compute_strengths(sup, [], []), False)
    assert score == 85 and any("no primary" in c for c in caps)


def test_fact_check_only_cap():
    sup = [cite(f"E{i}", f"K{i}", Authority.SECONDARY, fact_check=True) for i in range(4)]
    score, _, _, caps = score_claim(Verdict.SUPPORTED, sup, [], [], compute_strengths(sup, [], []), False)
    assert score == 60 and any("fact-check" in c for c in caps)


def test_injection_cap():
    sup = [cite("E1", "K1"), cite("E2", "K2"), cite("E3", "K3", injection=True)]
    score, _, _, caps = score_claim(Verdict.SUPPORTED, sup, [], [], compute_strengths(sup, [], []), False)
    assert score == 60 and any("injection" in c for c in caps)


def test_time_sensitive_without_recent_cap():
    sup = [cite("E1", "K1", recent=False, time_sensitive=True), cite("E2", "K2", recent=None, time_sensitive=True)]
    score, _, _, caps = score_claim(Verdict.SUPPORTED, sup, [], [], compute_strengths(sup, [], []), True)
    assert score == 60 and any("time-sensitive" in c for c in caps)
    sup_recent = [cite("E1", "K1", recent=True, time_sensitive=True), cite("E2", "K2", recent=True, time_sensitive=True)]
    score2, *_ = score_claim(Verdict.SUPPORTED, sup_recent, [], [], compute_strengths(sup_recent, [], []), True)
    assert score2 == 100


def test_opposing_evidence_penalises_score():
    sup = [cite("E1", "K1"), cite("E2", "K2")]
    ref = [cite("E3", "K3", Authority.OTHER, Directness.PARTIAL, Stance.REFUTES)]  # 0.24
    s_clean, *_ = score_claim(Verdict.SUPPORTED, sup, [], [], compute_strengths(sup, [], []), False)
    s_pen, *_ = score_claim(Verdict.SUPPORTED, sup, ref, [], compute_strengths(sup, ref, []), False)
    assert s_pen < s_clean and s_pen == round5(100 * 1.0 * (1 - 0.24))


def test_refuted_score_uses_refuting_side():
    ref = [cite("E1", "K1", stance=Stance.REFUTES), cite("E2", "K2", Authority.SECONDARY, stance=Stance.REFUTES)]
    score, *_ = score_claim(Verdict.REFUTED, [], ref, [], compute_strengths([], ref, []), False)
    assert score == 100


def test_misleading_capped_at_75():
    sup = [cite("E1", "K1"), cite("E2", "K2")]
    ctx = [cite("E3", "K3", stance=Stance.MISSING_CONTEXT), cite("E4", "K4", stance=Stance.MISSING_CONTEXT)]
    score, _, _, caps = score_claim(Verdict.MISLEADING, sup, [], ctx, compute_strengths(sup, [], ctx), False)
    assert score == 75 and any("misleading" in c for c in caps)


def test_conflicting_shows_both_sides_no_single_score():
    sup = [cite("E1", "K1")]
    ref = [cite("E2", "K2", Authority.SECONDARY, stance=Stance.REFUTES)]
    score, s, r, _ = score_claim(Verdict.CONFLICTING, sup, ref, [], compute_strengths(sup, ref, []), False)
    assert score is None and s == 70 and r == 70


def test_no_score_for_not_checkable_and_failures():
    for v in (Verdict.NOT_CHECKABLE, Verdict.RETRIEVAL_FAILED, Verdict.PROCESSING_FAILED):
        assert score_claim(v, [], [], [], Strengths(), False) == (None, None, None, [])


def test_insufficient_score_reflects_retrieved_evidence():
    sup = [cite("E1", "K1", Authority.OTHER)]
    score, *_ = score_claim(Verdict.INSUFFICIENT_EVIDENCE, sup, [], [], compute_strengths(sup, [], []), False)
    assert score == 40


def test_scores_bounded_and_deterministic_over_random_inputs():
    rng = random.Random(1234)
    options = list(itertools.product(Authority, Directness, [True, False], [True, False], [True, False, None]))
    for _ in range(300):
        cites = []
        for i in range(rng.randint(0, 8)):
            auth, direct, snip, inj, recent = rng.choice(options)
            stance = rng.choice([Stance.SUPPORTS, Stance.REFUTES, Stance.MISSING_CONTEXT])
            cites.append(cite(f"E{i}", f"K{rng.randint(1, 4)}", auth, direct, stance, snip, inj, recent=recent, time_sensitive=True))
        sup = [c for c in cites if c.stance == Stance.SUPPORTS]
        ref = [c for c in cites if c.stance == Stance.REFUTES]
        ctx = [c for c in cites if c.stance == Stance.MISSING_CONTEXT]
        st = compute_strengths(sup, ref, ctx)
        verdict, _ = select_verdict(st)
        first = score_claim(verdict, sup, ref, ctx, st, True)
        assert first == score_claim(verdict, sup, ref, ctx, st, True)
        for value in first[:3]:
            assert value is None or (0 <= value <= 100 and value % 5 == 0)


@pytest.mark.parametrize(
    "s,r,m,expected",
    [
        (0.9, 0.9, 0.0, Verdict.CONFLICTING),
        (0.2, 0.75, 0.0, Verdict.REFUTED),
        (0.75, 0.2, 0.6, Verdict.MISLEADING),
        (0.75, 0.2, 0.3, Verdict.SUPPORTED),
        (0.4, 0.4, 0.9, Verdict.INSUFFICIENT_EVIDENCE),
        (EVIDENCE_THRESHOLD, 0.0, 0.0, Verdict.SUPPORTED),
    ],
)
def test_select_verdict_rules(s, r, m, expected):
    assert select_verdict(Strengths(support=s, refute=r, context=m))[0] == expected


def _r(v: Verdict) -> ClaimResult:
    return ClaimResult(claim_id="C1", index=1, statement="s", source_excerpt="s", claim_type=ClaimType.FACTUAL,
                       checkability=Checkability.CHECKABLE, verdict=v)


@pytest.mark.parametrize(
    "verdicts,label",
    [
        ([Verdict.REFUTED, Verdict.MISLEADING, Verdict.PROCESSING_FAILED], SummaryLabel.CONTAINS_FALSE_CLAIMS),
        ([Verdict.MISLEADING, Verdict.SUPPORTED], SummaryLabel.CONTAINS_MISLEADING_CLAIMS),
        ([Verdict.SUPPORTED, Verdict.NOT_CHECKABLE], SummaryLabel.ALL_CHECKED_CLAIMS_SUPPORTED),
        ([Verdict.SUPPORTED, Verdict.CONFLICTING], SummaryLabel.PARTIALLY_RESOLVED),
        ([Verdict.RETRIEVAL_FAILED], SummaryLabel.PARTIALLY_RESOLVED),
        ([Verdict.NOT_CHECKABLE], SummaryLabel.NOTHING_TO_CHECK),
        ([], SummaryLabel.NOTHING_TO_CHECK),
    ],
)
def test_summary_labels(verdicts, label):
    assert summary_label([_r(v) for v in verdicts]) == label
