from __future__ import annotations

import pytest
from fakes import claim_dict, ext, judgment

from factcheck.config import Config
from factcheck.models import (
    Authority,
    Checkability,
    Claim,
    ClaimType,
    Directness,
    EvidencePassage,
    LLMCorrection,
    Verdict,
)
from factcheck.textnorm import extract_numbers, normalize_for_match, numbers_compatible
from factcheck.validation import (
    InputError,
    claim_figures,
    validate_claims,
    validate_correction,
    validate_input,
    validate_judgment,
)

# --------------------------------------------------------------------------- input


def test_validate_input_rejects_empty_and_too_long(config):
    with pytest.raises(InputError):
        validate_input("   ", config)
    with pytest.raises(InputError):
        validate_input("a" * (config.max_input_chars + 1), config)
    with pytest.raises(InputError):
        validate_input("12345 !!!", config)
    with pytest.raises(InputError):
        validate_input(None, config)  # type: ignore[arg-type]
    assert validate_input("  Paris is in France.  ", config) == "Paris is in France."


# --------------------------------------------------------------------------- normalisation


def test_normalization_is_documented_and_symmetric():
    assert normalize_for_match("The “Iron Lady” — Paris’s  tower") == normalize_for_match(
        'the "iron lady" - paris\'s tower'
    )
    assert normalize_for_match("zero​width") == "zerowidth"


@pytest.mark.parametrize(
    "text,values",
    [
        ("It cost 7.8 million francs in 1889.", [7.8e6, 1889.0]),
        ("about 45% of voters", [45.0]),
        ("3,245,000 residents", [3245000.0]),
        ("COVID19 cases", []),
    ],
)
def test_extract_numbers(text, values):
    assert extract_numbers(text) == values


def test_numbers_compatible_rounding_and_approximation():
    assert numbers_compatible(3.2e6, 3_245_000)  # rounding of "3.2 million"
    assert not numbers_compatible(1000, 1450)  # plain integers are exact
    assert not numbers_compatible(1889, 1887)
    assert numbers_compatible(300, 330, approximate=True)
    assert not numbers_compatible(300, 400, approximate=True)


def test_claim_figures_qualifiers():
    figs = claim_figures("More than 3 million people and about 45% of voters, under 10 cities, exactly 7 days")
    assert [q for _, q in figs] == ["min", "approx", "max", "exact"]


# --------------------------------------------------------------------------- claims

TEXT = (
    "The Eiffel Tower was completed in 1889. It is 330 metres tall. "
    "Paris is the most beautiful city in the world. Freedonia has 3.2 million residents."
)


def test_validate_claims_anchors_grounds_dedupes_and_classifies(config):
    raw = ext(
        claim_dict("The Eiffel Tower was completed in 1889.", queries=["a", "b", "c"]),
        claim_dict("The Eiffel Tower was completed in 1889", "Eiffel Tower was completed in 1889"),  # near-duplicate
        claim_dict("The Eiffel Tower is 330 metres tall.", "It is 330 metres tall."),
        claim_dict("Paris is the most beautiful city in the world.", checkability="OPINION"),
        claim_dict("The Eiffel Tower was moved to Lyon in 1950.", "The Eiffel Tower was moved to Lyon"),  # invented excerpt
        claim_dict("The Eiffel Tower is 500 metres tall.", "It is 330 metres tall."),  # invented figure
        claim_dict("Bananas are a good source of potassium.", "Paris is the most beautiful"),  # unrelated statement
        claim_dict("Freedonia has 3.2 million residents.", claim_type="NUMERIC"),
    )
    cv = validate_claims(raw, TEXT, config)
    assert [c.claim_id for c in cv.checkable] == ["C1", "C2", "C4"]
    assert [c.claim_id for c in cv.not_checkable] == ["C3"]
    reasons = " | ".join(d.reason for d in cv.dropped)
    assert "near-duplicate of C1" in reasons
    assert "does not occur in the input" in reasons
    assert "figures that are not in the input" in reasons
    assert "not grounded" in reasons
    assert len(cv.dropped) == 4
    assert cv.checkable[0].primary_queries == ["a", "b"]  # capped at 2 queries


def test_validate_claims_over_limit(config):
    cfg = config.model_copy(update={"max_checkable_claims": 2})
    text = "Alpha is one. Beta is two. Gamma is three."
    raw = ext(*(claim_dict(s) for s in ["Alpha is one.", "Beta is two.", "Gamma is three."]))
    cv = validate_claims(raw, text, cfg)
    assert len(cv.checkable) == 2 and len(cv.over_limit) == 1
    assert "over the limit" in cv.over_limit[0].reason


def test_validate_claims_tolerates_typographic_differences(config):
    text = "The “Iron Lady” — Paris’s tower — opened in 1889."
    raw = ext(claim_dict("The Iron Lady opened in 1889.", 'The "Iron Lady" - Paris\'s tower - opened in 1889.'))
    assert len(validate_claims(raw, text, config).checkable) == 1


# --------------------------------------------------------------------------- judgments


def claim(statement="The Eiffel Tower was completed in 1889.", ctype=ClaimType.FACTUAL) -> Claim:
    return Claim(claim_id="C1", index=1, statement=statement, source_excerpt=statement,
                 checkability=Checkability.CHECKABLE, claim_type=ctype)


def passage(eid: str, text: str, url="https://a.example.gov/x", cluster="C1-K1", **kw) -> EvidencePassage:
    return EvidencePassage(evidence_id=eid, url=url, domain="a.example.gov", text=text,
                           authority=kw.pop("authority", Authority.PRIMARY), cluster_id=cluster, **kw)


P1 = passage("C1-E1", "The Eiffel Tower was completed in 1889 for the World's Fair. It is in Paris.")
P2 = passage("C1-E2", "It is a popular myth that the Eiffel Tower was completed in 1889 by Napoleon.", url="https://b.example/y", cluster="C1-K2", authority=Authority.OTHER)


def a(eid, stance, quote, directness="EXPLICIT", **kw):
    return {"evidence_id": eid, "stance": stance, "directness": directness, "quote": quote, **kw}


def test_valid_supporting_quote_with_normalisation(config):
    j = judgment([a("C1-E1", "SUPPORTS", "  “The Eiffel Tower was completed   in 1889 for the World’s Fair.” ")])
    v = validate_judgment(j, claim(), [P1], config)
    assert v.ok and len(v.supporting) == 1 and not v.rejected
    assert v.supporting[0].weight == 1.0


def test_fabricated_quote_and_unknown_id_are_rejected_and_fatal(config):
    j = judgment([
        a("C1-E1", "SUPPORTS", "The Eiffel Tower was completed in 1889 by royal decree."),
        a("C1-E9", "SUPPORTS", "The Eiffel Tower was completed in 1889."),
    ])
    v = validate_judgment(j, claim(), [P1], config)
    assert not v.supporting
    assert any("verbatim" in r for r in v.rejected) and any("unknown evidence ID" in r for r in v.rejected)
    assert not v.ok  # most citations fabricated -> repair needed


def test_quote_too_short_rejected(config):
    j = judgment([a("C1-E1", "SUPPORTS", "1889"), a("C1-E1", "IRRELEVANT", "x")])
    v = validate_judgment(j, claim(), [P1], config)
    assert not v.supporting and "too short" in v.rejected[0]


def test_empty_assessments_are_fatal(config):
    v = validate_judgment(judgment([]), claim(), [P1], config)
    assert not v.ok


def test_debunking_quote_cannot_be_used_as_support(config):
    j = judgment([a("C1-E2", "SUPPORTS", "It is a popular myth that the Eiffel Tower was completed in 1889 by Napoleon.")])
    v = validate_judgment(j, claim(), [P2], config)
    assert not v.supporting and "debunking" in v.rejected[0]


def test_off_topic_passage_rejected(config):
    off = passage("C1-E1", "Bananas grew in 1889 in large plantations across the region, says the report.")
    j = judgment([a("C1-E1", "SUPPORTS", "Bananas grew in 1889 in large plantations across the region")])
    v = validate_judgment(j, claim(), [off], config)
    assert not v.supporting and "not about the claim" in v.rejected[0]


def test_explicit_directness_downgraded_when_quote_barely_overlaps(config):
    c = claim("The Eiffel Tower in Paris was completed in 1889 for the World's Fair exhibition.")
    p = passage("C1-E1", "The Eiffel Tower in Paris hosted the World's Fair exhibition. Work was completed in 1889 after two years.")
    j = judgment([a("C1-E1", "SUPPORTS", "Work was completed in 1889 after two years.")])
    v = validate_judgment(j, c, [p], config)
    assert v.supporting and v.supporting[0].directness == Directness.PARTIAL


NUM_CLAIM = claim("Freedonia has a population of 3.2 million.", ClaimType.NUMERIC)
NUM_P = passage("C1-E1", "The resident population of Freedonia was 3,245,000 on 30 June 2025, according to the statistics office.")
NUM_P2 = passage("C1-E2", "The 2020 census counted 2,950,000 residents in Freedonia, the statistics office said.", cluster="C1-K2")


def test_numeric_support_requires_figure_in_quote(config):
    ok = judgment([a("C1-E1", "SUPPORTS", "The resident population of Freedonia was 3,245,000 on 30 June 2025")])
    assert validate_judgment(ok, NUM_CLAIM, [NUM_P], config).supporting
    bad = judgment([a("C1-E1", "SUPPORTS", "The resident population of Freedonia was")])
    v = validate_judgment(bad, NUM_CLAIM, [NUM_P], config)
    assert not v.supporting and "claimed figure" in v.rejected[0]


def _nc(**kw):
    base = dict(claim_value=3.2e6, evidence_value=2.95e6, same_entity=True, same_metric=True, same_unit=True,
                same_scope=True, same_period=True)
    base.update(kw)
    return base


def test_numeric_refutation_rejected_when_measurements_differ(config):
    j = judgment([a("C1-E2", "REFUTES", "The 2020 census counted 2,950,000 residents in Freedonia",
                    numeric_comparison=_nc(same_period=False))])
    v = validate_judgment(j, NUM_CLAIM, [NUM_P2], config)
    assert not v.refuting and "measure different things" in v.rejected[0]


def test_numeric_refutation_rejected_without_comparison(config):
    j = judgment([a("C1-E2", "REFUTES", "The 2020 census counted 2,950,000 residents in Freedonia")])
    v = validate_judgment(j, NUM_CLAIM, [NUM_P2], config)
    assert not v.refuting and "lacks an explicit figure comparison" in v.rejected[0]


def test_numeric_refutation_rejected_when_figures_agree_within_rounding(config):
    j = judgment([a("C1-E1", "REFUTES", "The resident population of Freedonia was 3,245,000 on 30 June 2025",
                    numeric_comparison=_nc(evidence_value=3_245_000))])
    v = validate_judgment(j, NUM_CLAIM, [NUM_P], config)
    assert not v.refuting and "agree within rounding" in v.rejected[0]


def test_numeric_refutation_rejected_when_evidence_value_not_in_quote(config):
    j = judgment([a("C1-E2", "REFUTES", "The 2020 census counted 2,950,000 residents in Freedonia",
                    numeric_comparison=_nc(evidence_value=1.5e6))])
    v = validate_judgment(j, NUM_CLAIM, [NUM_P2], config)
    assert not v.refuting and "not in the quote" in v.rejected[0]


def test_numeric_safeguards_apply_even_if_claim_mislabelled(config):
    mislabelled = claim("Freedonia has a population of 3.2 million.", ClaimType.FACTUAL)
    j = judgment([a("C1-E2", "REFUTES", "The 2020 census counted 2,950,000 residents in Freedonia")])
    v = validate_judgment(j, mislabelled, [NUM_P2], config)
    assert not v.refuting and "lacks an explicit figure comparison" in v.rejected[0]


def test_year_refutation_needs_no_numeric_comparison(config):
    c = claim("The Eiffel Tower was completed in 1899.")
    j = judgment([a("C1-E1", "REFUTES", "The Eiffel Tower was completed in 1889 for the World's Fair.")])
    assert validate_judgment(j, c, [P1], config).refuting


def test_valid_numeric_refutation_accepted(config):
    c = claim("Freedonia had a population of 5 million in 2020.", ClaimType.NUMERIC)
    j = judgment([a("C1-E2", "REFUTES", "The 2020 census counted 2,950,000 residents in Freedonia",
                    numeric_comparison=_nc(claim_value=5e6))])
    v = validate_judgment(j, c, [NUM_P2], config)
    assert len(v.refuting) == 1, v.rejected


def test_time_sensitive_recency_recorded(config):
    c = Claim(claim_id="C1", index=1, statement=NUM_CLAIM.statement, source_excerpt=NUM_CLAIM.statement,
              checkability=Checkability.CHECKABLE, claim_type=ClaimType.NUMERIC, time_sensitive=True)
    p = NUM_P.model_copy(update={"published_date": "2020-01-01"})
    j = judgment([a("C1-E1", "SUPPORTS", "The resident population of Freedonia was 3,245,000 on 30 June 2025")])
    v = validate_judgment(j, c, [p], config)
    assert v.supporting[0].recent is False and v.supporting[0].weight == pytest.approx(0.6)


# --------------------------------------------------------------------------- corrections

REF_CLAIM = claim("The Eiffel Tower was completed in 1899.")
REF_P = passage("C1-E1", "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.")


def _refuting_validation(config):
    j = judgment([a("C1-E1", "REFUTES", "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.")], "REFUTED")
    v = validate_judgment(j, REF_CLAIM, [REF_P], config)
    assert v.refuting
    return v


def test_grounded_correction_accepted(config):
    v = _refuting_validation(config)
    text, note = validate_correction(LLMCorrection(text="The Eiffel Tower was completed in 1889, not 1899.",
                                                   evidence_ids=["C1-E1"]), REF_CLAIM, Verdict.REFUTED, v, [REF_P])
    assert text and "C1-E1" in note


def test_correction_with_ungrounded_figure_rejected(config):
    v = _refuting_validation(config)
    text, note = validate_correction(LLMCorrection(text="The tower was completed in 1887.", evidence_ids=["C1-E1"]),
                                     REF_CLAIM, Verdict.REFUTED, v, [REF_P])
    assert text is None and "ungrounded figure" in note


def test_correction_with_invented_details_rejected(config):
    v = _refuting_validation(config)
    text, note = validate_correction(
        LLMCorrection(text="Gustave Eiffel personally welded every rivet during a secret ceremony.", evidence_ids=["C1-E1"]),
        REF_CLAIM, Verdict.REFUTED, v, [REF_P])
    assert text is None and "not found in the cited evidence" in note


def test_correction_must_cite_validated_evidence(config):
    v = _refuting_validation(config)
    text, note = validate_correction(LLMCorrection(text="The Eiffel Tower was completed in 1889.", evidence_ids=["C1-E7"]),
                                     REF_CLAIM, Verdict.REFUTED, v, [REF_P])
    assert text is None and "did not cite validated" in note


def test_no_correction_for_supported_or_missing(config):
    v = _refuting_validation(config)
    assert validate_correction(LLMCorrection(text="x", evidence_ids=["C1-E1"]), REF_CLAIM, Verdict.SUPPORTED, v, [REF_P]) == (None, None)
    text, note = validate_correction(None, REF_CLAIM, Verdict.REFUTED, v, [REF_P])
    assert text is None and note.startswith("No correction could be established")
