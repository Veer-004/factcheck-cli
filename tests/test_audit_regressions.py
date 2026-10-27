"""Regression tests from the evidence/scoring/freshness/citation audit (2026-10-27).

Deterministic fixtures only; these establish pipeline behaviour, not real-world accuracy.
"""

from __future__ import annotations

from datetime import date

import pytest
from conftest import load_page
from fakes import FakeSearch, FakeLLM, claim_dict, extraction, result, rule_judge

from factcheck.config import Config
from factcheck.evidence import build_evidence, normalize_url
from factcheck.graph import verify_text
from factcheck.judging import select_verdict
from factcheck.models import (
    Authority,
    Checkability,
    Claim,
    ClaimType,
    EvidencePassage,
    LLMCorrection,
    LLMJudgment,
    SearchResult,
    Verdict,
)
from factcheck.prompts import JUDGE_SYSTEM
from factcheck.retrieval import TavilyProvider
from factcheck.scoring import compute_strengths, score_claim
from factcheck.textnorm import original_span
from factcheck.validation import validate_correction, validate_judgment

AS_OF = date(2026, 10, 27)
CFG = Config(as_of=AS_OF)
TEXT = ("The Eiffel Tower was completed in 1889 for the World's Fair in Paris. "
        "It was designed by the company of Gustave Eiffel and stands on the Champ de Mars. " * 3)


def claim(statement="The Eiffel Tower was completed in 1889.", **kw) -> Claim:
    base = dict(claim_id="C1", index=1, statement=statement, source_excerpt=statement,
                checkability=Checkability.CHECKABLE, claim_type=ClaimType.FACTUAL,
                primary_queries=["Eiffel Tower completed 1889"])
    base.update(kw)
    return Claim(**base)


def supports_all(passages, quote, c=None):
    assess = [{"evidence_id": p.evidence_id, "stance": "SUPPORTS", "directness": "EXPLICIT", "quote": quote}
              for p in passages]
    return validate_judgment(LLMJudgment.model_validate({"assessments": assess, "proposed_verdict": "SUPPORTED"}),
                             c or claim(), passages, CFG)


# --------------------------------------------------------------------------- 1. URL canonicalisation


SAME_PAGE = [
    "https://en.wikipedia.org/wiki/Eiffel_Tower",
    "http://en.wikipedia.org/wiki/Eiffel_Tower",
    "https://en.m.wikipedia.org/wiki/Eiffel_Tower",
    "https://EN.Wikipedia.org/wiki/Eiffel_Tower/",
    "https://en.wikipedia.org/wiki/Eiffel_Tower?utm_source=newsletter#History",
    "https://www.en.wikipedia.org/wiki/Eiffel_Tower",
]


def test_same_page_variants_share_one_canonical_key():
    assert len({normalize_url(u) for u in SAME_PAGE}) == 1
    assert normalize_url("https://amp.theguardian.com/x") == normalize_url("https://theguardian.com/x")


def test_canonicalisation_keeps_distinct_pages_and_registrable_domains():
    assert normalize_url("https://en.wikipedia.org/wiki/A") != normalize_url("https://en.wikipedia.org/wiki/B")
    assert normalize_url("https://m.com/page") == "https://m.com/page"  # 'm' is the registrable label here
    assert normalize_url("https://news.m.com/page") == "https://news.m.com/page"
    assert normalize_url("https://site.example/?id=2") != normalize_url("https://site.example/?id=3")


def test_duplicate_url_variants_become_one_source_and_one_passage():
    results = [SearchResult(url=u, title="Eiffel Tower - Wikipedia", content=TEXT) for u in SAME_PAGE]
    bundle = build_evidence(claim(), results, {}, CFG)
    assert bundle.sources_considered == 1 and len(bundle.passages) == 1


# --------------------------------------------------------------------------- 2. duplicates cannot inflate strength


def _passage(eid, url, cluster, text=TEXT, authority=Authority.SECONDARY):
    return EvidencePassage(evidence_id=eid, url=url, domain="x", text=text, authority=authority, cluster_id=cluster)


QUOTE = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."


def test_repeated_passages_from_one_source_do_not_add_strength_or_independence():
    one = supports_all([_passage("C1-E1", "https://a.example/p", "C1-K1")], QUOTE)
    many = supports_all([_passage(f"C1-E{i}", "https://a.example/p", "C1-K1") for i in range(1, 6)], QUOTE)
    s1 = compute_strengths(one.supporting, [], [])
    s5 = compute_strengths(many.supporting, [], [])
    assert len(many.supporting) == 5 and s5.support == s1.support == pytest.approx(0.75)
    score, _, _, caps = score_claim(Verdict.SUPPORTED, many.supporting, [], [], s5, False)
    assert score == 70 and any("one independent source" in c for c in caps)


def test_syndicated_snippet_of_a_page_is_not_an_independent_source():
    article = (
        "The city council approved the new tram line in March 2026 after a two-year consultation. "
        "Officials said construction will begin next spring and the line will connect the station to the university. "
        "The council vote was 31 to 9, with opposition members citing cost overruns on previous transport projects. "
    ) * 4
    snippet = ("The city council approved the new tram line in March 2026 after a two-year consultation. "
               "Officials said construction will begin next spring and the line will connect the station to the university.")
    c = claim("The city council approved the new tram line in March 2026.", primary_queries=["tram line approved"])
    wire, mirror = "https://wire.example/tram", "https://aggregator.example/tram"
    bundle = build_evidence(c, [SearchResult(url=wire, title="T", content=snippet),
                                SearchResult(url=mirror, title="T", content=snippet)],
                            {normalize_url(wire): article}, CFG)
    assert len({p.cluster_id for p in bundle.passages}) == 1
    v = supports_all(bundle.passages, "The city council approved the new tram line in March 2026 after a two-year consultation.", c)
    st = compute_strengths(v.supporting, [], [])
    assert st.support == pytest.approx(0.4)  # one "other" source, not two independent ones
    assert select_verdict(st)[0] == Verdict.INSUFFICIENT_EVIDENCE  # 0.52 would have crossed the 0.5 threshold


def test_independent_pages_sharing_one_sentence_stay_independent():
    shared = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris. "
    a = shared + ("Engineers studied the wrought iron lattice and its wind resistance over many decades of tests. " * 5)
    b = shared + ("Tourists queue for hours in summer to reach the summit viewing platforms and restaurants. " * 5)
    bundle = build_evidence(claim(), [SearchResult(url="https://one.example/a", title="A", content="x"),
                                      SearchResult(url="https://two.example/b", title="B", content="y")],
                            {normalize_url("https://one.example/a"): a, normalize_url("https://two.example/b"): b}, CFG)
    assert len({p.cluster_id for p in bundle.passages}) == 2


def test_duplicate_urls_through_the_full_graph_count_once(config):
    text = "The Eiffel Tower was completed in 1889."
    corpus = {"eiffel": [result(u, "Eiffel Tower - Wikipedia", TEXT) for u in SAME_PAGE[:4]]}
    llm = FakeLLM(extraction(claim_dict(text, queries=["Eiffel 1889"])),
                  rule_judge({"wikipedia": ("SUPPORTS", "EXPLICIT", "completed in 1889")}))
    [r] = verify_text(text, config, llm=llm, search=FakeSearch(corpus)).claims
    assert r.sources_considered == 1 and len({c.url for c in r.supporting}) == 1
    assert r.verdict == Verdict.INSUFFICIENT_EVIDENCE  # one snippet-only secondary source stays below threshold


# --------------------------------------------------------------------------- 3. freshness


def test_tavily_search_requests_publication_dates():
    class Fake:
        kwargs = {}

        def search(self, **kw):
            Fake.kwargs = kw
            return {"results": [{"url": "https://pmindia.gov.in/en/", "title": "PM", "content": "c",
                                 "published_date": "2026-10-01T08:00:00Z"}]}

    provider = TavilyProvider("tvly-test-not-real")
    provider._client = Fake()
    [res] = provider.search("current prime minister of India", 1)
    assert Fake.kwargs["include_published_date"] is True
    assert res.published_date == "2026-10-01T08:00:00Z"


PM_TEXT = ("Shri Narendra Modi is the Prime Minister of India. He was sworn in for a third term in June 2024. "
           "The Prime Minister's Office supports the Prime Minister in the discharge of official duties. ")


def _pm(published):
    c = claim("Narendra Modi is the current Prime Minister of India.", time_sensitive=True,
              primary_queries=["current Prime Minister of India"])
    p = EvidencePassage(evidence_id="C1-E1", url="https://pmindia.gov.in/en/", domain="pmindia.gov.in", text=PM_TEXT,
                        authority=Authority.PRIMARY, cluster_id="C1-K1", published_date=published)
    q = EvidencePassage(evidence_id="C1-E2", url="https://www.mea.gov.in/x", domain="mea.gov.in", text=PM_TEXT,
                        authority=Authority.PRIMARY, cluster_id="C1-K2", published_date=published)
    v = supports_all([p, q], "Shri Narendra Modi is the Prime Minister of India.", c)
    st = compute_strengths(v.supporting, [], [])
    return v, score_claim(Verdict.SUPPORTED, v.supporting, [], [], st, True)


def test_freshness_policy_undated_official_pages_are_capped():
    """Policy, not a defect: an undated page cannot show it is current, so the cap applies."""
    v, (score, _, _, caps) = _pm(None)
    assert all(c.recent is None and c.weight == pytest.approx(0.8) for c in v.supporting)
    assert score == 60 and any("no passage dated within one year" in c for c in caps)


def test_freshness_policy_recent_dated_official_pages_are_not_capped():
    v, (score, _, _, caps) = _pm("2026-09-15")
    assert all(c.recent is True and c.weight == 1.0 for c in v.supporting)
    assert score == 100 and not caps


def test_freshness_policy_stale_pages_are_weighted_down_and_capped():
    v, (score, _, _, caps) = _pm("2019-05-30")
    assert all(c.recent is False and c.weight == pytest.approx(0.6) for c in v.supporting)
    assert score == 60 and caps


# --------------------------------------------------------------------------- 4. citations and corrections


def test_rejected_citations_never_influence_strength_or_verdict():
    good = _passage("C1-E1", "https://a.example/p", "C1-K1", authority=Authority.OTHER)
    strong = _passage("C1-E2", "https://b.gov/p", "C1-K2", authority=Authority.PRIMARY)
    raw = LLMJudgment.model_validate({"proposed_verdict": "SUPPORTED", "assessments": [
        {"evidence_id": "C1-E1", "stance": "SUPPORTS", "directness": "EXPLICIT", "quote": QUOTE},
        # fabricated quote on the strongest source, and an unknown ID: both must be ignored
        {"evidence_id": "C1-E2", "stance": "SUPPORTS", "directness": "EXPLICIT",
         "quote": "Official records prove the Eiffel Tower was completed in 1889."},
        {"evidence_id": "C1-E9", "stance": "SUPPORTS", "directness": "EXPLICIT", "quote": QUOTE},
    ]})
    v = validate_judgment(raw, claim(), [good, strong], CFG)
    assert [c.evidence_id for c in v.supporting] == ["C1-E1"] and len(v.rejected) == 2
    st = compute_strengths(v.supporting, v.refuting, v.context)
    assert st.support == pytest.approx(0.4) and select_verdict(st)[0] == Verdict.INSUFFICIENT_EVIDENCE


def test_correction_cannot_be_grounded_in_a_rejected_citation():
    c = claim("The Eiffel Tower was completed in 1899.")
    p1 = _passage("C1-E1", "https://a.example/p", "C1-K1")
    p2 = _passage("C1-E2", "https://b.example/p", "C1-K2", text="The tower opened to the public in 1889 after construction.")
    raw = LLMJudgment.model_validate({"proposed_verdict": "REFUTED", "assessments": [
        {"evidence_id": "C1-E1", "stance": "REFUTES", "directness": "EXPLICIT", "quote": QUOTE},
        {"evidence_id": "C1-E2", "stance": "REFUTES", "directness": "EXPLICIT", "quote": "made up quote that is not there"},
    ]})
    v = validate_judgment(raw, c, [p1, p2], CFG)
    assert [x.evidence_id for x in v.refuting] == ["C1-E1"]
    text, note = validate_correction(LLMCorrection(text="The tower opened to the public in 1889.", evidence_ids=["C1-E2"]),
                                     c, Verdict.REFUTED, v, [p1, p2])
    assert text is None and "did not cite validated" in note


def test_displayed_quote_is_the_exact_source_text():
    p = _passage("C1-E1", "https://a.example/p", "C1-K1", text="Fact: The Eiffel Tower — “La dame de fer” — was completed in 1889.")
    model_copy = '  "the eiffel tower - "la dame de fer" -   was completed in 1889."  '
    v = supports_all([p], model_copy)
    assert v.supporting[0].quote == "The Eiffel Tower — “La dame de fer” — was completed in 1889."
    assert v.supporting[0].quote in p.text


def test_original_span_handles_misses_and_spacing():
    newline = chr(10)
    assert original_span(f"Alpha  beta{newline}gamma", "BETA gamma") == f"beta{newline}gamma"
    assert original_span("Alpha beta", "delta") is None
    assert original_span("", "x") is None


# --------------------------------------------------------------------------- 5. related-but-different entities


def test_lexical_checks_do_not_resolve_related_entities_documented_limitation():
    """'York' vs 'New York' (like 'Delhi' vs 'New Delhi') share tokens, so the deterministic
    topic/quote checks accept the passage. Entity resolution is delegated to the judge, which
    is instructed to mark passages about a different entity or place as IRRELEVANT."""
    c = claim("York is the most populous city in the United States.", primary_queries=["most populous US city"])
    p = _passage("C1-E1", "https://a.example/p", "C1-K1",
                 text="New York is the most populous city in the United States, with about 8.3 million residents.")
    v = supports_all([p], "New York is the most populous city in the United States", c)
    assert len(v.supporting) == 1  # accepted lexically - the judge is the safeguard
    assert "different entity, metric, place or time period" in JUDGE_SYSTEM


def test_judge_can_reject_a_different_entity_and_the_verdict_follows(config):
    text = "York is the most populous city in the United States."
    corpus = {"populous": [result("https://census.example.gov/nyc", "NYC", "New York is the most populous city in the "
                                  "United States, with about 8.3 million residents according to the census.")]}
    llm = FakeLLM(extraction(claim_dict(text, queries=["most populous city United States"])), rule_judge({}))
    [r] = verify_text(text, config, llm=llm, search=FakeSearch(corpus)).claims
    assert r.verdict == Verdict.INSUFFICIENT_EVIDENCE and not r.supporting


# --------------------------------------------------------------------------- 6. numbers that are names or dates
# Live finding (2026-10-27): official NASA/Smithsonian quotes for "The Apollo 11 mission landed
# ... on July 20, 1969" were rejected because the quote did not repeat "11" (a name) or the
# validator treated "20" (a date) as an amount. Only quantities must be stated in the quote.

from factcheck.validation import classified_claim_figures  # noqa: E402


@pytest.mark.parametrize("statement,kinds", [
    ("The Apollo 11 mission landed the first humans on the Moon on July 20, 1969.", ["label", "date", "year"]),
    ("Boeing 747 jets seat about 416 passengers.", ["label", "quantity"]),
    ("About 300 people attended on 4 July 2024.", ["quantity", "date", "year"]),
    ("The Eiffel Tower is 330 metres tall.", ["quantity"]),
    ("Freedonia has 3.2 million residents.", ["quantity"]),
    ("Over 45% of voters chose Windows 11.", ["quantity", "label"]),
    ("In 1969, 600 million people watched.", ["year", "quantity"]),
])
def test_claim_numbers_are_classified(statement, kinds):
    assert [k for _v, _q, k in classified_claim_figures(statement)] == kinds


APOLLO = claim("The Apollo 11 mission landed the first humans on the Moon on July 20, 1969.",
               primary_queries=["Apollo 11 moon landing date"])
SMITHSONIAN = ("Apollo 11 Bootprint Apollo 11 Timeline On July 20, 1969, a human walked on the Moon for the first "
               "time. Relive the full journey to and from the Moon with the timeline below.")


def test_official_quote_without_the_mission_number_is_accepted():
    p = _passage("C1-E1", "https://airandspace.si.edu/apollo-11-timeline", "C1-K1", text=SMITHSONIAN,
                 authority=Authority.PRIMARY)
    v = supports_all([p], "On July 20, 1969, a human walked on the Moon for the first time.", APOLLO)
    assert len(v.supporting) == 1, v.rejected


def test_name_label_must_still_appear_in_the_passage():
    p = _passage("C1-E1", "https://a.example/x", "C1-K1",
                 text="On July 20, 1969, a human walked on the Moon for the first time during a lunar mission.")
    v = supports_all([p], "On July 20, 1969, a human walked on the Moon for the first time", APOLLO)
    assert not v.supporting and "claimed figure 11" in v.rejected[0]


def test_wrong_date_is_still_rejected():
    p = _passage("C1-E1", "https://a.example/x", "C1-K1",
                 text="Apollo 11 landed on the Moon on July 24, 1969, after a four-day journey from Earth.")
    v = supports_all([p], "Apollo 11 landed on the Moon on July 24, 1969", APOLLO)
    assert not v.supporting and "claimed figure 20" in v.rejected[0]


def test_quantities_must_still_be_in_the_quote():
    c = claim("Boeing 747 jets seat about 416 passengers.", primary_queries=["Boeing 747 seats"])
    p = _passage("C1-E1", "https://a.example/x", "C1-K1",
                 text="The Boeing 747 is a wide-body airliner. Boeing 747 jets seat about 416 passengers in three classes.")
    v = supports_all([p], "The Boeing 747 is a wide-body airliner.", c)
    assert not v.supporting and "claimed figure 416" in v.rejected[0]
    v2 = supports_all([p], "Boeing 747 jets seat about 416 passengers in three classes.", c)
    assert len(v2.supporting) == 1


def test_report_distinguishes_rejected_citations_from_no_evidence():
    from factcheck.models import ClaimResult, Report, SummaryLabel
    from factcheck.report import render_text

    base = dict(claim_id="C1", index=1, statement="s", source_excerpt="s", claim_type=ClaimType.FACTUAL,
                checkability=Checkability.CHECKABLE, verdict=Verdict.INSUFFICIENT_EVIDENCE, score=0)
    rejected = Report(input_chars=1, summary_label=SummaryLabel.PARTIALLY_RESOLVED, summary_text="",
                      claims=[ClaimResult(**base, rejected_citations=6)])
    empty = Report(input_chars=1, summary_label=SummaryLabel.PARTIALLY_RESOLVED, summary_text="",
                   claims=[ClaimResult(**base)])
    assert "none of the AI model's citations could be verified" in " ".join(render_text(rejected).split())
    assert "No relevant evidence was found" in render_text(empty)
