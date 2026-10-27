from __future__ import annotations

from datetime import date

import pytest
from conftest import load_page

from factcheck.evidence import (
    build_evidence,
    chunk_text,
    classify_authority,
    dedupe_results,
    is_fact_check_source,
    is_recent,
    lexical_relevance,
    normalize_url,
    organisation_key,
    parse_date,
    registrable_domain,
)
from factcheck.models import Authority, Checkability, Claim, ClaimType, SearchResult


def make_claim(statement="The Eiffel Tower was completed in 1889.", claim_type=ClaimType.FACTUAL, **kw) -> Claim:
    return Claim(
        claim_id="C1", index=1, statement=statement, source_excerpt=statement,
        checkability=Checkability.CHECKABLE, claim_type=claim_type,
        primary_queries=kw.pop("queries", ["Eiffel Tower completed 1889"]), **kw,
    )


def test_normalize_url_strips_tracking_www_fragment_and_slash():
    a = normalize_url("https://WWW.Example.com/path/?utm_source=x&b=2&a=1#frag")
    b = normalize_url("https://example.com/path?a=1&b=2")
    assert a == b


def test_dedupe_results_limits_and_filters_unsafe():
    results = [SearchResult(url=f"https://site{i}.example/a", content="x") for i in range(10)]
    results.insert(1, SearchResult(url="https://site0.example/a/?utm_medium=y", content="dup"))
    results.insert(2, SearchResult(url="javascript:alert(1)", content="bad"))
    out = dedupe_results(results, limit=6)
    assert len(out) == 6
    assert all(r.url.startswith("https://") for r in out)
    assert len({normalize_url(r.url) for r in out}) == 6


def test_registrable_domain_and_org_key():
    assert registrable_domain("news.bbc.co.uk") == "bbc.co.uk"
    assert registrable_domain("www.nasa.gov") == "nasa.gov"
    assert organisation_key("bbc.com") == organisation_key("www.bbc.co.uk") == "bbc"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.census.gov/data", Authority.PRIMARY),
        ("https://www.ons.gov.uk/x", Authority.PRIMARY),
        ("https://www.who.int/news", Authority.PRIMARY),
        ("https://www.nature.com/articles/x", Authority.PRIMARY),
        ("https://www.reuters.com/world/x", Authority.SECONDARY),
        ("https://abcnews.go.com/story", Authority.SECONDARY),  # 'go' label is not a government marker here
        ("https://www.snopes.com/fact-check/x", Authority.SECONDARY),
        ("https://randomblog.example/post", Authority.OTHER),
    ],
)
def test_classify_authority_by_domain(url, expected):
    assert classify_authority(url, make_claim()) == expected


def test_company_is_primary_for_own_specs_but_not_for_comparative_marketing():
    spec = make_claim("Contoso's X1 laptop weighs 1.2 kilograms.", ClaimType.NUMERIC)
    comparative = make_claim("Contoso's X1 is the fastest laptop in the world.", ClaimType.COMPARATIVE)
    assert classify_authority("https://www.contoso.com/x1/specs", spec) == Authority.PRIMARY
    assert classify_authority("https://www.contoso.com/x1", comparative) == Authority.OTHER


def test_fact_check_detection_by_domain_and_path():
    assert is_fact_check_source("https://www.politifact.com/x")
    assert is_fact_check_source("https://www.reuters.com/fact-check/abc")
    assert not is_fact_check_source("https://www.reuters.com/world/abc")


def test_parse_date_formats_and_recency():
    assert parse_date("2025-06-30") == date(2025, 6, 30)
    assert parse_date("2025-06-30T10:00:00Z") == date(2025, 6, 30)
    assert parse_date("Mon, 30 Jun 2025 10:00:00 GMT") == date(2025, 6, 30)
    assert parse_date("garbage") is None
    as_of = date(2026, 10, 26)
    assert is_recent("2026-01-01", as_of) is True
    assert is_recent("2020-01-01", as_of) is False
    assert is_recent(None, as_of) is None


def test_chunk_text_respects_limit_and_drops_tiny_fragments():
    text = " ".join(f"Sentence number {i} talks about the Eiffel Tower in Paris." for i in range(100))
    chunks = chunk_text(text, 300)
    assert chunks and all(len(c) <= 300 for c in chunks)
    long_word_text = "x" * 2000
    assert all(len(c) <= 300 for c in chunk_text(long_word_text, 300))


def test_lexical_relevance_prefers_on_topic_text():
    claim = make_claim()
    on = lexical_relevance(claim, "The Eiffel Tower was completed in 1889 in Paris.")
    off = lexical_relevance(claim, "Bananas are rich in potassium and fibre.")
    assert on > 0.8 and off == 0.0


def _results():
    return [
        SearchResult(url="https://www.tourism.example.gov/eiffel", title="Official", content="Eiffel Tower completed in 1889."),
        SearchResult(url="https://www.bbc.co.uk/news/eiffel", title="News", content="The Eiffel Tower, completed in 1889, ..."),
        SearchResult(url="https://www.bbc.com/travel/eiffel", title="Travel", content="The Eiffel Tower was completed in 1889 for the fair."),
        SearchResult(url="https://trivia.example/eiffel", title="Trivia", content="snippet"),
    ]


def test_build_evidence_assigns_deterministic_ids_and_clusters(config):
    claim = make_claim()
    pages = {
        normalize_url("https://www.tourism.example.gov/eiffel"): load_page("eiffel_official.txt"),
        normalize_url("https://www.bbc.co.uk/news/eiffel"): load_page("eiffel_news.txt"),
        normalize_url("https://trivia.example/eiffel"): load_page("eiffel_injection.txt"),
    }
    b1 = build_evidence(claim, _results(), pages, config)
    b2 = build_evidence(claim, _results(), pages, config)
    assert [p.model_dump() for p in b1.passages] == [p.model_dump() for p in b2.passages]
    ids = [p.evidence_id for p in b1.passages]
    assert ids == [f"C1-E{i}" for i in range(1, len(ids) + 1)]
    assert len(ids) <= config.max_passages_for_judge
    assert all(len(p.text) <= config.max_passage_chars for p in b1.passages)

    by_url = {p.url: p for p in b1.passages}
    # bbc.co.uk and bbc.com are one organisation -> one independence cluster.
    assert by_url["https://www.bbc.co.uk/news/eiffel"].cluster_id == by_url["https://www.bbc.com/travel/eiffel"].cluster_id
    assert by_url["https://www.tourism.example.gov/eiffel"].cluster_id != by_url["https://www.bbc.co.uk/news/eiffel"].cluster_id
    # Page extraction failed for bbc.com -> snippet-only.
    assert by_url["https://www.bbc.com/travel/eiffel"].snippet_only
    assert not by_url["https://www.tourism.example.gov/eiffel"].snippet_only
    # Indirect injection on the trivia page is flagged.
    assert by_url["https://trivia.example/eiffel"].injection_flagged
    assert b1.injection_sources == 1
    assert by_url["https://www.tourism.example.gov/eiffel"].authority == Authority.PRIMARY


def test_syndicated_copies_share_a_cluster(config):
    claim = make_claim()
    body = load_page("eiffel_news.txt")
    results = [
        SearchResult(url="https://paper-one.example/a", content="x"),
        SearchResult(url="https://paper-two.example/b", content="y"),
    ]
    pages = {normalize_url(r.url): body for r in results}
    passages = build_evidence(claim, results, pages, config).passages
    assert len({p.cluster_id for p in passages}) == 1


def test_build_evidence_caps_urls_and_passages(config):
    claim = make_claim()
    results = [
        SearchResult(url=f"https://site{i}.example/eiffel", content="The Eiffel Tower was completed in 1889. " * 3)
        for i in range(12)
    ]
    bundle = build_evidence(claim, results, {}, config)
    assert bundle.sources_considered <= config.max_urls_per_claim
    assert len(bundle.passages) <= config.max_passages_for_judge
    assert all(p.snippet_only for p in bundle.passages)


def test_irrelevant_text_produces_no_passages(config):
    claim = make_claim()
    results = [SearchResult(url="https://x.example/a", content="Bananas are rich in potassium, fibre and vitamin B6.")]
    assert build_evidence(claim, results, {}, config).passages == []
