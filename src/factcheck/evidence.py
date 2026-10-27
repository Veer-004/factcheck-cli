"""Deterministic evidence processing.

Search results and extracted pages -> sanitised, chunked, ranked passages with
application-assigned evidence IDs, source-authority labels and independence
clusters. No model is involved in anything here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .config import Config
from .models import Authority, Claim, ClaimType, EvidencePassage, SearchResult
from .security import detect_injection, is_safe_url, sanitize_text
from .textnorm import content_tokens, extract_numbers, numbers_compatible, tokens

# --------------------------------------------------------------------------
# URLs and domains
# --------------------------------------------------------------------------

_TRACKING_PARAMS = re.compile(r"^(utm_[a-z]+|fbclid|gclid|mc_[a-z]+|ref|ref_src|cmpid|ocid)$", re.I)
_SECOND_LEVEL = {
    "co.uk", "ac.uk", "gov.uk", "org.uk", "nhs.uk", "com.au", "gov.au", "edu.au", "org.au",
    "co.in", "gov.in", "ac.in", "nic.in", "co.jp", "go.jp", "ac.jp", "com.br", "gov.br",
    "co.nz", "govt.nz", "ac.nz", "co.za", "gov.za", "com.cn", "gov.cn", "edu.cn", "gc.ca",
    "gov.sg", "edu.sg", "com.sg", "gouv.fr", "europa.eu",
}


_MOBILE_LABELS = {"m", "mobile", "amp"}


def normalize_url(url: str) -> str:
    """Canonical key for "the same page": the same page over http/https, with or without
    ``www.``, on a mobile host (``en.m.wikipedia.org``), with tracking parameters,
    fragments, a trailing slash or different host case all map to one key."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    labels = host.split(".")
    if len(labels) > 2:  # never touch the registrable part (e.g. m.com stays)
        labels = [lbl for i, lbl in enumerate(labels) if not (lbl in _MOBILE_LABELS and i < len(labels) - 2)]
    host = ".".join(labels)
    scheme = parts.scheme.lower()
    if scheme == "http":
        scheme = "https"
    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _TRACKING_PARAMS.match(k))
    )
    path = parts.path.rstrip("/") or ""
    return urlunsplit((scheme, host, path, query, ""))


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def registrable_domain(host: str) -> str:
    labels = host.lower().strip(".").split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def organisation_key(host: str) -> str:
    """Label before the public suffix: ``bbc.co.uk`` and ``bbc.com`` -> ``bbc``."""
    return registrable_domain(host).split(".")[0]


# --------------------------------------------------------------------------
# Source authority
# --------------------------------------------------------------------------

_PRIMARY_DOMAINS = {
    "who.int", "un.org", "worldbank.org", "imf.org", "oecd.org", "europa.eu",
    "nature.com", "science.org", "thelancet.com", "nejm.org", "bmj.com", "jamanetwork.com",
    "cell.com", "plos.org", "pnas.org", "doi.org", "arxiv.org", "springer.com",
    "sciencedirect.com", "wiley.com", "iea.org", "ipcc.ch", "cochranelibrary.com",
}
_SECONDARY_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", "pbs.org", "nytimes.com",
    "washingtonpost.com", "wsj.com", "theguardian.com", "ft.com", "economist.com",
    "bloomberg.com", "cnn.com", "cbsnews.com", "nbcnews.com", "abcnews.go.com", "aljazeera.com",
    "dw.com", "france24.com", "theatlantic.com", "time.com", "britannica.com", "wikipedia.org",
    "nationalgeographic.com", "scientificamerican.com", "newscientist.com", "smithsonianmag.com",
    "ourworldindata.org", "statista.com", "pewresearch.org", "theconversation.com",
    "thehindu.com", "indianexpress.com", "abc.net.au", "cbc.ca", "lemonde.fr", "spiegel.de",
}
_FACT_CHECK_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "leadstories.com",
    "checkyourfact.com", "africacheck.org", "healthfeedback.org", "sciencefeedback.co",
    "truthorfiction.com", "factcheck.afp.com", "boomlive.in", "altnews.in", "logically.ai",
}
_OFFICIAL_SECOND_LEVEL = {
    s for s in _SECOND_LEVEL if s.split(".")[0] in {"gov", "gouv", "govt", "go", "gc", "ac", "nic", "nhs", "edu", "europa"}
}
_FACT_CHECK_PATHS =re.compile(r"/(fact-?check|ap-fact-check|factcheck)(/|$)", re.I)


def _matches(host: str, domains: set[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def is_fact_check_source(url: str) -> bool:
    host = domain_of(url)
    return _matches(host, _FACT_CHECK_DOMAINS) or bool(_FACT_CHECK_PATHS.search(urlsplit(url).path))


def classify_authority(url: str, claim: Claim) -> Authority:
    """Authority depends on the domain *and* the claim.

    * Government, intergovernmental, academic and journal domains are primary.
    * An organisation's own site is a primary source for statements about
      itself (product specs, its own announcements) - but for comparative
      claims about itself it is self-interested and drops to ``OTHER``.
    * Established news/reference outlets and fact-check sites are secondary.
    """
    host = domain_of(url)
    labels = host.split(".")
    org = organisation_key(host)
    claim_terms = content_tokens(claim.statement)
    mentions_self = len(org) >= 4 and org in claim_terms

    if mentions_self and claim.claim_type == ClaimType.COMPARATIVE:
        return Authority.OTHER
    if (
        labels[-1] in {"gov", "mil", "edu", "int"}
        or ".".join(labels[-2:]) in _OFFICIAL_SECOND_LEVEL
        or _matches(host, _PRIMARY_DOMAINS)
    ):
        return Authority.PRIMARY
    if _matches(host, _SECONDARY_DOMAINS) or is_fact_check_source(url):
        return Authority.SECONDARY if not mentions_self else Authority.PRIMARY
    if mentions_self:
        return Authority.PRIMARY
    return Authority.OTHER


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------


def parse_date(value: str | None) -> date | None:
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(v[:10])
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(v).date()
    except (TypeError, ValueError, IndexError):
        return None


def is_recent(published: str | None, as_of: date, days: int = 365) -> bool | None:
    """True/False for dated sources, None when undated or unparseable."""
    d = parse_date(published)
    if d is None:
        return None
    return 0 <= (as_of - d).days <= days or (d > as_of and (d - as_of).days <= 2)


# --------------------------------------------------------------------------
# Chunking and ranking
# --------------------------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Greedy sentence packing into chunks of at most ``max_chars`` characters."""
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        current = ""
        for sent in _SENTENCE.split(para):
            while len(sent) > max_chars:  # very long "sentence": hard split on a space
                cut = sent.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            if not current:
                current = sent
            elif len(current) + 1 + len(sent) <= max_chars:
                current = f"{current} {sent}"
            else:
                chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
    return [c for c in chunks if len(c) >= 40]


def lexical_relevance(claim: Claim, text: str) -> float:
    """Deterministic relevance in [0, ~1.3]: weighted term coverage plus number bonus."""
    statement_terms = content_tokens(claim.statement)
    if not statement_terms:
        return 0.0
    query_terms = set()
    for q in claim.primary_queries + claim.fallback_queries:
        query_terms |= content_tokens(q)
    query_terms -= statement_terms
    passage_terms = set(tokens(text))
    covered = len(statement_terms & passage_terms) / len(statement_terms)
    q_cov = len(query_terms & passage_terms) / len(query_terms) if query_terms else 0.0
    score = 0.85 * covered + 0.15 * q_cov
    claim_numbers = extract_numbers(claim.statement)
    if claim_numbers:
        passage_numbers = extract_numbers(text)
        if any(numbers_compatible(a, b, approximate=True) for a in claim_numbers for b in passage_numbers):
            score += 0.2
        elif passage_numbers:
            score += 0.1  # numbers present: may be a contradicting figure
    return round(score, 6)


_AUTHORITY_RANK = {Authority.PRIMARY: 1.0, Authority.SECONDARY: 0.9, Authority.OTHER: 0.75}


def _shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    words = tokens(text)
    return {tuple(words[i : i + k]) for i in range(max(0, len(words) - k + 1))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


SYNDICATION_CONTAINMENT = 0.8
MIN_CONTAINMENT_SHINGLES = 10


def _containment(a: set, b: set) -> float:
    """Share of the smaller text's shingles found in the larger one.

    Jaccard alone misses syndication when one copy is a short search snippet and
    the other a full page: every snippet sentence may come from the page, yet the
    sets differ greatly in size. Very short texts are ignored to avoid merging
    independent sources that merely share a phrase.
    """
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    if len(small) < MIN_CONTAINMENT_SHINGLES:
        return 0.0
    return len(small & large) / len(small)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


@dataclass
class _Source:
    url: str
    key: str
    title: str
    published_date: str | None
    snippet: str
    page_text: str = ""
    authority: Authority = Authority.OTHER
    fact_check: bool = False
    injection: bool = False
    cluster: str = ""
    order: int = 0
    shingles: set = field(default_factory=set)


@dataclass
class EvidenceBundle:
    passages: list[EvidencePassage]
    sources_considered: int
    injection_sources: int


def dedupe_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
    """Drop unsafe URLs and duplicates (after URL normalisation); keep first-seen order."""
    seen: dict[str, SearchResult] = {}
    for r in results:
        if not is_safe_url(r.url):
            continue
        key = normalize_url(r.url)
        if key in seen:
            prev = seen[key]
            if not prev.content and r.content:
                seen[key] = prev.model_copy(update={"content": r.content})
            continue
        if len(seen) >= limit:
            continue
        seen[key] = r
    return list(seen.values())


def cluster_sources(sources: list[_Source]) -> None:
    """Union-find: same organisation, or near-duplicate text (syndication), => one cluster."""
    parent = list(range(len(sources)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            a, b = sources[i], sources[j]
            if organisation_key(domain_of(a.url)) == organisation_key(domain_of(b.url)):
                union(i, j)
            elif (_jaccard(a.shingles, b.shingles) >= 0.5
                  or _containment(a.shingles, b.shingles) >= SYNDICATION_CONTAINMENT):
                union(i, j)
    labels: dict[int, str] = {}
    for i, s in enumerate(sources):
        root = find(i)
        if root not in labels:
            labels[root] = f"K{len(labels) + 1}"
        s.cluster = labels[root]


def build_evidence(
    claim: Claim,
    results: list[SearchResult],
    pages: dict[str, str],
    config: Config,
) -> EvidenceBundle:
    """Turn raw retrieval output into ranked passages with deterministic IDs.

    ``pages`` maps *normalised* URLs to extracted page text.
    """
    unique = dedupe_results(results, config.max_urls_per_claim)
    sources: list[_Source] = []
    for order, r in enumerate(unique):
        key = normalize_url(r.url)
        page = sanitize_text(pages.get(key, ""), max_chars=60_000)
        snippet = sanitize_text(r.content or "", max_chars=config.max_passage_chars * 2)
        title = sanitize_text(r.title or "", max_chars=200)
        if not page and not snippet:
            continue
        src = _Source(
            url=r.url.strip(),
            key=key,
            title=title,
            published_date=r.published_date,
            snippet=snippet,
            page_text=page if len(page) >= 200 else "",
            order=order,
        )
        src.authority = classify_authority(src.url, claim)
        src.fact_check = is_fact_check_source(src.url)
        src.injection = detect_injection(src.page_text or src.snippet) or detect_injection(title)
        src.shingles = _shingles((src.page_text or src.snippet)[:4000])
        sources.append(src)

    cluster_sources(sources)

    candidates: list[tuple[float, int, int, _Source, str, bool]] = []
    for src in sources:
        snippet_only = not src.page_text
        chunks = chunk_text(src.page_text or src.snippet, config.max_passage_chars)
        scored = []
        for ci, chunk in enumerate(chunks):
            rel = lexical_relevance(claim, chunk)
            if rel <= 0.0:
                continue
            rank = rel * _AUTHORITY_RANK[src.authority] * (0.8 if snippet_only else 1.0)
            scored.append((rank, ci, chunk, rel))
        # At most two passages per source keeps the judge's view diverse.
        scored.sort(key=lambda t: (-t[0], t[1]))
        for rank, ci, chunk, rel in scored[:2]:
            candidates.append((rank, src.order, ci, src, chunk, snippet_only))

    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    selected = candidates[: config.max_passages_for_judge]

    passages = [
        EvidencePassage(
            evidence_id=f"{claim.claim_id}-E{n}",
            url=src.url,
            domain=domain_of(src.url),
            title=src.title,
            published_date=src.published_date,
            text=chunk,
            snippet_only=snippet_only,
            injection_flagged=src.injection,
            authority=src.authority,
            fact_check_site=src.fact_check,
            cluster_id=f"{claim.claim_id}-{src.cluster}",
            relevance=round(rank, 6),
        )
        for n, (rank, _o, _c, src, chunk, snippet_only) in enumerate(selected, start=1)
    ]
    return EvidenceBundle(
        passages=passages,
        sources_considered=len(sources),
        injection_sources=sum(1 for s in sources if s.injection),
    )
