"""Code-side validation of user input and of everything the model returns.

Nothing the model says is displayed or used for a verdict unless it passes
these checks. A verbatim quote is necessary but not sufficient: the quote and
its passage must also be on-topic for the claim, carry the claimed figures, and
not contradict the stance they are cited for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .evidence import is_recent
from .models import (
    Checkability,
    Citation,
    Claim,
    ClaimType,
    Directness,
    DroppedClaim,
    EvidencePassage,
    LLMAssessment,
    LLMCorrection,
    LLMExtraction,
    LLMJudgment,
    Stance,
    Verdict,
)
from .scoring import passage_weight
from .security import is_instruction, sanitize_text, sanitize_user_input
from .textnorm import (
    contains_normalized,
    content_tokens,
    extract_numbers,
    normalize_for_match,
    number_contexts,
    number_mentions,
    numbers_compatible,
    original_span,
    strip_quote_wrapping,
    subject_tokens,
)


class InputError(ValueError):
    """Invalid user input (CLI exit code 2)."""


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------


def validate_input(text: object, config: Config) -> str:
    if not isinstance(text, str):
        raise InputError("input must be text")
    cleaned = sanitize_user_input(text)
    if not cleaned:
        raise InputError("input is empty")
    if len(cleaned) > config.max_input_chars:
        raise InputError(f"input is {len(cleaned)} characters; the limit is {config.max_input_chars}")
    if not re.search(r"[^\W\d_]{2,}", cleaned):
        raise InputError("input contains no words to check")
    return cleaned


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------


@dataclass
class ClaimValidation:
    checkable: list[Claim] = field(default_factory=list)
    not_checkable: list[Claim] = field(default_factory=list)
    dropped: list[DroppedClaim] = field(default_factory=list)
    over_limit: list[DroppedClaim] = field(default_factory=list)


def _clean_queries(queries: list[str], limit: int) -> list[str]:
    out: list[str] = []
    for q in queries or []:
        if not isinstance(q, str):
            continue
        q = " ".join(sanitize_user_input(q).split())[:200]
        if q and normalize_for_match(q) not in {normalize_for_match(x) for x in out}:
            out.append(q)
        if len(out) >= limit:
            break
    return out


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def validate_claims(extraction: LLMExtraction, text: str, config: Config) -> ClaimValidation:
    """Anchor, ground, deduplicate and cap the extracted claims."""
    result = ClaimValidation()
    input_terms = content_tokens(text)
    input_numbers = extract_numbers(text)
    accepted: list[tuple[str, str]] = []  # (statement, claim_id)
    next_id = 1

    raw_claims = list(extraction.claims)
    for extra in raw_claims[config.max_extracted_claims :]:
        result.dropped.append(
            DroppedClaim(statement=extra.statement[:300], reason="exceeded the extraction cap")
        )

    for raw in raw_claims[: config.max_extracted_claims]:
        statement = " ".join(sanitize_user_input(raw.statement or "").split())[:500]
        excerpt = strip_quote_wrapping(sanitize_user_input(raw.source_excerpt or ""))
        drop = lambda reason: result.dropped.append(  # noqa: E731
            DroppedClaim(statement=statement[:300] or "(empty)", source_excerpt=excerpt[:300], reason=reason)
        )
        if len(content_tokens(statement)) == 0:
            drop("empty or contentless statement")
            continue
        if is_instruction(statement):
            drop("instruction-like text is not a factual claim")
            continue
        if len(normalize_for_match(excerpt)) < 8:
            drop("missing or too-short source excerpt")
            continue
        if not contains_normalized(text, excerpt):
            drop("source excerpt does not occur in the input (possible invented claim)")
            continue
        stmt_numbers = extract_numbers(statement)
        if any(not any(numbers_compatible(n, m) for m in input_numbers) for n in stmt_numbers):
            drop("statement contains figures that are not in the input")
            continue
        stmt_terms = content_tokens(statement)
        if len(stmt_terms & input_terms) / len(stmt_terms) < 0.6:
            drop("statement wording is not grounded in the input")
            continue
        dup = next((cid for s, cid in accepted if _token_jaccard(s, statement) >= 0.8
                    and sorted(extract_numbers(s)) == sorted(stmt_numbers)), None)
        if dup:
            drop(f"near-duplicate of {dup}")
            continue

        is_checkable = raw.checkability == Checkability.CHECKABLE
        if is_checkable and len(result.checkable) >= config.max_checkable_claims:
            result.over_limit.append(
                DroppedClaim(
                    statement=statement[:300],
                    source_excerpt=excerpt[:300],
                    reason=f"over the limit of {config.max_checkable_claims} checkable claims per run",
                )
            )
            continue

        primary = _clean_queries(raw.primary_queries, config.max_queries_per_round) or [statement[:200]]
        fallback = [
            q for q in _clean_queries(raw.fallback_queries, config.max_queries_per_round)
            if normalize_for_match(q) not in {normalize_for_match(p) for p in primary}
        ]
        claim = Claim(
            claim_id=f"C{next_id}",
            index=next_id,
            statement=statement,
            source_excerpt=excerpt,
            checkability=raw.checkability,
            claim_type=raw.claim_type,
            time_sensitive=bool(raw.time_sensitive),
            time_anchor=(sanitize_user_input(raw.time_anchor)[:80] or None) if raw.time_anchor else None,
            primary_queries=primary,
            fallback_queries=fallback,
        )
        next_id += 1
        accepted.append((statement, claim.claim_id))
        (result.checkable if is_checkable else result.not_checkable).append(claim)
    return result


# --------------------------------------------------------------------------
# Numbers in claims
# --------------------------------------------------------------------------

_APPROX = re.compile(r"\b(about|around|approximately|approx\.?|nearly|roughly|almost|some|circa|ca\.|~)\s*$", re.I)
_MIN = re.compile(r"\b(more than|over|at least|above|exceeds?|exceeding|upwards of)\s*$", re.I)
_MAX = re.compile(r"\b(less than|under|at most|below|fewer than|up to)\s*$", re.I)



def is_year_like(v: float) -> bool:
    return float(v).is_integer() and 1000 <= v <= 2100


def claim_figures(statement: str) -> list[tuple[float, str]]:
    """Numbers in a claim with a qualifier: exact | approx | min | max."""
    return [(value, q) for value, q, _kind in classified_claim_figures(statement)]


_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
           r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_DATE_BEFORE = re.compile(rf"\b{_MONTHS}\.?\s+$", re.I)
_DATE_AFTER = re.compile(rf"^(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTHS}\b", re.I)
# A capitalised word directly before the number names something ("Apollo 11",
# "Boeing 747", "Windows-11"); sentence-initial function words do not.
_NAME_BEFORE = re.compile(r"\b([A-Z][A-Za-z]*)[ -]$")
_NOT_NAMES = {
    "about", "around", "approximately", "nearly", "roughly", "almost", "over", "under", "more", "less",
    "fewer", "only", "just", "some", "exactly", "in", "on", "at", "by", "since", "until", "from", "to",
    "the", "a", "an", "of", "for", "with", "than", "and", "or", "after", "before", "during", "between",
    "top", "up", "at least", "at most",
}


def classified_claim_figures(statement: str) -> list[tuple[float, str, str]]:
    """Claim numbers as (value, qualifier, kind).

    kind: ``year`` (1000-2100 integer), ``date`` (day next to a month name, "July 20"),
    ``label`` (part of a name, "Apollo 11") or ``quantity``. Only quantities are
    measured amounts that a supporting quote must state; the other kinds identify
    *what* the claim is about and only need to appear in the cited passage.
    """
    out: list[tuple[float, str, str]] = []
    for value, before, after, suffix in number_contexts(statement):
        q = "approx" if _APPROX.search(before) else "min" if _MIN.search(before) else "max" if _MAX.search(before) else "exact"
        name = _NAME_BEFORE.search(before)
        if is_year_like(value):
            kind = "year"
        elif suffix:  # %, million, ... are always amounts
            kind = "quantity"
        elif float(value).is_integer() and 1 <= value <= 31 and (_DATE_BEFORE.search(before) or _DATE_AFTER.search(after)):
            kind = "date"
        elif q == "exact" and name and name.group(1).lower() not in _NOT_NAMES:
            kind = "label"
        else:
            kind = "quantity"
        out.append((value, q, kind))
    return out


def figure_agrees(claim_value: float, qualifier: str, evidence_value: float) -> bool:
    if qualifier == "min":
        return evidence_value >= claim_value * 0.995 or numbers_compatible(claim_value, evidence_value)
    if qualifier == "max":
        return evidence_value <= claim_value * 1.005 or numbers_compatible(claim_value, evidence_value)
    return numbers_compatible(claim_value, evidence_value, approximate=(qualifier == "approx"))


# --------------------------------------------------------------------------
# Judgments
# --------------------------------------------------------------------------

_DEBUNK = re.compile(r"\b(myth|debunked|false|untrue|hoax|misconception|incorrect|inaccurate|not true|no evidence)\b", re.I)
_MIN_QUOTE_CHARS = 20
_FABRICATION = "fabrication"


@dataclass
class JudgmentValidation:
    fatal: list[str] = field(default_factory=list)
    supporting: list[Citation] = field(default_factory=list)
    refuting: list[Citation] = field(default_factory=list)
    context: list[Citation] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    proposed_verdict: Verdict | None = None
    rationale: str = ""

    @property
    def ok(self) -> bool:
        return not self.fatal


def _check_assessment(a: LLMAssessment, claim: Claim, passage: EvidencePassage) -> tuple[str | None, str, Directness]:
    """Return (rejection_reason or None, rejection_kind, effective directness)."""
    quote = strip_quote_wrapping(a.quote or "")
    nq = normalize_for_match(quote)
    if len(nq) < _MIN_QUOTE_CHARS or len(nq.split()) < 4:
        return "quote is too short to verify", _FABRICATION, a.directness
    if len(quote) > len(passage.text):
        return "quote is longer than the passage", _FABRICATION, a.directness
    if not contains_normalized(passage.text, quote):
        return "quote does not appear verbatim in the cited passage", _FABRICATION, a.directness

    claim_terms = subject_tokens(claim.statement)
    quote_terms = content_tokens(quote)
    passage_terms = content_tokens(passage.text + " " + passage.title)
    if claim_terms and len(claim_terms & passage_terms) / len(claim_terms) < 0.34:
        return "passage is not about the claim's subject", "scope", a.directness
    if not (claim_terms & quote_terms) and not extract_numbers(quote):
        return "quote shares no terms or figures with the claim", "scope", a.directness

    directness = a.directness
    if (
        a.stance in (Stance.SUPPORTS, Stance.REFUTES)
        and directness == Directness.EXPLICIT
        and claim_terms
        and len(claim_terms & quote_terms) / len(claim_terms) < 0.34
    ):
        directness = Directness.PARTIAL

    classified = classified_claim_figures(claim.statement)
    figures = [(v, q) for v, q, _k in classified]
    quote_numbers = extract_numbers(quote)
    passage_numbers = extract_numbers(passage.text)

    if a.stance == Stance.SUPPORTS:
        if _DEBUNK.search(quote) and not _DEBUNK.search(claim.statement):
            return "quote reads as a debunking, not support", "stance", directness
        for value, qual, kind in classified:
            # Amounts must be in the quote; years, dates and name labels in the passage.
            pool = quote_numbers if kind == "quantity" else passage_numbers
            if not any(figure_agrees(value, qual, ev) for ev in pool):
                return f"supporting quote does not contain the claimed figure {value:g}", "stance", directness

    # Numeric safeguards apply whenever the claim hinges on a quantity - by label, or
    # because it states a non-year figure that the refuting quote answers with a number.
    numeric_claim = claim.claim_type == ClaimType.NUMERIC or (
        any(kind == "quantity" for _v, _q, kind in classified) and bool(quote_numbers)
    )
    if a.stance == Stance.REFUTES and numeric_claim and figures:
        nc = a.numeric_comparison
        if nc is None or nc.claim_value is None or nc.evidence_value is None:
            return "numeric refutation lacks an explicit figure comparison", "stance", directness
        if not all([nc.same_entity, nc.same_metric, nc.same_unit, nc.same_scope, nc.same_period]):
            return "figures measure different things (entity/metric/unit/scope/period differ)", "stance", directness
        if not any(numbers_compatible(nc.evidence_value, q) for q in quote_numbers):
            return "evidence figure is not in the quote", _FABRICATION, directness
        match = next(((v, q) for v, q in figures if numbers_compatible(v, nc.claim_value)), None)
        if match is None:
            return "compared claim figure does not appear in the claim", "stance", directness
        if figure_agrees(match[0], "approx" if nc.claim_is_approximate else match[1], nc.evidence_value):
            return "figures agree within rounding/approximation; not a refutation", "stance", directness
    return None, "", directness


def validate_judgment(
    raw: LLMJudgment, claim: Claim, passages: list[EvidencePassage], config: Config
) -> JudgmentValidation:
    out = JudgmentValidation(proposed_verdict=raw.proposed_verdict, rationale=sanitize_text(raw.rationale or "", 600))
    by_id = {p.evidence_id: p for p in passages}
    if passages and not raw.assessments:
        out.fatal.append("no evidence assessments were returned")
        return out

    seen: set[str] = set()
    considered = fabricated = 0
    for a in raw.assessments:
        eid = (a.evidence_id or "").strip().strip("[]")
        if a.stance == Stance.IRRELEVANT:
            seen.add(eid)
            continue
        considered += 1
        passage = by_id.get(eid)
        if passage is None:
            fabricated += 1
            out.rejected.append(f"{eid or '(blank)'}: unknown evidence ID")
            continue
        if eid in seen:
            out.rejected.append(f"{eid}: duplicate assessment ignored")
            continue
        seen.add(eid)
        reason, kind, directness = _check_assessment(a, claim, passage)
        if reason:
            fabricated += kind == _FABRICATION
            out.rejected.append(f"{eid}: {reason}")
            continue
        citation = Citation(
            evidence_id=eid,
            url=passage.url,
            title=passage.title,
            published_date=passage.published_date,
            # Display the source's exact characters, not the model's copy (validated equal above).
            quote=original_span(passage.text, strip_quote_wrapping(a.quote)) or strip_quote_wrapping(a.quote),
            stance=a.stance,
            directness=directness,
            authority=passage.authority,
            cluster_id=passage.cluster_id,
            snippet_only=passage.snippet_only,
            injection_flagged=passage.injection_flagged,
            fact_check_site=passage.fact_check_site,
            recent=is_recent(passage.published_date, config.as_of) if claim.time_sensitive else None,
        )
        citation.weight = passage_weight(citation, claim.time_sensitive)
        {Stance.SUPPORTS: out.supporting, Stance.REFUTES: out.refuting, Stance.MISSING_CONTEXT: out.context}[
            a.stance
        ].append(citation)

    if considered and fabricated * 2 >= considered and fabricated > 0:
        out.fatal.append(f"{fabricated} of {considered} citations reference unknown evidence or unverifiable quotes")
    return out


# --------------------------------------------------------------------------
# Corrections
# --------------------------------------------------------------------------


def validate_correction(
    raw: LLMCorrection | None,
    claim: Claim,
    verdict: Verdict,
    judgment: JudgmentValidation,
    passages: list[EvidencePassage],
) -> tuple[str | None, str | None]:
    """Return (correction, note). Only REFUTED/MISLEADING claims get corrections."""
    if verdict not in {Verdict.REFUTED, Verdict.MISLEADING}:
        return None, None
    unavailable = "No correction could be established from the validated evidence."
    if raw is None or not (raw.text or "").strip():
        return None, unavailable
    text = " ".join(sanitize_text(raw.text, 400).split())
    allowed = judgment.refuting if verdict == Verdict.REFUTED else judgment.refuting + judgment.context
    allowed_ids = {c.evidence_id for c in allowed}
    cited = [e.strip() for e in raw.evidence_ids if e.strip() in allowed_ids]
    if not cited:
        return None, unavailable + " (the proposed correction did not cite validated refuting/context evidence)"
    by_id = {p.evidence_id: p for p in passages}
    grounding_text = " ".join(by_id[e].text for e in cited if e in by_id)
    ground_numbers = extract_numbers(grounding_text) + extract_numbers(claim.statement)
    for n in extract_numbers(text):
        if not any(numbers_compatible(n, g) for g in ground_numbers):
            return None, unavailable + f" (the proposed correction contained an ungrounded figure: {n:g})"
    terms = content_tokens(text)
    ground_terms = content_tokens(grounding_text + " " + claim.statement)
    if terms and len(terms & ground_terms) / len(terms) < 0.6:
        return None, unavailable + " (the proposed correction contained details not found in the cited evidence)"
    return text, f"Grounded in {', '.join(sorted(set(cited)))}."
