"""Terminal and JSON rendering of a :class:`Report`.

``render_text`` is the human-facing view: it only *reads* the report (verdicts,
scores and citations are never changed here) and hides implementation detail
unless ``verbose`` is set. ``render_json`` is the machine interface and dumps the
complete report unchanged.
"""

from __future__ import annotations

import json
import re
import textwrap
from urllib.parse import urlsplit

from .evidence import parse_date
from .models import Checkability, Citation, ClaimResult, Report, SummaryLabel, Verdict

WIDTH = 88
INDENT = "   "
MAX_SOURCES_PER_SIDE = 3  # per claim and side in the normal view (verbose shows all)
MAX_QUOTE_CHARS = 280  # normal view; the full validated quote is always in the JSON

# The official verdict name, readable (same words as the JSON "verdict" field).
_VERDICT_TEXT = {v: v.value.replace("_", " ") for v in Verdict}

_CHECKABILITY_TEXT = {
    Checkability.OPINION: "an opinion",
    Checkability.PREDICTION: "a prediction about the future",
    Checkability.VAGUE: "too vague to verify",
    Checkability.OTHER_NOT_CHECKABLE: "not a verifiable statement of fact",
}

# Plain-language wording for score caps (scoring.py reasons).
_CAP_TEXT = {
    "only one independent source cluster": "only one independent source",
    "no primary-authority source": "no primary or official source",
    "only fact-check sites": "only fact-checking sites",
    "a contributing passage was flagged for possible prompt injection": "a source contained instruction-like text",
    "time-sensitive claim with no passage dated within one year": "no source dated within the last year",
    "misleading verdict": "misleading verdicts are capped",
}


def _wrap(text: str, indent: str = INDENT, subsequent: str | None = None) -> str:
    return textwrap.fill(" ".join(str(text).split()), width=WIDTH, initial_indent=indent,
                         subsequent_indent=indent if subsequent is None else subsequent)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _verb(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def _independent(cites: list[Citation]) -> int:
    return len({c.cluster_id for c in cites})


def _domain(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _short_quote(quote: str, limit: int) -> str:
    q = " ".join(quote.split())
    if len(q) <= limit:
        return q
    cut = q.rfind(" ", 0, limit)
    return q[: cut if cut > limit // 2 else limit].rstrip(",;:") + " …"


def _explanation(r: ClaimResult) -> str:
    s, ref, ctx = _independent(r.supporting), _independent(r.refuting), _independent(r.context)
    if r.verdict == Verdict.SUPPORTED:
        return f"The retrieved evidence supports this claim ({_plural(s, 'independent source')})."
    if r.verdict == Verdict.REFUTED:
        return f"The retrieved evidence contradicts this claim ({_plural(ref, 'independent source')})."
    if r.verdict == Verdict.MISLEADING:
        return ("Partly accurate, but the evidence shows it leaves out or distorts important context "
                f"({_plural(s, 'supporting source')}, {_plural(ctx, 'source')} giving context).")
    if r.verdict == Verdict.CONFLICTING:
        return (f"Credible sources disagree: {_plural(s, 'independent source')} "
                f"{_verb(s, 'supports', 'support')} it and {ref} {_verb(ref, 'contradicts', 'contradict')} it. "
                "No single verdict is given.")
    if r.verdict == Verdict.INSUFFICIENT_EVIDENCE:
        if not (r.supporting or r.refuting or r.context):
            if r.rejected_citations or r.evidence:
                return ("Sources were found, but none of the AI model's citations could be verified "
                        "against them, so no verdict was reached.")
            return "No relevant evidence was found, so the claim could not be confirmed or refuted."
        return "Some evidence was found, but not enough reliable evidence to confirm or refute the claim."
    if r.verdict == Verdict.NOT_CHECKABLE:
        what = _CHECKABILITY_TEXT.get(r.checkability, "not a verifiable statement of fact")
        return f"This statement is {what}, so it cannot be fact-checked."
    if r.verdict == Verdict.RETRIEVAL_FAILED:
        return ("The web search failed for this claim. This is a technical problem, not a judgement about "
                "whether the claim is true.")
    return f"This claim could not be checked: {_failure_summary(r.failure_reason or '')}."


def _failure_summary(reason: str) -> str:
    """Short, non-technical reason for PROCESSING_FAILED (full text in --verbose / JSON)."""
    low = reason.lower()
    if "budget" in low:
        return "the per-run search or model budget was used up"
    if "time limit" in low or "run time" in low:
        return "the run time limit was reached"
    if "remained invalid" in low:
        return "the AI model's answer failed validation twice"
    if "unavailable" in low or "timed out" in low or "no response" in low or "rate limit" in low:
        return "the AI model did not respond in time or was unavailable"
    if "step limit" in low:
        return "the processing step limit was reached"
    return "an internal processing error occurred"


def _evidence_block(title: str, cites: list[Citation], verbose: bool) -> list[str]:
    if not cites:
        return []
    # Group passages by source page so a page is listed once; strength already counts
    # each independent source once (scoring.side_strength), so this is display only.
    groups: dict[str, list[Citation]] = {}
    for c in cites:  # cites are ordered strongest first
        groups.setdefault(c.url, []).append(c)
    sources = list(groups.values())
    shown = sources if verbose else sources[:MAX_SOURCES_PER_SIDE]
    lines = [f"{INDENT}{title}:"]
    for n, group in enumerate(shown, start=1):
        first = group[0]
        meta = [_domain(first.url)]
        published = parse_date(first.published_date)
        if published:
            meta.append(published.isoformat())  # providers send ISO or RFC 2822 dates
        tags = []
        if any(c.snippet_only for c in group):
            tags.append("search snippet only")
        if any(c.fact_check_site for c in group):
            tags.append("fact-checking site")
        if any(c.injection_flagged for c in group):
            tags.append("page contained instruction-like text; given less weight")
        heading = f"{n}. {first.title or _domain(first.url)} ({', '.join(meta)})"
        lines.append(_wrap(heading, INDENT + "  ", INDENT + "     "))
        if tags:
            lines.append(_wrap(f"[{'; '.join(tags)}]", INDENT + "     "))
        for c in group if verbose else group[:1]:
            quote = c.quote if verbose else _short_quote(c.quote, MAX_QUOTE_CHARS)
            lines.append(_wrap(f'"{quote}"', INDENT + "     "))
            if verbose:
                lines.append(f"{INDENT}     ({c.evidence_id}; {c.authority.value.lower()}, {c.directness.value.lower()}, "
                             f"weight {c.weight:.2f}, cluster {c.cluster_id})")
        lines.append(f"{INDENT}     {first.url}")
    hidden = len(sources) - len(shown)
    if hidden:
        lines.append(f"{INDENT}  (+{_plural(hidden, 'more source')}; see --verbose or --json-out)")
    return lines


def _limits(r: ClaimResult) -> list[str]:
    reasons = []
    for cap in r.score_caps:
        reason = re.sub(r"^(support side |refute side )?capped at \d+: ", "", cap)
        text = _CAP_TEXT.get(reason, reason)
        if text not in reasons:
            reasons.append(text)
    return reasons


def _strength_lines(r: ClaimResult, disclaimer: str) -> list[str]:
    if r.verdict == Verdict.CONFLICTING and r.support_score is not None:
        lines = [_wrap(f"Evidence strength: support {r.support_score}/100, against {r.refute_score}/100 — "
                       f"{disclaimer}")]
    elif r.score is not None:
        lines = [_wrap(f"Evidence strength: {r.score}/100 — {disclaimer}")]
        if r.verdict == Verdict.INSUFFICIENT_EVIDENCE:
            lines.append(_wrap("This measures only the evidence that was found, not whether the claim is true.",
                               INDENT + "  "))
    else:
        return []
    limits = _limits(r)
    if limits:
        lines.append(_wrap(f"Limited by: {'; '.join(limits)}.", INDENT + "  "))
    return lines


def render_claim(r: ClaimResult, number: int, total: int, disclaimer: str, verbose: bool = False) -> list[str]:
    header = f"Claim {number} of {total}" if total > 1 else "Claim"
    lines = [_wrap(f"{header}: {r.statement}", "", INDENT)]
    if " ".join(r.source_excerpt.split()).casefold() != " ".join(r.statement.split()).casefold():
        lines.append(_wrap(f'From your text: "{r.source_excerpt}"'))
    lines.append(f"{INDENT}Verdict: {_VERDICT_TEXT[r.verdict]}")
    lines.append(_wrap(f"Explanation: {_explanation(r)}"))
    lines += _evidence_block("Supporting evidence", r.supporting, verbose)
    lines += _evidence_block("Contradicting evidence", r.refuting, verbose)
    lines += _evidence_block("Missing context", r.context, verbose)
    if r.correction:
        lines.append(_wrap(f"Correction: {r.correction}"))
    elif r.verdict in (Verdict.REFUTED, Verdict.MISLEADING):
        lines.append(_wrap("Correction: none could be established from the validated evidence."))
    lines += _strength_lines(r, disclaimer)
    return lines


def _diagnostics(report: Report) -> list[str]:
    out = ["", "DIAGNOSTICS (verbose)"]
    for r in report.claims:
        st = r.strengths
        out.append(f"{r.claim_id}: verdict={r.verdict.value} model_proposed="
                   f"{r.model_proposed_verdict.value if r.model_proposed_verdict else '-'} type={r.claim_type.value} "
                   f"time_sensitive={r.time_sensitive}")
        if st:
            out.append(f"  strengths: support={st.support:.3f} refute={st.refute:.3f} context={st.context:.3f}")
        out.append(f"  retrieval rounds={r.retrieval_rounds} search credits={r.search_credits} "
                   f"sources considered={r.sources_considered} rejected model citations={r.rejected_citations}")
        for cap in r.score_caps:
            out.append(_wrap(f"cap: {cap}", "  ", "    "))
        if r.failure_reason:
            out.append(_wrap(f"failure: {r.failure_reason}", "  ", "    "))
        if r.correction_note:
            out.append(_wrap(f"correction note: {r.correction_note}", "  ", "    "))
        for note in r.notes:
            out.append(_wrap(f"note: {note}", "  ", "    "))
    for d in report.dropped_claims:
        out.append(_wrap(f"dropped: {d.statement} ({d.reason})", "", "  "))
    s = report.stats
    out.append(f"run: {s.llm_calls} model call(s), {s.search_credits_used}/{s.search_credit_budget} search credits, "
               f"model {s.model or 'n/a'}, recency as of {report.as_of}"
               + (f", fallback {s.fallback_model} served {s.fallback_calls} call(s)" if s.fallback_model else ""))
    return out


def render_text(report: Report, verbose: bool = False) -> str:
    claims = report.claims
    out = ["FACT-CHECK REPORT", ""]
    out.append(f"OVERALL VERDICT: {report.summary_label.value.replace('_', ' ')}")
    out.append(_wrap(SUMMARY_SENTENCE.get(report.summary_label, report.summary_text), ""))
    if claims:
        counts: dict[str, int] = {}
        for r in claims:
            counts[_VERDICT_TEXT[r.verdict].lower()] = counts.get(_VERDICT_TEXT[r.verdict].lower(), 0) + 1
        out.append(f"{_plural(len(claims), 'claim')}: " + ", ".join(f"{n} {v}" for v, n in counts.items()))
    for n, r in enumerate(claims, start=1):
        out.append("")
        out += render_claim(r, n, len(claims), report.score_disclaimer, verbose)

    notices = []
    if report.over_limit_claims:
        notices.append(f"Not checked: {_plural(len(report.over_limit_claims), 'claim')} over the per-run limit "
                       "(raise it with --max-claims):")
        notices += [f"  - {d.statement}" for d in report.over_limit_claims]
    if report.dropped_claims and not verbose:
        n = len(report.dropped_claims)
        notices.append(f"Skipped: {_plural(n, 'extracted statement')} failed validation and "
                       f"{_verb(n, 'was', 'were')} not checked (details with --verbose).")
    for w in report.warnings:
        if "dropped by validation" in w or "over the per-run limit" in w:
            continue  # already shown above in plain language
        notices.append(f"Warning: {w}.")
    if notices:
        out.append("")
        out += [_wrap(n, "", "  ") for n in notices]

    out.append("")
    sources = sum(r.sources_considered for r in claims)
    out.append(f"Sources retrieved: {sources}   Processing time: {report.stats.elapsed_s:.1f} s")
    out.append(_wrap("Verdicts are generated automatically from web sources and may be wrong. "
                     "Check the cited sources before relying on them.", ""))
    if verbose:
        out += _diagnostics(report)
    return "\n".join(out)


SUMMARY_SENTENCE = {
    SummaryLabel.CONTAINS_FALSE_CLAIMS: "At least one claim is contradicted by the evidence.",
    SummaryLabel.CONTAINS_MISLEADING_CLAIMS: "No claim was refuted, but at least one is misleading.",
    SummaryLabel.ALL_CHECKED_CLAIMS_SUPPORTED: "Every checked claim is supported by the evidence.",
    SummaryLabel.PARTIALLY_RESOLVED: "Some claims could not be resolved; see the details below.",
    SummaryLabel.NOTHING_TO_CHECK: "The text contains no checkable factual claims.",
}


def render_json(report: Report) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
