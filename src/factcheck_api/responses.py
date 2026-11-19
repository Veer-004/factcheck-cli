"""Response layer: module reports -> ``FactCheckResponse``.

Only copies and re-arranges what the modules produced; nothing is computed
except verdict counts and the explanation sentence, which comes from the shared
report renderer (``factcheck.report``) so it matches the CLI word for word.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from factcheck.models import ClaimResult as ModuleClaim
from factcheck.models import Report, Verdict
from factcheck.report import _explanation  # the CLI's explanation sentence (single source of truth)

from .schemas import Citation, CitationValidation, ClaimResult, DroppedClaim, FactCheckResponse, Summary, TimeSpan

_COUNT_FIELD = {
    Verdict.SUPPORTED: "supported", Verdict.REFUTED: "refuted", Verdict.MISLEADING: "misleading",
    Verdict.CONFLICTING: "conflicting", Verdict.INSUFFICIENT_EVIDENCE: "insufficient_evidence",
    Verdict.NOT_CHECKABLE: "not_checkable", Verdict.RETRIEVAL_FAILED: "retrieval_failed",
    Verdict.PROCESSING_FAILED: "processing_failed",
}


def scrub(data: Any, replacements: dict[str, str]) -> Any:
    """Replace temporary paths everywhere in a JSON-able structure."""
    if not replacements:
        return data
    raw = json.dumps(data, ensure_ascii=False)
    for old, new in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
        if old:
            raw = raw.replace(json.dumps(old, ensure_ascii=False)[1:-1], json.dumps(new, ensure_ascii=False)[1:-1])
    return json.loads(raw)


def _summary(report: Optional[Report]) -> Summary:
    if report is None:
        return Summary()
    s = Summary(label=report.summary_label.value, text=report.summary_text, total_claims=len(report.claims),
                over_limit_claims=len(report.over_limit_claims), dropped_claims=len(report.dropped_claims))
    for r in report.claims:
        name = _COUNT_FIELD[r.verdict]
        setattr(s, name, getattr(s, name) + 1)
    return s


def _citations(r: ModuleClaim) -> list[Citation]:
    out = []
    for role, cites in (("supporting", r.supporting), ("refuting", r.refuting), ("context", r.context)):
        for c in cites:
            out.append(Citation(role=role, evidence_id=c.evidence_id, url=c.url, title=c.title,
                                published_date=c.published_date, quote=c.quote, stance=c.stance.value,
                                directness=c.directness.value, authority=c.authority.value,
                                snippet_only=c.snippet_only, fact_check_site=c.fact_check_site, recent=c.recent,
                                weight=c.weight))
    return out


def _claim(r: ModuleClaim, timestamps: list[TimeSpan], provenance: Optional[dict]) -> ClaimResult:
    return ClaimResult(
        claim_id=r.claim_id, claim=r.statement, source_excerpt=r.source_excerpt, claim_type=r.claim_type.value,
        checkability=r.checkability.value, time_sensitive=r.time_sensitive, verdict=r.verdict.value,
        evidence_score=r.score, support_score=r.support_score, refute_score=r.refute_score, score_caps=r.score_caps,
        strengths=r.strengths.model_dump() if r.strengths else None, explanation=_explanation(r),
        rationale=r.rationale, model_proposed_verdict=r.model_proposed_verdict.value if r.model_proposed_verdict else None,
        correction=r.correction, correction_note=r.correction_note, failure_reason=r.failure_reason, notes=r.notes,
        citations=_citations(r),
        citation_validation=CitationValidation(rejected_citations=r.rejected_citations,
                                               sources_considered=r.sources_considered,
                                               retrieval_rounds=r.retrieval_rounds, search_credits=r.search_credits),
        timestamps=timestamps, provenance=provenance)


def _base(input_type: str, report: Optional[Report], claims: list[ClaimResult], full: dict, *,
          filename: Optional[str], seconds: float, not_checked_reason: Optional[str] = None,
          warnings: Optional[list[str]] = None, notes: Optional[list[str]] = None) -> FactCheckResponse:
    dropped = [DroppedClaim(**d.model_dump()) for d in report.dropped_claims] if report else []
    over = [DroppedClaim(**d.model_dump()) for d in report.over_limit_claims] if report else []
    all_warnings = list(warnings or []) + (list(report.warnings) if report else [])
    all_notes = list(notes or []) + ([f"Evidence score: {report.score_disclaimer}"] if report else [])
    return FactCheckResponse(
        status="completed" if report is not None else "not_checked", input_type=input_type, filename=filename,
        summary=_summary(report), claims=claims, dropped_claims=dropped, over_limit_claims=over,
        not_checked_reason=not_checked_reason, warnings=all_warnings, notes=all_notes,
        processing_seconds=round(seconds, 2), report=full)


# ---------------------------------------------------------------- per module


def from_text(report: Report, seconds: float) -> FactCheckResponse:
    claims = [_claim(r, [], None) for r in report.claims]
    return _base("text", report, claims, report.model_dump(mode="json"), filename=None, seconds=seconds)


def from_image(ir: Any, filename: str, seconds: float, replacements: dict[str, str]) -> FactCheckResponse:
    full = scrub(ir.model_dump(mode="json"), replacements)
    full["extraction"]["image"] = filename
    traces = {t.claim_id: t for t in ir.claim_traces}
    claims = []
    report = ir.fact_check
    for r in report.claims if report else []:
        t = traces.get(r.claim_id)
        prov = {"modality": "image", **t.model_dump(mode="json", exclude={"claim_id"})} if t else None
        claims.append(_claim(r, [], prov))
    return _base("image", report, claims, full, filename=filename, seconds=seconds,
                 not_checked_reason=ir.not_checked_reason, warnings=list(ir.extraction.warnings),
                 notes=[ir.ocr_confidence_note])


def from_video(vr: Any, filename: str, seconds: float, replacements: dict[str, str]) -> FactCheckResponse:
    full = scrub(vr.model_dump(mode="json"), replacements)
    full["extraction"]["media"]["path"] = filename
    sources = {s.claim_id: s for s in vr.claim_sources}
    claims = []
    report = vr.fact_check
    for r in report.claims if report else []:
        src = sources.get(r.claim_id)
        spans = [TimeSpan(modality=s.modality.value, start=s.start, end=s.end, word_level=s.word_level, text=s.text,
                          confidence=s.confidence, reliable=s.reliable,
                          segment_ids=[s.segment_id] if s.segment_id is not None else [],
                          observation_id=s.observation_id, frame_times=s.frame_times)
                 for s in (src.spans if src else [])]
        prov = {"modality": "video", **scrub(src.model_dump(mode="json", exclude={"claim_id"}), replacements)} if src else None
        claims.append(_claim(r, spans, prov))
    return _base("video", report, claims, full, filename=filename, seconds=seconds,
                 not_checked_reason=vr.not_checked_reason, warnings=list(vr.extraction.warnings),
                 notes=[vr.timestamp_note, vr.confidence_note])


def from_audio(ar: Any, filename: str, seconds: float, replacements: dict[str, str]) -> FactCheckResponse:
    full = scrub(ar.model_dump(mode="json"), replacements)
    full["media"]["path"] = filename
    sources = {s.claim_id: s for s in ar.claim_sources}
    claims = []
    report = ar.fact_check
    for r in report.claims if report else []:
        src = sources.get(r.claim_id)
        spans = [TimeSpan(modality="speech", start=s.start, end=s.end, word_level=s.word_level, text=s.transcript,
                          confidence=s.confidence, reliable=s.reliable, segment_ids=s.segment_ids)
                 for s in (src.spans if src else [])]
        prov = {"modality": "audio", **src.model_dump(mode="json", exclude={"claim_id"})} if src else None
        claims.append(_claim(r, spans, prov))
    return _base("audio", report, claims, full, filename=filename, seconds=seconds,
                 not_checked_reason=ar.not_checked_reason, warnings=list(ar.warnings),
                 notes=[ar.timestamp_note, ar.confidence_note])
