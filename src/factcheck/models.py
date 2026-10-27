"""Typed data models.

Two families of models live here:

* ``LLM*`` models describe what the language model is asked to return. They are
  deliberately permissive (the API strips most string-length constraints), and
  everything in them is treated as untrusted until validated in code.
* Application models (``Claim``, ``EvidencePassage``, ``ClaimResult``,
  ``Report`` ...) are produced only by deterministic application code.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    MISLEADING = "MISLEADING"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_CHECKABLE = "NOT_CHECKABLE"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    PROCESSING_FAILED = "PROCESSING_FAILED"


RESOLVED_VERDICTS = {Verdict.SUPPORTED, Verdict.REFUTED, Verdict.MISLEADING}
UNRESOLVED_VERDICTS = {
    Verdict.CONFLICTING,
    Verdict.INSUFFICIENT_EVIDENCE,
    Verdict.RETRIEVAL_FAILED,
    Verdict.PROCESSING_FAILED,
}
TECHNICAL_FAILURES = {Verdict.RETRIEVAL_FAILED, Verdict.PROCESSING_FAILED}


class SummaryLabel(str, Enum):
    CONTAINS_FALSE_CLAIMS = "CONTAINS_FALSE_CLAIMS"
    CONTAINS_MISLEADING_CLAIMS = "CONTAINS_MISLEADING_CLAIMS"
    ALL_CHECKED_CLAIMS_SUPPORTED = "ALL_CHECKED_CLAIMS_SUPPORTED"
    PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
    NOTHING_TO_CHECK = "NOTHING_TO_CHECK"


class Checkability(str, Enum):
    CHECKABLE = "CHECKABLE"
    OPINION = "OPINION"
    PREDICTION = "PREDICTION"
    VAGUE = "VAGUE"
    OTHER_NOT_CHECKABLE = "OTHER_NOT_CHECKABLE"


class ClaimType(str, Enum):
    FACTUAL = "FACTUAL"
    NUMERIC = "NUMERIC"
    COMPARATIVE = "COMPARATIVE"
    CAUSAL = "CAUSAL"
    QUOTE_ATTRIBUTION = "QUOTE_ATTRIBUTION"
    OTHER = "OTHER"


class Stance(str, Enum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    # Passage supplies material context the claim omits or distorts.
    MISSING_CONTEXT = "MISSING_CONTEXT"
    IRRELEVANT = "IRRELEVANT"


class Directness(str, Enum):
    EXPLICIT = "EXPLICIT"
    PARTIAL = "PARTIAL"
    INFERENTIAL = "INFERENTIAL"


class Authority(str, Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    OTHER = "OTHER"


# --------------------------------------------------------------------------
# LLM output schemas (untrusted)
# --------------------------------------------------------------------------


class LLMClaim(BaseModel):
    statement: str = Field(description="One atomic, self-contained, checkable statement.")
    source_excerpt: str = Field(
        description="Exact contiguous text copied verbatim from the user's input that this claim comes from."
    )
    checkability: Checkability
    claim_type: ClaimType
    time_sensitive: bool = Field(description="True if the truth of the claim can change over time.")
    time_anchor: Optional[str] = Field(
        default=None, description="Date or period the claim refers to, if stated or implied; else null."
    )
    primary_queries: list[str] = Field(default_factory=list, description="Up to 2 web search queries.")
    fallback_queries: list[str] = Field(
        default_factory=list, description="Up to 2 alternative queries, phrased differently."
    )


class LLMExtraction(BaseModel):
    claims: list[LLMClaim] = Field(default_factory=list)


class LLMNumericComparison(BaseModel):
    claim_value: Optional[float] = None
    evidence_value: Optional[float] = None
    claim_unit: Optional[str] = None
    evidence_unit: Optional[str] = None
    same_entity: bool = False
    same_metric: bool = False
    same_unit: bool = False
    same_scope: bool = False
    same_period: bool = False
    claim_is_approximate: bool = False


class LLMAssessment(BaseModel):
    evidence_id: str
    stance: Stance
    directness: Directness
    quote: str = Field(description="Verbatim substring of the passage that justifies the stance.")
    numeric_comparison: Optional[LLMNumericComparison] = None
    explanation: str = ""


class LLMCorrection(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class LLMJudgment(BaseModel):
    assessments: list[LLMAssessment] = Field(default_factory=list)
    proposed_verdict: Verdict
    rationale: str = ""
    correction: Optional[LLMCorrection] = None


# --------------------------------------------------------------------------
# Application models (produced by deterministic code)
# --------------------------------------------------------------------------


class Claim(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    index: int
    statement: str
    source_excerpt: str
    checkability: Checkability
    claim_type: ClaimType
    time_sensitive: bool = False
    time_anchor: Optional[str] = None
    primary_queries: list[str] = Field(default_factory=list)
    fallback_queries: list[str] = Field(default_factory=list)


class DroppedClaim(BaseModel):
    statement: str
    source_excerpt: str = ""
    reason: str


class SearchResult(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    published_date: Optional[str] = None
    score: Optional[float] = None


class ExtractedPage(BaseModel):
    url: str
    raw_content: str = ""


class EvidencePassage(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    url: str
    domain: str
    title: str = ""
    published_date: Optional[str] = None
    text: str
    snippet_only: bool = False
    injection_flagged: bool = False
    authority: Authority = Authority.OTHER
    fact_check_site: bool = False
    cluster_id: str
    relevance: float = 0.0


class Citation(BaseModel):
    evidence_id: str
    url: str
    title: str = ""
    published_date: Optional[str] = None
    quote: str
    stance: Stance
    directness: Directness
    authority: Authority
    cluster_id: str
    snippet_only: bool = False
    injection_flagged: bool = False
    fact_check_site: bool = False
    recent: Optional[bool] = None
    weight: float = 0.0


class Strengths(BaseModel):
    support: float = 0.0
    refute: float = 0.0
    context: float = 0.0


class ClaimResult(BaseModel):
    claim_id: str
    index: int
    statement: str
    source_excerpt: str
    claim_type: ClaimType
    checkability: Checkability
    time_sensitive: bool = False
    verdict: Verdict
    rationale: str = ""
    model_proposed_verdict: Optional[Verdict] = None
    supporting: list[Citation] = Field(default_factory=list)
    refuting: list[Citation] = Field(default_factory=list)
    context: list[Citation] = Field(default_factory=list)
    correction: Optional[str] = None
    correction_note: Optional[str] = None
    score: Optional[int] = None
    score_caps: list[str] = Field(default_factory=list)
    support_score: Optional[int] = None
    refute_score: Optional[int] = None
    strengths: Optional[Strengths] = None
    failure_reason: Optional[str] = None
    notes: list[str] = Field(default_factory=list)
    rejected_citations: int = 0
    # Passages shown to the judge (JSON output / audit; not printed in the terminal report).
    evidence: list[EvidencePassage] = Field(default_factory=list)
    retrieval_rounds: int = 0
    search_credits: int = 0
    sources_considered: int = 0


class RunStats(BaseModel):
    llm_calls: int = 0
    search_credits_used: int = 0
    search_credit_budget: int = 0
    elapsed_s: float = 0.0
    model: str = ""
    # Local fallback: which model (if configured) and how many calls it actually served.
    fallback_model: str = ""
    fallback_calls: int = 0


SCORE_DISCLAIMER = "experimental rubric; not a probability and not calibrated."


class Report(BaseModel):
    input_chars: int
    summary_label: SummaryLabel
    summary_text: str
    claims: list[ClaimResult] = Field(default_factory=list)
    dropped_claims: list[DroppedClaim] = Field(default_factory=list)
    over_limit_claims: list[DroppedClaim] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: RunStats = Field(default_factory=RunStats)
    score_disclaimer: str = SCORE_DISCLAIMER
    as_of: str = ""
