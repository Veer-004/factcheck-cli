"""Request and response schemas (the contract the frontend relies on).

All four endpoints return ``FactCheckResponse``. Field names are the same for
every input type; fields that do not apply are ``null`` or ``[]``:

* ``timestamps`` is ``[]`` for text and image (they have no time axis);
* ``provenance`` is ``null`` for text;
* ``filename`` is ``null`` for text.

Verdict labels, scores, explanations, citations and validation counts are
copied from the module's report without reinterpretation, and the complete
module report is included under ``report`` (only temporary file paths are
replaced with the uploaded file's name).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

InputType = Literal["text", "image", "video", "audio"]


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text or claim(s) to fact-check (UTF-8).",
                      examples=["The Eiffel Tower was completed in 1889. Water boils at 50 degrees Celsius."])
    max_claims: Optional[int] = Field(None, ge=1, le=12, description="Checkable claims to verify (default 6).")
    search_budget: Optional[int] = Field(None, ge=0, le=40, description="Web-search credits for this run (default 40).")
    auto_plan: bool = Field(False, description="Let the AI planning agent choose `max_claims` / `search_budget` "
                                               "(only for values that are not given).")


class Summary(BaseModel):
    """Verdict counts over ``claims`` plus the module's overall label."""

    label: Optional[str] = Field(None, description="Overall label, e.g. `CONTAINS_FALSE_CLAIMS`, "
                                                   "`ALL_CHECKED_CLAIMS_SUPPORTED`, `PARTIALLY_RESOLVED`, "
                                                   "`NOTHING_TO_CHECK`; null when nothing was fact-checked.")
    text: Optional[str] = Field(None, description="One-sentence overall summary from the report.")
    total_claims: int = 0
    supported: int = 0
    refuted: int = 0
    misleading: int = 0
    conflicting: int = 0
    insufficient_evidence: int = 0
    not_checkable: int = 0
    retrieval_failed: int = 0
    processing_failed: int = 0
    over_limit_claims: int = Field(0, description="Checkable claims not verified because of the per-run limit.")
    dropped_claims: int = Field(0, description="Extracted claims removed by validation (see `dropped_claims`).")


class Citation(BaseModel):
    """A quote validated against the retrieved page (verbatim) and accepted for the verdict."""

    role: Literal["supporting", "refuting", "context"]
    evidence_id: str
    url: str
    title: str = ""
    published_date: Optional[str] = None
    quote: str
    stance: str
    directness: str
    authority: str
    snippet_only: bool = False
    fact_check_site: bool = False
    recent: Optional[bool] = None
    weight: float = 0.0


class CitationValidation(BaseModel):
    rejected_citations: int = Field(0, description="Model citations rejected by validation (not verbatim, "
                                                   "off-topic, wrong figure, ...).")
    sources_considered: int = 0
    retrieval_rounds: int = 0
    search_credits: int = 0


class TimeSpan(BaseModel):
    """Where and when a claim's source text was spoken (video/audio) or shown on screen (video)."""

    modality: Literal["speech", "screen"]
    start: float = Field(description="Seconds from the start of the media (approximate).")
    end: float
    word_level: bool = Field(False, description="True when narrowed to the claim's own words.")
    text: str = Field(description="The transcript segment(s) or on-screen text the claim came from.")
    confidence: float = Field(description="Recognition confidence (speech or OCR), not factual confidence.")
    reliable: bool = Field(description="Confidence at or above the module's threshold.")
    segment_ids: list[int] = Field(default_factory=list)
    observation_id: Optional[int] = Field(None, description="On-screen text block id (video).")
    frame_times: list[float] = Field(default_factory=list, description="Sampled frames where the text was seen.")


class ClaimResult(BaseModel):
    claim_id: str
    claim: str = Field(description="The atomic statement that was checked.")
    source_excerpt: str = Field(description="The exact words in the input the claim was taken from.")
    claim_type: str
    checkability: str
    time_sensitive: bool = False
    verdict: str = Field(description="SUPPORTED | REFUTED | MISLEADING | CONFLICTING | INSUFFICIENT_EVIDENCE | "
                                     "NOT_CHECKABLE | RETRIEVAL_FAILED | PROCESSING_FAILED (unchanged).")
    evidence_score: Optional[int] = Field(None, description="Experimental evidence-strength rubric 0-100; not a "
                                                            "probability. Null when not scored.")
    support_score: Optional[int] = None
    refute_score: Optional[int] = None
    score_caps: list[str] = Field(default_factory=list)
    strengths: Optional[dict[str, float]] = None
    explanation: str = Field(description="Plain-language explanation, identical to the CLI report.")
    rationale: str = ""
    model_proposed_verdict: Optional[str] = None
    correction: Optional[str] = None
    correction_note: Optional[str] = None
    failure_reason: Optional[str] = None
    notes: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    citation_validation: CitationValidation
    timestamps: list[TimeSpan] = Field(default_factory=list)
    provenance: Optional[dict[str, Any]] = Field(
        None, description="Modality-specific trace: image = OCR lines/blocks and OCR confidence; video/audio = "
                          "the module's claim source (spans, merged duplicates). Null for text.")


class DroppedClaim(BaseModel):
    statement: str
    source_excerpt: str = ""
    reason: str


class FactCheckResponse(BaseModel):
    status: Literal["completed", "not_checked"] = Field(
        description="`completed`: the fact-check ran (whatever the verdicts). `not_checked`: the input was "
                    "processed but nothing could be checked (no text/speech, not English, everything below the "
                    "recognition threshold); see `not_checked_reason`.")
    input_type: InputType
    filename: Optional[str] = Field(None, description="The uploaded file's name (display only).")
    summary: Summary
    claims: list[ClaimResult] = Field(default_factory=list)
    dropped_claims: list[DroppedClaim] = Field(default_factory=list)
    over_limit_claims: list[DroppedClaim] = Field(default_factory=list)
    not_checked_reason: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="How to read scores and confidences.")
    processing_seconds: float
    run_plan: Optional["PlanResponse"] = Field(
        None, description="The run plan the AI planning agent chose on the server (`auto_plan`); null otherwise.")
    report: dict[str, Any] = Field(description="The complete module report (same as the CLI's --json-out), "
                                               "with temporary file paths replaced by the file name.")


class PlanRequest(BaseModel):
    mode: InputType
    text: Optional[str] = Field(None, max_length=20000, description="The text to check (text mode).")
    duration_s: Optional[float] = Field(None, ge=0, le=86400,
                                        description="Media length in seconds (audio/video), when known.")
    use_ai: bool = Field(False, description="Text only: ask the AI planning agent (one Gemini request) instead of "
                                            "the instant rule-based estimate. For files use "
                                            "`POST /api/v1/plan/{image|video|audio}`.")


class PlanResponse(BaseModel):
    """Suggested run settings. Pass `max_claims` and `search_budget` to the fact-check endpoint."""

    mode: InputType
    max_claims: int = Field(ge=1, le=12)
    search_budget: int = Field(ge=0, le=40)
    estimated_claims: int = Field(description="Checkable claims the planner expects in the input.")
    estimated_credits: int = Field(description="Tavily credits a typical run of this plan uses.")
    basis: Literal["agent", "text", "duration", "typical", "default"] = Field(
        description="What the estimate is based on: `agent` = the AI planner read the content.")
    planner: Literal["agent", "rules"] = Field(
        "rules", description="`agent`: the AI planning agent decided; `rules`: the rule-based fallback.")
    model: Optional[str] = Field(None, description="The model the planning agent used.")
    deferred: bool = Field(False, description="For files: the AI planner decides after reading the extracted "
                                              "text on the server; these numbers are only a preview.")
    statements: list[str] = Field(default_factory=list,
                                  description="The checkable statements the agent found (text and media).")
    credits_left: Optional[int] = Field(None, description="Tavily credits left this month; null if unknown.")
    credits_limit: Optional[int] = Field(None, description="The Tavily plan's monthly credits; null if unknown.")
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ModuleStatus(BaseModel):
    available: bool
    missing: list[str] = Field(default_factory=list, description="Python packages that are not installed.")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    modules: dict[str, ModuleStatus]
    fact_check_configured: bool = Field(description="Whether the credentials needed for fact-checking are set "
                                                    "(their values are never shown).")
