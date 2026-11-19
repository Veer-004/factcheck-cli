// Mirrors src/factcheck_api/schemas.py (FactCheckResponse) and the parts of the module reports the
// UI reads. Everything optional in the backend is optional (or nullable) here.

import type { Mode } from "./config";

export const VERDICTS = [
  "SUPPORTED",
  "REFUTED",
  "MISLEADING",
  "CONFLICTING",
  "INSUFFICIENT_EVIDENCE",
  "NOT_CHECKABLE",
  "RETRIEVAL_FAILED",
  "PROCESSING_FAILED",
] as const;
export type Verdict = (typeof VERDICTS)[number];

export interface Summary {
  label: string | null;
  text: string | null;
  total_claims: number;
  supported: number;
  refuted: number;
  misleading: number;
  conflicting: number;
  insufficient_evidence: number;
  not_checkable: number;
  retrieval_failed: number;
  processing_failed: number;
  over_limit_claims: number;
  dropped_claims: number;
}

export interface Citation {
  role: "supporting" | "refuting" | "context";
  evidence_id: string;
  url: string;
  title: string;
  published_date: string | null;
  quote: string;
  stance: string;
  directness: string;
  authority: string;
  snippet_only: boolean;
  fact_check_site: boolean;
  recent: boolean | null;
  weight: number;
}

export interface CitationValidation {
  rejected_citations: number;
  sources_considered: number;
  retrieval_rounds: number;
  search_credits: number;
}

export interface TimeSpan {
  modality: "speech" | "screen";
  start: number;
  end: number;
  word_level: boolean;
  text: string;
  confidence: number;
  reliable: boolean;
  segment_ids: number[];
  observation_id: number | null;
  frame_times: number[];
}

export interface ClaimResult {
  claim_id: string;
  claim: string;
  source_excerpt: string;
  claim_type: string;
  checkability: string;
  time_sensitive: boolean;
  verdict: string; // a Verdict; kept as string so an unknown future value still renders
  evidence_score: number | null;
  support_score: number | null;
  refute_score: number | null;
  score_caps: string[];
  strengths: Record<string, number> | null;
  explanation: string;
  rationale: string;
  model_proposed_verdict: string | null;
  correction: string | null;
  correction_note: string | null;
  failure_reason: string | null;
  notes: string[];
  citations: Citation[];
  citation_validation: CitationValidation;
  timestamps: TimeSpan[];
  provenance: Record<string, unknown> | null;
}

export interface DroppedClaim {
  statement: string;
  source_excerpt: string;
  reason: string;
}

export interface FactCheckResponse {
  status: "completed" | "not_checked";
  input_type: Mode;
  filename: string | null;
  summary: Summary;
  claims: ClaimResult[];
  dropped_claims: DroppedClaim[];
  over_limit_claims: DroppedClaim[];
  not_checked_reason: string | null;
  warnings: string[];
  notes: string[];
  processing_seconds: number;
  /** The plan used for this run (server-side AI plan, or attached by the page for text / manual runs). */
  run_plan?: RunPlan | null;
  report: Record<string, unknown>;
}

export interface HealthResponse {
  status: "ok";
  version: string;
  modules: Record<string, { available: boolean; missing: string[] }>;
  fact_check_configured: boolean;
}

export interface ApiErrorBody {
  status: "error";
  error: { code: string; message: string; details: Array<Record<string, unknown>> };
  request_id: string | null;
}

// ---- module-report fragments read for media context (all defensive: may be absent) ----

export interface OcrLine {
  line_no: number;
  block_no: number;
  text: string;
  score: number;
  bbox: [number, number, number, number];
  reliable: boolean;
}

export interface TranscriptSegment {
  segment_id: number;
  start: number;
  end: number;
  text: string;
  confidence: number;
  reliable: boolean;
}

export interface ScreenObservation {
  observation_id: number;
  text: string;
  min_score: number;
  reliable: boolean;
  frame_times: number[];
}

// ---- run planner (POST /api/v1/plan; see src/factcheck_api/planner.py) ----

export interface RunPlan {
  mode: "text" | "image" | "video" | "audio";
  max_claims: number;
  search_budget: number;
  estimated_claims: number;
  estimated_credits: number;
  basis: "agent" | "text" | "duration" | "typical" | "default";
  /** agent = the AI planning agent decided; rules = rule-based fallback; manual = set by the user (client-side only). */
  planner: "agent" | "rules" | "manual";
  model: string | null;
  /** For files: the AI agent decides on the server after reading the extracted text; numbers are a preview. */
  deferred: boolean;
  statements: string[];
  credits_left: number | null;
  credits_limit: number | null;
  reasons: string[];
  warnings: string[];
}

/** Settings sent with a fact-check request (all optional on the server). `auto_plan` lets the server's AI
 *  planning agent choose the values that are not given. */
export interface RunOptions {
  max_claims?: number;
  search_budget?: number;
  auto_plan?: boolean;
}
