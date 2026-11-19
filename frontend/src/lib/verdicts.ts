// Visual language for the backend's verdict values (factcheck.models.Verdict).

import { PALETTE } from "./palette";

export interface VerdictStyle {
  label: string;
  color: string; // hex, used for text, borders and the distribution bar
  description: string;
}

export const VERDICT_STYLES: Record<string, VerdictStyle> = {
  SUPPORTED: { label: "Supported", color: PALETTE.supported, description: "The retrieved evidence supports the claim." },
  REFUTED: { label: "Refuted", color: PALETTE.refuted, description: "The retrieved evidence contradicts the claim." },
  MISLEADING: { label: "Misleading", color: PALETTE.warn, description: "Partly accurate but leaves out or distorts context." },
  CONFLICTING: { label: "Conflicting", color: PALETTE.conflict, description: "Credible sources disagree." },
  INSUFFICIENT_EVIDENCE: { label: "Insufficient evidence", color: PALETTE.muted,
                           description: "Not enough reliable evidence to confirm or refute." },
  NOT_CHECKABLE: { label: "Not checkable", color: PALETTE.neutral, description: "An opinion, prediction or vague statement." },
  RETRIEVAL_FAILED: { label: "Search failed", color: PALETTE.failed, description: "A technical problem, not a judgement." },
  PROCESSING_FAILED: { label: "Not processed", color: PALETTE.failed, description: "A technical problem, not a judgement." },
};

export function verdictStyle(verdict: string): VerdictStyle {
  return VERDICT_STYLES[verdict] ?? { label: verdict.replace(/_/g, " ").toLowerCase(), color: PALETTE.neutral, description: "" };
}

export const SUMMARY_LABELS: Record<string, string> = {
  CONTAINS_FALSE_CLAIMS: "Contains false claims",
  CONTAINS_MISLEADING_CLAIMS: "Contains misleading claims",
  ALL_CHECKED_CLAIMS_SUPPORTED: "All checked claims supported",
  PARTIALLY_RESOLVED: "Partially resolved",
  NOTHING_TO_CHECK: "Nothing to check",
};

export const SUMMARY_COUNT_KEYS: Array<[string, keyof import("./types").Summary]> = [
  ["SUPPORTED", "supported"],
  ["REFUTED", "refuted"],
  ["MISLEADING", "misleading"],
  ["CONFLICTING", "conflicting"],
  ["INSUFFICIENT_EVIDENCE", "insufficient_evidence"],
  ["NOT_CHECKABLE", "not_checkable"],
  ["RETRIEVAL_FAILED", "retrieval_failed"],
  ["PROCESSING_FAILED", "processing_failed"],
];
