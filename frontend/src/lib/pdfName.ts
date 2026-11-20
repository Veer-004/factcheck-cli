// File names for the downloaded PDF report. Kept out of pdfReport.ts so the dialog does not load jsPDF.

import type { FactCheckResponse } from "./types";

export const MAX_NAME_CHARS = 80;
// Characters Windows, macOS and Linux reject in file names, plus control characters.
// eslint-disable-next-line no-control-regex
const FORBIDDEN = /[\\/:*?"<>|\u0000-\u001f]/g;
const RESERVED = /^(con|prn|aux|nul|com\d|lpt\d)$/i;

const TYPE_LABEL: Record<FactCheckResponse["input_type"], string> = {
  text: "Text", image: "Image", video: "Video", audio: "Audio",
};

/** A readable default, e.g. "FactCheck - Video - agni-launch - 20 Nov 2026". */
export function suggestedPdfName(r: FactCheckResponse, now = new Date()): string {
  const date = now.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
  const source = r.filename ? r.filename.replace(/\.[^.]+$/, "") : null;
  const parts = ["FactCheck", TYPE_LABEL[r.input_type], source, date].filter(Boolean) as string[];
  return cleanPdfName(parts.join(" - ")) || "FactCheck report";
}

/** Make a user-typed name safe to save: drop forbidden characters, a typed ".pdf", and trailing dots/spaces. */
export function cleanPdfName(raw: string): string {
  let name = raw.replace(FORBIDDEN, "").replace(/\s+/g, " ").trim();
  name = name.replace(/\.pdf$/i, "").replace(/[. ]+$/, "").trim();
  if (RESERVED.test(name)) name = `${name}-report`;
  return name.slice(0, MAX_NAME_CHARS).trim();
}

/** What is wrong with the name, or null when it can be saved. */
export function pdfNameProblem(raw: string): string | null {
  if (!raw.trim()) return "Enter a name for the report.";
  if (!cleanPdfName(raw)) return "Use at least one letter or number.";
  if (raw.replace(/\.pdf$/i, "").trim().length > MAX_NAME_CHARS) return `Keep the name under ${MAX_NAME_CHARS} characters.`;
  return null;
}

/** True when cleaning changed what the user typed (shown as a hint, not an error). */
export function nameWasAdjusted(raw: string): boolean {
  const base = raw.replace(/\.pdf$/i, "");
  return new RegExp(FORBIDDEN.source).test(base) || /\S[. ]*\.$/.test(base.trim());
}
