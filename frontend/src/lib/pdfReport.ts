// The downloadable investigation report, built in the browser from the API response, in the same
// Deep Ocean look as the site: dark pages, glass cards, teal accents, the sonar mark and the same
// verdict colours (src/lib/palette.ts). jsPDF is loaded only when the user downloads a report.

import type { jsPDF as JsPDF } from "jspdf";
import { domainOf, fmtDate, fmtTime, safeHref, titleCase } from "./format";
import { hexToRgb, PALETTE } from "./palette";
import type { Citation, ClaimResult, FactCheckResponse, OcrLine, ScreenObservation, TranscriptSegment } from "./types";
import { SUMMARY_COUNT_KEYS, SUMMARY_LABELS, verdictStyle } from "./verdicts";

type RGB = [number, number, number];
type Style = "normal" | "bold" | "italic" | "bolditalic";

const PAGE_W = 210;
const PAGE_H = 297;
const M = 16; // margin (mm)
const CW = PAGE_W - 2 * M; // content width
const TOP = 26; // content starts below the running header
const BOTTOM = PAGE_H - 18; // and ends above the footer
const PT = 0.3528; // mm per point

const C = {
  abyss: hexToRgb(PALETTE.abyss),
  deep: hexToRgb(PALETTE.deep),
  card: [14, 33, 41] as RGB, // between deep and glass
  cardHi: [18, 44, 52] as RGB,
  edge: [16, 70, 72] as RGB, // teal-tinted hairline on dark
  teal: hexToRgb(PALETTE.teal),
  tealDeep: hexToRgb(PALETTE.tealDeep),
  ice: hexToRgb(PALETTE.ice),
  muted: hexToRgb(PALETTE.muted),
  warn: hexToRgb(PALETTE.warn),
};
const vRGB = (v: string): RGB => hexToRgb(verdictStyle(v).color);

// The built-in PDF fonts cover Latin-1 / Windows-1252 only; map common symbols and drop the rest.
const REPLACE: Record<string, string> = {
  "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...",
  "•": "-", "→": "->", "←": "<-", "≈": "~", "≤": "<=", "≥": ">=", "−": "-",
  " ": " ", " ": " ", " ": " ", "​": "", "′": "'", "″": '"', "₹": "Rs ",
};
function clean(s: string | null | undefined): string {
  if (!s) return "";
  return [...s.normalize("NFC")]
    .map((ch) => REPLACE[ch] ?? (ch.charCodeAt(0) <= 0xff ? ch : "?"))
    .join("")
    .replace(/[\u0000-\u0008\u000B-\u001F]/g, "");
}
const trunc = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 3).trimEnd()}...` : s);

// ---------------------------------------------------------------- drawing helpers

class Writer {
  y = TOP;
  page = 0;
  constructor(public d: JsPDF, private meta: string) {}

  alpha(o: number) {
    const GState = (this.d as unknown as { GState: new (o: object) => unknown }).GState;
    this.d.setGState(new GState({ opacity: o, "stroke-opacity": o }) as never);
  }

  /** Dark page background with a soft teal glow and the running header. */
  paint(cover = false) {
    const d = this.d;
    d.setFillColor(...C.abyss);
    d.rect(0, 0, PAGE_W, PAGE_H, "F");
    this.alpha(cover ? 0.07 : 0.045);
    d.setFillColor(...C.teal);
    d.circle(cover ? 160 : 10, cover ? 70 : -40, cover ? 78 : 62, "F");
    this.alpha(cover ? 0.04 : 0.025);
    d.circle(cover ? 160 : 10, cover ? 70 : -40, cover ? 108 : 92, "F");
    this.alpha(1);
    if (!cover) {
      mark(d, M + 2.6, 12.2, 2.6);
      this.font(7, "bold", C.ice);
      d.setCharSpace(0.5);
      d.text("FACTCHECK", M + 7, 13.3);
      const brandW = d.getTextWidth("FACTCHECK") + 9 * 0.5; // text width + letter spacing
      this.font(7, "normal", C.muted);
      d.setCharSpace(0.5);
      d.text("INVESTIGATION REPORT", M + 7 + brandW + 3, 13.3);
      d.text(clean(this.meta).toUpperCase(), PAGE_W - M, 13.3, { align: "right" });
      d.setCharSpace(0);
      this.alpha(0.6);
      d.setDrawColor(...C.edge);
      d.setLineWidth(0.3);
      d.line(M, 17.5, PAGE_W - M, 17.5);
      this.alpha(1);
    }
  }

  newPage() {
    this.d.addPage();
    this.page++;
    this.paint();
    this.y = TOP;
  }

  ensure(h: number) {
    if (this.y + h > BOTTOM) this.newPage();
  }

  font(size: number, style: Style = "normal", color: RGB = C.ice, family = "helvetica") {
    this.d.setFont(family, style);
    this.d.setFontSize(size);
    this.d.setTextColor(...color);
  }

  lines(s: string, size: number, width: number, style: Style = "normal", family = "helvetica"): string[] {
    this.d.setFont(family, style);
    this.d.setFontSize(size);
    return this.d.splitTextToSize(clean(s), width);
  }

  /** Wrapped paragraph, page-breaking line by line. */
  text(s: string, o: { size?: number; style?: Style; color?: RGB; x?: number; width?: number; family?: string; lead?: number } = {}) {
    const size = o.size ?? 9.5;
    const x = o.x ?? M;
    const lh = size * PT * (o.lead ?? 1.4);
    for (const line of this.lines(s, size, o.width ?? CW - (x - M), o.style, o.family)) {
      this.ensure(lh);
      this.font(size, o.style, o.color ?? C.ice, o.family);
      this.d.text(line, x, this.y + lh * 0.75);
      this.y += lh;
    }
  }

  eyebrow(s: string, color: RGB = C.muted, x = M) {
    this.ensure(14);
    this.font(7, "bold", color);
    this.d.setCharSpace(0.7);
    this.d.text(clean(s).toUpperCase(), x, this.y + 2.6);
    this.d.setCharSpace(0);
    this.y += 5.2;
  }

  /** Glass card: dark fill, teal-tinted hairline, optional verdict accent on the left. */
  card(x: number, y: number, w: number, h: number, accent?: RGB, fill: RGB = C.card) {
    const d = this.d;
    d.setFillColor(...fill);
    d.setDrawColor(...C.edge);
    d.setLineWidth(0.3);
    d.roundedRect(x, y, w, h, 2.6, 2.6, "FD");
    this.alpha(0.28);
    d.setDrawColor(...C.ice);
    d.setLineWidth(0.15);
    d.line(x + 5, y + 0.15, x + Math.min(w * 0.45, 70), y + 0.15); // top light edge
    this.alpha(1);
    if (accent) {
      d.setFillColor(...accent);
      d.roundedRect(x, y + 2, 1.1, h - 4, 0.5, 0.5, "F");
    }
  }

  /** Verdict pill like the site's badge: tinted fill, coloured border and text. */
  pill(label: string, color: RGB, x: number, y: number): number {
    const t = clean(label).toUpperCase();
    this.font(6.8, "bold", color);
    this.d.setCharSpace(0.4);
    const w = this.d.getTextWidth(t) + 0.4 * t.length * 0.35 + 8;
    this.alpha(0.16);
    this.d.setFillColor(...color);
    this.d.roundedRect(x, y, w, 5.6, 2.8, 2.8, "F");
    this.alpha(1);
    this.d.setDrawColor(...color);
    this.d.setLineWidth(0.25);
    this.d.roundedRect(x, y, w, 5.6, 2.8, 2.8, "S");
    this.d.setFillColor(...color);
    this.d.circle(x + 3, y + 2.8, 0.8, "F");
    this.d.text(t, x + 5, y + 3.85);
    this.d.setCharSpace(0);
    return w;
  }

  /** Evidence meter: label, value and a bar in the verdict colour. */
  meter(label: string, value: number, color: RGB, x: number, y: number, w = 34) {
    this.font(6.5, "bold", C.muted);
    this.d.setCharSpace(0.5);
    this.d.text(label.toUpperCase(), x, y + 2);
    this.d.setCharSpace(0);
    this.font(10, "bold", color, "courier");
    this.d.text(String(value), x + w, y + 2.4, { align: "right" });
    this.d.setFillColor(...C.deep);
    this.d.roundedRect(x, y + 4, w, 1.6, 0.8, 0.8, "F");
    this.d.setFillColor(...color);
    this.d.roundedRect(x, y + 4, Math.max(1.6, (w * Math.min(100, value)) / 100), 1.6, 0.8, 0.8, "F");
  }
}

/** The sonar brand mark. */
function mark(d: JsPDF, cx: number, cy: number, r: number) {
  d.setDrawColor(...C.teal);
  d.setLineWidth(r * 0.12);
  d.circle(cx, cy, r, "S");
  d.setLineWidth(r * 0.16);
  d.circle(cx, cy, r * 0.5, "S");
  d.setDrawColor(...C.ice);
  d.line(cx, cy, cx + r * 0.78, cy - r * 0.58);
  d.setFillColor(...C.teal);
  d.circle(cx, cy, r * 0.16, "F");
}

// ---------------------------------------------------------------- cover

function cover(w: Writer, r: FactCheckResponse) {
  const d = w.d;
  const s = r.summary;
  w.paint(true);

  // sonar rings, sweep and blips (decoration)
  const cx = 158, cy = 74;
  for (const [rad, o] of [[62, 0.35], [47, 0.25], [32, 0.2], [17, 0.16]] as const) {
    w.alpha(o);
    d.setDrawColor(...C.teal);
    d.setLineWidth(0.35);
    d.circle(cx, cy, rad, "S");
  }
  w.alpha(0.16);
  d.line(cx - 66, cy, cx + 66, cy);
  d.line(cx, cy - 66, cx, cy + 66);
  w.alpha(0.18);
  d.setFillColor(...C.teal);
  d.triangle(cx, cy, cx + 62, cy - 20, cx + 56, cy + 30, "F");
  w.alpha(1);
  for (const [bx, by] of [[cx + 30, cy - 18], [cx - 22, cy + 26], [cx + 12, cy + 38], [cx - 36, cy - 14]]) {
    d.setFillColor(...C.teal);
    d.circle(bx, by, 1.2, "F");
  }
  d.setFillColor(...C.ice);
  d.circle(cx, cy, 1.6, "F");

  // brand
  mark(d, M + 5, 22, 5);
  w.font(18, "bold", C.ice);
  d.text("FactCheck", M + 13.5, 24.2);
  w.font(7, "bold", C.muted);
  d.setCharSpace(0.8);
  d.text("INVESTIGATION REPORT", M + 13.5, 29.2);
  d.setCharSpace(0);

  // headline
  w.y = 112;
  w.eyebrow(r.status === "completed" ? "Overall result" : "Investigation finished - nothing checkable", C.teal);
  const title = s.label ? SUMMARY_LABELS[s.label] ?? s.label : "No claims could be checked";
  w.text(title, { size: 30, style: "bold", lead: 1.12 });
  w.y += 2;
  if (s.text) w.text(s.text, { size: 10.5, color: C.muted, width: 150 });
  if (r.not_checked_reason) w.text(r.not_checked_reason, { size: 10.5, color: C.muted, width: 150 });

  // verdict distribution
  const parts = SUMMARY_COUNT_KEYS.map(([v, k]) => ({ v, n: s[k] as number })).filter((p) => p.n > 0);
  if (s.total_claims > 0) {
    w.y += 6;
    let x = M;
    d.setFillColor(...C.deep);
    d.roundedRect(M, w.y, CW, 3, 1.5, 1.5, "F");
    for (const p of parts) {
      const wd = (p.n / s.total_claims) * CW;
      d.setFillColor(...vRGB(p.v));
      d.rect(x, w.y, wd, 3, "F");
      x += wd;
    }
    w.y += 7;
    let lx = M;
    for (const p of parts) {
      const label = `${p.n} ${verdictStyle(p.v).label.toLowerCase()}`;
      d.setFillColor(...vRGB(p.v));
      d.circle(lx + 1, w.y + 1.6, 1, "F");
      w.font(8.5, "normal", C.ice);
      d.text(label, lx + 3.5, w.y + 2.6);
      lx += d.getTextWidth(label) + 10;
    }
    w.y += 8;
  }

  // stat cards
  const sources = new Set(r.claims.flatMap((c) => c.citations.map((x) => x.url))).size;
  const stats: Array<[string, string]> = [
    ["Claims checked", String(s.total_claims)],
    ["Processing", `${r.processing_seconds.toFixed(0)} s`],
    ["Sources cited", String(sources)],
    ["Input", titleCase(r.input_type)],
  ];
  const gap = 4;
  const sw = (CW - gap * 3) / 4;
  stats.forEach(([k, v], i) => {
    const x = M + i * (sw + gap);
    w.card(x, w.y, sw, 20);
    w.font(17, "bold", C.ice, "courier");
    d.text(clean(v), x + 4, w.y + 10.5);
    w.font(6.3, "bold", C.muted);
    d.setCharSpace(0.5);
    d.text(k.toUpperCase(), x + 4, w.y + 16);
    d.setCharSpace(0);
  });
  w.y += 27;

  // details
  const rep = r.report as Record<string, any>;
  const model = (rep?.stats?.model ?? rep?.fact_check?.stats?.model) as string | undefined;
  const details: Array<[string, string]> = [
    ...(r.filename ? [["File", r.filename] as [string, string]] : []),
    ["Generated", new Date().toLocaleString()],
    ...(model ? [["AI model", model] as [string, string]] : []),
    ...(s.over_limit_claims ? [["Not verified", `${s.over_limit_claims} claim(s) over the per-run limit`] as [string, string]] : []),
    ...(s.dropped_claims ? [["Removed", `${s.dropped_claims} claim(s) removed by validation`] as [string, string]] : []),
  ];
  for (const [k, v] of details) {
    w.font(7, "bold", C.muted);
    d.setCharSpace(0.5);
    d.text(k.toUpperCase(), M, w.y + 3);
    d.setCharSpace(0);
    w.font(9, "normal", C.ice);
    d.text(trunc(clean(v), 90), M + 30, w.y + 3);
    w.y += 6;
  }

  if (r.warnings.length) {
    w.y += 4;
    const lines = r.warnings.flatMap((x) => w.lines(`- ${x}`, 8.5, CW - 10));
    const h = 10 + lines.length * 4.2;
    w.card(M, w.y, CW, h, C.warn);
    w.y += 4;
    w.eyebrow("Warnings", C.warn, M + 5);
    lines.forEach((l) => {
      w.font(8.5, "normal", C.ice);
      d.text(l, M + 5, w.y + 3);
      w.y += 4.2;
    });
  }

  // footer note on the cover
  w.font(7.5, "normal", C.muted);
  d.text("Verdicts are generated automatically from web sources and can be wrong. Read the cited sources before relying on them.",
         M, PAGE_H - 22, { maxWidth: CW });
}

// ---------------------------------------------------------------- claims

function claimBlock(w: Writer, c: ClaimResult, i: number, total: number) {
  const d = w.d;
  const color = vRGB(c.verdict);
  const titleLines = w.lines(c.claim, 12, CW - 58, "bold");
  const headH = 14 + titleLines.length * 5.6;
  w.ensure(headH + 24);
  const top = w.y;
  w.card(M, top, CW, headH, color, C.cardHi);
  w.font(7, "bold", C.muted);
  d.setCharSpace(0.6);
  d.text(`CLAIM ${String(i + 1).padStart(2, "0")} / ${String(total).padStart(2, "0")}`, M + 6, top + 7);
  d.setCharSpace(0);
  w.pill(verdictStyle(c.verdict).label, color, M + 38, top + 3.4);
  titleLines.forEach((l, k) => {
    w.font(12, "bold", C.ice);
    d.text(l, M + 6, top + 15 + k * 5.6);
  });
  // score
  const mx = PAGE_W - M - 42;
  if (c.evidence_score !== null) w.meter("Evidence", c.evidence_score, color, mx, top + 5);
  else {
    if (c.support_score !== null) w.meter("Support", c.support_score, vRGB("SUPPORTED"), mx, top + 4);
    if (c.refute_score !== null) w.meter("Against", c.refute_score, vRGB("REFUTED"), mx, top + 14);
  }
  w.y = top + headH + 5;

  w.text(c.explanation, { size: 9.5 });

  if (c.correction) {
    w.y += 3;
    const lines = w.lines(c.correction, 9.5, CW - 14);
    const note = c.correction_note ? w.lines(c.correction_note, 7.5, CW - 14) : [];
    const h = 11 + lines.length * 4.6 + note.length * 3.6;
    w.ensure(h);
    const y0 = w.y;
    w.alpha(0.08);
    d.setFillColor(...C.teal);
    d.roundedRect(M, y0, CW, h, 2.6, 2.6, "F");
    w.alpha(1);
    d.setDrawColor(...C.tealDeep);
    d.setLineWidth(0.3);
    d.roundedRect(M, y0, CW, h, 2.6, 2.6, "S");
    w.y = y0 + 3;
    w.eyebrow("Correction", C.teal, M + 6);
    lines.forEach((l) => {
      w.font(9.5, "normal", C.ice);
      d.text(l, M + 6, w.y + 3.4);
      w.y += 4.6;
    });
    note.forEach((l) => {
      w.font(7.5, "normal", C.muted);
      d.text(l, M + 6, w.y + 2.8);
      w.y += 3.6;
    });
    w.y = y0 + h;
  }

  w.y += 4;
  w.eyebrow("From the input");
  w.text(`"${c.source_excerpt}"`, { size: 9, style: "italic", color: C.muted });

  if (c.timestamps.length) {
    w.y += 3;
    w.eyebrow("Where it appears");
    for (const t of [...c.timestamps].sort((a, b) => a.start - b.start)) {
      const ids = t.modality === "speech"
        ? (t.segment_ids.length ? `segment ${t.segment_ids.join(", ")}` : "")
        : (t.observation_id !== null ? `text block ${t.observation_id}` : "");
      const conf = `${t.modality === "speech" ? "speech" : "OCR"} confidence ${t.confidence.toFixed(2)}${t.reliable ? "" : " (below threshold)"}`;
      w.ensure(10);
      w.font(8.5, "bold", C.teal, "courier");
      d.text(`${fmtTime(t.start)}-${fmtTime(t.end)}`, M, w.y + 3);
      w.font(7.5, "normal", C.muted);
      d.text(clean(`${t.modality === "speech" ? "Spoken" : "On screen"}${ids ? ` - ${ids}` : ""} - ${conf}`), M + 34, w.y + 3);
      w.y += 4.5;
      w.text(t.text, { size: 8.5, x: M + 34 });
      w.y += 1;
    }
  }

  const prov = c.provenance as { modality?: string; line_numbers?: number[]; min_ocr_score?: number | null; ocr_reliable?: boolean } | null;
  if (prov?.modality === "image" && prov.line_numbers?.length) {
    w.y += 2;
    w.text(`Read from OCR line(s) ${prov.line_numbers.join(", ")}${typeof prov.min_ocr_score === "number" ? ` - OCR confidence ${prov.min_ocr_score.toFixed(2)}` : ""}${prov.ocr_reliable === false ? " (below threshold)" : ""}`,
           { size: 8.5, color: C.muted });
  }

  const roles: Array<[Citation["role"], string, RGB]> = [
    ["supporting", "Supporting evidence", vRGB("SUPPORTED")],
    ["refuting", "Contradicting evidence", vRGB("REFUTED")],
    ["context", "Missing context", vRGB("MISLEADING")],
  ];
  for (const [role, title, rc] of roles) {
    const cites = c.citations.filter((x) => x.role === role);
    if (!cites.length) continue;
    w.y += 3;
    w.eyebrow(`${title} - ${cites.length}`, rc);
    cites.forEach((x) => source(w, x, rc));
  }

  const v = c.citation_validation;
  const vlines = [
    `${v.sources_considered} source(s) considered - ${v.rejected_citations} model citation(s) rejected - ${v.retrieval_rounds} search round(s) - ${v.search_credits} search credit(s)`,
    c.model_proposed_verdict && c.model_proposed_verdict !== c.verdict
      ? `The AI proposed "${verdictStyle(c.model_proposed_verdict).label}"; the deterministic verdict rules selected "${verdictStyle(c.verdict).label}".` : null,
    c.failure_reason ? `Reason: ${c.failure_reason}` : null,
    ...c.score_caps.map((s) => `Score ${s}`),
    ...c.notes,
    `${titleCase(c.claim_type)} claim${c.time_sensitive ? " - time-sensitive" : ""}`,
  ].filter((x): x is string => !!x);
  // keep the whole validation block on one page
  const vh = 7 + vlines.flatMap((l) => w.lines(l, 7.8, CW)).length * 7.8 * PT * 1.35;
  w.y += 2;
  w.ensure(vh);
  w.eyebrow("Validation");
  vlines.forEach((l) => w.text(l, { size: 7.8, color: C.muted, lead: 1.35 }));
  w.y += 8;
}

/** One citation as a small glass card: quote, title, metadata, clickable link. */
function source(w: Writer, x: Citation, rc: RGB) {
  const d = w.d;
  const inner = CW - 12;
  const quote = w.lines(`"${x.quote}"`, 9, inner);
  const title = w.lines(trunc(x.title || domainOf(x.url), 120), 8.3, inner, "bold");
  const meta = [domainOf(x.url), titleCase(x.authority), titleCase(x.directness), fmtDate(x.published_date),
                x.snippet_only ? "snippet only" : null, x.fact_check_site ? "fact-check site" : null].filter(Boolean).join("  -  ");
  const h = 6 + quote.length * 4.3 + title.length * 3.9 + 8.6;
  w.ensure(h + 2);
  const y0 = w.y;
  w.card(M, y0, CW, h, rc);
  let y = y0 + 6;
  quote.forEach((l) => {
    w.font(9, "normal", C.ice);
    d.text(l, M + 6, y);
    y += 4.3;
  });
  y += 0.6;
  title.forEach((l) => {
    w.font(8.3, "bold", C.ice);
    d.text(l, M + 6, y);
    y += 3.9;
  });
  w.font(7.3, "normal", C.muted);
  d.text(clean(meta), M + 6, y + 0.2);
  const href = safeHref(x.url);
  w.font(7.3, "normal", href ? C.teal : C.muted);
  const shown = clean(trunc(x.url, 105));
  if (href) d.textWithLink(shown, M + 6, y + 4.3, { url: href });
  else d.text(shown, M + 6, y + 4.3);
  w.y = y0 + h + 2.2;
}

function unverified(w: Writer, r: FactCheckResponse) {
  if (!r.over_limit_claims.length && !r.dropped_claims.length) return;
  w.ensure(24);
  if (r.over_limit_claims.length) {
    w.eyebrow("Found but not verified - per-run claim limit", C.warn);
    r.over_limit_claims.forEach((d) => w.text(`- ${d.statement}`, { size: 9 }));
    w.y += 4;
  }
  if (r.dropped_claims.length) {
    w.eyebrow("Removed by validation");
    r.dropped_claims.forEach((d) => {
      w.text(`- ${d.statement}`, { size: 9 });
      w.text(d.reason, { size: 7.8, color: C.muted, x: M + 3 });
    });
    w.y += 4;
  }
}

function notes(w: Writer, r: FactCheckResponse) {
  const all = [...r.notes,
    "Verdicts are generated automatically from web sources and can be wrong. Read the cited sources before relying on them."];
  const lines = all.flatMap((n) => w.lines(n, 8, CW - 12));
  const h = 12 + lines.length * 3.9;
  w.ensure(h + 2);
  const y0 = w.y;
  w.card(M, y0, CW, h);
  w.y = y0 + 4;
  w.eyebrow("How to read this report", C.teal, M + 6);
  lines.forEach((l) => {
    w.font(8, "normal", C.muted);
    w.d.text(l, M + 6, w.y + 2.8);
    w.y += 3.9;
  });
  w.y = y0 + h + 4;
}

function appendix(w: Writer, r: FactCheckResponse) {
  const rep = r.report as Record<string, any>;
  type Row = [string, string, string]; // [time/line, confidence, text]
  const conf = (x: number, ok: boolean) => `${x.toFixed(2)}${ok ? "" : "*"}`;
  const blocks: Array<{ title: string; rows: Row[] }> = [];
  if (r.input_type === "image") {
    const lines = (rep?.extraction?.quality?.lines ?? []) as OcrLine[];
    blocks.push({ title: "Appendix - text read from the image (OCR)",
                  rows: lines.map((l): Row => [`line ${l.line_no}`, conf(l.score, l.reliable), l.text]) });
  } else if (r.input_type === "video") {
    const segs = (rep?.extraction?.audio?.segments ?? []) as TranscriptSegment[];
    const obs = (rep?.extraction?.screen?.observations ?? []) as ScreenObservation[];
    blocks.push({ title: "Appendix - speech transcript",
                  rows: segs.map((s): Row => [fmtTime(s.start), conf(s.confidence, s.reliable), s.text]) });
    blocks.push({ title: "Appendix - on-screen text",
                  rows: obs.map((o): Row => [fmtTime(o.frame_times[0] ?? 0), conf(o.min_score, o.reliable), o.text.replace(/\n/g, " / ")]) });
  } else if (r.input_type === "audio") {
    const t = rep?.transcription;
    const segs = (t?.segments ?? []) as TranscriptSegment[];
    const lang = t?.language?.detected ? ` (language ${t.language.detected}${typeof t.language.probability === "number" ? `, p=${t.language.probability.toFixed(2)}` : ""})` : "";
    blocks.push({ title: `Appendix - transcript${lang}`,
                  rows: segs.map((s): Row => [fmtTime(s.start), conf(s.confidence, s.reliable), s.text]) });
  }
  for (const block of blocks) {
    if (!block.rows.length) continue;
    w.newPage();
    w.eyebrow(block.title, C.teal);
    w.text("Confidence is recognition confidence, not truth. * = below the checking threshold.", { size: 7.8, color: C.muted });
    w.y += 3;
    block.rows.forEach(([a, b, text], k) => {
      const lines = w.lines(text, 8.5, CW - 34);
      const h = Math.max(1, lines.length) * 4 + 2.2;
      w.ensure(h);
      if (k % 2 === 0) {
        w.alpha(0.5);
        w.d.setFillColor(...C.card);
        w.d.rect(M - 2, w.y - 0.6, CW + 4, h, "F");
        w.alpha(1);
      }
      w.font(8, "bold", C.teal, "courier");
      w.d.text(a, M, w.y + 3);
      w.font(7.5, "normal", b.endsWith("*") ? C.warn : C.muted, "courier");
      w.d.text(b, M + 20, w.y + 3);
      lines.forEach((l, i) => {
        w.font(8.5, "normal", C.ice);
        w.d.text(l, M + 34, w.y + 3 + i * 4);
      });
      w.y += h;
    });
  }
}

function footers(d: JsPDF) {
  const n = d.getNumberOfPages();
  for (let p = 1; p <= n; p++) {
    d.setPage(p);
    d.setDrawColor(...C.edge);
    d.setLineWidth(0.3);
    d.line(M, PAGE_H - 13, PAGE_W - M, PAGE_H - 13);
    d.setFont("helvetica", "normal");
    d.setFontSize(7);
    d.setTextColor(...C.muted);
    d.text("FactCheck  -  generated automatically; verify with the cited sources", M, PAGE_H - 8.5);
    d.setFont("courier", "bold");
    d.setFontSize(8);
    d.setTextColor(...C.teal);
    d.text(`${String(p).padStart(2, "0")} / ${String(n).padStart(2, "0")}`, PAGE_W - M, PAGE_H - 8.5, { align: "right" });
  }
}

/** Build the PDF document (exported for testing). */
export async function buildPdfReport(r: FactCheckResponse): Promise<JsPDF> {
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ unit: "mm", format: "a4", compress: true });
  doc.setProperties({ title: `FactCheck report - ${r.input_type}`, subject: "Fact-check investigation report", creator: "FactCheck" });
  const w = new Writer(doc, `${titleCase(r.input_type)}${r.filename ? `  -  ${trunc(r.filename, 40)}` : ""}`);
  cover(w, r);
  if (r.claims.length) {
    w.newPage();
    w.eyebrow(`Claims - ${r.claims.length}`, C.teal);
    w.y += 1;
    const ordered = r.input_type === "video" || r.input_type === "audio"
      ? [...r.claims].sort((a, b) => (a.timestamps[0]?.start ?? Infinity) - (b.timestamps[0]?.start ?? Infinity))
      : r.claims;
    ordered.forEach((c, i) => claimBlock(w, c, i, ordered.length));
  } else {
    w.newPage();
  }
  unverified(w, r);
  notes(w, r);
  appendix(w, r);
  footers(doc);
  return doc;
}

export async function downloadPdfReport(r: FactCheckResponse): Promise<void> {
  const doc = await buildPdfReport(r);
  const base = (r.filename ?? r.input_type).replace(/\.[^.]+$/, "").replace(/[^\w-]+/g, "-").slice(0, 40) || r.input_type;
  doc.save(`factcheck-report-${base}-${new Date().toISOString().slice(0, 10)}.pdf`);
}
