import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, Lightbulb, MonitorPlay, Mic, ScanText } from "lucide-react";
import { useState } from "react";
import { fmtTime, titleCase } from "../../lib/format";
import type { ClaimResult, TimeSpan } from "../../lib/types";
import { verdictStyle } from "../../lib/verdicts";
import { StrikeReplace } from "../fx/TextFx";
import { ClaimScore } from "./EvidenceScore";
import { SourcesList } from "./SourcesList";
import { VerdictBadge } from "./VerdictBadge";

interface Props {
  claim: ClaimResult;
  index: number;
  onSeek?: (t: number) => void;
  onHover?: (id: string | null) => void;
  defaultOpen?: boolean;
}

function SpanRow({ span, onSeek }: { span: TimeSpan; onSeek?: (t: number) => void }) {
  const Icon = span.modality === "speech" ? Mic : MonitorPlay;
  const ids = span.modality === "speech"
    ? (span.segment_ids.length ? `segment ${span.segment_ids.join(", ")}` : null)
    : (span.observation_id !== null ? `text block ${span.observation_id}` : null);
  return (
    <li className="flex flex-col gap-2 rounded-xl border border-line bg-abyss/40 p-3 sm:flex-row sm:items-start sm:gap-4">
      <button
        type="button"
        disabled={!onSeek}
        onClick={() => onSeek?.(span.start)}
        className="inline-flex shrink-0 items-center gap-2 self-start rounded-xl border border-teal/30 bg-teal/10 px-2.5 py-1 font-mono text-xs text-teal transition-colors enabled:hover:bg-teal/20 disabled:cursor-default"
        title={onSeek ? "Play from here" : undefined}
      >
        <Icon className="h-3.5 w-3.5" />
        {fmtTime(span.start)}–{fmtTime(span.end)}
      </button>
      <div className="min-w-0 flex-1">
        <p className="text-sm leading-relaxed text-ice/90">{span.text}</p>
        <p className="mt-1 text-xs text-muted">
          {[span.modality === "speech" ? "Spoken" : "Shown on screen", ids, span.word_level ? "word-level timing" : null,
            `${span.modality === "speech" ? "speech" : "OCR"} confidence ${span.confidence.toFixed(2)}${span.reliable ? "" : " (below threshold)"}`]
            .filter(Boolean).join(" · ")}
        </p>
      </div>
    </li>
  );
}

export function ClaimCard({ claim: c, index, onSeek, onHover, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const style = verdictStyle(c.verdict);
  const prov = c.provenance as { modality?: string; line_numbers?: number[]; min_ocr_score?: number | null; ocr_reliable?: boolean;
                                 merged?: unknown[] } | null;
  const v = c.citation_validation;
  const first = c.timestamps[0];

  return (
    <motion.article
      layout="position"
      onMouseEnter={() => onHover?.(c.claim_id)}
      onMouseLeave={() => onHover?.(null)}
      className="glass relative overflow-hidden rounded-2xl"
    >
      <span aria-hidden className="absolute inset-y-0 left-0 w-[3px]" style={{ background: style.color }} />
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-start gap-4 p-4 text-left sm:items-center sm:p-5"
      >
        <span className="mt-0.5 font-mono text-xs text-muted sm:mt-0">{String(index + 1).padStart(2, "0")}</span>
        <div className="min-w-0 flex-1">
          <p className="font-medium leading-snug text-ice">{c.claim}</p>
          <div className="mt-2.5 flex flex-wrap items-center gap-2">
            <VerdictBadge verdict={c.verdict} />
            {first && <span className="font-mono text-micro text-teal">@ {fmtTime(first.start)}</span>}
            {c.timestamps.length > 1 && <span className="text-micro text-muted">+{c.timestamps.length - 1} more</span>}
          </div>
        </div>
        <div className="hidden sm:block">
          <ClaimScore verdict={c.verdict} score={c.evidence_score} support={c.support_score} refute={c.refute_score} />
        </div>
        <ChevronDown className={`mt-1 h-5 w-5 shrink-0 text-muted transition-transform duration-300 sm:mt-0 ${open ? "rotate-180" : ""}`} />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="space-y-6 border-t border-line px-4 pb-6 pt-5 sm:px-5 sm:pl-12">
              <div className="flex items-start justify-between gap-4">
                <p className="text-sm leading-relaxed text-ice/90">{c.explanation}</p>
                <div className="sm:hidden">
                  <ClaimScore verdict={c.verdict} score={c.evidence_score} support={c.support_score} refute={c.refute_score} />
                </div>
              </div>

              {c.correction && (
                <div className="flex gap-3 rounded-xl border border-teal/25 bg-teal/[0.06] p-4">
                  <Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-teal" />
                  <div className="min-w-0">
                    <p className="eyebrow !text-teal">Correction</p>
                    <div className="mt-2">
                      {c.verdict === "REFUTED" ? (
                        <StrikeReplace from={c.claim} to={c.correction} />
                      ) : (
                        <p className="text-sm leading-relaxed">{c.correction}</p>
                      )}
                    </div>
                    {c.correction_note && <p className="mt-1 text-xs text-muted">{c.correction_note}</p>}
                  </div>
                </div>
              )}

              <div>
                <p className="eyebrow">From the input</p>
                <p className="mt-2 text-sm italic leading-relaxed text-muted">“{c.source_excerpt}”</p>
              </div>

              {c.timestamps.length > 0 && (
                <div>
                  <p className="eyebrow">Where it appears</p>
                  <ul className="mt-3 space-y-2">
                    {[...c.timestamps].sort((a, b) => a.start - b.start).map((s, i) => <SpanRow key={i} span={s} onSeek={onSeek} />)}
                  </ul>
                  {!!prov?.merged?.length && (
                    <p className="mt-2 text-xs text-muted">{prov.merged.length} repeat(s) of this claim were merged and checked once.</p>
                  )}
                </div>
              )}

              {prov?.modality === "image" && prov.line_numbers && prov.line_numbers.length > 0 && (
                <div className="flex items-start gap-3 text-sm text-muted">
                  <ScanText className="mt-0.5 h-4 w-4 shrink-0 text-teal" />
                  <p>
                    Read from OCR line{prov.line_numbers.length > 1 ? "s" : ""} {prov.line_numbers.join(", ")}
                    {typeof prov.min_ocr_score === "number" && <> · OCR confidence {prov.min_ocr_score.toFixed(2)}</>}
                    {prov.ocr_reliable === false && " (below threshold)"}
                  </p>
                </div>
              )}

              <SourcesList citations={c.citations} />

              {(v.sources_considered > 0 || v.rejected_citations > 0 || c.notes.length > 0 || c.failure_reason || c.score_caps.length > 0) && (
                <details className="group rounded-xl border border-line bg-abyss/30 px-4 py-3">
                  <summary className="cursor-pointer list-none text-xs text-muted transition-colors hover:text-ice">
                    <span className="eyebrow">Validation details</span>
                  </summary>
                  <div className="mt-3 space-y-2 text-xs leading-relaxed text-muted">
                    <p className="font-mono">
                      {v.sources_considered} sources considered · {v.rejected_citations} model citation(s) rejected · {v.retrieval_rounds} search round(s) · {v.search_credits} search credit(s)
                    </p>
                    {c.model_proposed_verdict && c.model_proposed_verdict !== c.verdict && (
                      <p>The AI proposed {verdictStyle(c.model_proposed_verdict).label.toLowerCase()}; the deterministic verdict rules selected {style.label.toLowerCase()}.</p>
                    )}
                    {c.failure_reason && <p>Reason: {c.failure_reason}</p>}
                    {c.score_caps.map((s) => <p key={s}>Score {s}</p>)}
                    {c.notes.map((n) => <p key={n}>{n}</p>)}
                    <p>{titleCase(c.claim_type)} claim{c.time_sensitive ? " · time-sensitive" : ""}</p>
                  </div>
                </details>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.article>
  );
}
