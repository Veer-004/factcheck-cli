import { motion, useReducedMotion } from "framer-motion";
import { BrainCircuit, AlertTriangle, FileDown, FileSearch, PencilLine, Plus } from "lucide-react";
import { describePlan } from "../../lib/runPlan";
import { useToast } from "../ui/Toast";
import { useCallback, useMemo, useRef, useState } from "react";
import { suggestedPdfName } from "../../lib/pdfName";
import { PdfNameDialog } from "./PdfNameDialog";
import { useObjectUrl } from "../../hooks/useObjectUrl";
import type { FactCheckResponse } from "../../lib/types";
import { SUMMARY_COUNT_KEYS, SUMMARY_LABELS, verdictStyle } from "../../lib/verdicts";
import { CountUp, ScrambleText } from "../fx/TextFx";
import { ClaimCard } from "./ClaimCard";
import { audioData, ImageEvidence, MediaEvidence, videoData } from "./MediaEvidence";

interface Props {
  result: FactCheckResponse;
  file: File | null;
  onNew: () => void;
  onEdit: () => void;
}

function PdfButton({ result }: { result: FactCheckResponse }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const toast = useToast();
  const defaultName = useMemo(() => suggestedPdfName(result), [result]);
  // Start loading jsPDF while the user types the name.
  const preload = () => void import("../../lib/pdfReport");

  const download = async (name: string) => {
    setBusy(true);
    try {
      const { downloadPdfReport } = await import("../../lib/pdfReport");
      await downloadPdfReport(result, name);
      setOpen(false);
      toast("PDF downloaded", `Saved as ${name}.pdf`, "success");
    } catch {
      toast("The PDF couldn't be created", "Please try again. Your report is still shown on this page.");
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <button onClick={() => { preload(); setOpen(true); }} onMouseEnter={preload} className="btn btn-glass">
        <FileDown className="h-4 w-4 text-teal" /> Download PDF report
      </button>
      <PdfNameDialog open={open} defaultName={defaultName} busy={busy}
                     onCancel={() => setOpen(false)} onConfirm={(name) => void download(name)} />
    </>
  );
}

function Distribution({ result }: { result: FactCheckResponse }) {
  const s = result.summary;
  const parts = SUMMARY_COUNT_KEYS.map(([v, k]) => ({ v, n: s[k] as number })).filter((p) => p.n > 0);
  if (!s.total_claims) return null;
  return (
    <div>
      <div className="flex h-2.5 overflow-hidden rounded-full bg-abyss/80">
        {parts.map((p, i) => (
          <motion.span key={p.v} initial={{ width: 0 }} animate={{ width: `${(p.n / s.total_claims) * 100}%` }}
                       transition={{ duration: 0.9, delay: 0.3 + i * 0.08, ease: [0.16, 1, 0.3, 1] }}
                       style={{ background: verdictStyle(p.v).color }} />
        ))}
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-sm">
        {parts.map((p) => (
          <li key={p.v} className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full" style={{ background: verdictStyle(p.v).color }} />
            <span className="font-mono">{p.n}</span>
            <span className="text-muted">{verdictStyle(p.v).label.toLowerCase()}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function InvestigationReport({ result, file, onNew, onEdit }: Props) {
  const reduce = useReducedMotion();
  const url = useObjectUrl(file);
  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const s = result.summary;
  const report = result.report as Record<string, any>;
  const media = result.input_type === "video" ? videoData(report) : result.input_type === "audio" ? audioData(report) : null;

  const seek = useCallback((t: number) => {
    const el = mediaRef.current;
    if (!el) return;
    el.currentTime = t;
    void el.play().catch(() => {});
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  const reveal = (i: number) => ({
    initial: reduce ? false : { opacity: 0, y: 18 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, delay: 0.1 + i * 0.07, ease: [0.16, 1, 0.3, 1] as const },
  });
  const sorted = media
    ? [...result.claims].sort((a, b) => (a.timestamps[0]?.start ?? Infinity) - (b.timestamps[0]?.start ?? Infinity))
    : result.claims;

  return (
    <div className="space-y-6">
      {/* header */}
      <motion.header {...reveal(0)} className="panel hud overflow-hidden p-6 sm:p-8">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-2xl">
            <p className="eyebrow flex items-center gap-2">
              <FileSearch className="h-3.5 w-3.5 text-teal" />
              {result.status === "completed" ? "Investigation complete" : "Investigation finished · nothing checkable"}
              <span className="text-muted/60">· {result.input_type}{result.filename ? ` · ${result.filename}` : ""}</span>
            </p>
            <h2 className="h-section mt-4">
              <ScrambleText text={s.label ? SUMMARY_LABELS[s.label] ?? s.label : "No claims could be checked"} delay={200} duration={1000} />
            </h2>
            {s.text && <p className="mt-3 text-muted">{s.text}</p>}
            {result.not_checked_reason && <p className="mt-3 text-muted">{result.not_checked_reason}</p>}
          </div>
          <div className="flex shrink-0 gap-6 font-mono">
            <div><p className="text-3xl text-ice"><CountUp value={s.total_claims} duration={800} delay={300} /></p><p className="text-micro uppercase tracking-wider text-muted">claims checked</p></div>
            <div><p className="text-3xl text-ice"><CountUp value={result.processing_seconds} duration={1200} delay={300} /><span className="text-base text-muted">s</span></p><p className="text-micro uppercase tracking-wider text-muted">processing</p></div>
          </div>
        </div>
        <div className="mt-7"><Distribution result={result} /></div>
        {result.run_plan && (
          <p className="mt-4 flex flex-wrap items-center gap-2 text-xs text-muted">
            <BrainCircuit className="h-3.5 w-3.5 text-teal" />
            <span className="eyebrow !text-micro">Run plan</span>
            <span className="text-ice/90">{describePlan(result.run_plan)}</span>
            {result.run_plan.reasons[0] && result.run_plan.planner === "agent" && (
              <span className="basis-full text-muted sm:basis-auto">— {result.run_plan.reasons[0]}</span>
            )}
          </p>
        )}
        <div className="mt-7 flex flex-wrap gap-2.5">
          <button onClick={onNew} className="btn btn-primary"><Plus className="h-4 w-4" /> New investigation</button>
          <button onClick={onEdit} className="btn btn-glass"><PencilLine className="h-4 w-4" /> Edit input</button>
          <PdfButton result={result} />
        </div>
      </motion.header>

      {/* warnings */}
      {result.warnings.length > 0 && (
        <motion.ul {...reveal(1)} className="space-y-2">
          {result.warnings.map((w) => (
            <li key={w} className="glass flex gap-3 rounded-2xl px-4 py-3 text-sm text-ice/90">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" /> {w}
            </li>
          ))}
        </motion.ul>
      )}

      {/* media context */}
      {result.input_type === "image" && (
        <motion.div {...reveal(2)}><ImageEvidence url={url} report={report} claims={result.claims} hovered={hovered} /></motion.div>
      )}
      {media && (result.input_type === "video" || result.input_type === "audio") && (
        <motion.div {...reveal(2)}>
          <MediaEvidence kind={result.input_type} url={url} mediaRef={mediaRef} duration={media.duration}
                         claims={result.claims} segments={media.segments}
                         observations={"observations" in media ? media.observations : []} onSeek={seek} />
        </motion.div>
      )}

      {/* claims */}
      {sorted.length > 0 && (
        <section>
          <motion.h3 {...reveal(3)} className="eyebrow mb-3 px-1">Claims{media ? " · in order of appearance" : ""}</motion.h3>
          <div className="space-y-3">
            {sorted.map((c, i) => (
              <motion.div key={c.claim_id} {...reveal(4 + i)}>
                <ClaimCard claim={c} index={i} onSeek={media ? seek : undefined} onHover={setHovered} defaultOpen={sorted.length === 1} />
              </motion.div>
            ))}
          </div>
        </section>
      )}

      {/* not verified */}
      {(result.over_limit_claims.length > 0 || result.dropped_claims.length > 0) && (
        <motion.section {...reveal(5 + sorted.length)} className="panel p-5 sm:p-6">
          {result.over_limit_claims.length > 0 && (
            <>
              <h3 className="eyebrow">Found but not verified · per-run claim limit</h3>
              <ul className="mt-3 space-y-1.5 text-sm text-ice/85">
                {result.over_limit_claims.map((d) => <li key={d.statement} className="flex gap-2"><span className="text-muted">—</span>{d.statement}</li>)}
              </ul>
            </>
          )}
          {result.dropped_claims.length > 0 && (
            <details className={result.over_limit_claims.length ? "mt-5" : ""}>
              <summary className="eyebrow cursor-pointer list-none hover:text-ice">Removed by validation · {result.dropped_claims.length}</summary>
              <ul className="mt-3 space-y-2 text-sm">
                {result.dropped_claims.map((d) => (
                  <li key={d.statement + d.reason}><span className="text-ice/85">{d.statement}</span> <span className="text-xs text-muted">— {d.reason}</span></li>
                ))}
              </ul>
            </details>
          )}
        </motion.section>
      )}

      {result.notes.length > 0 && (
        <ul className="space-y-1 px-1 text-xs leading-relaxed text-muted">
          {result.notes.map((n) => <li key={n}>{n}</li>)}
          <li>Verdicts are generated automatically from web sources and can be wrong — read the cited sources.</li>
        </ul>
      )}
    </div>
  );
}
