import { AnimatePresence, motion } from "framer-motion";
import { BrainCircuit, Calculator, ChevronDown, Minus, Plus, RefreshCw, SlidersHorizontal, Sparkles, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { LIMITS, type PlanSource, type RunPlanState } from "../../hooks/useRunPlan";
import type { Mode } from "../../lib/config";

const SOURCES: Array<{ id: PlanSource; label: string; icon: typeof Calculator }> = [
  { id: "rules", label: "Rule-based", icon: Calculator },
  { id: "ai", label: "AI", icon: Sparkles },
  { id: "manual", label: "Manual", icon: SlidersHorizontal },
];

const READING: Record<Mode, string> = {
  text: "The AI agent is reading the text…",
  image: "Reading the image with OCR, then the AI agent decides…",
  audio: "Transcribing the audio, then the AI agent decides…",
  video: "Extracting speech and on-screen text, then the AI agent decides…",
};

function Stepper({ id, label, value, min, max, hint, onChange, disabled }: {
  id: string;
  label: string;
  value: number | null;
  min: number;
  max: number;
  hint: string;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  const v = value ?? min;
  return (
    <div className="rounded-xl border border-line bg-abyss/40 p-3">
      <label htmlFor={id} className="eyebrow !text-micro">{label}</label>
      <div className="mt-2 flex items-center gap-2">
        <button type="button" onClick={() => onChange(v - 1)} disabled={disabled || value === null || v <= min}
                aria-label={`Decrease ${label.toLowerCase()}`} className="btn btn-glass btn-sm !h-8 !w-8 !p-0">
          <Minus className="h-3.5 w-3.5" />
        </button>
        <input id={id} type="number" inputMode="numeric" min={min} max={max} value={value ?? ""} placeholder="–"
               disabled={disabled} onChange={(e) => e.target.value !== "" && onChange(Number(e.target.value))}
               className="h-card w-full min-w-0 bg-transparent text-center font-mono tabular-nums text-ice outline-none placeholder:text-muted [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none" />
        <button type="button" onClick={() => onChange(v + 1)} disabled={disabled || value === null || v >= max}
                aria-label={`Increase ${label.toLowerCase()}`} className="btn btn-glass btn-sm !h-8 !w-8 !p-0">
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>
      <p className="mt-2 text-xs text-muted">{hint}</p>
    </div>
  );
}

function Elapsed() {
  const [s, setS] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setS((x) => x + 1), 1000);
    return () => clearInterval(t);
  }, []);
  return <span className="font-mono tabular-nums">{s}s</span>;
}

/** Run settings for every input type: an instant rule-based estimate, an on-demand AI plan, or manual values. */
export function RunSettings({ run, mode, disabled }: { run: RunPlanState; mode: Mode; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const { source, rules, aiPlan, aiStatus, shown } = run;
  const plan = source === "ai" ? aiPlan : source === "rules" ? rules : null;
  const working = aiStatus === "working";

  const subtitle = source === "manual" ? "Your own numbers are used for this run"
    : source === "rules" ? (rules ? `Instant estimate: about ${rules.estimated_claims} checkable claim${rules.estimated_claims === 1 ? "" : "s"}`
      : run.rulesFailed ? "Planner unreachable: the server's defaults (6 claims, 40 credits) apply" : "Estimating…")
    : working ? READING[mode]
    : aiPlan ? `The AI agent found ${aiPlan.statements.length} checkable statement${aiPlan.statements.length === 1 ? "" : "s"}${aiPlan.model ? ` · ${aiPlan.model}` : ""}`
    : run.aiStale ? "The input changed: calculate again to update the AI plan"
    : "Click “Calculate with AI” to let the agent read the input and decide";

  const warnings = [
    ...(source === "ai" && !aiPlan && !working && rules ? ["Until the AI plan is calculated, the rule-based estimate is used."] : []),
    ...(plan?.warnings ?? []),
  ];

  return (
    <section aria-label="Run settings" className="glass rounded-2xl p-4 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-xl border border-teal/30 bg-teal/10 text-teal">
            <BrainCircuit className={`h-4 w-4 ${working ? "animate-pulse" : ""}`} />
          </span>
          <div className="min-w-0">
            <p className="eyebrow">Run settings</p>
            <p className="text-xs text-muted" aria-live="polite">{subtitle}</p>
          </div>
        </div>
        <div role="radiogroup" aria-label="How the run is sized" className="inline-flex rounded-full border border-line bg-abyss/40 p-0.5">
          {SOURCES.map(({ id, label, icon: Icon }) => (
            <button key={id} type="button" role="radio" aria-checked={source === id} disabled={disabled}
                    onClick={() => run.choose(id)}
                    className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs transition-colors ${source === id ? "bg-teal/15 text-ice" : "text-muted hover:text-ice"}`}>
              <Icon className="h-3.5 w-3.5" /> {label}
            </button>
          ))}
        </div>
      </div>

      {source === "ai" && (
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button type="button" onClick={() => void run.calculateWithAi()} disabled={disabled || working}
                  className="btn btn-primary btn-sm">
            {working ? <RefreshCw className="relative h-3.5 w-3.5 animate-spin" />
              : <Sparkles className="relative h-3.5 w-3.5" />}
            <span className="relative">{working ? "Calculating…" : aiPlan ? "Recalculate with AI" : "Calculate with AI"}</span>
          </button>
          {working && <span className="text-xs text-muted">{mode === "text" ? "Usually a few seconds" : "Takes about as long as the extraction step"} · <Elapsed /></span>}
          {!working && mode !== "text" && !aiPlan && (
            <span className="text-xs text-muted">The file is read on the server first (one Gemini request, no search credits).</span>
          )}
          {!working && mode === "text" && !aiPlan && <span className="text-xs text-muted">One Gemini request, no search credits.</span>}
        </div>
      )}

      {run.aiError && source === "ai" && !working && (
        <p className="mt-3 flex items-start gap-2 text-xs text-refuted"><TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {run.aiError}</p>
      )}

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <Stepper id="run-claims" label="Claims to check" value={shown?.max_claims ?? null} disabled={disabled || working}
                 min={LIMITS.claims.min} max={LIMITS.claims.max} onChange={(v) => run.change("max_claims", v)}
                 hint={`At most ${LIMITS.claims.max} per run; every part of the input gets a turn.`} />
        <Stepper id="run-budget" label="Search credits" value={shown?.search_budget ?? null} disabled={disabled || working}
                 min={LIMITS.budget.min} max={LIMITS.budget.max} onChange={(v) => run.change("search_budget", v)}
                 hint={plan ? `Shared equally between the claims; typically about ${plan.estimated_credits} used.` : "Shared equally between the claims."} />
      </div>
      {source !== "manual" && shown && <p className="mt-2 text-micro text-muted">Change a number to switch to Manual.</p>}

      {warnings.map((w) => (
        <p key={w} className="mt-3 flex items-start gap-2 text-xs text-warn">
          <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {w}
        </p>
      ))}

      {plan && (plan.reasons.length > 0 || plan.statements.length > 0) && (
        <div className="mt-3">
          <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
                  className="flex items-center gap-1 text-xs text-muted transition-colors hover:text-ice">
            Why these numbers?
            <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`} />
          </button>
          <AnimatePresence initial={false}>
            {open && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                <ul>
                  {plan.reasons.map((r) => (
                    <li key={r} className="mt-2 flex gap-2 text-xs text-muted">
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-teal" /> {r}
                    </li>
                  ))}
                </ul>
                {plan.statements.length > 0 && (
                  <>
                    <p className="eyebrow !text-micro mt-4">Checkable statements the agent found</p>
                    <ol data-lenis-prevent className="mt-2 max-h-48 space-y-1 overflow-auto pr-1">
                      {plan.statements.map((st, i) => (
                        <li key={i} className="flex gap-2 text-xs text-ice/90">
                          <span className="w-5 shrink-0 text-right font-mono text-teal">{i + 1}</span> {st}
                        </li>
                      ))}
                    </ol>
                    {plan.statements.length > plan.max_claims && (
                      <p className="mt-2 text-xs text-muted">
                        More than {plan.max_claims} statements: the run picks claims round-robin, so every item still gets checked first.
                      </p>
                    )}
                  </>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </section>
  );
}
