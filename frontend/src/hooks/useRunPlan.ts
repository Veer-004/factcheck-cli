import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getFilePlan, getPlan } from "../lib/api";
import type { MediaMode, Mode } from "../lib/config";
import type { RunOptions, RunPlan } from "../lib/types";

export const LIMITS = { claims: { min: 1, max: 12 }, budget: { min: 0, max: 40 } } as const;
const RULES_DEBOUNCE_MS = 350;

/** Where the numbers come from. */
export type PlanSource = "rules" | "ai" | "manual";
export type AiStatus = "idle" | "working" | "ready" | "failed";

interface Values {
  max_claims: number;
  search_budget: number;
}

/**
 * Run settings for the next investigation, for every input type.
 *
 * - **Rule-based** (default): an instant estimate, refreshed whenever the input changes. No model call.
 * - **AI**: only when the user clicks "Calculate with AI". Text is read directly; for a file the server first
 *   reads it (OCR / transcription), then the AI planning agent decides. A changed input makes it stale.
 * - **Manual**: the user's own numbers (editing a value in any mode switches here).
 */
export function useRunPlan(mode: Mode, text: string, file: File | null, durationS: number | null, ready: boolean) {
  const [source, setSource] = useState<PlanSource>("rules");
  const [rules, setRules] = useState<RunPlan | null>(null);
  const [rulesFailed, setRulesFailed] = useState(false);
  const [ai, setAi] = useState<{ plan: RunPlan; key: string } | null>(null);
  const [aiStatus, setAiStatus] = useState<AiStatus>("idle");
  const [aiError, setAiError] = useState<string | null>(null);
  const [manual, setManual] = useState<Values>({ max_claims: 6, search_budget: 40 });
  const aiController = useRef<AbortController | null>(null);

  // What the AI plan was calculated for; a different input makes it stale.
  const inputKey = mode === "text" ? `text:${text}` : `${mode}:${file ? `${file.name}:${file.size}:${file.lastModified}` : ""}`;

  // Rule-based estimate: always on, instant.
  const planText = mode === "text" ? text : "";
  useEffect(() => {
    if (!ready) {
      setRules(null);
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      getPlan({ mode, text: planText || undefined, duration_s: durationS }, controller.signal)
        .then((p) => {
          setRules(p);
          setRulesFailed(false);
        })
        .catch((e) => {
          if (e?.kind !== "aborted") {
            setRules(null);
            setRulesFailed(true);
          }
        });
    }, mode === "text" ? RULES_DEBOUNCE_MS : 0);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [mode, planText, durationS, ready]);

  // A different input (or mode) cancels a running AI calculation and goes back to the rule-based estimate
  // (the AI plan was for the old input). The user's manual numbers stay.
  useEffect(() => {
    setSource((s) => (s === "ai" ? "rules" : s));
    return () => aiController.current?.abort();
  }, [inputKey]);

  const aiPlan = ai && ai.key === inputKey ? ai.plan : null;
  const aiStale = !!ai && ai.key !== inputKey;

  const calculateWithAi = useCallback(async () => {
    aiController.current?.abort();
    const controller = new AbortController();
    aiController.current = controller;
    setSource("ai");
    setAiStatus("working");
    setAiError(null);
    const key = inputKey;
    try {
      const plan = mode === "text"
        ? await getPlan({ mode, text, useAi: true }, controller.signal)
        : await getFilePlan(mode as MediaMode, file as File, controller.signal);
      setAi({ plan, key });
      setAiStatus("ready");
    } catch (e) {
      if (e instanceof ApiError && e.kind === "aborted") {
        if (aiController.current === controller) setAiStatus("idle"); // cancelled by an input change
        return;
      }
      setAiStatus("failed");
      setAiError(e instanceof ApiError ? e.message : "The AI planner could not be reached.");
    }
  }, [inputKey, mode, text, file]);

  // AI not calculated (yet): the rule estimate is shown and sent.
  const active: RunPlan | null = source === "ai" ? aiPlan ?? rules : source === "rules" ? rules : null;

  /** Numbers shown in the steppers. */
  const shown: Values | null = source === "manual" ? manual
    : active ? { max_claims: active.max_claims, search_budget: active.search_budget } : null;

  /** What the request sends: always explicit numbers; nothing (server defaults) only if no plan exists. */
  const options: RunOptions | undefined = useMemo(() => {
    if (source === "manual") return { ...manual };
    const p = (source === "ai" ? aiPlan : null) ?? rules; // AI not calculated (yet): the rule estimate is used
    return p ? { max_claims: p.max_claims, search_budget: p.search_budget } : undefined;
  }, [source, manual, aiPlan, rules]);

  /** The plan remembered with the result (report header and PDF). */
  const usedPlan: RunPlan | null = useMemo(() => {
    if (source === "manual") {
      return {
        mode, ...manual, planner: "manual", basis: "default", model: null, deferred: false, statements: [],
        estimated_claims: rules?.estimated_claims ?? 0, estimated_credits: 0, credits_left: rules?.credits_left ?? null,
        credits_limit: rules?.credits_limit ?? null, reasons: ["Set manually before the run."], warnings: [],
      };
    }
    return (source === "ai" ? aiPlan : null) ?? rules;
  }, [source, manual, aiPlan, rules, mode]);

  const choose = (s: PlanSource) => {
    if (s === "manual" && source !== "manual" && shown) setManual(shown);
    setSource(s);
  };

  /** Editing a value switches to manual, starting from the numbers currently shown. */
  const change = (key: keyof Values, value: number) => {
    const lim = key === "max_claims" ? LIMITS.claims : LIMITS.budget;
    const base = shown ?? manual;
    setManual({ ...base, [key]: Math.max(lim.min, Math.min(lim.max, Math.round(value))) });
    setSource("manual");
  };

  return {
    source, choose, change, shown, options, usedPlan, calculateWithAi,
    rules, rulesFailed, aiPlan, aiStatus: aiStale && aiStatus === "ready" ? "idle" : aiStatus, aiStale, aiError,
  };
}

export type RunPlanState = ReturnType<typeof useRunPlan>;
