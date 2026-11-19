import type { RunPlan } from "./types";

/** "12 claims · 38 credits · AI planner (gemini-3.5-flash-lite)" */
export function describePlan(p: RunPlan): string {
  const who = p.planner === "agent" ? `AI planner${p.model ? ` (${p.model})` : ""}`
    : p.planner === "manual" ? "set manually" : "rule-based estimate";
  return `${p.max_claims} claim${p.max_claims === 1 ? "" : "s"} · ${p.search_budget} credits · ${who}`;
}
