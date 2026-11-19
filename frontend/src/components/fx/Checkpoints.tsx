import { Check } from "lucide-react";
import type { Checkpoint } from "../../lib/progress";

/** Checkpoint rail: real observed stages, with the estimated one marked. */
export function Checkpoints({ items }: { items: Checkpoint[] }) {
  return (
    <ol className="grid w-full grid-cols-2 gap-3 sm:grid-cols-4" aria-label="Investigation checkpoints">
      {items.map((c, i) => {
        const done = c.status === "done";
        const active = c.status === "active";
        return (
          <li key={c.id} className={`relative rounded-xl px-3 py-3 text-left transition-all duration-500 ${active ? "glass-strong hairline-glow" : "glass"} ${c.status === "pending" ? "opacity-55" : ""}`}>
            <div className="flex items-center gap-2">
              <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full border font-mono text-micro transition-colors ${
                done ? "border-teal bg-teal text-abyss" : active ? "border-teal text-teal" : "border-line text-muted"}`}>
                {done ? <Check className="h-3.5 w-3.5" strokeWidth={3} /> : active ? <span className="h-2 w-2 animate-pulse rounded-full bg-teal" /> : i + 1}
              </span>
              <span className={`text-xs font-semibold leading-tight ${active || done ? "text-ice" : "text-muted"}`}>{c.label}</span>
            </div>
            <p className="mt-1.5 min-h-[1rem] pl-8 font-mono text-micro text-muted">
              {c.detail}
              {c.estimated && active && <span className="ml-1.5 rounded border border-line px-1 text-micro uppercase tracking-wider">est. fill</span>}
            </p>
          </li>
        );
      })}
    </ol>
  );
}
