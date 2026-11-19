import { verdictStyle } from "../../lib/verdicts";

export function VerdictBadge({ verdict, size = "md" }: { verdict: string; size?: "sm" | "md" }) {
  const s = verdictStyle(verdict);
  return (
    <span
      title={s.description}
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border font-mono uppercase tracking-[0.12em] ${
        size === "sm" ? "px-2 py-0.5 text-micro" : "px-2.5 py-1 text-micro"
      }`}
      style={{ color: s.color, borderColor: `${s.color}55`, background: `${s.color}14` }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: s.color }} />
      {s.label}
    </span>
  );
}
