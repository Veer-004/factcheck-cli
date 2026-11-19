import { motion, useReducedMotion } from "framer-motion";
import { verdictStyle } from "../../lib/verdicts";
import { CountUp } from "../fx/TextFx";

interface GaugeProps {
  score: number;
  verdict: string;
  size?: number;
  label?: string;
}

/** Evidence-strength gauge (0-100). The backend's score is an experimental rubric, not a probability. */
export function EvidenceGauge({ score, verdict, size = 56, label }: GaugeProps) {
  const reduce = useReducedMotion();
  const color = verdictStyle(verdict).color;
  const r = 20;
  const c = 2 * Math.PI * r;
  const arc = c * 0.75; // 270° gauge
  const value = Math.max(0, Math.min(100, score)) / 100;
  return (
    <div className="flex flex-col items-center" title="Evidence strength: an experimental rubric, not a probability">
      <svg width={size} height={size} viewBox="0 0 48 48" className="-rotate-[135deg]" aria-label={`${label ?? "Evidence strength"} ${score} of 100`}>
        <circle cx="24" cy="24" r={r} fill="none" stroke="rgba(140,166,176,0.16)" strokeWidth="3.5" strokeDasharray={`${arc} ${c}`} strokeLinecap="round" />
        <motion.circle
          cx="24" cy="24" r={r} fill="none" stroke={color} strokeWidth="3.5" strokeLinecap="round"
          strokeDasharray={`${arc} ${c}`}
          initial={reduce ? false : { strokeDashoffset: arc }}
          animate={{ strokeDashoffset: arc * (1 - value) }}
          transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
        />
      </svg>
      <span className="-mt-[2.4rem] font-mono text-sm font-medium" style={{ color }}><CountUp value={score} duration={1100} /></span>
      <span className="mt-[1.1rem] text-micro uppercase tracking-wider text-muted">{label ?? "evidence"}</span>
    </div>
  );
}

/** Score area of a claim: one gauge, two (support vs. against) for CONFLICTING, or nothing. */
export function ClaimScore({ verdict, score, support, refute }: {
  verdict: string;
  score: number | null;
  support: number | null;
  refute: number | null;
}) {
  if (score !== null) return <EvidenceGauge score={score} verdict={verdict} />;
  if (support !== null || refute !== null) {
    return (
      <div className="flex gap-2">
        {support !== null && <EvidenceGauge score={support} verdict="SUPPORTED" size={46} label="support" />}
        {refute !== null && <EvidenceGauge score={refute} verdict="REFUTED" size={46} label="against" />}
      </div>
    );
  }
  return null;
}
