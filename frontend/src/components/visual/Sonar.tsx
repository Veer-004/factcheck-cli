import type { CSSProperties } from "react";

const SWEEP_S = 6; // one sweep revolution; blips light up as the beam passes them

interface Blip {
  angle: number; // degrees, 0 = top, clockwise
  radius: number; // 0-1 of the sonar radius
}

const BLIPS: Blip[] = [
  { angle: 38, radius: 0.62 },
  { angle: 112, radius: 0.38 },
  { angle: 205, radius: 0.74 },
  { angle: 287, radius: 0.5 },
  { angle: 330, radius: 0.27 },
];

interface SonarProps {
  className?: string;
  /** "active" = stronger sweep and pings while an investigation runs. */
  variant?: "ambient" | "active";
}

/** Decorative sonar: rings, a rotating sweep, expanding pings and blips timed to the beam. */
export function Sonar({ className = "", variant = "ambient" }: SonarProps) {
  const active = variant === "active";
  return (
    <div aria-hidden className={`relative aspect-square ${className}`}>
      {/* rings */}
      {[1, 0.75, 0.5, 0.25].map((s) => (
        <div
          key={s}
          className="absolute rounded-full border"
          style={{
            inset: `${(1 - s) * 50}%`,
            borderColor: `rgba(0,213,192,${s === 1 ? 0.22 : 0.12})`,
          }}
        />
      ))}
      {/* tick ring */}
      <svg viewBox="0 0 100 100" className="animate-spin-slower absolute inset-0 h-full w-full">
        {Array.from({ length: 72 }, (_, i) => (
          <line
            key={i}
            x1="50"
            y1="1.2"
            x2="50"
            y2={i % 6 === 0 ? 3.6 : 2.3}
            stroke="rgba(140,166,176,0.35)"
            strokeWidth={i % 6 === 0 ? 0.35 : 0.18}
            transform={`rotate(${i * 5} 50 50)`}
          />
        ))}
      </svg>
      {/* dashed counter-rotating ring */}
      <svg viewBox="0 0 100 100" className="animate-spin-rev absolute inset-[9%] h-[82%] w-[82%]">
        <circle cx="50" cy="50" r="49" fill="none" stroke="rgba(0,213,192,0.25)" strokeWidth="0.3" strokeDasharray="1 3" />
      </svg>
      {/* crosshair */}
      <div className="absolute inset-y-[4%] left-1/2 w-px -translate-x-1/2 bg-[rgba(140,166,176,0.10)]" />
      <div className="absolute inset-x-[4%] top-1/2 h-px -translate-y-1/2 bg-[rgba(140,166,176,0.10)]" />
      {/* sweep */}
      <div className="absolute inset-0 overflow-hidden rounded-full">
        <div
          className="animate-sweep absolute inset-0"
          style={{
            background: `conic-gradient(from 0deg, rgba(0,213,192,${active ? 0.42 : 0.26}) 0deg, rgba(0,213,192,0.06) 50deg, transparent 75deg)`,
            animationDuration: `${SWEEP_S}s`,
          }}
        />
      </div>
      {/* pings */}
      {[0, 1.5, 3].map((d) => (
        <div
          key={d}
          className="animate-ping-ring absolute inset-0 rounded-full border border-teal/40"
          style={{ animationDelay: `${d}s`, animationDuration: active ? "3s" : "4.5s" }}
        />
      ))}
      {/* blips */}
      {BLIPS.map((b, i) => {
        const rad = ((b.angle - 90) * Math.PI) / 180;
        const style: CSSProperties = {
          left: `${50 + Math.cos(rad) * b.radius * 50}%`,
          top: `${50 + Math.sin(rad) * b.radius * 50}%`,
          animationDelay: `${(b.angle / 360) * SWEEP_S}s`,
          animationDuration: `${SWEEP_S}s`,
        };
        return (
          <span key={i} className="animate-blip absolute -ml-1 -mt-1 h-2 w-2 rounded-full bg-teal shadow-[0_0_12px_2px_rgba(0,213,192,0.6)]" style={style} />
        );
      })}
      {/* core */}
      <div className="absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-ice shadow-[0_0_18px_4px_rgba(0,213,192,0.55)]" />
    </div>
  );
}
