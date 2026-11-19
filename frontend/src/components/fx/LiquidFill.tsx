import { animate, useMotionValue, useReducedMotion, useTransform, motion } from "framer-motion";
import { useEffect, useId, type ReactNode } from "react";
import { PALETTE } from "../../lib/palette";

interface Props {
  level: number; // 0-100
  size?: number;
  children?: ReactNode; // centred label
}

const R = 96; // inner radius in the 240x240 viewBox
const C = 120;

// One wave period is 120 units; the path is 3 periods wide so it can scroll seamlessly.
function wavePath(amp: number, phase: number) {
  let d = `M -120 0`;
  for (let x = -120; x <= 360; x += 10) {
    d += ` L ${x} ${amp * Math.sin(((x + phase) / 120) * Math.PI * 2)}`;
  }
  return `${d} L 360 260 L -120 260 Z`;
}

/** Liquid Fill: a glass orb whose liquid level eases to `level`, with moving waves and bubbles. */
export function LiquidFill({ level, size = 260, children }: Props) {
  const id = useId().replace(/:/g, "");
  const reduce = useReducedMotion();
  const mv = useMotionValue(level);
  useEffect(() => {
    const ctl = animate(mv, level, reduce ? { duration: 0 } : { type: "spring", stiffness: 40, damping: 18 });
    return () => ctl.stop();
  }, [level, mv, reduce]);
  // top of the liquid: y from bottom (C+R) to top (C-R)
  const y = useTransform(mv, (v) => C + R - (Math.max(0, Math.min(100, v)) / 100) * 2 * R);

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg viewBox="0 0 240 240" className="absolute inset-0 h-full w-full" aria-hidden>
        <defs>
          <clipPath id={`clip-${id}`}>
            <circle cx={C} cy={C} r={R} />
          </clipPath>
          <linearGradient id={`liquid-${id}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={PALETTE.tealBright} stopOpacity="0.95" />
            <stop offset="0.45" stopColor={PALETTE.teal} stopOpacity="0.75" />
            <stop offset="1" stopColor={PALETTE.tealDeep} stopOpacity="0.9" />
          </linearGradient>
          <linearGradient id={`rim-${id}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor={PALETTE.ice} stopOpacity="0.55" />
            <stop offset="0.5" stopColor={PALETTE.teal} stopOpacity="0.15" />
            <stop offset="1" stopColor={PALETTE.teal} stopOpacity="0.5" />
          </linearGradient>
          <radialGradient id={`glass-${id}`} cx="0.35" cy="0.3" r="0.8">
            <stop offset="0" stopColor={PALETTE.ice} stopOpacity="0.10" />
            <stop offset="1" stopColor={PALETTE.glass} stopOpacity="0.35" />
          </radialGradient>
        </defs>

        {/* vessel */}
        <circle cx={C} cy={C} r={R + 10} fill="none" stroke="rgba(0,213,192,0.12)" strokeWidth="1" strokeDasharray="2 5" />
        <circle cx={C} cy={C} r={R} fill={`url(#glass-${id})`} />

        {/* liquid */}
        <g clipPath={`url(#clip-${id})`}>
          <motion.g style={{ y }}>
            <g className={reduce ? "" : "animate-wave-slow"}>
              <path d={wavePath(7, 30)} fill={PALETTE.teal} opacity="0.28" />
            </g>
            <g className={reduce ? "" : "animate-wave"}>
              <path d={wavePath(5, 0)} fill={`url(#liquid-${id})`} />
            </g>
          </motion.g>
          {!reduce && [0, 1, 2, 3, 4, 5].map((i) => (
            <circle key={i} className="animate-bubble" cx={70 + i * 20} cy={C + R} r={1.6 + (i % 3)}
                    fill={PALETTE.ice} opacity="0.5" style={{ animationDelay: `${i * 0.7}s`, animationDuration: `${3.5 + (i % 3)}s` }} />
          ))}
        </g>

        {/* glass rim + sheen */}
        <circle cx={C} cy={C} r={R} fill="none" stroke={`url(#rim-${id})`} strokeWidth="2" />
        <path d={`M ${C - 62} ${C - 58} A ${R - 14} ${R - 14} 0 0 1 ${C - 6} ${C - 84}`} fill="none" stroke={PALETTE.ice}
              strokeOpacity="0.35" strokeWidth="3" strokeLinecap="round" />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">{children}</div>
    </div>
  );
}
