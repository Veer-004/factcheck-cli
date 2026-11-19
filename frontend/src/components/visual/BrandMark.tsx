import { PALETTE } from "../../lib/palette";

/** Logo: a small sonar lens with a beam. */
export function BrandMark({ className = "h-7 w-7" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <circle cx="16" cy="16" r="13" fill="none" stroke={PALETTE.teal} strokeOpacity=".3" strokeWidth="1.4" />
      <circle cx="16" cy="16" r="7" fill="none" stroke={PALETTE.teal} strokeWidth="1.7" />
      <path d="M16 16 L26 8.5" stroke={PALETTE.ice} strokeWidth="1.7" strokeLinecap="round" />
      <circle cx="16" cy="16" r="2" fill={PALETTE.teal} />
    </svg>
  );
}
