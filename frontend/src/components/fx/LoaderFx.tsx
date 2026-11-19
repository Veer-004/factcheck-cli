// Small loading indicators. None of them shows a percentage: the API reports no progress.

import type { Mode } from "../../lib/config";

/** Indeterminate Bar: a light sweeping along a track, for "working, no ETA". */
export function IndeterminateBar({ className = "" }: { className?: string }) {
  return (
    <div className={`relative h-[3px] w-full overflow-hidden rounded-full bg-[rgba(140,166,176,0.14)] ${className}`} aria-hidden>
      <span className="animate-indeterminate absolute inset-y-0 w-1/3 rounded-full bg-[linear-gradient(90deg,transparent,var(--color-teal),transparent)]" />
    </div>
  );
}

/** Typing Dots. */
export function TypingDots({ className = "" }: { className?: string }) {
  return (
    <span className={`inline-flex items-center gap-1 ${className}`} aria-hidden>
      {[0, 1, 2].map((i) => (
        <span key={i} className="animate-typing h-1.5 w-1.5 rounded-full bg-teal" style={{ animationDelay: `${i * 0.16}s` }} />
      ))}
    </span>
  );
}

/** Equalizer Bars (audio). */
function Equalizer() {
  const bars = [0.55, 0.9, 0.4, 1, 0.65, 0.85, 0.35, 0.75, 0.5, 0.95, 0.45, 0.7];
  return (
    <div className="flex h-7 items-end gap-[3px]" aria-hidden>
      {bars.map((h, i) => (
        <span key={i} className="animate-eq w-[3px] origin-bottom rounded-full bg-teal/80"
              style={{ height: `${h * 100}%`, animationDelay: `${(i % 5) * 0.11}s`, animationDuration: `${0.7 + (i % 4) * 0.12}s` }} />
      ))}
    </div>
  );
}

/** Dot Grid Wave (video frames being sampled). */
function DotGridWave() {
  const cols = 14;
  const rows = 3;
  return (
    <div className="grid gap-[5px]" style={{ gridTemplateColumns: `repeat(${cols}, 4px)` }} aria-hidden>
      {Array.from({ length: cols * rows }, (_, i) => (
        <span key={i} className="animate-dotwave h-1 w-1 rounded-full bg-teal" style={{ animationDelay: `${(i % cols) * 0.08 + Math.floor(i / cols) * 0.12}s` }} />
      ))}
    </div>
  );
}

/** A frame with a scan line passing over it (image being read). */
function ScanFrame() {
  return (
    <div className="relative h-7 w-24 overflow-hidden rounded-md border border-teal/30" aria-hidden>
      <div className="absolute inset-x-2 top-1.5 space-y-1">
        <div className="h-[3px] w-4/5 rounded bg-[rgba(140,166,176,0.3)]" />
        <div className="h-[3px] w-3/5 rounded bg-[rgba(140,166,176,0.3)]" />
        <div className="h-[3px] w-2/3 rounded bg-[rgba(140,166,176,0.3)]" />
      </div>
      <span className="animate-scanx absolute inset-y-0 w-6 bg-[linear-gradient(90deg,transparent,rgba(0,213,192,0.55),transparent)]" />
    </div>
  );
}

/** One quiet accent per input type, shown under the loader's status line. */
export function ModeAccent({ mode }: { mode: Mode }) {
  if (mode === "audio") return <Equalizer />;
  if (mode === "video") return <DotGridWave />;
  if (mode === "image") return <ScanFrame />;
  return <TypingDots />;
}
