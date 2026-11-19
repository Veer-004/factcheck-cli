// Text effects. Every effect renders its final text for screen readers and is static under
// prefers-reduced-motion.

import { AnimatePresence, motion, useInView, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";

const UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&*";
const LOWER = "abcdefghijklmnopqrstuvwxyz0123456789#%&*";

function glyphFor(ch: string): string {
  const set = ch === ch.toLowerCase() ? LOWER : UPPER; // keep roughly the same width as the real letter
  return set[Math.floor(Math.random() * set.length)];
}

/** Scramble Decode: random glyphs resolve into the text, left to right. The final text reserves the
 *  space, so the scrambled frames never change the layout. */
export function ScrambleText({ text, delay = 0, duration = 1100, className = "" }: {
  text: string;
  delay?: number; // ms
  duration?: number; // ms
  className?: string;
}) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(reduce ? text : "");

  useEffect(() => {
    if (reduce) {
      setShown(text);
      return;
    }
    let raf = 0;
    let start = 0;
    const tick = (t: number) => {
      if (!start) start = t;
      const p = Math.min(1, (t - start - delay) / duration);
      const done = p < 0 ? 0 : Math.floor(p * text.length);
      setShown(text.split("").map((ch, i) => (i < done || ch === " " ? ch : glyphFor(ch))).join(""));
      if (p < 1) raf = requestAnimationFrame(tick);
      else setShown(text);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [text, delay, duration, reduce]);

  return (
    <span className={className}>
      <span className="sr-only">{text}</span>
      {splitWords(text, shown).map(([finalWord, current], i) => (
        <span key={i} aria-hidden>
          {i > 0 && " "}
          <span className="relative inline-block">
            <span className="invisible select-none">{finalWord}</span>
            <span className="absolute left-0 top-0 whitespace-nowrap">{current}</span>
          </span>
        </span>
      ))}
    </span>
  );
}

/** Pairs each word of the final text with the same-length slice of the current (scrambled) text. */
function splitWords(text: string, shown: string): Array<[string, string]> {
  const out: Array<[string, string]> = [];
  let pos = 0;
  for (const word of text.split(" ")) {
    out.push([word, (shown || text).slice(pos, pos + word.length)]);
    pos += word.length + 1;
  }
  return out;
}

/** Mask Rise: the line slides up from behind an invisible edge. */
export function MaskRise({ children, delay = 0, className = "" }: { children: ReactNode; delay?: number; className?: string }) {
  const reduce = useReducedMotion();
  return (
    <span className={`block overflow-hidden pb-[0.08em] ${className}`}>
      <motion.span
        className="block"
        initial={reduce ? false : { y: "105%" }}
        animate={{ y: "0%" }}
        transition={{ duration: 1, delay, ease: [0.16, 1, 0.3, 1] }}
      >
        {children}
      </motion.span>
    </span>
  );
}

/** Blur In + Word Stagger. `parts` keeps inline emphasis (hl = highlighted). */
export function BlurInWords({ parts, delay = 0, className = "" }: {
  parts: Array<{ t: string; hl?: boolean }>;
  delay?: number;
  className?: string;
}) {
  const reduce = useReducedMotion();
  let i = 0;
  return (
    <p className={className}>
      {parts.map((part, pi) =>
        part.t.split(/(\s+)/).map((w, wi) => {
          if (!w.trim()) return w;
          const n = i++;
          return (
            <motion.span
              key={`${pi}-${wi}`}
              className={`inline-block ${part.hl ? "text-ice" : ""}`}
              initial={reduce ? false : { opacity: 0, filter: "blur(10px)", y: 6 }}
              animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
              transition={{ duration: 0.6, delay: delay + n * 0.028, ease: "easeOut" }}
            >
              {w}
            </motion.span>
          );
        }),
      )}
    </p>
  );
}

/** Rotating Word: cycles through words with a vertical flip. */
export function RotatingWord({ words, interval = 2200, className = "" }: { words: string[]; interval?: number; className?: string }) {
  const reduce = useReducedMotion();
  const [i, setI] = useState(0);
  useEffect(() => {
    if (reduce) return;
    const t = setInterval(() => setI((x) => (x + 1) % words.length), interval);
    return () => clearInterval(t);
  }, [words.length, interval, reduce]);
  const longest = words.reduce((a, b) => (b.length > a.length ? b : a), "");
  return (
    <span className={`relative inline-grid overflow-hidden align-bottom ${className}`} aria-label={words.join(", ")}>
      <span className="invisible col-start-1 row-start-1" aria-hidden>{longest}</span>
      <AnimatePresence initial={false}>
        <motion.span
          key={words[i]}
          aria-hidden
          className="col-start-1 row-start-1"
          initial={{ y: "100%", opacity: 0 }}
          animate={{ y: "0%", opacity: 1 }}
          exit={{ y: "-100%", opacity: 0 }}
          transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
        >
          {words[i]}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}

/** Marker Highlight: a translucent teal marker sweeps under the text when it scrolls into view. */
export function MarkerHighlight({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const reduce = useReducedMotion();
  return (
    <span
      ref={ref}
      className="bg-no-repeat px-1 [box-decoration-break:clone]"
      style={{
        backgroundImage: "linear-gradient(transparent 58%, rgba(0,213,192,0.22) 58%, rgba(0,213,192,0.22) 92%, transparent 92%)",
        backgroundSize: inView || reduce ? "100% 100%" : "0% 100%",
        transition: "background-size 1.1s cubic-bezier(0.16,1,0.3,1) 0.25s",
      }}
    >
      {children}
    </span>
  );
}

/** Number Count-up: animates from 0 to `value` once mounted. */
export function CountUp({ value, duration = 1100, decimals = 0, delay = 0 }: {
  value: number;
  duration?: number;
  decimals?: number;
  delay?: number;
}) {
  const reduce = useReducedMotion();
  const [n, setN] = useState(reduce ? value : 0);
  useEffect(() => {
    if (reduce) {
      setN(value);
      return;
    }
    let raf = 0;
    let start = 0;
    const tick = (t: number) => {
      if (!start) start = t;
      const p = Math.max(0, Math.min(1, (t - start - delay) / duration));
      setN(value * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration, delay, reduce]);
  return <>{n.toFixed(decimals)}</>;
}

/** Typewriter: types the text, with a blinking caret. Restarts when the text changes. */
export function Typewriter({ text, speed = 32, className = "" }: { text: string; speed?: number; className?: string }) {
  const reduce = useReducedMotion();
  const [k, setK] = useState(reduce ? text.length : 0);
  useEffect(() => {
    if (reduce) {
      setK(text.length);
      return;
    }
    setK(0);
    const t = setInterval(() => setK((x) => (x >= text.length ? x : x + 1)), speed);
    return () => clearInterval(t);
  }, [text, speed, reduce]);
  return (
    <span className={className} aria-label={text}>
      <span aria-hidden>{text.slice(0, k)}</span>
      <span aria-hidden className="ml-0.5 inline-block w-[0.08em] animate-caret self-stretch bg-teal align-[-0.12em]" style={{ height: "1em" }} />
    </span>
  );
}

/** Strike & Replace: the refuted statement is struck through, then the correction is written in. */
export function StrikeReplace({ from, to }: { from: string; to: string }) {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });
  const go = inView || !!reduce;
  return (
    <div ref={ref} className="space-y-2">
      <p className="relative inline text-sm leading-relaxed text-muted">
        <span>{from}</span>
        <motion.span
          aria-hidden
          className="absolute left-0 top-1/2 h-[1.5px] w-full origin-left bg-refuted"
          initial={reduce ? false : { scaleX: 0 }}
          animate={go ? { scaleX: 1 } : undefined}
          transition={{ duration: 0.7, delay: 0.2, ease: [0.65, 0, 0.35, 1] }}
        />
      </p>
      <motion.p
        className="text-sm leading-relaxed text-ice"
        initial={reduce ? false : { opacity: 0, x: -8, filter: "blur(6px)" }}
        animate={go ? { opacity: 1, x: 0, filter: "blur(0px)" } : undefined}
        transition={{ duration: 0.6, delay: 0.85 }}
      >
        {to}
      </motion.p>
    </div>
  );
}
