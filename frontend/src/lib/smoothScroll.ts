// Lenis smooth scrolling. One instance for the page; components scroll through `scrollToTarget`
// so jumps (anchors, page changes) use the same eased motion. Inner scroll areas (transcripts,
// OCR lists) opt out with the `data-lenis-prevent` attribute.

import Lenis from "lenis";

let lenis: Lenis | null = null;

export function startSmoothScroll(): () => void {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return () => {};
  lenis = new Lenis({
    lerp: 0.085, // lower = silkier, higher = snappier
    wheelMultiplier: 0.95,
    touchMultiplier: 1.4,
    smoothWheel: true,
    anchors: false, // anchors are handled by scrollToTarget
  });
  let raf = 0;
  const loop = (t: number) => {
    lenis?.raf(t);
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
  return () => {
    cancelAnimationFrame(raf);
    lenis?.destroy();
    lenis = null;
  };
}

/** Scroll to an element id, or to the top when `id` is null. Instant when `immediate`. */
export function scrollToTarget(id: string | null, immediate = false) {
  const el = id ? document.getElementById(id) : null;
  if (id && !el) return;
  if (lenis) {
    lenis.scrollTo(el ?? 0, { offset: el ? -88 : 0, immediate, duration: immediate ? 0 : 1.4,
                              easing: (x) => 1 - Math.pow(1 - x, 4) });
    return;
  }
  if (el) el.scrollIntoView({ behavior: immediate ? "auto" : "smooth", block: "start" });
  else window.scrollTo({ top: 0 });
}

/** Pause/resume page scrolling (e.g. while a modal is open). */
export function setScrollLocked(locked: boolean) {
  if (!lenis) return;
  if (locked) lenis.stop();
  else lenis.start();
}
