import { useEffect, useRef } from "react";

/**
 * Fixed atmospheric backdrop: two soft light pools and slowly rising "marine snow".
 * One small canvas, ~70 particles, paused when the tab is hidden; static under reduced motion.
 */
const VIDEO_SRC = `${import.meta.env.BASE_URL}media/ocean-loop.mp4`;
const POSTER_SRC = `${import.meta.env.BASE_URL}media/ocean-poster.jpg`;

export function OceanBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0;
    let h = 0;
    let raf = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    type P = { x: number; y: number; r: number; a: number; vy: number; phase: number };
    let particles: P[] = [];

    const resize = () => {
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.round(Math.min(45, (w * h) / 36000)); // a foreground layer over the video's own particles
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        r: 0.4 + Math.random() * 1.4,
        a: 0.08 + Math.random() * 0.35,
        vy: 0.05 + Math.random() * 0.18,
        phase: Math.random() * Math.PI * 2,
      }));
    };

    const draw = (t: number) => {
      ctx.clearRect(0, 0, w, h);
      for (const p of particles) {
        if (!reduced) {
          p.y -= p.vy;
          p.x += Math.sin(t / 4000 + p.phase) * 0.08;
          if (p.y < -4) {
            p.y = h + 4;
            p.x = Math.random() * w;
          }
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(190, 240, 245, ${p.a})`;
        ctx.fill();
      }
      if (!reduced) raf = requestAnimationFrame(draw);
    };

    const onVisibility = () => {
      cancelAnimationFrame(raf);
      if (!document.hidden) raf = requestAnimationFrame(draw);
    };

    resize();
    raf = requestAnimationFrame(draw);
    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-abyss">
      {/* Looping underwater video (rendered by scripts/make_ocean_video.py). Under reduced motion only
          the still poster frame is shown. A tint keeps text readable over it. */}
      <video
        className="absolute inset-0 h-full w-full scale-105 object-cover opacity-90"
        src={reduced ? undefined : VIDEO_SRC}
        poster={POSTER_SRC}
        autoPlay={!reduced}
        muted
        loop
        playsInline
        preload="auto"
        disablePictureInPicture
        tabIndex={-1}
      />
      <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(7,20,26,0.25)_0%,rgba(7,20,26,0.55)_55%,rgba(7,20,26,0.85)_100%)]" />
      <div className="absolute -left-[20%] -top-[30%] h-[80vh] w-[80vw] rounded-full bg-[radial-gradient(closest-side,rgba(0,213,192,0.10),transparent)]" />
      <div className="absolute -bottom-[35%] right-[-15%] h-[85vh] w-[75vw] rounded-full bg-[radial-gradient(closest-side,rgba(30,90,120,0.22),transparent)]" />
      {/* faint perspective grid "sea floor" */}
      <div className="absolute inset-x-[-25%] bottom-0 h-[48vh] [perspective:520px]">
        <div className="animate-grid h-full w-full origin-bottom [transform:rotateX(64deg)] bg-[size:64px_64px] bg-[linear-gradient(rgba(0,213,192,0.07)_1px,transparent_1px),linear-gradient(90deg,rgba(0,213,192,0.07)_1px,transparent_1px)] [mask-image:linear-gradient(to_top,black_10%,transparent_85%)]" />
      </div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_40%,rgba(3,10,13,0.75))]" />
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
    </div>
  );
}
