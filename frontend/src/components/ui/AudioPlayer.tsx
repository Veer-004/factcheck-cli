import { Gauge, Pause, Play, RotateCcw, RotateCw, Volume2, VolumeX } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type CSSProperties, type MutableRefObject, type RefObject } from "react";

const RATES = [0.75, 1, 1.25, 1.5, 2];
const BARS = 48;

function clock(s: number): string {
  if (!Number.isFinite(s) || s < 0) s = 0;
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  const mm = String(m).padStart(h ? 2 : 1, "0"), ss = String(sec).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

// A fixed, file-specific "waveform" silhouette (decorative: the real audio is not analysed).
function silhouette(seed: string): number[] {
  let h = 2166136261;
  for (const ch of seed) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return Array.from({ length: BARS }, (_, i) => {
    h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
    const wave = 0.45 + 0.35 * Math.sin(i / 2.3);
    return Math.max(0.32, Math.min(1, wave + ((h % 1000) / 1000 - 0.5) * 0.6));
  });
}

interface Props {
  src: string;
  /** Receives the underlying <audio> element (the report seeks it when a timestamp is clicked). */
  audioRef?: RefObject<HTMLAudioElement | null> | MutableRefObject<HTMLAudioElement | HTMLVideoElement | null>;
  title?: string;
  onLoadedMetadata?: (duration: number) => void;
  className?: string;
}

/** Audio player in the site's glass style, replacing the browser's default controls. */
export function AudioPlayer({ src, audioRef, title, onLoadedMetadata, className = "" }: Props) {
  const el = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [buffered, setBuffered] = useState(0);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);
  const [rate, setRate] = useState(1);
  const [error, setError] = useState(false);
  const bars = useMemo(() => silhouette(src), [src]);

  const attach = (node: HTMLAudioElement | null) => {
    el.current = node;
    if (audioRef) (audioRef as MutableRefObject<HTMLAudioElement | null>).current = node;
  };

  useEffect(() => {
    const a = el.current;
    if (!a) return;
    const sync = () => {
      setPlaying(!a.paused && !a.ended);
      setTime(a.currentTime);
      if (Number.isFinite(a.duration)) setDuration(a.duration);
      setVolume(a.volume);
      setMuted(a.muted);
      setRate(a.playbackRate);
      if (a.buffered.length) setBuffered(a.buffered.end(a.buffered.length - 1));
    };
    const events = ["play", "pause", "ended", "timeupdate", "durationchange", "loadedmetadata", "volumechange", "ratechange", "progress", "seeked"];
    events.forEach((e) => a.addEventListener(e, sync));
    const onError = () => setError(true);
    a.addEventListener("error", onError);
    sync();
    return () => {
      events.forEach((e) => a.removeEventListener(e, sync));
      a.removeEventListener("error", onError);
    };
  }, [src]);

  const a = () => el.current;
  const toggle = () => {
    const x = a();
    if (!x) return;
    if (x.paused) void x.play().catch(() => setError(true));
    else x.pause();
  };
  const seekTo = (t: number) => {
    const x = a();
    if (x && Number.isFinite(t)) x.currentTime = Math.max(0, Math.min(duration || x.duration || 0, t));
  };
  const nudge = (d: number) => seekTo((a()?.currentTime ?? 0) + d);
  const cycleRate = () => {
    const x = a();
    if (x) x.playbackRate = RATES[(RATES.indexOf(x.playbackRate) + 1) % RATES.length] ?? 1;
  };

  const progress = duration ? (time / duration) * 100 : 0;
  const loaded = duration ? Math.min(100, (buffered / duration) * 100) : 0;
  const vol = muted ? 0 : volume;

  return (
    <div className={`rounded-2xl border border-line bg-abyss/50 p-4 sm:p-5 ${className}`}
         onKeyDown={(e) => {
           if (e.target instanceof HTMLInputElement) return; // the sliders handle their own keys
           if (e.key === " " || e.key === "k") { e.preventDefault(); toggle(); }
           if (e.key === "ArrowLeft") nudge(-5);
           if (e.key === "ArrowRight") nudge(5);
         }}>
      <audio ref={attach} src={src} preload="metadata" className="hidden"
             onLoadedMetadata={(e) => onLoadedMetadata?.(e.currentTarget.duration)} />

      <div className="flex items-center gap-4">
        <button type="button" onClick={toggle} disabled={error} aria-label={playing ? "Pause" : "Play"}
                className="relative grid h-12 w-12 shrink-0 place-items-center rounded-full bg-teal text-abyss shadow-[0_0_24px_-6px_rgba(0,213,192,0.8)] transition-transform hover:scale-105 active:scale-95 disabled:opacity-40">
          {playing && <span className="absolute inset-0 animate-ping rounded-full bg-teal/30" aria-hidden />}
          {playing ? <Pause className="relative h-5 w-5 fill-current" /> : <Play className="relative ml-0.5 h-5 w-5 fill-current" />}
        </button>

        <div className="min-w-0 flex-1">
          {title && <p className="truncate text-sm font-semibold text-ice/90" title={title}>{title}</p>}
          {/* waveform silhouette, filled up to the playhead; clicking it seeks */}
          <div className="relative mt-1 flex h-9 cursor-pointer items-center justify-between gap-px" aria-hidden
               onClick={(e) => {
                 const r = e.currentTarget.getBoundingClientRect();
                 seekTo(((e.clientX - r.left) / r.width) * duration);
               }}>
            {bars.map((h, i) => {
              const on = ((i + 0.5) / BARS) * 100 <= progress;
              return (
                <span key={i} className={`w-[3px] origin-center rounded-full transition-colors duration-200 ${on ? "bg-teal" : "bg-muted/25"} ${playing && on ? "animate-eq" : ""}`}
                      style={{ height: `${h * 100}%`, animationDelay: `${(i % 5) * 0.11}s`, animationDuration: `${0.8 + (i % 4) * 0.12}s` }} />
              );
            })}
          </div>
        </div>
      </div>

      {/* seek bar */}
      <div className="mt-3 flex items-center gap-3">
        <span className="w-12 shrink-0 font-mono text-xs tabular-nums text-teal">{clock(time)}</span>
        <input type="range" min={0} max={duration || 0} step={0.1} value={Math.min(time, duration || 0)}
               onChange={(e) => seekTo(Number(e.target.value))} disabled={!duration}
               aria-label="Seek" aria-valuetext={`${clock(time)} of ${clock(duration)}`}
               className="media-range flex-1"
               style={{ "--p": `${progress}%`, "--b": `${loaded}%` } as CSSProperties} />
        <span className="w-12 shrink-0 text-right font-mono text-xs tabular-nums text-muted">{clock(duration)}</span>
      </div>

      {/* secondary controls */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1">
          <button type="button" onClick={() => nudge(-10)} aria-label="Back 10 seconds" className="btn btn-ghost btn-sm !px-2.5">
            <RotateCcw className="h-3.5 w-3.5" /> <span className="font-mono text-micro">10</span>
          </button>
          <button type="button" onClick={() => nudge(10)} aria-label="Forward 10 seconds" className="btn btn-ghost btn-sm !px-2.5">
            <span className="font-mono text-micro">10</span> <RotateCw className="h-3.5 w-3.5" />
          </button>
          <button type="button" onClick={cycleRate} aria-label={`Playback speed ${rate}x`} title="Playback speed"
                  className="btn btn-ghost btn-sm !px-2.5">
            <Gauge className="h-3.5 w-3.5" /> <span className="font-mono text-micro">{rate}×</span>
          </button>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" onClick={() => { const x = a(); if (x) x.muted = !x.muted; }}
                  aria-label={vol === 0 ? "Unmute" : "Mute"} className="grid h-8 w-8 place-items-center rounded-xl text-muted transition-colors hover:text-ice">
            {vol === 0 ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
          </button>
          <input type="range" min={0} max={1} step={0.05} value={vol} aria-label="Volume"
                 onChange={(e) => { const x = a(); if (x) { x.volume = Number(e.target.value); x.muted = Number(e.target.value) === 0; } }}
                 className="media-range w-24" style={{ "--p": `${vol * 100}%`, "--b": "0%" } as CSSProperties} />
        </div>
      </div>

      {error && <p className="mt-3 text-xs text-warn">This browser can't play this audio format here, but it can still be fact-checked.</p>}
    </div>
  );
}
