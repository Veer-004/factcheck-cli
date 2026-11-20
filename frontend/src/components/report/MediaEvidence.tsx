import { Mic, MonitorPlay, ScanText } from "lucide-react";
import { useEffect, useMemo, useState, type RefObject } from "react";
import { fmtTime } from "../../lib/format";
import { AudioPlayer } from "../ui/AudioPlayer";
import { PALETTE } from "../../lib/palette";
import type { ClaimResult, OcrLine, ScreenObservation, TranscriptSegment } from "../../lib/types";
import { verdictStyle } from "../../lib/verdicts";
type Report = Record<string, any>; // module report (see factcheck_api schemas: `report`)

// ---------------------------------------------------------------- report readers (defensive)

export function imageData(report: Report) {
  const ex = report?.extraction;
  if (!ex) return null;
  return {
    width: Number(ex.width) || 0,
    height: Number(ex.height) || 0,
    text: String(ex.text ?? ""),
    lines: (ex.quality?.lines ?? []) as OcrLine[],
    threshold: ex.quality?.threshold as number | undefined,
  };
}

export function videoData(report: Report) {
  const ex = report?.extraction;
  if (!ex) return null;
  return {
    duration: (ex.media?.duration_s as number | null) ?? null,
    segments: (ex.audio?.segments ?? []) as TranscriptSegment[],
    observations: (ex.screen?.observations ?? []) as ScreenObservation[],
    language: ex.audio?.language as string | undefined,
  };
}

export function audioData(report: Report) {
  const t = report?.transcription;
  if (!t) return null;
  return {
    duration: (report.media?.duration_s as number | null) ?? null,
    segments: (t.segments ?? []) as TranscriptSegment[],
    language: t.language?.detected as string | undefined,
    languageP: t.language?.probability as number | undefined,
  };
}

// ---------------------------------------------------------------- image

export function ImageEvidence({ url, report, claims, hovered }: {
  url: string | null;
  report: Report;
  claims: ClaimResult[];
  hovered: string | null;
}) {
  const data = imageData(report);
  if (!data) return null;
  const lit = new Set<number>();
  for (const c of claims) {
    if (hovered && c.claim_id !== hovered) continue;
    for (const n of ((c.provenance?.line_numbers as number[] | undefined) ?? [])) lit.add(n);
  }
  return (
    <section className="panel hud p-4 sm:p-6">
      <div className="flex items-center justify-between gap-3">
        <h3 className="eyebrow flex items-center gap-2"><ScanText className="h-3.5 w-3.5 text-teal" /> Text read from the image</h3>
        <span className="text-xs text-muted">{data.lines.length} OCR line{data.lines.length === 1 ? "" : "s"}</span>
      </div>
      <div className="mt-4 grid gap-5 lg:grid-cols-[1.2fr_1fr]">
        {url && data.width > 0 && (
          <div className="relative overflow-hidden rounded-xl border border-line bg-black/30">
            <img src={url} alt="Investigated image" className="block w-full" />
            <svg viewBox={`0 0 ${data.width} ${data.height}`} className="absolute inset-0 h-full w-full" preserveAspectRatio="none" aria-hidden>
              {data.lines.map((l) => {
                const [x0, y0, x1, y1] = l.bbox;
                const on = lit.has(l.line_no);
                return (
                  <rect key={l.line_no} x={x0} y={y0} width={x1 - x0} height={y1 - y0} rx={Math.min(8, (y1 - y0) / 4)}
                        fill={on ? "rgba(0,213,192,0.16)" : "transparent"}
                        stroke={on ? PALETTE.teal : "rgba(229,250,255,0.25)"}
                        strokeWidth={Math.max(1.5, data.width / 500)}
                        vectorEffect="non-scaling-stroke"
                        style={{ transition: "all 250ms" }} />
                );
              })}
            </svg>
          </div>
        )}
        <ol data-lenis-prevent className="max-h-[26rem] space-y-1.5 overflow-auto pr-1 text-sm">
          {data.lines.map((l) => (
            <li key={l.line_no} className={`flex gap-3 rounded-xl px-2.5 py-1.5 transition-colors ${lit.has(l.line_no) ? "bg-teal/10" : ""}`}>
              <span className="w-5 shrink-0 text-right font-mono text-xs text-muted">{l.line_no}</span>
              <span className="flex-1 text-ice/90">{l.text}</span>
              <span className={`shrink-0 font-mono text-xs ${l.reliable ? "text-muted" : "text-warn"}`} title="OCR confidence">
                {l.score.toFixed(2)}
              </span>
            </li>
          ))}
        </ol>
      </div>
      <p className="mt-4 text-xs text-muted">Hover a claim below to see the lines it was read from. OCR confidence describes how reliably characters were recognised — not whether the text is true.</p>
    </section>
  );
}

// ---------------------------------------------------------------- video / audio

interface MediaProps {
  kind: "video" | "audio";
  url: string | null;
  mediaRef: RefObject<HTMLVideoElement | HTMLAudioElement | null>;
  duration: number | null;
  claims: ClaimResult[];
  segments: TranscriptSegment[];
  observations?: ScreenObservation[];
  onSeek: (t: number) => void;
}

function useCurrentTime(ref: RefObject<HTMLMediaElement | null>) {
  const [t, setT] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const on = () => setT(el.currentTime);
    el.addEventListener("timeupdate", on);
    return () => el.removeEventListener("timeupdate", on);
  }, [ref]);
  return t;
}

export function MediaEvidence({ kind, url, mediaRef, duration, claims, segments, observations = [], onSeek }: MediaProps) {
  const now = useCurrentTime(mediaRef);
  const [elDuration, setElDuration] = useState<number | null>(null);
  const total = duration || elDuration || Math.max(1, ...segments.map((s) => s.end), ...claims.flatMap((c) => c.timestamps.map((t) => t.end)));
  const [tab, setTab] = useState<"speech" | "screen">("speech");

  const markers = useMemo(() => claims.flatMap((c, i) => c.timestamps.map((s) => ({ claim: c, i, s }))), [claims]);
  const rows: Array<"speech" | "screen"> = kind === "video" ? ["speech", "screen"] : ["speech"];

  return (
    <section className="panel hud p-4 sm:p-6">
      <h3 className="eyebrow flex items-center gap-2">
        {kind === "video" ? <MonitorPlay className="h-3.5 w-3.5 text-teal" /> : <Mic className="h-3.5 w-3.5 text-teal" />}
        {kind === "video" ? "The video, claim by claim" : "The recording, claim by claim"}
      </h3>

      {url && (
        <div className="mt-4 overflow-hidden rounded-xl border border-line bg-black/40">
          {kind === "video" ? (
            <video ref={mediaRef as RefObject<HTMLVideoElement>} src={url} controls className="max-h-[26rem] w-full"
                   onLoadedMetadata={(e) => setElDuration(e.currentTarget.duration)} />
          ) : (
            <AudioPlayer src={url} audioRef={mediaRef} onLoadedMetadata={setElDuration} className="!rounded-none !border-0" />
          )}
        </div>
      )}

      {/* timeline of claims */}
      {markers.length > 0 && (
        <div className="mt-5">
          <div className="flex justify-between font-mono text-micro text-muted">
            <span>00:00.0</span><span>{fmtTime(total)}</span>
          </div>
          <div className="mt-1.5 space-y-1.5">
            {rows.map((row) => (
              <div key={row} className="flex items-center gap-2">
                <span className="w-12 shrink-0 text-micro uppercase tracking-wider text-muted">{row === "speech" ? "spoken" : "screen"}</span>
                <div className="relative h-6 flex-1 rounded-md bg-abyss/70">
                  <span className="absolute inset-y-0 w-px bg-ice/70" style={{ left: `${Math.min(100, (now / total) * 100)}%` }} />
                  {markers.filter((m) => m.s.modality === row).map((m, k) => (
                    <button key={k} type="button" onClick={() => onSeek(m.s.start)}
                            title={`${fmtTime(m.s.start)} · ${m.claim.claim}`}
                            className="absolute inset-y-1 min-w-[6px] rounded-md opacity-80 transition-opacity hover:opacity-100"
                            style={{ left: `${(m.s.start / total) * 100}%`, width: `${Math.max(0.8, ((m.s.end - m.s.start) / total) * 100)}%`,
                                     background: verdictStyle(m.claim.verdict).color }}>
                      <span className="sr-only">Claim {m.i + 1} at {fmtTime(m.s.start)}</span>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="mt-2 text-xs text-muted">Each mark is a claim, coloured by its verdict. Click one to play from there.</p>
        </div>
      )}

      {/* transcript / on-screen text */}
      {(segments.length > 0 || observations.length > 0) && (
        <div className="mt-6">
          {kind === "video" && observations.length > 0 && (
            <div className="mb-3 inline-flex rounded-full border border-line p-0.5 text-xs">
              {(["speech", "screen"] as const).map((t) => (
                <button key={t} type="button" onClick={() => setTab(t)}
                        className={`rounded-full px-3 py-1 transition-colors ${tab === t ? "bg-teal/15 text-ice" : "text-muted hover:text-ice"}`}>
                  {t === "speech" ? `Transcript · ${segments.length}` : `On-screen text · ${observations.length}`}
                </button>
              ))}
            </div>
          )}
          {tab === "speech" || kind === "audio" ? (
            <ol data-lenis-prevent className="max-h-80 space-y-1 overflow-auto pr-1">
              {segments.map((s) => {
                const active = now >= s.start && now < s.end;
                return (
                  <li key={s.segment_id}>
                    <button type="button" onClick={() => onSeek(s.start)}
                            className={`flex w-full gap-3 rounded-xl px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-teal/[0.06] ${active ? "bg-teal/10" : ""}`}>
                      <span className="w-14 shrink-0 font-mono text-xs text-teal">{fmtTime(s.start)}</span>
                      <span className={`flex-1 ${s.reliable ? "text-ice/90" : "text-muted"}`}>{s.text}</span>
                      <span className={`shrink-0 font-mono text-xs ${s.reliable ? "text-muted" : "text-warn"}`} title="Speech confidence">
                        {s.confidence.toFixed(2)}
                      </span>
                    </button>
                  </li>
                );
              })}
              {segments.length === 0 && <li className="px-2.5 text-sm text-muted">No speech was recognised.</li>}
            </ol>
          ) : (
            <ol data-lenis-prevent className="max-h-80 space-y-1 overflow-auto pr-1">
              {observations.map((o) => (
                <li key={o.observation_id}>
                  <button type="button" onClick={() => onSeek(o.frame_times[0] ?? 0)}
                          className="flex w-full gap-3 rounded-xl px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-teal/[0.06]">
                    <span className="w-14 shrink-0 font-mono text-xs text-teal">{fmtTime(o.frame_times[0] ?? 0)}</span>
                    <span className={`flex-1 whitespace-pre-line ${o.reliable ? "text-ice/90" : "text-muted"}`}>{o.text}</span>
                    <span className={`shrink-0 font-mono text-xs ${o.reliable ? "text-muted" : "text-warn"}`} title="OCR confidence">
                      {o.min_score.toFixed(2)}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          )}
          <p className="mt-3 text-xs text-muted">Times are approximate. Confidence describes how reliably words were recognised (amber = below the checking threshold) — not whether they are true.</p>
        </div>
      )}
    </section>
  );
}
