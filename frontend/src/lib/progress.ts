// Checkpoint model for the loading screen.
//
// Checkpoints are real, observed events: the upload's bytes, the moment the request was fully sent,
// the response's bytes, and its arrival. Only the fill level *inside* the server stage is estimated
// (the API reports no progress while it works): it rises toward the next checkpoint on an easing
// curve scaled to the usual processing time, and never reaches it until the response really arrives.

import type { Transfer } from "../hooks/useInvestigation";
import type { Mode } from "./config";

export type CheckpointStatus = "done" | "active" | "pending";

export interface Checkpoint {
  id: "upload" | "server" | "receive" | "ready";
  label: string;
  status: CheckpointStatus;
  detail: string | null; // real figures only (bytes, %, elapsed time)
  estimated?: boolean;
}

// Fill bands (percent of the vessel) for each stage.
const BAND = { upload: [6, 24], server: [24, 88], receive: [88, 97] } as const;

/** Typical server time in seconds (measured on the development laptop; see README). */
export function expectedServerSeconds(mode: Mode, mediaSeconds: number | null): number {
  if (mode === "video") return Math.max(45, (mediaSeconds ?? 60) * 1.1 + 30);
  if (mode === "audio") return Math.max(40, (mediaSeconds ?? 120) * 0.15 + 35);
  if (mode === "image") return 35;
  return 30;
}

function fmtMB(n: number) {
  return n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;
}

function fmtSecs(ms: number) {
  const s = Math.floor(ms / 1000);
  return s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
}

export function computeProgress(mode: Mode, mediaSeconds: number | null, t: Transfer, now: number):
  { level: number; checkpoints: Checkpoint[]; stage: Checkpoint["id"] } {
  const up = t.upload;
  const upFrac = up && up.total > 0 ? Math.min(1, up.loaded / up.total) : t.sentAt ? 1 : 0;
  const downFrac = t.receivedAt ? 1 : t.download?.total ? Math.min(1, t.download.loaded / t.download.total) : t.download ? 0.5 : 0;

  let stage: Checkpoint["id"] = "upload";
  if (t.receivedAt) stage = "ready";
  else if (t.download) stage = "receive";
  else if (t.sentAt) stage = "server";

  let level: number;
  if (stage === "upload") level = BAND.upload[0] + (BAND.upload[1] - BAND.upload[0]) * upFrac;
  else if (stage === "server") {
    const elapsed = (now - (t.sentAt ?? now)) / 1000;
    const k = 1 - Math.exp(-elapsed / expectedServerSeconds(mode, mediaSeconds));
    level = BAND.server[0] + (BAND.server[1] - BAND.server[0]) * k;
  } else if (stage === "receive") level = BAND.receive[0] + (BAND.receive[1] - BAND.receive[0]) * downFrac;
  else level = 100;

  const order = ["upload", "server", "receive", "ready"] as const;
  const idx = order.indexOf(stage);
  const status = (i: number): CheckpointStatus => (i < idx || stage === "ready" ? "done" : i === idx ? "active" : "pending");
  const serverMs = t.sentAt ? (t.download?.loaded || t.receivedAt ? (t.receivedAt ?? now) : now) - t.sentAt : 0;

  const checkpoints: Checkpoint[] = [
    {
      id: "upload",
      label: mode === "text" ? "Text submitted" : "Uploading evidence",
      status: status(0),
      detail: up ? (upFrac >= 1 ? fmtMB(up.total) : `${Math.round(upFrac * 100)}% of ${fmtMB(up.total)}`) : null,
    },
    {
      id: "server",
      label: "Analysing on the server",
      status: status(1),
      detail: t.sentAt ? fmtSecs(serverMs) : null,
      estimated: true,
    },
    {
      id: "receive",
      label: "Receiving the report",
      status: status(2),
      detail: t.download?.total ? `${Math.round(downFrac * 100)}%` : t.receivedAt ? "done" : null,
    },
    { id: "ready", label: "Report ready", status: status(3), detail: null },
  ];
  return { level, checkpoints, stage };
}
