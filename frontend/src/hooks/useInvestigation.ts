import { useCallback, useRef, useState } from "react";
import { ApiError, verify, type TransferEvent } from "../lib/api";
import type { Mode } from "../lib/config";
import type { FactCheckResponse, RunOptions, RunPlan } from "../lib/types";

export interface Inputs {
  text: string;
  files: Record<Exclude<Mode, "text">, File | null>;
  /** Media length in seconds, when the browser could read it (used only for time estimates). */
  durations: Record<Exclude<Mode, "text">, number | null>;
}

/** What the browser has actually observed about the request (drives the checkpoints). */
export interface Transfer {
  upload: { loaded: number; total: number } | null;
  sentAt: number | null; // request fully sent: the server is working
  download: { loaded: number; total: number | null } | null;
  receivedAt: number | null; // response fully received
}

export type Investigation =
  | { phase: "idle" }
  | { phase: "running"; mode: Mode; startedAt: number; file: File | null; mediaSeconds: number | null; transfer: Transfer }
  | { phase: "done"; mode: Mode; result: FactCheckResponse; file: File | null }
  | { phase: "error"; mode: Mode; error: ApiError };

// After the response arrives the orb fills to the top before the report is shown.
const FINISH_MS = 900;

/** One investigation at a time. Lives in App, so changing pages never drops a running request. */
export function useInvestigation() {
  const [state, setState] = useState<Investigation>({ phase: "idle" });
  const controller = useRef<AbortController | null>(null);
  const running = useRef(false);

  const start = useCallback(async (mode: Mode, inputs: Inputs, options?: RunOptions, plan?: RunPlan | null) => {
    if (running.current) return; // duplicate submissions are ignored
    running.current = true;
    const file = mode === "text" ? null : inputs.files[mode];
    const mediaSeconds = mode === "text" ? null : inputs.durations[mode];
    controller.current = new AbortController();
    const startedAt = Date.now();
    let transfer: Transfer = { upload: null, sentAt: null, download: null, receivedAt: null };
    const publish = () => setState({ phase: "running", mode, startedAt, file, mediaSeconds, transfer });
    publish();
    const onEvent = (e: TransferEvent) => {
      if (e.stage === "upload") transfer = { ...transfer, upload: { loaded: e.loaded, total: e.total } };
      else if (e.stage === "processing") transfer = { ...transfer, sentAt: transfer.sentAt ?? Date.now() };
      else transfer = { ...transfer, download: { loaded: e.loaded, total: e.total } };
      publish();
    };
    try {
      const raw = await verify(mode, { text: inputs.text, file }, options, controller.current.signal, onEvent);
      // The server reports the plan only when its AI agent chose it; otherwise keep the one the page used.
      const result = { ...raw, run_plan: raw.run_plan ?? plan ?? null };
      transfer = { ...transfer, sentAt: transfer.sentAt ?? Date.now(), receivedAt: Date.now() };
      publish(); // last checkpoint reached: the liquid fills to the top
      await new Promise((r) => setTimeout(r, FINISH_MS));
      setState({ phase: "done", mode, result, file });
    } catch (e) {
      const error = e instanceof ApiError ? e : new ApiError("invalid_response", "Something went wrong while reading the result.");
      if (error.kind === "aborted") setState({ phase: "idle" });
      else setState({ phase: "error", mode, error });
    } finally {
      running.current = false;
      controller.current = null;
    }
  }, []);

  const stopWaiting = useCallback(() => controller.current?.abort(), []);
  const reset = useCallback(() => setState({ phase: "idle" }), []);

  return { state, start, stopWaiting, reset };
}
