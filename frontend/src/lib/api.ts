// All communication with the FastAPI backend goes through this module.
//
// Fact-check requests use XMLHttpRequest rather than fetch because it reports real upload and
// download progress; those events drive the checkpoints of the loading screen. The server itself
// reports no progress while it works, so nothing between "uploaded" and "response arriving" is
// measured.

import { API_BASE_URL, REQUEST_TIMEOUT_MS, type MediaMode, type Mode } from "./config";
import type { ApiErrorBody, FactCheckResponse, HealthResponse } from "./types";

export type ApiErrorKind = "http" | "network" | "timeout" | "aborted" | "invalid_response";

export class ApiError extends Error {
  constructor(
    public kind: ApiErrorKind,
    message: string,
    public status?: number,
    public code?: string,
    public details: Array<Record<string, unknown>> = [],
    public requestId?: string | null,
  ) {
    super(message);
  }

  get retryable(): boolean {
    return this.kind === "network" || this.kind === "timeout" || this.status === 500 || this.status === 503;
  }
}

/** Real, observed transfer events (never estimates). */
export type TransferEvent =
  | { stage: "upload"; loaded: number; total: number }
  | { stage: "processing" } // the whole request has been sent; the server is working
  | { stage: "download"; loaded: number; total: number | null };

function isErrorBody(x: unknown): x is ApiErrorBody {
  const b = x as ApiErrorBody;
  return !!b && b.status === "error" && typeof b.error?.message === "string";
}

function isReport(x: unknown): x is FactCheckResponse {
  const r = x as FactCheckResponse;
  return !!r && (r.status === "completed" || r.status === "not_checked") && Array.isArray(r.claims)
    && typeof r.summary === "object" && r.summary !== null;
}

function networkError() {
  return new ApiError("network", `Could not reach the FactCheck server at ${API_BASE_URL}. Is the backend running?`);
}

function post(path: string, body: XMLHttpRequestBodyInit, headers: Record<string, string>, timeoutMs: number,
              signal?: AbortSignal, onEvent?: (e: TransferEvent) => void): Promise<FactCheckResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}${path}`);
    xhr.timeout = timeoutMs;
    xhr.responseType = "text";
    for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v);

    let sent = false;
    const markSent = () => {
      if (!sent) {
        sent = true;
        onEvent?.({ stage: "processing" });
      }
    };
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onEvent?.({ stage: "upload", loaded: e.loaded, total: e.total });
    };
    xhr.upload.onload = markSent;
    xhr.onprogress = (e) => {
      markSent();
      onEvent?.({ stage: "download", loaded: e.loaded, total: e.lengthComputable ? e.total : null });
    };
    xhr.onload = () => {
      let parsed: unknown = null;
      try {
        parsed = JSON.parse(xhr.responseText);
      } catch {
        parsed = null;
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        if (isErrorBody(parsed)) {
          reject(new ApiError("http", parsed.error.message, xhr.status, parsed.error.code, parsed.error.details,
                              parsed.request_id));
        } else {
          reject(new ApiError("http", `The server answered with HTTP ${xhr.status}.`, xhr.status));
        }
        return;
      }
      if (!isReport(parsed)) {
        reject(new ApiError("invalid_response", "The server returned a response this page cannot read.", xhr.status));
        return;
      }
      resolve(parsed);
    };
    xhr.onerror = () => reject(networkError());
    xhr.ontimeout = () => reject(new ApiError("timeout",
      `No response after ${Math.round(timeoutMs / 60000)} minutes. The server may still be working on it.`));
    xhr.onabort = () => reject(new ApiError("aborted", "Stopped waiting for the result."));
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(body);
  });
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  signal?.addEventListener("abort", () => controller.abort(), { once: true });
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal: controller.signal });
    if (!res.ok) throw new ApiError("http", `HTTP ${res.status}`, res.status);
    return (await res.json()) as HealthResponse;
  } catch (e) {
    throw e instanceof ApiError ? e : networkError();
  } finally {
    clearTimeout(timer);
  }
}

export function verifyText(text: string, signal?: AbortSignal, onEvent?: (e: TransferEvent) => void) {
  return post("/api/v1/fact-check/text", JSON.stringify({ text }), { "Content-Type": "application/json" },
              REQUEST_TIMEOUT_MS.text, signal, onEvent);
}

export function verifyFile(mode: MediaMode, file: File, signal?: AbortSignal, onEvent?: (e: TransferEvent) => void) {
  const form = new FormData();
  form.append("file", file, file.name);
  return post(`/api/v1/fact-check/${mode}`, form, {}, REQUEST_TIMEOUT_MS[mode], signal, onEvent);
}

export function verify(mode: Mode, input: { text: string; file: File | null }, signal?: AbortSignal,
                       onEvent?: (e: TransferEvent) => void): Promise<FactCheckResponse> {
  if (mode === "text") return verifyText(input.text, signal, onEvent);
  if (!input.file) return Promise.reject(new ApiError("http", "Choose a file first.", 400, "missing_file"));
  return verifyFile(mode, input.file, signal, onEvent);
}
