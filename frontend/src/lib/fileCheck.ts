// Per-mode upload constraints, checked in the browser before anything is uploaded:
// extension allow-list, declared MIME family, the file's real content (magic bytes), size, and media length.
// The backend validates everything again; this only stops obviously wrong files early with a clear message.

import { EXTENSIONS, MAX_UPLOAD_MB, type MediaMode } from "./config";
import { fmtBytes } from "./format";

export type Sniffed = "image" | "video" | "audio" | "av" | "document" | "unknown"; // av = audio or video container

// Backend defaults: factcheck-video --max-duration 600 s, factcheck-audio --max-duration 3600 s.
export const MAX_SECONDS: Record<Exclude<MediaMode, "image">, number> = { video: 600, audio: 3600 };

const NOUN: Record<Sniffed, string> = {
  image: "an image", video: "a video", audio: "an audio file", av: "a media file", document: "a document",
  unknown: "an unrecognised file",
};
const MODE_FOR: Partial<Record<Sniffed, string>> = { image: "Image", video: "Video", audio: "Audio" };

function ascii(b: Uint8Array, start: number, len: number) {
  return String.fromCharCode(...b.slice(start, start + len));
}

/** Identify a file by its first bytes (container signatures). */
export function sniffBytes(b: Uint8Array): Sniffed {
  const has = (...sig: number[]) => sig.every((v, i) => b[i] === v);
  if (has(0x89, 0x50, 0x4e, 0x47) || has(0xff, 0xd8, 0xff) || ascii(b, 0, 4) === "GIF8" || ascii(b, 0, 2) === "BM"
      || has(0x49, 0x49, 0x2a, 0x00) || has(0x4d, 0x4d, 0x00, 0x2a)) return "image";
  if (ascii(b, 0, 4) === "RIFF") {
    const kind = ascii(b, 8, 4);
    if (kind === "WEBP") return "image";
    if (kind === "WAVE") return "audio";
    if (kind === "AVI ") return "video";
  }
  if (ascii(b, 4, 4) === "ftyp") {
    const brand = ascii(b, 8, 4);
    if (["heic", "heix", "mif1", "msf1", "avif"].includes(brand)) return "image";
    if (["M4A ", "M4B ", "M4P "].includes(brand)) return "audio";
    return "av"; // mp4/mov/3gp; can be audio-only
  }
  if (has(0x1a, 0x45, 0xdf, 0xa3) || ascii(b, 0, 4) === "OggS" || has(0x30, 0x26, 0xb2, 0x75)) return "av";
  if (ascii(b, 0, 3) === "FLV" || has(0x00, 0x00, 0x01, 0xba) || has(0x00, 0x00, 0x01, 0xb3)
      || (b[0] === 0x47 && b[188] === 0x47)) return "video";
  if (ascii(b, 0, 3) === "ID3" || (b[0] === 0xff && (b[1] & 0xe0) === 0xe0) || ascii(b, 0, 4) === "fLaC"
      || (ascii(b, 0, 4) === "FORM" && ascii(b, 8, 3) === "AIF") || ascii(b, 0, 5) === "#!AMR" || ascii(b, 0, 4) === "caff"
      || has(0x0b, 0x77)) return "audio";
  if (ascii(b, 0, 4) === "%PDF" || has(0x50, 0x4b, 0x03, 0x04) || ascii(b, 0, 5) === "<?xml" || ascii(b, 0, 1) === "{") return "document";
  return "unknown";
}

async function sniffFile(file: File): Promise<Sniffed> {
  const buf = new Uint8Array(await file.slice(0, 256).arrayBuffer());
  return sniffBytes(buf);
}

/** Media length from the browser's own decoder; null when it can't tell (the server will check). */
function mediaSeconds(file: File, kind: "video" | "audio"): Promise<number | null> {
  return new Promise((resolve) => {
    const el = document.createElement(kind);
    const url = URL.createObjectURL(file);
    const done = (v: number | null) => {
      clearTimeout(timer);
      URL.revokeObjectURL(url);
      el.removeAttribute("src");
      resolve(v);
    };
    const timer = setTimeout(() => done(null), 5000);
    el.preload = "metadata";
    el.onloadedmetadata = () => done(Number.isFinite(el.duration) ? el.duration : null);
    el.onerror = () => done(null);
    el.src = url;
  });
}

export type FileCheck = { ok: true; seconds: number | null } | { ok: false; title: string; body: string };

function reject(title: string, body: string): FileCheck {
  return { ok: false, title, body };
}

export async function checkFile(mode: MediaMode, file: File): Promise<FileCheck> {
  const ext = file.name.includes(".") ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase() : "";
  const wanted = NOUN[mode];
  const allowed = EXTENSIONS[mode];
  if (!allowed.includes(ext)) {
    return reject(`Only ${wanted === "an audio file" ? "audio files" : `${mode}s`} can be used here`,
                  `${ext || "A file without an extension"} isn't accepted in ${MODE_FOR[mode]} mode. Allowed: ${allowed.map((e) => e.slice(1)).join(", ")}.`);
  }
  const family = (file.type || "").split("/")[0];
  if (family && family !== mode && file.type !== "application/octet-stream"
      && !(mode === "audio" && (file.type === "video/webm" || file.type === "video/ogg" || file.type === "video/mp4"))) {
    const other = family === "image" ? "an image" : family === "video" ? "a video" : family === "audio" ? "an audio file" : "not a media file";
    return reject(`This is ${other}, not ${wanted}`, `${MODE_FOR[mode]} mode only takes ${mode === "audio" ? "audio files" : `${mode}s`}.`
                  + (MODE_FOR[family as Sniffed] ? ` Use ${MODE_FOR[family as Sniffed]} mode for it.` : ""));
  }
  if (file.size === 0) return reject("The file is empty", "Choose a file that contains data.");
  const limit = MAX_UPLOAD_MB[mode] * 1024 * 1024;
  if (file.size > limit) return reject("The file is too large", `It is ${fmtBytes(file.size)}; the limit is ${MAX_UPLOAD_MB[mode]} MB.`);

  const real = await sniffFile(file);
  const mismatch =
    (mode === "image" && real !== "image" && real !== "unknown") ||
    (mode === "video" && (real === "image" || real === "audio" || real === "document")) ||
    (mode === "audio" && (real === "image" || real === "video" || real === "document"));
  if (mismatch) {
    const hint = MODE_FOR[real] ? ` Use ${MODE_FOR[real]} mode for it.` : "";
    return reject(`This file is ${NOUN[real]}, not ${wanted}`,
                  `Its content doesn't match ${MODE_FOR[mode]} mode, whatever its name says.${hint}`);
  }

  if (mode === "image") return { ok: true, seconds: null };
  const seconds = await mediaSeconds(file, mode);
  if (seconds !== null && seconds > MAX_SECONDS[mode]) {
    const mins = Math.round(MAX_SECONDS[mode] / 60);
    return reject(`The ${mode === "video" ? "video" : "recording"} is too long`,
                  `It is ${Math.round(seconds / 60)} min; the limit is ${mins} min. Trim it and try again.`);
  }
  return { ok: true, seconds };
}

/** The file picker's accept attribute: the MIME family plus the allowed extensions. */
export function acceptFor(mode: MediaMode): string {
  return [`${mode}/*`, ...EXTENSIONS[mode]].join(",");
}
