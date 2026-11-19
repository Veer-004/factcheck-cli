// Frontend configuration (Vite env vars, see .env.example). The backend enforces its own limits;
// these only let the UI explain them and reject obviously invalid files before uploading.

export type Mode = "text" | "image" | "video" | "audio";
export type MediaMode = Exclude<Mode, "text">;

const env = import.meta.env;

function num(value: string | undefined, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

export const API_BASE_URL = (env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

// The shared engine's default input limit (factcheck.config.Config.max_input_chars).
export const MAX_TEXT_CHARS = num(env.VITE_MAX_TEXT_CHARS, 4000);

export const MAX_UPLOAD_MB: Record<MediaMode, number> = {
  image: num(env.VITE_MAX_IMAGE_MB, 25),
  video: num(env.VITE_MAX_VIDEO_MB, 500),
  audio: num(env.VITE_MAX_AUDIO_MB, 500),
};

// Same allow-lists as the backend (src/factcheck_api/uploads.py).
export const EXTENSIONS: Record<MediaMode, string[]> = {
  image: [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"],
  video: [".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".ts", ".3gp", ".wmv", ".flv"],
  audio: [".mp3", ".wav", ".wave", ".m4a", ".m4b", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aif",
          ".aiff", ".aifc", ".amr", ".mka", ".weba", ".caf", ".ac3", ".mp2"],
};

// How long the browser waits for a result. Generous on purpose: the backend's own limits are
// 180 s for the fact-check plus up to 900 s (video) / 1800 s (audio) for extraction.
export const REQUEST_TIMEOUT_MS: Record<Mode, number> = {
  text: 10 * 60_000,
  image: 10 * 60_000,
  video: 45 * 60_000,
  audio: 45 * 60_000,
};

// Measured on the development laptop (README, "Long-running requests"); shown as guidance only.
export const TYPICAL_DURATION: Record<Mode, string> = {
  text: "15–60 seconds",
  image: "15–60 seconds",
  video: "about as long as the video itself, sometimes longer",
  audio: "around a minute for a few minutes of speech",
};

export const GITHUB_URL = "https://github.com/Veer-004/factcheck-cli";
