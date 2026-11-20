import { AnimatePresence, motion } from "framer-motion";
import { AudioPlayer } from "../ui/AudioPlayer";
import { AudioLines, Clapperboard, ImageIcon, RefreshCw, ShieldCheck, Trash2, UploadCloud } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";
import { useObjectUrl } from "../../hooks/useObjectUrl";
import { EXTENSIONS, MAX_UPLOAD_MB, type MediaMode } from "../../lib/config";
import { acceptFor, checkFile, MAX_SECONDS } from "../../lib/fileCheck";
import { fmtBytes, fmtTime } from "../../lib/format";
import { useToast } from "../ui/Toast";

interface Props {
  mode: MediaMode;
  file: File | null;
  seconds: number | null;
  onChange: (f: File | null, seconds: number | null) => void;
  disabled?: boolean;
}

const NOUN: Record<MediaMode, string> = { image: "an image", video: "a video", audio: "an audio file" };
const ICON = { image: ImageIcon, video: Clapperboard, audio: AudioLines };

export function FileUpload({ mode, file, seconds, onChange, disabled }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [checking, setChecking] = useState(false);
  const toast = useToast();
  const url = useObjectUrl(file);
  const Icon = ICON[mode];

  const accept = async (f: File | undefined) => {
    if (!f) return;
    setChecking(true);
    try {
      const r = await checkFile(mode, f);
      if (!r.ok) toast(r.title, r.body);
      else onChange(f, r.seconds);
    } finally {
      setChecking(false);
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    if (!disabled) void accept(e.dataTransfer.files[0]);
  };

  const limits = `${EXTENSIONS[mode].map((e) => e.slice(1)).join(" · ")} — up to ${MAX_UPLOAD_MB[mode]} MB`
    + (mode === "video" ? `, ${MAX_SECONDS.video / 60} min` : mode === "audio" ? `, ${MAX_SECONDS.audio / 60} min · English speech` : "");

  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <p className="eyebrow whitespace-nowrap">{mode} file</p>
        <p className="chip" title="Files are checked by their content, not just their name">
          <ShieldCheck className="h-3 w-3 text-teal" /> {mode === "audio" ? "audio" : `${mode}s`} only
          <span className="hidden sm:inline">— checked by content</span>
        </p>
      </div>
      <input
        ref={input}
        type="file"
        className="hidden"
        accept={acceptFor(mode)}
        onChange={(e) => {
          void accept(e.target.files?.[0]);
          e.target.value = ""; // allow choosing the same file again
        }}
      />
      <AnimatePresence mode="wait" initial={false}>
        {!file ? (
          <motion.button
            key="drop"
            type="button"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            disabled={disabled || checking}
            onClick={() => input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
            className={`dropzone mt-3 ${drag ? "dropzone-active" : ""}`}
          >
            {(drag || checking) && <span className="animate-scan absolute inset-x-0 h-1/2 bg-[linear-gradient(transparent,rgba(0,213,192,0.12),transparent)]" />}
            <span className="relative grid h-14 w-14 place-items-center rounded-2xl border border-teal/30 bg-teal/10 text-teal">
              {checking ? <span className="h-5 w-5 animate-spin rounded-full border-2 border-teal/30 border-t-teal" /> : drag ? <UploadCloud className="h-6 w-6" /> : <Icon className="h-6 w-6" />}
            </span>
            <span className="relative mt-4 font-semibold">{checking ? "Checking the file…" : `Drop ${NOUN[mode]} here, or click to browse`}</span>
            <span className="relative mt-2 max-w-md text-xs leading-relaxed text-muted">{limits}</span>
          </motion.button>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="glass mt-3 overflow-hidden rounded-2xl"
          >
            <div className="grid place-items-center bg-black/30">
              {url && mode === "image" && <img src={url} alt="Selected image" className="max-h-[22rem] w-auto object-contain" />}
              {url && mode === "video" && <video src={url} controls className="max-h-[22rem] w-full" />}
              {url && mode === "audio" && <AudioPlayer src={url} className="m-4 w-[calc(100%-2rem)] sm:m-5 sm:w-[calc(100%-2.5rem)]" />}
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-4 py-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold" title={file.name}>{file.name}</p>
                <p className="font-mono text-xs text-muted">
                  {fmtBytes(file.size)}{seconds !== null ? ` · ${fmtTime(seconds)}` : ""} · verified {mode}
                </p>
              </div>
              <div className="flex gap-2">
                <button type="button" disabled={disabled || checking} onClick={() => input.current?.click()} className="btn btn-glass btn-sm">
                  <RefreshCw className="h-3.5 w-3.5" /> Replace
                </button>
                <button type="button" disabled={disabled} onClick={() => onChange(null, null)} className="btn btn-glass btn-sm btn-danger">
                  <Trash2 className="h-3.5 w-3.5" /> Remove
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
