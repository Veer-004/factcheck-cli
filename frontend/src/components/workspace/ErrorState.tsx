import { motion } from "framer-motion";
import { AlertOctagon, CloudOff, FileWarning, PencilLine, RotateCcw, ServerCrash, TimerOff } from "lucide-react";
import type { ApiError } from "../../lib/api";

interface Props {
  error: ApiError;
  onRetry: () => void;
  onEdit: () => void;
}

function describe(e: ApiError) {
  if (e.kind === "network") return { icon: CloudOff, title: "The FactCheck server can't be reached" };
  if (e.kind === "timeout") return { icon: TimerOff, title: "The investigation took too long" };
  if (e.kind === "invalid_response") return { icon: ServerCrash, title: "The result couldn't be read" };
  switch (e.status) {
    case 400:
      return { icon: FileWarning, title: "This input can't be investigated" };
    case 413:
      return { icon: FileWarning, title: "The file is too large" };
    case 415:
      return { icon: FileWarning, title: "Unsupported file type" };
    case 422:
      return { icon: FileWarning, title: "The request was incomplete" };
    case 503:
      return { icon: CloudOff, title: "A required service is unavailable" };
    default:
      return { icon: AlertOctagon, title: "The investigation failed" };
  }
}

const HINT: Record<string, string> = {
  service_not_configured: "The server is missing its fact-checking API keys. Check the backend's .env file.",
  external_service_error: "An external service (the language model or web search) failed. Trying again later often helps.",
  dependency_unavailable: "A speech or OCR model isn't installed on the server.",
};

export function ErrorState({ error, onRetry, onEdit }: Props) {
  const { icon: Icon, title } = describe(error);
  const fields = error.kind === "http" && error.status === 422
    ? error.details.map((d) => (Array.isArray(d.loc) ? d.loc.slice(1).join(".") : "")).filter(Boolean)
    : [];
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="panel hud px-6 py-10 sm:px-10"
      role="alert"
    >
      <div className="mx-auto flex max-w-xl flex-col items-center text-center">
        <span className="grid h-14 w-14 place-items-center rounded-full border border-refuted/30 bg-refuted/10 text-refuted">
          <Icon className="h-6 w-6" />
        </span>
        <h2 className="h-card mt-5">{title}</h2>
        <p className="mt-3 text-sm leading-relaxed text-muted">{error.message}</p>
        {error.code && HINT[error.code] && <p className="mt-2 text-sm leading-relaxed text-muted">{HINT[error.code]}</p>}
        {fields.length > 0 && <p className="mt-2 font-mono text-xs text-muted">Fields: {fields.join(", ")}</p>}
        {error.requestId && <p className="mt-4 font-mono text-micro text-muted/70">Request ID {error.requestId}</p>}
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          {error.retryable && (
            <button onClick={onRetry} className="btn btn-primary">
              <RotateCcw className="h-4 w-4" /> Try again
            </button>
          )}
          <button onClick={onEdit} className="btn btn-glass">
            <PencilLine className="h-4 w-4" /> {error.retryable ? "Change input" : "Back to input"}
          </button>
        </div>
      </div>
    </motion.div>
  );
}
