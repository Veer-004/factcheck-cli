import { motion } from "framer-motion";
import { ScanSearch } from "lucide-react";

interface Props {
  label: string;
  onClick: () => void;
  disabled: boolean;
  busy: boolean;
  hint?: string | null;
}

export function VerifyButton({ label, onClick, disabled, busy, hint }: Props) {
  return (
    <div className="flex flex-col items-stretch gap-3 sm:flex-row sm:items-center sm:justify-between">
      <p className="text-xs text-muted" aria-live="polite">{hint}</p>
      <motion.button
        whileTap={disabled ? undefined : { scale: 0.97 }}
        onClick={onClick}
        disabled={disabled}
        aria-busy={busy}
        className="group btn btn-primary btn-lg"
      >
        <span className="relative grid h-5 w-5 place-items-center">
          {busy ? (
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-abyss/30 border-t-abyss" />
          ) : (
            <ScanSearch className="h-5 w-5 transition-transform group-hover:rotate-12" />
          )}
        </span>
        <span className="relative">{busy ? "Investigating…" : label}</span>
      </motion.button>
    </div>
  );
}
