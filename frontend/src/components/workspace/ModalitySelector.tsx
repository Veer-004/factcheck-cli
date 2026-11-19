import { motion } from "framer-motion";
import type { Mode } from "../../lib/config";
import { MODES } from "./modes";

interface Props {
  mode: Mode;
  onChange: (m: Mode) => void;
  disabled?: boolean;
}

export function ModalitySelector({ mode, onChange, disabled }: Props) {
  return (
    <div role="radiogroup" aria-label="What do you want to investigate?" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {MODES.map(({ id, title, icon: Icon, description }) => {
        const active = id === mode;
        return (
          <button
            key={id}
            role="radio"
            aria-checked={active}
            disabled={disabled}
            onClick={() => onChange(id)}
            className={`group relative overflow-hidden rounded-2xl p-4 text-left transition-all duration-300 disabled:cursor-not-allowed disabled:opacity-50 sm:p-5 ${
              active ? "glass-strong" : "glass hover:-translate-y-0.5 hover:border-teal/25"
            }`}
          >
            {active && (
              <motion.span
                layoutId="mode-active"
                className="absolute inset-0 rounded-2xl hairline-glow"
                transition={{ type: "spring", stiffness: 380, damping: 32 }}
              />
            )}
            <span
              className={`relative grid h-10 w-10 place-items-center rounded-xl border transition-colors ${
                active ? "border-teal/50 bg-teal/15 text-teal" : "border-line text-muted group-hover:text-teal"
              }`}
            >
              <Icon className="h-5 w-5" />
            </span>
            <span className="relative mt-4 block font-semibold">{title}</span>
            <span className="relative mt-1 hidden text-xs leading-relaxed text-muted sm:block">{description}</span>
          </button>
        );
      })}
    </div>
  );
}
