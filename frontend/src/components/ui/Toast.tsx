import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, X } from "lucide-react";
import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";

interface ToastItem {
  id: number;
  title: string;
  body?: string;
}

const ToastContext = createContext<(title: string, body?: string) => void>(() => {});

export function useToast() {
  return useContext(ToastContext);
}

/** Small glass notifications for input problems (wrong file type, too large, ...). */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const next = useRef(1);

  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((x) => x.id !== id)), []);
  const push = useCallback((title: string, body?: string) => {
    const id = next.current++;
    setItems((xs) => [...xs.slice(-2), { id, title, body }]);
    setTimeout(() => dismiss(id), 6000);
  }, [dismiss]);

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed inset-x-0 bottom-4 z-50 flex flex-col items-center gap-2 px-4 sm:bottom-6" aria-live="polite">
        <AnimatePresence>
          {items.map((t) => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, y: 16, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.98 }}
              className="pointer-events-auto flex w-full max-w-md items-start gap-3 rounded-xl px-4 py-3 glass-strong"
              role="status"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" />
              <div className="flex-1 text-sm">
                <p className="font-semibold">{t.title}</p>
                {t.body && <p className="mt-0.5 text-muted">{t.body}</p>}
              </div>
              <button onClick={() => dismiss(t.id)} className="text-muted hover:text-ice" aria-label="Dismiss">
                <X className="h-4 w-4" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}
