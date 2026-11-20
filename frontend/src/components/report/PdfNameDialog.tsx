import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { FileDown, FileText, X } from "lucide-react";
import { useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";
import { cleanPdfName, MAX_NAME_CHARS, nameWasAdjusted, pdfNameProblem } from "../../lib/pdfName";

interface Props {
  open: boolean;
  defaultName: string;
  busy: boolean;
  onCancel: () => void;
  onConfirm: (name: string) => void;
}

/** "Name your report" dialog shown before the PDF is created. Rendered in a portal so it sits above
 *  every transformed / blurred layer of the page. */
export function PdfNameDialog({ open, defaultName, busy, onCancel, onConfirm }: Props) {
  const reduce = useReducedMotion();
  const [name, setName] = useState(defaultName);
  const [touched, setTouched] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const hintId = useId();

  // Fresh default each time it opens; select the text so typing replaces it.
  useEffect(() => {
    if (!open) return;
    setName(defaultName);
    setTouched(false);
    const t = setTimeout(() => input.current?.select(), 60);
    const previous = document.activeElement as HTMLElement | null;
    return () => {
      clearTimeout(t);
      previous?.focus?.();
    };
  }, [open, defaultName]);

  // Esc closes the dialog wherever the focus is.
  useEffect(() => {
    if (!open) return;
    const onEsc = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onCancel();
    };
    window.addEventListener("keydown", onEsc);
    return () => window.removeEventListener("keydown", onEsc);
  }, [open, busy, onCancel]);

  const problem = pdfNameProblem(name);
  const finalName = `${cleanPdfName(name) || "FactCheck report"}.pdf`;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (problem || busy) return;
    onConfirm(cleanPdfName(name));
  };

  // Tab stays inside the dialog.
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "Tab" || !dialog.current) return;
    const items = [...dialog.current.querySelectorAll<HTMLElement>("input, button:not([disabled])")];
    if (!items.length) return;
    const first = items[0], last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div key="pdf-name" data-lenis-prevent onKeyDown={onKeyDown}
                    className="fixed inset-0 z-[80] grid place-items-center px-4"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: reduce ? 0 : 0.25 }}>
          <button type="button" aria-label="Close" tabIndex={-1} onClick={() => !busy && onCancel()}
                  className="absolute inset-0 cursor-default bg-abyss/70 backdrop-blur-sm" />
          <motion.div ref={dialog} role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={hintId}
                      className="glass-strong hud relative w-full max-w-lg rounded-3xl p-6 sm:p-8"
                      initial={reduce ? false : { opacity: 0, y: 18, scale: 0.97 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={reduce ? { opacity: 0 } : { opacity: 0, y: 10, scale: 0.98 }}
                      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}>
            <button type="button" onClick={onCancel} disabled={busy} aria-label="Close"
                    className="absolute right-4 top-4 grid h-8 w-8 place-items-center rounded-xl text-muted transition-colors hover:text-ice">
              <X className="h-4 w-4" />
            </button>

            <div className="flex items-center gap-3">
              <span className="grid h-11 w-11 place-items-center rounded-2xl border border-teal/30 bg-teal/10 text-teal">
                <FileText className="h-5 w-5" />
              </span>
              <div>
                <p className="eyebrow">Download PDF report</p>
                <h2 id={titleId} className="h-card mt-1">Name your report</h2>
              </div>
            </div>

            <form onSubmit={submit} className="mt-6" noValidate>
              <label htmlFor="pdf-name" className="eyebrow !text-micro">File name</label>
              <div className="relative mt-2">
                <input id="pdf-name" ref={input} value={name} maxLength={MAX_NAME_CHARS + 10} spellCheck={false}
                       autoComplete="off" disabled={busy}
                       onChange={(e) => setName(e.target.value)} onBlur={() => setTouched(true)}
                       aria-invalid={touched && !!problem} aria-describedby={hintId}
                       className="field !pr-14" placeholder="e.g. Eiffel Tower claims - November" />
                <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 rounded-md border border-line px-1.5 py-0.5 font-mono text-micro text-muted">
                  .pdf
                </span>
              </div>

              <div id={hintId} className="mt-2 min-h-[2.5rem] text-xs" aria-live="polite">
                {touched && problem ? (
                  <p className="text-refuted">{problem}</p>
                ) : (
                  <>
                    <p className="text-muted">
                      Saved as <span className="break-all font-mono text-ice/90">{finalName}</span>
                    </p>
                    {nameWasAdjusted(name) && (
                      <p className="mt-1 text-warn">Characters that can't be used in file names ( \ / : * ? " &lt; &gt; | ) are removed.</p>
                    )}
                  </>
                )}
              </div>

              <div className="mt-5 flex flex-col-reverse gap-2.5 sm:flex-row sm:justify-end">
                <button type="button" onClick={onCancel} disabled={busy} className="btn btn-ghost">Cancel</button>
                <button type="submit" disabled={busy || (touched && !!problem)} className="group btn btn-primary">
                  {busy ? <span className="relative h-4 w-4 animate-spin rounded-full border-2 border-abyss/30 border-t-abyss" />
                    : <FileDown className="relative h-4 w-4 transition-transform group-hover:translate-y-0.5" />}
                  <span className="relative">{busy ? "Creating PDF…" : "Download PDF"}</span>
                </button>
              </div>
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  );
}
