import { AnimatePresence, motion } from "framer-motion";
import type { Health } from "../../hooks/useHealth";
import type { Inputs, Investigation } from "../../hooks/useInvestigation";
import { MAX_TEXT_CHARS, type MediaMode, type Mode } from "../../lib/config";
import { ScrambleText } from "../fx/TextFx";
import { InvestigationReport } from "../report/InvestigationReport";
import { BackendStatus } from "./BackendStatus";
import { ErrorState } from "./ErrorState";
import { FileUpload } from "./FileUpload";
import { InvestigationLoader } from "./InvestigationLoader";
import { ModalitySelector } from "./ModalitySelector";
import { MODE_BY_ID } from "./modes";
import { TextInput } from "./TextInput";
import { VerifyButton } from "./VerifyButton";

interface Props {
  mode: Mode;
  setMode: (m: Mode) => void;
  inputs: Inputs;
  setInputs: (f: (i: Inputs) => Inputs) => void;
  investigation: Investigation;
  onVerify: () => void;
  onStopWaiting: () => void;
  onReset: () => void;
  health: Health;
  onRecheck: () => void;
}

function readiness(mode: Mode, inputs: Inputs): string | null {
  if (mode === "text") {
    const t = inputs.text.trim();
    if (!t) return "Enter some text to verify.";
    if (inputs.text.length > MAX_TEXT_CHARS) return `Shorten the text to ${MAX_TEXT_CHARS.toLocaleString()} characters or fewer.`;
    return null;
  }
  return inputs.files[mode] ? null : `Choose ${mode === "image" ? "an image" : mode === "video" ? "a video" : "an audio file"} to investigate.`;
}

export function InvestigationWorkspace(p: Props) {
  const { investigation: inv } = p;
  const running = inv.phase === "running";
  const problem = readiness(p.mode, p.inputs);
  const view = inv.phase === "idle" ? "input" : inv.phase;

  return (
    <main className="mx-auto max-w-5xl px-4 pb-24 pt-28 sm:px-8">
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                  className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="eyebrow">Investigation workspace</p>
          <h1 className="h-page mt-3">
            What should we <ScrambleText text="investigate?" delay={250} duration={900} className="italic text-teal" />
          </h1>
        </div>
        <BackendStatus health={p.health} onRecheck={p.onRecheck} />
      </motion.div>

      {view === "input" && (
        <div className="mt-8">
          <ModalitySelector mode={p.mode} onChange={p.setMode} disabled={running} />
        </div>
      )}

      <div className="mt-6">
        {/* In a hidden tab the browser pauses animation frames, so "wait" would hold the finished report
            back until the user returns; swap immediately instead. */}
        <AnimatePresence mode={typeof document !== "undefined" && document.hidden ? "sync" : "wait"}>
          {view === "input" && (
            <motion.section
              key={`input-${p.mode}`}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
              className="panel hud space-y-6 p-5 sm:p-8"
            >
              {p.mode === "text" ? (
                <TextInput value={p.inputs.text} onChange={(text) => p.setInputs((i) => ({ ...i, text }))} />
              ) : (
                <FileUpload
                  mode={p.mode}
                  file={p.inputs.files[p.mode as MediaMode]}
                  seconds={p.inputs.durations[p.mode as MediaMode]}
                  onChange={(f, seconds) => p.setInputs((i) => ({
                    ...i,
                    files: { ...i.files, [p.mode]: f },
                    durations: { ...i.durations, [p.mode]: f ? seconds : null },
                  }))}
                />
              )}
              <VerifyButton label={MODE_BY_ID[p.mode].verb} onClick={p.onVerify} disabled={!!problem || running} busy={running}
                            hint={problem ?? (p.health.state === "offline" ? "The server looks offline; verification will fail until it is running." : null)} />
            </motion.section>
          )}

          {inv.phase === "running" && (
            <InvestigationLoader key="loader" mode={inv.mode} startedAt={inv.startedAt} fileName={inv.file?.name}
                                 mediaSeconds={inv.mediaSeconds} transfer={inv.transfer} onStopWaiting={p.onStopWaiting} />
          )}

          {inv.phase === "error" && (
            <ErrorState key="error" error={inv.error} onRetry={p.onVerify} onEdit={p.onReset} />
          )}

          {inv.phase === "done" && (
            <motion.div key="report" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <InvestigationReport
                result={inv.result}
                file={inv.file}
                onEdit={p.onReset}
                onNew={() => {
                  p.setInputs((i) => (inv.mode === "text" ? { ...i, text: "" } : { ...i, files: { ...i.files, [inv.mode]: null } }));
                  p.onReset();
                }}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}
