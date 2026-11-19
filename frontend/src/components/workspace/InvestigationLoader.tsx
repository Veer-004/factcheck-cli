import { motion, useReducedMotion } from "framer-motion";
import { Check } from "lucide-react";
import { useEffect, useState } from "react";
import type { Transfer } from "../../hooks/useInvestigation";
import { TYPICAL_DURATION, type Mode } from "../../lib/config";
import { fmtElapsed } from "../../lib/format";
import { computeProgress } from "../../lib/progress";
import { Checkpoints } from "../fx/Checkpoints";
import { LiquidFill } from "../fx/LiquidFill";
import { ModeAccent } from "../fx/LoaderFx";
import { Typewriter } from "../fx/TextFx";

// What the server does during the analysis stage. The API answers once, at the end, so this is a
// description of the pipeline, not a progress report (the UI says so).
const PIPELINE: Record<Mode, string[]> = {
  text: ["Extracting checkable claims", "Searching the web for evidence", "Verifying every quote against its source",
         "Applying the verdict rules"],
  image: ["Reading the image with OCR", "Extracting checkable claims", "Searching the web for evidence",
          "Verifying every quote against its source", "Applying the verdict rules"],
  video: ["Transcribing the speech", "Reading on-screen text from sampled frames", "Extracting checkable claims",
          "Searching the web for evidence", "Verifying every quote against its source", "Applying the verdict rules"],
  audio: ["Identifying the spoken language", "Transcribing English speech", "Extracting checkable claims",
          "Searching the web for evidence", "Verifying every quote against its source", "Applying the verdict rules"],
};

interface Props {
  mode: Mode;
  startedAt: number;
  fileName?: string | null;
  mediaSeconds: number | null;
  transfer: Transfer;
  onStopWaiting: () => void;
}

export function InvestigationLoader({ mode, startedAt, fileName, mediaSeconds, transfer, onStopWaiting }: Props) {
  const reduce = useReducedMotion();
  const [now, setNow] = useState(Date.now());
  const [i, setI] = useState(0);
  const steps = PIPELINE[mode];

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 250);
    const s = setInterval(() => setI((x) => (x + 1) % steps.length), 3600);
    return () => {
      clearInterval(t);
      clearInterval(s);
    };
  }, [steps.length]);

  const { level, checkpoints, stage } = computeProgress(mode, mediaSeconds, transfer, now);
  const active = checkpoints.find((c) => c.status === "active");
  const elapsed = now - startedAt;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="panel hud px-5 py-10 sm:px-10"
      role="status"
      aria-live="polite"
    >
      <div className="mx-auto flex max-w-3xl flex-col items-center text-center">
        <p className="eyebrow"><span className="text-shimmer">Investigation in progress</span></p>

        <div className="relative mt-6">
          <div className="absolute inset-[-14%] rounded-full bg-[radial-gradient(closest-side,rgba(0,213,192,0.16),transparent)]" />
          <LiquidFill level={level} size={250}>
            <div className="drop-shadow-[0_2px_10px_rgba(7,20,26,0.9)]">
              {stage === "ready" ? (
                <Check className="mx-auto h-10 w-10 text-ice" strokeWidth={2.5} />
              ) : (
                <p className="font-mono text-3xl text-ice">
                  {stage === "upload" && transfer.upload ? `${Math.round((transfer.upload.loaded / Math.max(1, transfer.upload.total)) * 100)}%`
                    : stage === "receive" && transfer.download?.total ? `${Math.round((transfer.download.loaded / transfer.download.total) * 100)}%`
                    : fmtElapsed(elapsed)}
                </p>
              )}
              <p className="mt-1 text-micro uppercase tracking-[0.2em] text-ice/80">{active?.label ?? "Report ready"}</p>
            </div>
          </LiquidFill>
        </div>

        <div className="mt-7 flex min-h-[3.25rem] w-full items-center justify-center">
          <p className="h-card text-ice">
            {stage === "ready" ? "Preparing the investigation report"
              : stage === "server" && !reduce ? <Typewriter text={steps[i]} />
              : stage === "server" ? "The server is analysing your input"
              : stage === "receive" ? "Receiving the results" : mode === "text" ? "Sending your text" : `Uploading ${fileName ?? "your file"}`}
          </p>
        </div>
        <div className="mt-2 flex h-8 items-center justify-center"><ModeAccent mode={mode} /></div>

        <div className="mt-6 w-full"><Checkpoints items={checkpoints} /></div>

        <p className="mt-6 max-w-xl text-xs leading-relaxed text-muted">
          Checkpoints are real events seen by your browser. While the server analyses, it reports no progress, so the
          liquid only rises toward the next checkpoint (marked <span className="font-mono">est. fill</span>) and the
          server-side steps above describe the pipeline rather than track it. This usually takes {TYPICAL_DURATION[mode]}.
          {elapsed > 90_000 && " Longer recordings take longer; video and audio jobs run one at a time — you can keep browsing."}
        </p>

        <button onClick={onStopWaiting} className="btn btn-ghost mt-6 text-xs">Stop waiting</button>
      </div>
    </motion.div>
  );
}
