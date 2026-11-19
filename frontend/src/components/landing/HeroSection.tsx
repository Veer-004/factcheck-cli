import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { ArrowRight, AudioLines, ChevronDown, Clapperboard, FileText, ImageIcon, ScanSearch } from "lucide-react";
import { BlurInWords, MaskRise, RotatingWord, ScrambleText } from "../fx/TextFx";
import { Sonar } from "../visual/Sonar";

interface HeroProps {
  onStart: () => void;
  onHowItWorks: () => void;
}

// Purely decorative: redacted "evidence fragments" drifting around the sonar. They contain no text
// or numbers, so they can never be mistaken for analysed content.
const FRAGMENTS = [
  { label: "claim", icon: FileText, pos: "left-[-6%] top-[12%]", bars: [86, 64, 40], delay: 0.9 },
  { label: "ocr line", icon: ImageIcon, pos: "right-[-8%] top-[24%]", bars: [70, 48], delay: 1.1 },
  { label: "speech segment", icon: AudioLines, pos: "left-[2%] bottom-[16%]", bars: [92, 58, 74], delay: 1.3 },
  { label: "source", icon: ScanSearch, pos: "right-[-2%] bottom-[8%]", bars: [60, 84], delay: 1.5 },
];

const MODES = [
  { icon: FileText, label: "Text" },
  { icon: ImageIcon, label: "Image" },
  { icon: Clapperboard, label: "Video" },
  { icon: AudioLines, label: "Audio" },
];

export function HeroSection({ onStart, onHowItWorks }: HeroProps) {
  const reduce = useReducedMotion();
  const rise = (delay: number) => ({
    initial: reduce ? false : { opacity: 0, y: 24 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.9, delay, ease: [0.16, 1, 0.3, 1] as const },
  });

  // Scroll parallax (smoothed by Lenis): text drifts up and fades, the sonar sinks more slowly.
  const { scrollY } = useScroll();
  const textY = useTransform(scrollY, [0, 700], [0, reduce ? 0 : -90]);
  const textOpacity = useTransform(scrollY, [0, 600], [1, reduce ? 1 : 0.25]);
  const sonarY = useTransform(scrollY, [0, 700], [0, reduce ? 0 : 140]);
  const sonarRotate = useTransform(scrollY, [0, 700], [0, reduce ? 0 : 18]);

  return (
    <section className="relative mx-auto grid min-h-[100svh] max-w-6xl items-center gap-10 px-5 pb-16 pt-28 sm:px-8 lg:grid-cols-[1.08fr_0.92fr] lg:gap-6 lg:pt-24">
      <motion.div className="relative z-10" style={{ y: textY, opacity: textOpacity }}>
        <motion.p {...rise(0.1)} className="eyebrow flex items-center gap-3">
          <span className="h-px w-10 bg-teal/60" />
          <span className="text-shimmer">Built to investigate</span>
          <RotatingWord words={["text", "images", "video", "audio"]} className="text-teal" />
        </motion.p>

        <h1 className="mt-6 font-display text-[clamp(3rem,8.5vw,6.4rem)] font-light leading-[0.95] tracking-[-0.035em]">
          <MaskRise delay={0.2}>Question</MaskRise>
          <MaskRise delay={0.32}>everything.</MaskRise>
          <MaskRise delay={0.48} className="mt-2 italic text-teal">
            <ScrambleText text="Verify anything." delay={350} duration={1200} />
          </MaskRise>
        </h1>

        <BlurInWords
          delay={0.9}
          className="mt-7 max-w-xl text-lead leading-relaxed text-muted"
          parts={[
            { t: "FactCheck pulls the factual claims out of " },
            { t: "text, images, videos and audio,", hl: true },
            { t: " searches the web for evidence and checks every quote against its source before it reaches a verdict — with a paper trail you can inspect." },
          ]}
        />

        <motion.div {...rise(1.25)} className="mt-9 flex flex-wrap items-center gap-4">
          <button onClick={onStart} className="group btn btn-primary btn-lg">
            <ScanSearch className="relative h-5 w-5" />
            <span className="relative">Start an Investigation</span>
            <ArrowRight className="relative h-4 w-4 transition-transform group-hover:translate-x-1" />
          </button>
          <button onClick={onHowItWorks} className="btn btn-ghost">
            How it works <ChevronDown className="h-4 w-4" />
          </button>
        </motion.div>

        <motion.ul {...rise(1.4)} className="mt-12 flex flex-wrap gap-2" aria-label="Supported inputs">
          {MODES.map(({ icon: Icon, label }) => (
            <li key={label} className="chip glass !text-xs">
              <Icon className="h-3.5 w-3.5 text-teal" /> {label}
            </li>
          ))}
        </motion.ul>
      </motion.div>

      {/* sonar composition */}
      <motion.div
        style={{ y: sonarY, rotate: sonarRotate }}
        initial={reduce ? false : { opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.4, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="relative mx-auto w-full max-w-[min(520px,88vw)]"
      >
        <div className="absolute inset-[-12%] rounded-full bg-[radial-gradient(closest-side,rgba(0,213,192,0.12),transparent)]" />
        <Sonar className="w-full" />
        {FRAGMENTS.map(({ label, icon: Icon, pos, bars, delay }) => (
          <motion.div
            key={label}
            aria-hidden
            initial={reduce ? false : { opacity: 0, y: 14 }}
            animate={reduce ? { opacity: 1 } : { opacity: 1, y: [0, -8, 0] }}
            transition={reduce ? undefined : { opacity: { delay, duration: 0.8 }, y: { delay, duration: 7, repeat: Infinity, ease: "easeInOut" } }}
            className={`glass absolute ${pos} hidden w-40 rounded-2xl p-3 sm:block`}
          >
            <div className="eyebrow flex items-center gap-1.5 !text-micro !tracking-[0.18em]">
              <Icon className="h-3 w-3 text-teal" /> {label}
            </div>
            <div className="mt-2.5 space-y-1.5">
              {bars.map((wd, i) => (
                <div key={i} className="h-1.5 rounded-full bg-[rgba(140,166,176,0.22)]" style={{ width: `${wd}%` }} />
              ))}
            </div>
          </motion.div>
        ))}
      </motion.div>
    </section>
  );
}
