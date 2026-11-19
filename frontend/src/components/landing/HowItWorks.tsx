import { motion, useReducedMotion } from "framer-motion";
import { AudioLines, Clapperboard, FileSearch, FileText, ImageIcon, Quote, Scale, ScanText } from "lucide-react";
import { MarkerHighlight } from "../fx/TextFx";

// Describes what the backend actually does (see README, "How it works").
const STEPS = [
  {
    icon: ScanText,
    title: "Read the evidence",
    body: "Text is read as written. Images go through OCR; videos through local speech recognition and on-screen text reading; audio through English speech recognition — each with recognition confidence kept.",
  },
  {
    icon: FileSearch,
    title: "Isolate the claims",
    body: "Checkable factual statements are separated from opinions, greetings and calls to action. Every claim stays tied to the exact words it came from.",
  },
  {
    icon: Quote,
    title: "Search and quote",
    body: "Web evidence is retrieved for each claim. Every quote the AI cites is verified word-for-word against the source; invented or off-topic citations are rejected.",
  },
  {
    icon: Scale,
    title: "Rule on the verdict",
    body: "Deterministic rules — not the model alone — pick one of eight verdicts. The evidence score is an experimental rubric of evidence strength, not a probability of truth.",
  },
];

const INPUTS = [
  { icon: FileText, title: "Text", body: "Paste a paragraph, a post or a single claim." },
  { icon: ImageIcon, title: "Image", body: "Screenshots, memes, posters — claims are read from the pixels and traced to their lines." },
  { icon: Clapperboard, title: "Video", body: "Speech and on-screen text, de-duplicated, with timestamps for every claim." },
  { icon: AudioLines, title: "Audio", body: "English podcasts, interviews and voice notes, traced to the exact words spoken." },
];

export function HowItWorks() {
  const reduce = useReducedMotion();
  const reveal = (i: number) => ({
    initial: reduce ? false : { opacity: 0, y: 28 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true, margin: "-60px" },
    transition: { duration: 0.7, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] as const },
  });

  return (
    <section id="how-it-works" className="relative mx-auto max-w-6xl scroll-mt-24 px-5 py-24 sm:px-8">
      <motion.div {...reveal(0)} className="max-w-2xl">
        <p className="eyebrow">How it works</p>
        <h2 className="h-page mt-4">
          Every verdict comes with a <MarkerHighlight><span className="italic text-teal">paper trail.</span></MarkerHighlight>
        </h2>
      </motion.div>

      <ol className="relative mt-14 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <div aria-hidden className="absolute left-0 right-0 top-[2.15rem] hidden h-px bg-[linear-gradient(90deg,transparent,rgba(0,213,192,0.35),transparent)] lg:block" />
        {STEPS.map(({ icon: Icon, title, body }, i) => (
          <motion.li key={title} {...reveal(i + 1)} className="glass relative rounded-2xl p-6 transition-transform duration-300 hover:-translate-y-1">
            <div className="flex items-center gap-3">
              <span className="grid h-11 w-11 place-items-center rounded-full border border-teal/30 bg-abyss text-teal">
                <Icon className="h-5 w-5" />
              </span>
              <span className="font-mono text-xs text-muted">0{i + 1}</span>
            </div>
            <h3 className="mt-5 text-lg font-semibold">{title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">{body}</p>
          </motion.li>
        ))}
      </ol>

      <motion.div {...reveal(1)} className="mt-24 grid gap-10 lg:grid-cols-[0.8fr_1.2fr] lg:items-end">
        <div>
          <p className="eyebrow">Four ways in</p>
          <h2 className="h-page mt-4">
            One investigation,<br />
            <span className="italic text-teal">any medium.</span>
          </h2>
        </div>
        <ul className="grid gap-3 sm:grid-cols-2">
          {INPUTS.map(({ icon: Icon, title, body }) => (
            <li key={title} className="glass flex gap-4 rounded-2xl p-5 transition-transform duration-300 hover:-translate-y-1">
              <Icon className="mt-0.5 h-5 w-5 shrink-0 text-teal" />
              <div>
                <h3 className="font-semibold">{title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-muted">{body}</p>
              </div>
            </li>
          ))}
        </ul>
      </motion.div>
    </section>
  );
}
