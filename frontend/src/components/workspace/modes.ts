import { AudioLines, Clapperboard, FileText, ImageIcon, type LucideIcon } from "lucide-react";
import type { Mode } from "../../lib/config";

export interface ModeInfo {
  id: Mode;
  title: string;
  icon: LucideIcon;
  description: string;
  verb: string; // verify button label
}

export const MODES: ModeInfo[] = [
  { id: "text", title: "Text", icon: FileText, verb: "Verify text",
    description: "Paste a paragraph, post or claim. Each factual statement is checked separately." },
  { id: "image", title: "Image", icon: ImageIcon, verb: "Investigate image",
    description: "Screenshots, memes, posters. Text is read with OCR, then fact-checked." },
  { id: "video", title: "Video", icon: Clapperboard, verb: "Investigate video",
    description: "Speech and on-screen text are extracted and timestamped." },
  { id: "audio", title: "Audio", icon: AudioLines, verb: "Investigate audio",
    description: "English speech is transcribed and traced to the exact words spoken." },
];

export const MODE_BY_ID = Object.fromEntries(MODES.map((m) => [m.id, m])) as Record<Mode, ModeInfo>;
