// The Deep Ocean palette as values, for the places that need raw colours (SVG attributes, inline
// styles, canvas, the PDF). Must match the tokens in src/index.css (@theme).

export const PALETTE = {
  abyss: "#07141A",
  deep: "#0B1C24",
  glass: "#14252D",
  teal: "#00D5C0",
  tealBright: "#3FF3E0",
  tealDeep: "#035E63",
  ice: "#E5FAFF",
  muted: "#8CA6B0",
  supported: "#5BE3A6",
  refuted: "#FF7D6E",
  warn: "#F2B84B",
  conflict: "#7EC8FF",
  neutral: "#A3ADB1",
  failed: "#C9A27E",
} as const;

export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
