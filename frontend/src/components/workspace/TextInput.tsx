import { MAX_TEXT_CHARS } from "../../lib/config";

interface Props {
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
}

export function TextInput({ value, onChange, disabled }: Props) {
  const over = value.length > MAX_TEXT_CHARS;
  return (
    <div>
      <label htmlFor="claim-text" className="eyebrow">Text to verify</label>
      <textarea
        id="claim-text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        rows={8}
        placeholder={"Paste the text you want checked — a paragraph, a social media post or a single claim.\n\nExample: The Eiffel Tower was completed in 1889. Water boils at 50 °C at sea level."}
        className="field mt-3 resize-y text-base leading-relaxed disabled:opacity-60"
      />
      <div className="mt-2 flex justify-between gap-4 text-xs text-muted">
        <span>Opinions, questions and greetings are recognised and not fact-checked.</span>
        <span className={`shrink-0 font-mono ${over ? "text-refuted" : ""}`}>
          {value.length.toLocaleString()} / {MAX_TEXT_CHARS.toLocaleString()}
        </span>
      </div>
    </div>
  );
}
