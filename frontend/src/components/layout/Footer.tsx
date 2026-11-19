import { GITHUB_URL } from "../../lib/config";
import { BrandMark } from "../visual/BrandMark";

export function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-5 py-8 text-sm text-muted sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <div className="flex items-center gap-2.5">
          <BrandMark className="h-5 w-5" />
          <span className="font-display text-ice">FactCheck</span>
          <span className="hidden sm:inline">· college project</span>
        </div>
        <p className="max-w-md text-xs leading-relaxed">
          Verdicts are generated automatically from web sources and can be wrong. Always read the cited sources.
        </p>
        <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className="text-xs hover:text-ice">
          Source on GitHub ↗
        </a>
      </div>
    </footer>
  );
}
