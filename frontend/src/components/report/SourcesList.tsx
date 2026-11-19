import { ExternalLink, ShieldCheck } from "lucide-react";
import { domainOf, fmtDate, safeHref, titleCase } from "../../lib/format";
import { PALETTE } from "../../lib/palette";
import type { Citation } from "../../lib/types";

const ROLE: Record<Citation["role"], { title: string; color: string }> = {
  supporting: { title: "Supporting evidence", color: PALETTE.supported },
  refuting: { title: "Contradicting evidence", color: PALETTE.refuted },
  context: { title: "Missing context", color: PALETTE.warn },
};

function SourceItem({ c }: { c: Citation }) {
  const href = safeHref(c.url);
  const date = fmtDate(c.published_date);
  const meta = [domainOf(c.url), titleCase(c.authority), titleCase(c.directness), date].filter(Boolean);
  return (
    <li className="rounded-xl border border-line bg-abyss/40 p-4 transition-colors hover:border-teal/25">
      <blockquote className="border-l-2 pl-3 text-sm leading-relaxed text-ice/90" style={{ borderColor: ROLE[c.role].color }}>
        “{c.quote}”
      </blockquote>
      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
        {href ? (
          <a href={href} target="_blank" rel="noopener noreferrer"
             className="inline-flex max-w-full items-center gap-1 font-semibold text-ice hover:text-teal">
            <span className="truncate">{c.title || domainOf(c.url)}</span>
            <ExternalLink className="h-3 w-3 shrink-0" />
          </a>
        ) : (
          <span className="font-semibold text-ice">{c.title || c.url}</span>
        )}
        <span>{meta.join(" · ")}</span>
        {c.fact_check_site && <span className="inline-flex items-center gap-1 text-teal"><ShieldCheck className="h-3 w-3" />fact-check site</span>}
        {c.snippet_only && <span title="Only the search snippet could be read">snippet only</span>}
      </div>
    </li>
  );
}

/** Validated citations, grouped by role (only roles that have citations are shown). */
export function SourcesList({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <div className="space-y-5">
      {(Object.keys(ROLE) as Citation["role"][]).map((role) => {
        const items = citations.filter((c) => c.role === role);
        if (!items.length) return null;
        return (
          <section key={role}>
            <h4 className="eyebrow flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: ROLE[role].color }} />
              {ROLE[role].title} <span className="text-muted/60">· {items.length}</span>
            </h4>
            <ul className="mt-3 space-y-2.5">
              {items.map((c, i) => <SourceItem key={`${c.evidence_id}-${i}`} c={c} />)}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
