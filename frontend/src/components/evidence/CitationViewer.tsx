import { Quote } from "lucide-react";
import type { SourceItem } from "@/types";

/**
 * Inline citation chips (future: click a citation to jump to the evidence).
 */
export function CitationViewer({ sources }: { sources: SourceItem[] }) {
  if (!sources.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Quote className="h-3.5 w-3.5 text-muted" />
      {sources.slice(0, 5).map((s) => (
        <span
          key={s.doc_id}
          className="rounded border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-muted hover:text-accent"
          title={`${s.subject} — score ${s.score.toFixed(2)}`}
        >
          {s.doc_id}
        </span>
      ))}
    </div>
  );
}
