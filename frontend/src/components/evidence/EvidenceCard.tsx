import { CheckCircle2, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import type { SourceItem } from "@/types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/utils/cn";
import { confidenceLabel, dominantComponent } from "@/utils/formatters";
import { useDraftStore } from "@/store/useDraftStore";

export function EvidenceCard({
  source,
  selected,
  isGraph,
}: {
  source: SourceItem;
  selected: boolean;
  isGraph: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const selectEvidence = useDraftStore((s) => s.selectEvidence);
  const dominant = dominantComponent(source);

  const scores = [
    source.dense_score != null && ["Dense", `${source.dense_score.toFixed(3)}`],
    source.bm25_score != null && ["BM25", `${source.bm25_score.toFixed(3)}`],
    source.rrf_score != null && ["RRF", `${source.rrf_score.toFixed(3)}`],
    source.rerank_score != null && ["Rerank", `${source.rerank_score.toFixed(3)}`],
  ].filter(Boolean) as Array<[string, string]>;

  return (
    <div
      className={cn(
        "rounded-lg border bg-surface-2 p-2.5 transition-colors",
        selected ? "border-accent/60 ring-1 ring-accent/40" : "border-border"
      )}
      onClick={() => selectEvidence(selected ? null : source)}
      title="Click to highlight this evidence for the current sentence"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-xs font-semibold text-accent">{source.doc_id}</p>
          <p className="truncate text-[11px] text-muted">{source.subject || "—"}</p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {isGraph && <Badge variant="accent" className="text-[9px]">graph</Badge>}
          <Badge variant={selected ? "success" : "default"} className="text-[9px]">
            {confidenceLabel(source.score)}
          </Badge>
        </div>
      </div>

      <div className="mt-1.5 flex flex-wrap gap-1">
        {scores.map(([label, val]) => (
          <span
            key={label}
            className={cn(
              "rounded px-1.5 py-0.5 font-mono text-[9px]",
              dominant === label
                ? "bg-accent/20 text-accent"
                : "bg-surface text-muted"
            )}
          >
            {label} {val}
          </span>
        ))}
      </div>

      <button
        className="mt-2 flex w-full items-center gap-1 text-[10px] text-muted hover:text-foreground"
        onClick={(e) => {
          e.stopPropagation();
          setExpanded((v) => !v);
        }}
      >
        {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        {expanded ? "Collapse" : "Expand"} transcript
      </button>

      {expanded && (
        <div className="mt-2 max-h-44 space-y-1.5 overflow-y-auto rounded border border-border bg-background/60 p-2">
          <p className="text-[10px] font-medium text-foreground/80">{source.question}</p>
          <p className="whitespace-pre-wrap border-t border-border pt-1 text-[10px] leading-relaxed text-muted">
            {source.answer}
          </p>
        </div>
      )}

      {selected && (
        <div className="mt-2 flex items-center gap-1 text-[10px] text-success">
          <CheckCircle2 className="h-3 w-3" />
          Linked to selected sentence
        </div>
      )}
    </div>
  );
}
