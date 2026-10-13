import { ArrowDown, Share2 } from "lucide-react";

/**
 * Reserved Graph tab — no visualization yet (GraphRAG build is paused).
 * The layout is ready to host an interactive graph + traversal view later.
 */
export function GraphPlaceholder() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-lg border border-dashed border-border text-muted">
        <Share2 className="h-6 w-6" />
      </div>
      <p className="text-sm font-medium">Graph visualization</p>
      <p className="max-w-[220px] text-xs text-muted">
        Interactive entities, relationships and traversal paths will appear here
        once GraphRAG is built.
      </p>
      <div className="mt-2 w-full max-w-[240px] space-y-1 rounded border border-border bg-surface-2 p-2 text-left font-mono text-[10px] text-muted">
        <p>Question</p>
        <p className="flex items-center gap-1 pl-2"><ArrowDown className="h-3 w-3 shrink-0" aria-hidden="true" />Entities</p>
        <p className="flex items-center gap-1 pl-3"><ArrowDown className="h-3 w-3 shrink-0" aria-hidden="true" />Relationships</p>
        <p className="flex items-center gap-1 pl-4"><ArrowDown className="h-3 w-3 shrink-0" aria-hidden="true" />Traversal</p>
        <p className="flex items-center gap-1 pl-5"><ArrowDown className="h-3 w-3 shrink-0" aria-hidden="true" />Generated answer</p>
      </div>
    </div>
  );
}
