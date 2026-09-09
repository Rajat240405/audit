import { AlertTriangle, Loader2 } from "lucide-react";
import { GraphPlaceholder } from "./GraphPlaceholder";
import { GraphViewPanel } from "./GraphView";
import { hasGraphResults } from "@/lib/graphModel";
import type { SourceItem } from "@/types";

export interface GraphTabProps {
  /** Sources of the CURRENT answer (graph provenance lives here). */
  sources: SourceItem[];
  /** A query is in flight. */
  loading?: boolean;
  /** Stream/query error, if any. */
  error?: string | null;
  /** Retrieval mode of the current request. */
  isGraphMode: boolean;
}

/**
 * Graph tab state machine.
 *
 *   error                      -> error panel
 *   loading & nothing yet      -> spinner
 *   graph provenance present   -> interactive node-link view
 *   graph mode, no provenance  -> "no graph matches" (honest empty)
 *   hybrid mode                -> GraphPlaceholder (genuine empty state)
 *
 * Hybrid results never reach GraphViewPanel: the check is on the presence of
 * real graph provenance, not on the requested mode alone.
 */
export function GraphTab({ sources, loading, error, isGraphMode }: GraphTabProps) {
  const ready = hasGraphResults(sources);

  if (error) {
    return (
      <div
        data-testid="graph-error"
        className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center"
      >
        <div className="flex h-12 w-12 items-center justify-center rounded-lg border border-dashed border-danger/50 text-danger">
          <AlertTriangle className="h-6 w-6" />
        </div>
        <p className="text-sm font-medium">Graph unavailable</p>
        <p className="max-w-[260px] text-xs text-muted">{error}</p>
      </div>
    );
  }

  if (loading && !ready) {
    return (
      <div
        data-testid="graph-loading"
        className="flex h-full flex-col items-center justify-center gap-2 text-center"
      >
        <Loader2 className="h-5 w-5 animate-spin text-muted" />
        <p className="text-xs text-muted">Traversing the knowledge graph…</p>
      </div>
    );
  }

  if (ready) {
    return <GraphViewPanel sources={sources} />;
  }

  if (isGraphMode) {
    return (
      <div
        data-testid="graph-empty-nomatch"
        className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center"
      >
        <p className="text-sm font-medium">No graph matches</p>
        <p className="max-w-[240px] text-xs text-muted">
          This query resolved no entities in the knowledge graph. Try naming an
          organisation, programme, facility or place.
        </p>
      </div>
    );
  }

  return <GraphPlaceholder />;
}
