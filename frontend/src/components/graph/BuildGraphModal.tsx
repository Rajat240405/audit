import { useQuery } from "@tanstack/react-query";
import { Boxes, RefreshCw } from "lucide-react";
import { Modal } from "@/components/common/Modal";
import { useAppStore } from "@/store/useAppStore";
import { fetchGraphBuildStatus } from "@/api/graph";
import { Button } from "@/components/ui/button";
import { formatNumber } from "@/utils/formatters";
import type { GraphBuildState } from "@/types";

const STATE_LABEL: Record<GraphBuildState, string> = {
  idle: "Idle",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  interrupted: "Interrupted",
};

const STATE_CLASS: Record<GraphBuildState, string> = {
  idle: "text-muted",
  running: "text-blue-500",
  completed: "text-emerald-500",
  failed: "text-red-500",
  interrupted: "text-amber-500",
};

function formatElapsed(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

/**
 * Build Graph modal — live GraphRAG build progress.
 *
 * The build runs on the backend (CLI, usually detached); this view only
 * reports what the build process itself persisted into the checkpoint, so it
 * is correct after a refresh, after an app restart, and while a nohup build
 * is running. The UI never owns build state and cannot start a build.
 */
export function BuildGraphModal() {
  const open = useAppStore((s) => s.buildModalOpen);
  const setOpen = useAppStore((s) => s.setBuildModalOpen);

  const { data, refetch, isFetching } = useQuery({
    queryKey: ["graph-build-status"],
    queryFn: fetchGraphBuildStatus,
    // Poll faster while a build is live; back off when it is not.
    refetchInterval: open ? (q) => (q.state.data?.running ? 3000 : 15000) : false,
    enabled: open,
  });

  const state: GraphBuildState = data?.state ?? "idle";
  const pct = Math.min(100, Math.max(0, data?.percent ?? 0));

  return (
    <Modal open={open} onOpenChange={setOpen} title="Build Graph — progress">
      <div className="space-y-3 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-xs uppercase text-muted">Status</span>
          <span className={`font-medium ${STATE_CLASS[state]}`}>
            {STATE_LABEL[state]}
            {data?.mode ? ` · ${data.mode}` : ""}
          </span>
        </div>

        {/* GraphRAG Build: 37% — 981 / 2648 documents */}
        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="font-mono text-sm">
              {pct.toFixed(1)}% — {formatNumber(data?.documents_processed ?? 0)} /{" "}
              {formatNumber(data?.total ?? 0)} documents
            </span>
            <span className="text-xs text-muted">
              {formatElapsed(data?.elapsed_seconds)}
            </span>
          </div>
          <div
            className="h-2 w-full overflow-hidden rounded bg-surface-2"
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div
              className={`h-full transition-all duration-500 ${
                state === "failed"
                  ? "bg-red-500"
                  : state === "interrupted"
                    ? "bg-amber-500"
                    : state === "completed"
                      ? "bg-emerald-500"
                      : "bg-blue-500"
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Stat label="Newly processed" value={formatNumber(data?.added ?? 0)} />
          <Stat label="Reconciled / changed" value={formatNumber(data?.reconciled ?? 0)} />
          <Stat label="Skipped unchanged" value={formatNumber(data?.skipped_unchanged ?? 0)} />
          <Stat label="Failed" value={formatNumber(data?.failed ?? 0)} />
        </div>

        {(data?.current_doc || data?.last_doc) && (
          <div className="rounded border border-border bg-surface-2 p-2 text-xs">
            <span className="text-muted">
              {data.current_doc ? "Current document: " : "Last processed: "}
            </span>
            <span className="font-mono">{data.current_doc ?? data.last_doc}</span>
          </div>
        )}

        {data?.error && (
          <div className="rounded border border-red-500/40 bg-red-500/10 p-2 text-xs text-red-500">
            {data.error}
          </div>
        )}

        <div className="rounded border border-border bg-surface-2 p-2 text-xs text-muted">
          {data?.checkpoint_exists ? (
            <>
              Checkpoint: <span className="font-mono">{data.path}</span>
            </>
          ) : (
            "No checkpoint found. Start a build on the backend: python -m src.graphrag.cli build"
          )}
        </div>

        <div className="flex justify-end gap-2">
          <Button variant="secondary" size="sm" onClick={() => refetch()} disabled={isFetching}>
            <RefreshCw className={`h-3.5 w-3.5 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button size="sm" onClick={() => setOpen(false)}>
            <Boxes className="h-3.5 w-3.5" />
            Close
          </Button>
        </div>
      </div>
    </Modal>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-border bg-surface-2 px-2 py-1.5">
      <p className="text-[10px] uppercase text-muted">{label}</p>
      <p className="font-mono text-sm">{value}</p>
    </div>
  );
}
