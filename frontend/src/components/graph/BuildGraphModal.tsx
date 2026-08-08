import { useQuery } from "@tanstack/react-query";
import { Boxes, RefreshCw } from "lucide-react";
import { Modal } from "@/components/common/Modal";
import { useAppStore } from "@/store/useAppStore";
import { fetchGraphBuildStatus } from "@/api/graph";
import { Button } from "@/components/ui/button";
import { formatNumber } from "@/utils/formatters";

/**
 * Build Graph modal — live build progress from the GraphRAG checkpoint.
 * (Does not open Neo4j; reads the checkpoint JSON + status endpoint.)
 */
export function BuildGraphModal() {
  const open = useAppStore((s) => s.buildModalOpen);
  const setOpen = useAppStore((s) => s.setBuildModalOpen);

  const { data, refetch, isFetching } = useQuery({
    queryKey: ["graph-build-status"],
    queryFn: fetchGraphBuildStatus,
    refetchInterval: open ? 5000 : false,
    enabled: open,
  });

  return (
    <Modal open={open} onOpenChange={setOpen} title="Build Graph — progress">
      <div className="space-y-3 text-sm">
        <div className="grid grid-cols-2 gap-2">
          <Stat label="Documents processed" value={data ? formatNumber(data.documents_processed) : "—"} />
          <Stat label="Nodes created" value={data && data.checkpoint_exists ? "—" : "—"} />
          <Stat label="Relationships created" value={data && data.checkpoint_exists ? "—" : "—"} />
          <Stat label="Failed" value={data ? formatNumber(data.failed) : "—"} />
        </div>

        <div className="rounded border border-border bg-surface-2 p-2 text-xs text-muted">
          {data?.checkpoint_exists ? (
            <>
              Checkpoint: <span className="font-mono">{data.path}</span>
              <br />
              {data.total ? `${formatNumber(data.documents_processed)} of ${formatNumber(data.total)} documents` : "No checkpoint entries yet."}
            </>
          ) : (
            "No checkpoint found. Run `graphrag build` on the backend to start."
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
