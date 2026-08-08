import { Check, Circle, Loader2 } from "lucide-react";
import { usePipelineStore } from "@/store/usePipelineStore";
import { cn } from "@/utils/cn";
import { motion } from "framer-motion";

/**
 * Live pipeline visualization — mode-aware (Hybrid stages vs Graph stages).
 * Driven by SSE status events from the backend.
 */
export function PipelineView() {
  const stages = usePipelineStore((s) => s.stages);
  const running = usePipelineStore((s) => s.running);
  const message = usePipelineStore((s) => s.message);

  if (stages.length === 0) {
    return (
      <div className="px-4 py-6 text-center text-xs text-muted">
        Run a query to see the retrieval pipeline.
      </div>
    );
  }

  return (
    <div className="space-y-1 p-3">
      {stages.map((stage, i) => (
        <div key={stage.key} className="flex items-center gap-2.5">
          {/* connector */}
          {i < stages.length - 1 && (
            <div
              className={cn(
                "ml-[7px] h-4 w-0.5",
                stage.status === "done" ? "bg-success/50" : "bg-border"
              )}
            />
          )}
          <StageIcon status={stage.status} />
          <div className="flex min-w-0 flex-1 items-center justify-between">
            <span
              className={cn(
                "text-xs",
                stage.status === "done"
                  ? "text-foreground"
                  : stage.status === "running"
                    ? "text-accent"
                    : "text-muted"
              )}
            >
              {stage.label}
            </span>
            {stage.count != null && stage.status === "done" && (
              <span className="font-mono text-[10px] text-muted">{stage.count}</span>
            )}
          </div>
        </div>
      ))}

      {running && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-2 rounded border border-accent/30 bg-accent/10 px-2 py-1 text-[11px] text-accent"
        >
          {message || "Processing…"}
        </motion.div>
      )}
    </div>
  );
}

function StageIcon({ status }: { status: "pending" | "running" | "done" | "error" }) {
  if (status === "done") return <Check className="h-3.5 w-3.5 shrink-0 text-success" />;
  if (status === "running")
    return <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-accent" />;
  if (status === "error") return <Circle className="h-3.5 w-3.5 shrink-0 text-danger" />;
  return <Circle className="h-3.5 w-3.5 shrink-0 text-border" />;
}
