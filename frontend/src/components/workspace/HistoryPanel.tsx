import { History, RotateCcw } from "lucide-react";
import { useDraftStore } from "@/store/useDraftStore";
import { formatDate } from "@/utils/formatters";
import { cn } from "@/utils/cn";

/**
 * Answer history — every snapshot of the draft lives here: generated answers,
 * AI edits, stopped generations, and an automatic "Session open" snapshot on
 * every reload. Clicking an entry restores the whole answer onto the drafting
 * canvas.
 */
export function HistoryPanel({ onRestored }: { onRestored?: () => void }) {
  const versions = useDraftStore((s) => s.versions);
  const activeVersion = useDraftStore((s) => s.activeVersion);
  const restoreVersion = useDraftStore((s) => s.restoreVersion);

  const open = (id: string) => {
    restoreVersion(id);
    onRestored?.();
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-1.5 px-3 py-2 text-xs font-semibold text-muted">
        <History className="h-3.5 w-3.5" />
        History
      </div>

      <div className="flex-1 space-y-1 overflow-y-auto px-2 pb-2">
        {versions.length === 0 ? (
          <p className="px-2 py-6 text-center text-xs text-muted">
            No history yet. Generated answers, AI edits, and reload snapshots
            appear here automatically.
          </p>
        ) : (
          [...versions].reverse().map((v) => (
            <div
              key={v.id}
              role="button"
              tabIndex={0}
              onClick={() => open(v.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  open(v.id);
                }
              }}
              title="Open this answer on the canvas"
              className={cn(
                "flex cursor-pointer items-center justify-between rounded-md border border-border px-2.5 py-2 hover:bg-surface-2",
                activeVersion === v.id && "border-accent/40 bg-accent/10"
              )}
            >
              <div className="min-w-0">
                <p className="truncate text-xs font-medium">{v.label}</p>
                {v.description && (
                  <p className="truncate text-[10px] text-accent">{v.description}</p>
                )}
                <p className="text-[10px] text-muted">{formatDate(v.createdAt)}</p>
              </div>
              <RotateCcw className="h-3 w-3 shrink-0 text-muted" />
            </div>
          ))
        )}
      </div>
    </div>
  );
}
