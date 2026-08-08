import { History, RotateCcw } from "lucide-react";
import { useDraftStore } from "@/store/useDraftStore";
import { Button } from "@/components/ui/button";
import { formatDate } from "@/utils/formatters";
import { cn } from "@/utils/cn";

/**
 * Version history for the current draft — save, restore, and (future)
 * side-by-side compare.
 */
export function VersionPanel() {
  const versions = useDraftStore((s) => s.versions);
  const activeVersion = useDraftStore((s) => s.activeVersion);
  const content = useDraftStore((s) => s.content);
  const saveVersion = useDraftStore((s) => s.saveVersion);
  const restoreVersion = useDraftStore((s) => s.restoreVersion);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-3 py-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-muted">
          <History className="h-3.5 w-3.5" />
          Versions
        </div>
        <Button size="sm" variant="secondary" onClick={() => saveVersion()} disabled={!content.trim()}>
          Save version
        </Button>
      </div>

      <div className="flex-1 space-y-1 overflow-y-auto px-2 pb-2">
        {versions.length === 0 ? (
          <p className="px-2 py-6 text-center text-xs text-muted">No versions yet.</p>
        ) : (
          [...versions].reverse().map((v) => (
            <div
              key={v.id}
              className={cn(
                "flex items-center justify-between rounded-md border border-border px-2.5 py-2",
                activeVersion === v.id && "border-accent/40 bg-accent/10"
              )}
            >
              <div className="min-w-0">
                <p className="truncate text-xs font-medium">{v.label}</p>
                <p className="text-[10px] text-muted">{formatDate(v.createdAt)}</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                title="Restore this version"
                onClick={() => restoreVersion(v.id)}
              >
                <RotateCcw className="h-3 w-3" />
              </Button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
