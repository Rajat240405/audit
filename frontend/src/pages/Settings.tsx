import { useAppStore } from "@/store/useAppStore";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Modal } from "@/components/common/Modal";

/** Collapsible Settings panel — advanced config hidden from the header. */
export function Settings() {
  const open = useAppStore((s) => s.settingsOpen);
  const setOpen = useAppStore((s) => s.setSettingsOpen);
  const app = useAppStore();

  return (
    <Modal open={open} onOpenChange={setOpen} title="Settings">
      <div className="space-y-4 text-sm">
        <div>
          <p className="mb-1 text-xs font-semibold text-muted">Active configuration</p>
          <div className="grid grid-cols-2 gap-2">
            <Setting label="Provider" value={app.provider} />
            <Setting label="Model family" value={app.modelFamily} />
            <Setting label="Model" value={app.model} />
            <Setting label="Execution profile" value={app.mode} />
            <Setting label="Retrieval mode" value={app.retrievalMode} />
            <Setting label="Draft style" value={app.draftStyle} />
          </div>
        </div>

        <div>
          <p className="mb-1 text-xs font-semibold text-muted">Backend</p>
          <div className="flex items-center gap-2">
            <Badge variant={app.backendOnline === false ? "danger" : "success"}>
              {app.backendOnline === false ? "offline" : "online"}
            </Badge>
            <span className="text-xs text-muted">{app.gpu}</span>
          </div>
        </div>

        <div className="border-t border-border pt-3">
          <p className="mb-2 text-xs font-semibold text-muted">Advanced</p>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                setOpen(false);
                app.setBuildModalOpen(true);
              }}
            >
              Build Graph
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  );
}

function Setting({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-border bg-surface-2 px-2 py-1.5">
      <p className="text-[10px] uppercase text-muted">{label}</p>
      <p className="truncate font-mono text-xs">{value}</p>
    </div>
  );
}
