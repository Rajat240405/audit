import { useEffect, useState } from "react";
import { useChatStream } from "@/hooks/useChatStream";
import { useChatActionsStore } from "@/store/useChatActionsStore";
import { useDraftStore } from "@/store/useDraftStore";
import { useAppStore } from "@/store/useAppStore";
import { DraftCanvas } from "@/components/workspace/DraftCanvas";
import { NotesEditor } from "@/components/workspace/NotesEditor";
import { HistoryPanel } from "@/components/workspace/HistoryPanel";
import { ExportMenu } from "@/components/workspace/ExportMenu";
import { EvidencePanel } from "@/components/evidence/EvidencePanel";
import { DocViewerModal } from "@/components/evidence/DocViewerModal";
import { PipelineView } from "@/components/pipeline/PipelineView";
import { Metrics } from "@/components/pipeline/Metrics";
import { GraphPlaceholder } from "@/components/graph/GraphPlaceholder";
import { ModelActivityPanel } from "@/components/activity/ModelActivityPanel";
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";

const TABS = [
  { key: "draft", label: "Draft" },
  { key: "history", label: "History" },
  { key: "sources", label: "Sources" },
  { key: "pipeline", label: "RAG Pipeline" },
  { key: "notes", label: "Notes" },
  { key: "graph", label: "Graph" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

/**
 * Center workspace — the drafting canvas is the heart. Nav tabs switch
 * between Draft / Sources / RAG Pipeline / Notes / Graph.
 */
export function Workspace() {
  const { send, stop, running } = useChatStream();
  // Tab state is LOCAL (useState) so it always works even if the actions
  // store is a stale copy; we mirror the local tab + a setter into the store
  // as an optional bridge for the Model Activity panel's "Go to canvas".
  const [tab, setTab] = useState<TabKey>("draft");
  const [copied, setCopied] = useState(false);
  const isGraph = useAppStore((s) => s.retrievalMode) === "graph";

  // expose the streaming actions + tab bridge to the sidebar/activity panel
  useEffect(() => {
    useChatActionsStore.setState({
      send,
      stop,
      running,
      tab,
      setTab: (t: string) => setTab(t as TabKey),
    });
  }, [send, stop, running, tab]);

  const copy = async () => {
    const c = useDraftStore.getState().content;
    if (!c) return;
    try {
      await navigator.clipboard.writeText(c);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <section className="flex min-h-0 flex-1 flex-col bg-background">
      <nav className="flex shrink-0 items-center justify-between border-b border-border bg-surface px-6 py-2">
        <div className="flex gap-5 text-xs font-semibold uppercase tracking-wide text-muted">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={cn(
                "pb-1",
                tab === t.key
                  ? "border-b-2 border-foreground text-foreground"
                  : "hover:text-foreground"
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={copy}>
            {copied ? "Copied ✓" : "Copy"}
          </Button>
          <ExportMenu />
        </div>
      </nav>

      <div className="min-h-0 flex-1">
        {tab === "draft" && <DraftCanvas />}
        {tab === "history" && <HistoryPanel onRestored={() => setTab("draft")} />}
        {tab === "sources" && (
          <div className="h-full">
            <EvidencePanel />
          </div>
        )}
        {tab === "pipeline" && (
          <div className="h-full space-y-4 overflow-y-auto p-4">
            <PipelineView />
            <Metrics isGraph={isGraph} />
          </div>
        )}
        {tab === "notes" && <NotesTab />}
        {tab === "graph" && <GraphPlaceholder />}
      </div>
      {/* Full-document reader — opened from the Sources tab or Cross-Verify Facts */}
      <DocViewerModal />
      {/* Live model activity — thinking, received context, go-to-canvas */}
      <ModelActivityPanel />
    </section>
  );
}

function NotesTab() {
  return <NotesEditor />;
}
