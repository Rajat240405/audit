import { useEffect, useState } from "react";
import { useChatStream } from "@/hooks/useChatStream";
import { useChatActionsStore } from "@/store/useChatActionsStore";
import { useDraftStore } from "@/store/useDraftStore";
import { useAppStore } from "@/store/useAppStore";
import { DraftCanvas } from "@/components/workspace/DraftCanvas";
import { VersionPanel } from "@/components/workspace/VersionPanel";
import { ExportMenu } from "@/components/workspace/ExportMenu";
import { EvidencePanel } from "@/components/evidence/EvidencePanel";
import { PipelineView } from "@/components/pipeline/PipelineView";
import { Metrics } from "@/components/pipeline/Metrics";
import { GraphPlaceholder } from "@/components/graph/GraphPlaceholder";
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";

const TABS = [
  { key: "draft", label: "Draft" },
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
  const [tab, setTab] = useState<TabKey>("draft");
  const [showVersions, setShowVersions] = useState(false);
  const [copied, setCopied] = useState(false);
  const isGraph = useAppStore((s) => s.retrievalMode) === "graph";

  // expose the streaming actions to the sidebar's query input
  useEffect(() => {
    useChatActionsStore.setState({ send, stop, running });
  }, [send, stop, running]);

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
          <Button variant="outline" size="sm" onClick={() => setShowVersions((v) => !v)}>
            Versions
          </Button>
          <Button variant="outline" size="sm" onClick={copy}>
            {copied ? "Copied ✓" : "Copy"}
          </Button>
          <ExportMenu />
        </div>
      </nav>

      <div className="min-h-0 flex-1">
        {tab === "draft" && (showVersions ? <VersionPanel /> : <DraftCanvas />)}
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
    </section>
  );
}

function NotesTab() {
  const [notes, setNotes] = useState("");
  return (
    <div className="flex h-full flex-col p-4">
      <p className="mb-2 text-xs font-semibold text-muted">Personal workspace</p>
      <textarea
        className="min-h-0 flex-1 resize-none rounded-md border border-border bg-surface px-3 py-2 text-sm text-foreground placeholder:text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        placeholder="Bookmarks, important findings, needs-verification items, scientist feedback…"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
      />
    </div>
  );
}
