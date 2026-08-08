import { FileSearch } from "lucide-react";
import { useDraftStore } from "@/store/useDraftStore";
import { EvidenceCard } from "./EvidenceCard";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAppStore } from "@/store/useAppStore";

/**
 * Evidence tab — the audit-transparency heart. Each retrieved document shows
 * title, ID, confidence, component scores and highlighted transcript. In
 * graph mode the card surface switches to entity/relationship emphasis.
 */
export function EvidencePanel() {
  const sources = useDraftStore((s) => s.sources);
  const selected = useDraftStore((s) => s.selectedEvidence);
  const retrievalMode = useAppStore((s) => s.retrievalMode);

  if (sources.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
        <FileSearch className="h-8 w-8 text-muted" />
        <p className="text-xs text-muted">
          {retrievalMode === "graph"
            ? "Entities, relationships and supporting documents will appear here when you run a GraphRAG query."
            : "Retrieved chunks, confidence scores and highlighted evidence will appear here after a query."}
        </p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="space-y-2 p-3">
        {sources.map((s) => (
          <EvidenceCard
            key={s.doc_id}
            source={s}
            selected={selected?.doc_id === s.doc_id}
            isGraph={retrievalMode === "graph"}
          />
        ))}
      </div>
    </ScrollArea>
  );
}
