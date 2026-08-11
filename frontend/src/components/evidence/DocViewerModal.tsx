import { FileText } from "lucide-react";
import { useDocViewerStore } from "@/store/useDocViewerStore";
import { Modal } from "@/components/common/Modal";
import { Badge } from "@/components/ui/badge";
import { confidenceLabel } from "@/utils/formatters";

/**
 * Full-document viewer — opens the complete parliamentary Q&A record
 * (original question + full answer) that a retrieved source came from.
 * The whole text is already delivered with each answer; this is the
 * audit view that lets you check the LLM's claims against the original.
 */
export function DocViewerModal() {
  const source = useDocViewerStore((s) => s.source);
  const close = useDocViewerStore((s) => s.close);

  return (
    <Modal
      open={!!source}
      onOpenChange={(open) => !open && close()}
      title={source ? `Document ${source.doc_id}` : "Document"}
      // Viewport-relative sizing: ~80% of the user's actual screen. vw/vh
      // scale with the browser's zoom level (75%, 100%, 125%…) so the reader
      // always fits without clipping or forcing the user to zoom out.
      className="flex h-[80vh] w-[80vw] max-w-[1100px] flex-col"
    >
      {source && (
        <div className="flex min-h-0 flex-1 flex-col gap-3">
          {/* Header: identity + provenance */}
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-muted">
            <Badge variant="accent">{source.doc_id}</Badge>
            <span>
              <span className="font-medium text-foreground">Ministry:</span> {source.ministry}
            </span>
            <span className="hidden text-border sm:inline">|</span>
            <span className="truncate">{source.subject}</span>
            <span className="ml-auto shrink-0">
              Confidence <span className="font-semibold text-foreground">{confidenceLabel(source.score)}</span>
            </span>
          </div>

          {/* Original question */}
          <div className="shrink-0 rounded-lg border border-border bg-surface-2/60 p-3">
            <p className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-muted">
              <FileText className="h-3 w-3" />
              Original question
            </p>
            <p className="mt-1.5 max-h-[18vh] overflow-y-auto whitespace-pre-wrap text-[13px] leading-relaxed text-foreground">
              {source.question}
            </p>
          </div>

          {/* Full answer — the document the LLM answered from */}
          <div className="flex min-h-0 flex-1 flex-col">
            <p className="text-[10px] font-bold uppercase tracking-widest text-muted">
              Full answer (as retrieved)
            </p>
            <div className="mt-1.5 min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap rounded-lg border border-border bg-background/60 p-3 text-[13px] leading-relaxed text-foreground/90">
              {source.answer}
            </div>
          </div>

          {source.answer.includes("[MATCHED SECTION OF THIS ANSWER]") && (
            <p className="shrink-0 text-[10px] text-muted">
              This document is longer than the model's context — the answer above includes the
              matched section that actually drove the LLM's response.
            </p>
          )}
        </div>
      )}
    </Modal>
  );
}
