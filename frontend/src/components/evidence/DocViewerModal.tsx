import { FileText } from "lucide-react";
import { useDocViewerStore } from "@/store/useDocViewerStore";
import { Modal } from "@/components/common/Modal";
import { Badge } from "@/components/ui/badge";
import { confidenceLabel } from "@/utils/formatters";
import {
  hasMatchedSections,
  matchedTextUnits,
  matchedTotal,
  parentEvidenceDiffers,
  unitTypeLabel,
} from "@/lib/matchedChunks";

/**
 * Full-document viewer — opens the complete parliamentary Q&A record
 * (original question + full answer) that a retrieved source came from.
 * The whole text is already delivered with each answer; this is the
 * audit view that lets you check the LLM's claims against the original.
 *
 * When the server reports matched sections, the matched text is the primary
 * view and the assembled parent evidence (what the answer was generated from)
 * is collapsed below it. Without matched sections the original display is shown.
 */
export function DocViewerModal() {
  const source = useDocViewerStore((s) => s.source);
  const close = useDocViewerStore((s) => s.close);

  const matched = source ? matchedTextUnits(source) : [];
  const showMatched = source ? hasMatchedSections(source) : false;
  const total = source ? matchedTotal(source) : 0;
  const parentDiffers = source ? parentEvidenceDiffers(source) : false;

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
            {source.date && (
              <>
                <span className="hidden text-border sm:inline">|</span>
                <span>
                  <span className="font-medium text-foreground">Date:</span> {source.date}
                </span>
              </>
            )}
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

          {showMatched ? (
            <>
              {/* Primary: the matched section(s) this document was retrieved for */}
              <div className="flex min-h-0 flex-1 flex-col">
                <p className="text-[10px] font-bold uppercase tracking-widest text-muted">
                  Matched {matched.length > 1 ? "sections" : "section"}
                  {total > matched.length ? ` (showing ${matched.length} of ${total})` : ""}
                </p>
                <div className="mt-1.5 flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
                  {matched.map((m) => (
                    <div
                      key={m.chunk_id}
                      className="rounded-lg border border-accent/40 bg-background/60 p-3"
                    >
                      <p className="mb-1.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[10px] text-muted">
                        <span className="font-semibold text-accent">{unitTypeLabel(m.unit_type)}</span>
                        <span className="font-mono">{m.chunk_id}</span>
                        {m.page != null && <span>Page {m.page}</span>}
                        {m.reranked && <span className="font-semibold text-foreground/80">Top reranked</span>}
                      </p>
                      <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-foreground/90">
                        {m.text}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Secondary: the assembled parent evidence, collapsed by default.
                  Omitted when it is the same text as the matched section. */}
              {parentDiffers ? (
                <details className="shrink-0 rounded-lg border border-border bg-background/40">
                  <summary className="cursor-pointer select-none px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-muted hover:text-foreground">
                    Parent evidence (as used for the answer)
                  </summary>
                  <div className="max-h-[28vh] overflow-y-auto whitespace-pre-wrap border-t border-border p-3 text-[13px] leading-relaxed text-foreground/90">
                    {source.answer}
                  </div>
                </details>
              ) : (
                <p className="shrink-0 text-[10px] text-muted">
                  The matched section is the full evidence for this document.
                </p>
              )}
            </>
          ) : (
            <>
              {/* Full answer — the document the LLM answered from */}
              <div className="flex min-h-0 flex-1 flex-col">
                <p className="text-[10px] font-bold uppercase tracking-widest text-muted">
                  Full answer (as retrieved)
                </p>
                <div className="mt-1.5 min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap rounded-lg border border-border bg-background/60 p-3 text-[13px] leading-relaxed text-foreground/90">
                  {source.answer}
                </div>
              </div>
            </>
          )}

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
