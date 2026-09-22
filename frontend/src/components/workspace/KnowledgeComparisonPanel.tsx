import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useDraftStore } from "@/store/useDraftStore";
import { useToastStore } from "@/store/useToastStore";
import type { KnowledgeMatchCard, SourceItem } from "@/types";

/** Compact date for the card header (ISO string -> "12 Aug 2026"). */
function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso || "—";
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

/**
 * Side-by-side comparison of Saved Knowledge vs the fresh RAG answer.
 *
 * Layout priority: the two ANSWER bodies get the majority of the vertical
 * space (with their own internal scrolling); the header strip — paging,
 * contributors, dates — stays compact (~70–100px). The panel exists to READ
 * and COMPARE answers, not to admire metadata.
 *
 * "Use This" makes the chosen answer the current Draft via the draft store.
 * It never re-saves anything back into the knowledge base.
 */
export function KnowledgeComparisonPanel() {
  const knowledge = useDraftStore((s) => s.knowledge);
  const content = useDraftStore((s) => s.content);
  const streamingText = useDraftStore((s) => s.streamingText);
  const isStreaming = useDraftStore((s) => s.isStreaming);
  const useAnswer = useDraftStore((s) => s.useKnowledgeAnswer);
  const pushToast = useToastStore((s) => s.push);
  const [page, setPage] = useState(0);

  if (!knowledge || knowledge.matches.length === 0) return null;

  const matches = knowledge.matches;
  const card: KnowledgeMatchCard = matches[Math.min(page, matches.length - 1)];
  const freshText = knowledge.fresh?.content ?? content ?? streamingText;
  const freshSources = knowledge.fresh?.sources ?? useDraftStore.getState().sources;

  const pickSaved = () => {
    // Saved sources are citation-identity shaped; widen them into the full
    // SourceItem form the canvas/Facts panel expects.
    const sources: SourceItem[] = card.sources.map((src) => ({
      doc_id: src.doc_id,
      ministry: src.ministry,
      subject: src.subject,
      date: null,
      score: src.score ?? 0,
      question: "",
      answer: "",
      dense_score: null,
      bm25_score: null,
      rrf_score: null,
      rerank_score: null,
    }));
    useAnswer(card.answer, sources);
    pushToast("success", "Saved answer applied to the draft");
  };
  const pickFresh = () => {
    if (!freshText) return;
    useAnswer(freshText, freshSources);
    pushToast("success", "Fresh RAG answer applied to the draft");
  };

  return (
    <div
      data-testid="knowledge-comparison"
      className="relative mb-3 grid shrink-0 grid-cols-2 gap-3 rounded-lg border border-accent/40 bg-surface p-3 shadow-sm"
      style={{ maxHeight: "60vh" }}
    >
      <button
        onClick={() => useDraftStore.getState().setKnowledge([], null)}
        title="Dismiss comparison"
        aria-label="Dismiss comparison"
        className="absolute -top-2 -right-2 z-10 rounded-full border border-border bg-surface px-1.5 text-xs leading-4 text-muted shadow-sm hover:text-foreground"
      >
        ✕
      </button>
      {/* ── Saved Knowledge ─────────────────────────────────────────────── */}
      <div className="flex min-h-0 min-w-0 flex-col rounded-md border border-border bg-surface">
        {/* compact header: paging + contributors + date (~70–100px) */}
        <div className="shrink-0 space-y-1 border-b border-border bg-accent/5 px-3 py-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-accent">
              Saved Knowledge
            </span>
            {matches.length > 1 && (
              <span className="flex items-center gap-1 text-xs text-muted">
                <button
                  data-testid="kn-prev"
                  onClick={() => setPage((p) => (p - 1 + matches.length) % matches.length)}
                  className="rounded border border-border px-1.5 leading-5 hover:bg-surface-2"
                  aria-label="Previous saved answer"
                >
                  ◀
                </button>
                <span data-testid="kn-page">
                  {Math.min(page, matches.length - 1) + 1} of {matches.length}
                </span>
                <button
                  data-testid="kn-next"
                  onClick={() => setPage((p) => (p + 1) % matches.length)}
                  className="rounded border border-border px-1.5 leading-5 hover:bg-surface-2"
                  aria-label="Next saved answer"
                >
                  ▶
                </button>
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-muted">
            <span data-testid="kn-contributors" className="font-medium text-foreground/80">
              Saved by {card.contributors.join(", ")}
            </span>
            <span>·</span>
            <span data-testid="kn-date">{fmtDate(card.updated_at || card.created_at)}</span>
            <span>·</span>
            <span>
              {card.sources.length} source{card.sources.length === 1 ? "" : "s"}
            </span>
          </div>
        </div>

        {/* the ANSWER gets the majority of the height, scrolling internally */}
        <div
          data-testid="kn-saved-answer"
          className="min-h-0 flex-1 overflow-y-auto px-3 py-2 text-sm leading-relaxed text-foreground/90"
        >
          {card.answer}
        </div>

        <div className="shrink-0 border-t border-border px-3 py-2">
          <Button data-testid="kn-use-saved" size="sm" className="w-full" onClick={pickSaved}>
            Use This
          </Button>
        </div>
      </div>

      {/* ── Fresh RAG ───────────────────────────────────────────────────── */}
      <div className="flex min-h-0 min-w-0 flex-col rounded-md border border-border bg-surface">
        <div className="shrink-0 space-y-1 border-b border-border bg-surface-2/60 px-3 py-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-muted">
              Fresh RAG
            </span>
            <span className="text-[11px] text-muted">{fmtDate(new Date().toISOString())}</span>
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-muted">
            <span data-testid="kn-fresh-label">Generated from current corpus</span>
            <span>·</span>
            <span>
              {freshSources.length} source{freshSources.length === 1 ? "" : "s"}
            </span>
            {isStreaming && <span className="text-accent">· generating…</span>}
          </div>
        </div>

        <div
          data-testid="kn-fresh-answer"
          className="min-h-0 flex-1 overflow-y-auto px-3 py-2 text-sm leading-relaxed text-foreground/90"
        >
          {freshText || "Generating…"}
        </div>

        <div className="shrink-0 border-t border-border px-3 py-2">
          <Button
            data-testid="kn-use-fresh"
            size="sm"
            variant="outline"
            className="w-full"
            onClick={pickFresh}
            disabled={!freshText}
          >
            Use This
          </Button>
        </div>
      </div>
    </div>
  );
}
