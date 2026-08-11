import { useRef, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertTriangle, CheckCircle2, FileText } from "lucide-react";
import { useDraftStore } from "@/store/useDraftStore";
import { useEditDraft } from "@/hooks/useEditDraft";
import { useEditStore } from "@/store/useEditStore";
import { useDocViewerStore } from "@/store/useDocViewerStore";
import { Toolbar } from "./Toolbar";
import { buildGroundingReport } from "@/services/grounding";
import { cn } from "@/utils/cn";
import type { GroundingClaim, SourceItem } from "@/types";

/**
 * Drafting canvas — the heart of the workstation. White document card with a
 * header, live streaming markdown body, docked AI editing tools, and
 * FORMALIZE / CROSS-VERIFY FACTS actions.
 */
export function DraftCanvas() {
  const content = useDraftStore((s) => s.content);
  const streamingText = useDraftStore((s) => s.streamingText);
  const isStreaming = useDraftStore((s) => s.isStreaming);
  const grounding = useDraftStore((s) => s.grounding);
  const sources = useDraftStore((s) => s.sources);
  const { edit, pendingEdit, accept, reject } = useEditDraft();
  const editing = useEditStore((s) => s.editing);
  const [showClaims, setShowClaims] = useState(false);
  const streamRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    streamRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [streamingText]);

  const verified = grounding.filter((g) => g.found).length;
  const score = grounding.length ? Math.round((verified / grounding.length) * 100) : null;

  const reverify = () => {
    const { content: c, sources } = useDraftStore.getState();
    useDraftStore.setState({ grounding: buildGroundingReport(c, sources) });
    setShowClaims((v) => !v);
  };

  const display = content || streamingText;

  return (
    <div className="flex h-full flex-col p-6">
      {/* Canvas card */}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-t-lg border border-border bg-surface shadow-sm">
        <div className="flex items-center justify-between border-b border-border bg-surface-2/60 px-4 py-2">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-muted">
            Drafting Canvas
          </span>
          <span className="text-[10px] text-muted">
            BGE-M3 • 2-STAGE RAG
            {score != null && (
              <span
                className={cn(
                  "ml-2 font-semibold",
                  score >= 80 ? "text-success" : score >= 40 ? "text-warning" : "text-danger"
                )}
              >
                • Grounding {score}%
              </span>
            )}
          </span>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-6">
          {display ? (
            <div ref={streamRef} className="md-body text-[15px] leading-relaxed text-foreground">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{display}</ReactMarkdown>
              {isStreaming && (
                <span className="inline-block h-4 w-0.5 animate-pulse bg-accent align-middle" />
              )}
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-muted">
              Workspace idle. Awaiting query...
            </div>
          )}
        </div>
      </div>

      {/* AI-edit review panel — side-by-side comparison */}
      {pendingEdit && (
        <div className="mt-2 rounded-lg border border-accent/40 bg-surface shadow-sm">
          <div className="flex items-center justify-between border-b border-border bg-surface-2/60 px-4 py-2">
            <span className="text-xs font-semibold text-foreground">
              Compare: "{pendingEdit.label}" edit
            </span>
            <span className="text-[11px] text-muted">Pick the version you want to keep</span>
          </div>
          <div className="grid grid-cols-2 gap-px bg-border">
            {/* ORIGINAL */}
            <div className="bg-surface">
              <div className="flex items-center justify-between bg-surface-2/40 px-3 py-1.5">
                <span className="text-[11px] font-bold uppercase tracking-wide text-muted">Original</span>
                <button
                  onClick={reject}
                  className="rounded bg-surface-2 px-2 py-0.5 text-[11px] font-semibold text-foreground hover:bg-border"
                >
                  Keep Original
                </button>
              </div>
              <div className="max-h-64 overflow-y-auto p-3">
                <div className="md-body text-[13px] leading-relaxed text-foreground/80">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{pendingEdit.original}</ReactMarkdown>
                </div>
              </div>
            </div>
            {/* EDITED */}
            <div className="bg-surface">
              <div className="flex items-center justify-between bg-accent/10 px-3 py-1.5">
                <span className="text-[11px] font-bold uppercase tracking-wide text-accent">Edited</span>
                <button
                  onClick={accept}
                  className="rounded bg-success px-2 py-0.5 text-[11px] font-semibold text-white hover:opacity-90"
                >
                  Use Edited
                </button>
              </div>
              <div className="max-h-64 overflow-y-auto p-3">
                <div className="md-body text-[13px] leading-relaxed text-foreground/80">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{pendingEdit.revised}</ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Docked editing tools */}
      <div className="space-y-3 rounded-b-lg border border-t-0 border-border bg-surface p-3 shadow-sm">
        <Toolbar editing={editing} />
        {showClaims && <ClaimsList claims={grounding} sources={sources} />}
        <div className="flex gap-3">
          <button
            onClick={() => edit("Rewrite this draft in a formal official register.")}
            disabled={!content || editing}
            className="rounded-md border border-border bg-surface px-4 py-2 text-xs font-bold uppercase text-foreground shadow-sm hover:bg-surface-2 disabled:opacity-50"
          >
            Formalize
          </button>
          <button
            onClick={reverify}
            disabled={!content}
            className="rounded-md border border-border bg-surface px-4 py-2 text-xs font-bold uppercase text-foreground shadow-sm hover:bg-surface-2 disabled:opacity-50"
          >
            Cross-Verify Facts
          </button>
        </div>
      </div>
    </div>
  );
}

function ClaimsList({
  claims,
  sources,
}: {
  claims: GroundingClaim[];
  sources: SourceItem[];
}) {
  const openDoc = useDocViewerStore((s) => s.openDoc);

  if (claims.length === 0) {
    return <p className="text-[11px] text-muted">No claims to verify.</p>;
  }

  const findSource = (docId: string) => sources.find((s) => s.doc_id === docId);

  return (
    <div className="max-h-40 space-y-1 overflow-y-auto rounded-md border border-border bg-surface-2 p-2">
      {claims.map((c, i) => {
        const src = c.source ? findSource(c.source) : undefined;
        return (
          <div key={i} className="flex items-start gap-1.5 text-[11px]">
            {c.found ? (
              <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0 text-success" />
            ) : (
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-warning" />
            )}
            <span className={c.found ? "text-foreground/80" : "text-warning"}>
              {c.text}
              {c.found && c.source ? (
                <span className="text-muted">
                  {" — "}
                  {src ? (
                    <button
                      className="text-accent hover:underline"
                      onClick={() => openDoc(src)}
                      title={`Open ${c.source} to verify this claim`}
                    >
                      {c.source}
                    </button>
                  ) : (
                    c.source
                  )}
                </span>
              ) : null}
              {!c.found ? " — not found in sources" : ""}
            </span>
            {src && (
              <button
                className="ml-auto shrink-0 text-muted hover:text-accent"
                onClick={() => openDoc(src)}
                title="Open the source document"
              >
                <FileText className="h-3 w-3" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
