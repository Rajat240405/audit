import { useState } from "react";
import { Pencil } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { markdownComponents } from "@/components/common/markdownComponents";
import remarkGfm from "remark-gfm";
import { useDraftStore } from "@/store/useDraftStore";
import { FactVerificationPanel } from "./FactVerificationPanel";
import { DeepVerifyBanner } from "./DeepVerifyBanner";
import { useSessionStore } from "@/store/useSessionStore";
import { KnowledgeComparisonPanel } from "@/components/workspace/KnowledgeComparisonPanel";
import { useStickyAnchorScroll } from "@/hooks/useStickyScroll";
import { useToastStore } from "@/store/useToastStore";
import { useEditDraft } from "@/hooks/useEditDraft";
import { useEditStore } from "@/store/useEditStore";
import { Toolbar } from "./Toolbar";
import { refreshGrounding, summarizeGrounding } from "@/lib/serverGrounding";
import { useAppStore } from "@/store/useAppStore";
import { cn } from "@/utils/cn";

/**
 * Drafting canvas — the heart of the workstation. White document card with a
 * header, live streaming markdown body, and docked AI editing tools.
 *
 * The canvas shows ONLY the final draft/answer. Retrieved sources / citations
 * are deliberately NOT rendered here — they stay available internally for RAG
 * and verification (see the Sources tab and the grounding score below), so this
 * is purely a presentation choice. The "Cross-Verify Facts" action asks the
 * backend (/api/verify) to re-check the draft and updates the score and the
 * facts panel without showing the source list.
 */
export function DraftCanvas() {
  const content = useDraftStore((s) => s.content);
  const streamingText = useDraftStore((s) => s.streamingText);
  const isStreaming = useDraftStore((s) => s.isStreaming);
  const grounding = useDraftStore((s) => s.grounding);
  const groundingFor = useDraftStore((s) => s.groundingFor);
  // Lifted into the store so a NEW QUERY can close it (answer-scoped state).
  const factsOpen = useDraftStore((s) => s.factsOpen);
  const setFactsOpen = useDraftStore((s) => s.setFactsOpen);
  const { edit, pendingEdit, accept, reject } = useEditDraft();
  const editing = useEditStore((s) => s.editing);
  const pushToast = useToastStore((s) => s.push);
  // Auto-follow the stream ONLY while the user is at the bottom. Scrolling up
  // during thinking/generation now keeps the user's position instead of
  // dragging them back down on every token; returning to the bottom resumes
  // the follow. (Previously this force-scrolled on every token.)
  const { ref: streamRef } = useStickyAnchorScroll<HTMLDivElement>([streamingText]);

  // The score is only meaningful for rows verified against the CURRENT draft.
  // Counts come from the same summary the facts panel shows, so the two never
  // disagree. Term presence rows are not facts and are not in the score.
  const groundingStale = groundingFor != null && groundingFor !== content;
  const summary = summarizeGrounding(grounding);
  const score =
    !groundingStale && summary.facts > 0
      ? Math.round((summary.grounded / summary.facts) * 100)
      : null;
  const [verifying, setVerifying] = useState(false);

  // Cross-Verify asks the BACKEND (/api/verify) to check the current draft
  // against its sources, using the same depth as the automatic check. The
  // browser does not extract or judge claims itself. The rewritten text in the
  // response is ignored, so this button never changes the canvas.
  const crossVerify = async () => {
    const { content: answer, sources, activeMessageId } = useDraftStore.getState();
    if (!answer || verifying) return;
    const depth = useAppStore.getState().mode === "deep" ? "full" : "light";
    setVerifying(true);
    const res = await refreshGrounding(answer, sources, depth);
    setVerifying(false);

    // Discard the result if the canvas moved on while the request ran.
    const now = useDraftStore.getState();
    if (now.content !== answer || now.activeMessageId !== activeMessageId) return;

    if (!res.ok) {
      pushToast("error", `Cross-verification failed: ${res.error}`);
      return;
    }
    now.setGrounding(res.claims, answer);
    // The claim list is shown in the persistent panel below; the toast is
    // supplementary feedback and intentionally carries only the summary.
    setFactsOpen(true);

    const sum = summarizeGrounding(res.claims);
    if (sum.facts === 0) {
      pushToast("info", "Cross-verification complete — no factual claims found to check.");
    } else if (sum.grounded === sum.facts) {
      pushToast("success", `Cross-verification complete — all ${sum.facts} factual claim(s) supported by sources.`);
    } else {
      const parts = [
        sum.conflicting > 0 ? `${sum.conflicting} conflicting` : "",
        sum.needsReview > 0 ? `${sum.needsReview} need review` : "",
      ].filter(Boolean);
      pushToast(
        "error",
        `Cross-verification: ${sum.grounded} of ${sum.facts} factual claims supported; ${parts.join(", ")}.`
      );
    }
  };

  // Direct manual editing of the canvas (no AI). On save the content is set
  // in-place and the active session's last assistant message is updated so
  // the sidebar transcript matches.
  const [editingText, setEditingText] = useState<string | null>(null);
  const startEdit = () => setEditingText(content);
  const cancelEdit = () => setEditingText(null);
  const saveEdit = () => {
    if (editingText == null) return;
    const text = editingText;
    useDraftStore.getState().setContent(text);
    // update the SPECIFIC message this canvas is bound to (session + message)
    // so the sidebar transcript always matches the canvas edit
    const draft = useDraftStore.getState();
    const sessions = useSessionStore.getState();
    const sid = draft.activeSessionId;
    const mid = draft.activeMessageId;
    if (sid && mid) {
      const active = sessions.sessions.find((s) => s.id === sid);
      const msg = active?.messages.find((m) => m.id === mid);
      if (msg) sessions.updateMessage(sid, mid, { content: text });
    }
    setEditingText(null);
    pushToast("success", "Draft saved — synced to chat");
  };

  const display = content || streamingText;

  return (
    <div className="flex h-full flex-col p-6">
      {/* Canvas card */}
      {/* data-tour: product-tour anchor for "Editing the draft". */}
      <div
        data-tour="draft-canvas"
        className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-t-lg border border-border bg-surface shadow-sm"
      >
        <div className="flex items-center justify-between border-b border-border bg-surface-2/60 px-4 py-2">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-muted">
            Drafting Canvas
          </span>
          <span className="flex items-center gap-2">
            {content && !isStreaming && editingText == null && (
              <button
                onClick={startEdit}
                className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-bold text-foreground shadow-sm hover:bg-surface-2"
              >
                <Pencil className="h-3 w-3" aria-hidden="true" />Edit
              </button>
            )}
            <span className="text-[10px] text-muted">
            BGE-M3 • 2-STAGE RAG
            {groundingStale && (
              <span className="ml-2 font-semibold text-muted">
                • Grounding stale — re-run Cross-Verify
              </span>
            )}
            {score != null && (
              <span
                className={cn(
                  "ml-2 font-semibold",
                  score >= 80 ? "text-success" : score >= 40 ? "text-warning" : "text-danger"
                )}
              >
                • {summary.grounded} of {summary.facts} facts supported ({score}%)
              </span>
            )}
            </span>
          </span>
        </div>
        {/* Saved Knowledge vs Fresh RAG comparison — shown only when the
            backend found strong saved matches for this query. Answer-scoped:
            a new query clears it. */}
        <KnowledgeComparisonPanel />

        <div data-testid="canvas-scroll" className="min-h-0 flex-1 overflow-y-auto p-6">
          {editingText != null ? (
            <div className="flex h-full flex-col gap-2">
              <textarea
                value={editingText}
                onChange={(e) => setEditingText(e.target.value)}
                className="min-h-0 flex-1 resize-none rounded-md border border-border bg-background/60 p-3 font-mono text-[13px] leading-relaxed text-foreground focus:outline-none"
              />
              <div className="flex justify-end gap-2">
                <button onClick={cancelEdit} className="rounded border border-border px-3 py-1 text-xs font-semibold text-muted hover:bg-surface-2">
                  Cancel
                </button>
                <button onClick={saveEdit} className="rounded bg-success px-3 py-1 text-xs font-semibold text-white hover:opacity-90">
                  Save
                </button>
              </div>
            </div>
          ) : display ? (
            <div ref={streamRef} className="md-body text-[15px] leading-relaxed text-foreground">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={markdownComponents}
              >
                {display}
              </ReactMarkdown>
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
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={markdownComponents}
                  >
                    {pendingEdit.original}
                  </ReactMarkdown>
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
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={markdownComponents}
                  >
                    {pendingEdit.revised}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Deep-mode verification progress — Wait / Start anyway */}
      <DeepVerifyBanner />

      {/* Docked editing tools */}
      {factsOpen && (
        <FactVerificationPanel onClose={() => setFactsOpen(false)} />
      )}

      <div className="space-y-3 rounded-b-lg border border-t-0 border-border bg-surface p-3 shadow-sm">
        <Toolbar editing={editing} />
        {/* data-tour: product-tour anchor for "Cross-Verify Facts". */}
        <div className="flex gap-3" data-tour="draft-verify">
          <button
            onClick={() => edit("Rewrite this draft in a formal official register.")}
            disabled={!content || editing}
            className="rounded-md border border-border bg-surface px-4 py-2 text-xs font-bold uppercase text-foreground shadow-sm hover:bg-surface-2 disabled:opacity-50"
          >
            Formalize
          </button>
          <button
            onClick={() => void crossVerify()}
            disabled={!content || verifying}
            className="rounded-md border border-border bg-surface px-4 py-2 text-xs font-bold uppercase text-foreground shadow-sm hover:bg-surface-2 disabled:opacity-50"
          >
            {verifying ? "Verifying…" : "Cross-Verify Facts"}
          </button>
        </div>
      </div>
    </div>
  );
}
