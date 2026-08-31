import { useRef, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useDraftStore } from "@/store/useDraftStore";
import { useSessionStore } from "@/store/useSessionStore";
import { useToastStore } from "@/store/useToastStore";
import { useEditDraft } from "@/hooks/useEditDraft";
import { useEditStore } from "@/store/useEditStore";
import { Toolbar } from "./Toolbar";
import { buildGroundingReport } from "@/services/grounding";
import { cn } from "@/utils/cn";

/**
 * Drafting canvas — the heart of the workstation. White document card with a
 * header, live streaming markdown body, and docked AI editing tools.
 *
 * The canvas shows ONLY the final draft/answer. Retrieved sources / citations
 * are deliberately NOT rendered here — they stay available internally for RAG
 * and verification (see the Sources tab and the grounding score below), so this
 * is purely a presentation choice. The "Cross-Verify Facts" action recomputes
 * the grounding report (updating the score) without showing the source list.
 */
export function DraftCanvas() {
  const content = useDraftStore((s) => s.content);
  const streamingText = useDraftStore((s) => s.streamingText);
  const isStreaming = useDraftStore((s) => s.isStreaming);
  const grounding = useDraftStore((s) => s.grounding);
  const { edit, pendingEdit, accept, reject } = useEditDraft();
  const editing = useEditStore((s) => s.editing);
  const pushToast = useToastStore((s) => s.push);
  const streamRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    streamRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [streamingText]);

  const verified = grounding.filter((g) => g.found).length;
  const score = grounding.length ? Math.round((verified / grounding.length) * 100) : null;

  // Re-run the client-side grounding verification on the current draft against
  // its retrieved sources. Updates the Grounding score only — it does NOT
  // reveal the sources/citations list inside the canvas (Sources tab remains
  // the home for that).
  const crossVerify = () => {
    const { content: c, sources } = useDraftStore.getState();
    if (!c) return;
    const report = buildGroundingReport(c, sources);
    useDraftStore.setState({ grounding: report });

    const unverified = report.filter((r) => !r.found);
    if (unverified.length === 0) {
      pushToast("success", "Cross-verification complete — all claims grounded in sources.");
    } else {
      const claims = unverified.map((r) => `• ${r.text}`).join("\n");
      pushToast(
        "error",
        `Cross-verification: ${unverified.length} claim(s) not found in sources:\n${claims}`
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
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-t-lg border border-border bg-surface shadow-sm">
        <div className="flex items-center justify-between border-b border-border bg-surface-2/60 px-4 py-2">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-muted">
            Drafting Canvas
          </span>
          <span className="flex items-center gap-2">
            {content && !isStreaming && editingText == null && (
              <button
                onClick={startEdit}
                className="rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-bold text-foreground shadow-sm hover:bg-surface-2"
              >
                ✏️ Edit
              </button>
            )}
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
          </span>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-6">
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
                components={{
                  table: ({ children }) => (
                    <div className="overflow-x-auto">
                      <table>{children}</table>
                    </div>
                  ),
                }}
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
                    components={{
                      table: ({ children }) => (
                        <div className="overflow-x-auto">
                          <table>{children}</table>
                        </div>
                      ),
                    }}
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
                    components={{
                      table: ({ children }) => (
                        <div className="overflow-x-auto">
                          <table>{children}</table>
                        </div>
                      ),
                    }}
                  >
                    {pendingEdit.revised}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Docked editing tools */}
      <div className="space-y-3 rounded-b-lg border border-t-0 border-border bg-surface p-3 shadow-sm">
        <Toolbar editing={editing} />
        <div className="flex gap-3">
          <button
            onClick={() => edit("Rewrite this draft in a formal official register.")}
            disabled={!content || editing}
            className="rounded-md border border-border bg-surface px-4 py-2 text-xs font-bold uppercase text-foreground shadow-sm hover:bg-surface-2 disabled:opacity-50"
          >
            Formalize
          </button>
          <button
            onClick={crossVerify}
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
