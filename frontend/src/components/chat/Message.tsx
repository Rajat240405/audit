import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "@/types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/utils/cn";
import { saveKnowledge } from "@/api/model";
import { useSessionStore } from "@/store/useSessionStore";
import { useDraftStore } from "@/store/useDraftStore";
import { usePipelineStore } from "@/store/usePipelineStore";

/** Message card in the left assistant column (matches the Stitch design). */
export function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const [saving, setSaving] = useState(false);

  // Live state for the assistant message currently being generated. The
  // session transcript only receives `content` at the END of the stream
  // (onMeta), so without this the card renders a static "…" for the whole
  // request and the real backend phases never appear in the left chat.
  //
  // `activeMessageId` is bound by useChatStream at stream start, so exactly
  // ONE message can ever be the live one.
  const isLive = useDraftStore((s) => s.activeMessageId) === message.id;
  const streamingText = useDraftStore((s) => (isLive ? s.streamingText : ""));
  // usePipelineStore.message is set by onStatus from the SSE "status" events —
  // it is REPLACED on every phase, so it is inherently a single status with no
  // accumulated history.
  const phaseMessage = usePipelineStore((s) => (isLive ? s.message : ""));

  // Streamed tokens win over the phase label: the moment real answer text
  // arrives, "Generating answer…" disappears and the answer takes its place.
  const liveBody = isLive ? streamingText : "";
  const body = message.content || liveBody;

  const onSaveKnowledge = async () => {
    if (saving) return;
    setSaving(true);
    try {
      // find the preceding USER message = the question for this answer
      const sessions = useSessionStore.getState();
      const active = sessions.sessions.find((s) => s.id === sessions.activeSessionId);
      let question = "";
      if (active) {
        const idx = active.messages.findIndex((m) => m.id === message.id);
        for (let i = idx - 1; i >= 0; i--) {
          if (active.messages[i].role === "user") {
            question = active.messages[i].content;
            break;
          }
        }
      }
      if (!question) question = message.content.slice(0, 80);
      await saveKnowledge({ question, answer: message.content, sources: message.sources ?? [] });
      alert("Saved to Knowledge ✓");
    } catch (e) {
      alert("Save failed: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setSaving(false);
    }
  };

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-xl rounded-tr-none bg-accent/10 px-4 py-2.5 text-sm leading-relaxed text-foreground">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <div
        className={cn(
          "rounded-xl border bg-surface p-4 shadow-sm",
          message.meta?.is_fallback ? "border-warning/50" : "border-border"
        )}
      >
        {body ? (
          <div className="md-body text-sm text-foreground/90">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // Keep wide tables from overflowing the message card: wrap each
                // table in its own horizontally-scrollable container.
                table: ({ children }) => (
                  <div className="overflow-x-auto">
                    <table>{children}</table>
                  </div>
                ),
              }}
            >
              {body}
            </ReactMarkdown>
          </div>
        ) : isLive && phaseMessage ? (
          <p
            data-testid="chat-progress-status"
            className="flex items-center gap-2 text-sm text-muted"
          >
            <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
            {phaseMessage}
          </p>
        ) : (
          <p className="text-sm text-muted">…</p>
        )}
      </div>

      {message.meta && (
        <div className="flex flex-wrap items-center gap-1.5 px-1">
          <Badge variant="muted">
            {message.meta.provider} · {message.meta.model}
          </Badge>
          <Badge variant="muted">{message.meta.profile}</Badge>
          {message.meta.total_tokens != null && (
            <Badge variant="muted">{message.meta.total_tokens} tok</Badge>
          )}
          {message.meta.is_fallback && <Badge variant="warning">fallback</Badge>}
        </div>
      )}

      <div className="flex gap-2 pl-1">
        <button
          className="rounded-full border border-border bg-surface-2 px-3 py-1 text-[11px] text-foreground/70 hover:bg-surface"
          onClick={() => {
            try {
              void navigator.clipboard.writeText(message.content);
            } catch {
              /* clipboard unavailable */
            }
          }}
        >
          Copy
        </button>
        <button
          className="rounded-full border border-border bg-surface-2 px-3 py-1 text-[11px] text-foreground/70 hover:bg-surface disabled:opacity-50"
          onClick={onSaveKnowledge}
          disabled={saving}
        >
          {saving ? "Saving…" : "Save to Knowledge"}
        </button>
      </div>
    </div>
  );
}
