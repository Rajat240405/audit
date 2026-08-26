import { Plus, Trash2 } from "lucide-react";
import { useSessionStore } from "@/store/useSessionStore";
import { useFilteredSessions } from "@/hooks/useSessions";
import { useChatActionsStore } from "@/store/useChatActionsStore";
import { Message } from "@/components/chat/Message";
import { ChatInput } from "@/components/chat/ChatInput";

/**
 * Left sidebar — the "assistant" column (matches the Stitch design):
 * session pills on top, chat history in the middle, query input docked at
 * the bottom.
 */
export function Sidebar() {
  const sessions = useFilteredSessions();
  const activeId = useSessionStore((s) => s.activeSessionId);
  const setActive = useSessionStore((s) => s.setActive);
  const createSession = useSessionStore((s) => s.createSession);
  const deleteSession = useSessionStore((s) => s.deleteSession);
  const renameSession = useSessionStore((s) => s.renameSession);
  const active = sessions.find((s) => s.id === activeId) ?? null;

  const send = useChatActionsStore((s) => s.send);
  const stop = useChatActionsStore((s) => s.stop);
  const running = useChatActionsStore((s) => s.running);

  const newSession = () => {
    const id = createSession();
    setActive(id);
  };

  const rename = () => {
    if (!active) return;
    const name = window.prompt("Session name", active.title);
    if (name && name.trim()) renameSession(active.id, name.trim());
  };

  return (
    <aside className="flex w-96 shrink-0 flex-col justify-between border-r border-border bg-surface">
      {/* Session tabs */}
      <div className="flex items-center gap-1 overflow-x-auto border-b border-border p-2">
        <button
          className="shrink-0 rounded-full bg-foreground px-3 py-1 text-xs font-semibold text-background"
          onClick={newSession}
          title="Start a new session"
        >
          {active?.title ?? "Default Session"}
        </button>
        <button
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-border text-muted hover:bg-surface-2"
          onClick={newSession}
          title="New session"
        >
          <Plus className="h-3 w-3" />
        </button>
        <button
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-border text-muted hover:bg-surface-2"
          onClick={rename}
          title="Rename session"
        >
          R
        </button>
        <button
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-border text-muted hover:bg-surface-2"
          onClick={() => activeId && deleteSession(activeId)}
          title="Delete session"
        >
          <Trash2 className="h-3 w-3" />
        </button>
      </div>

      {/* Chat history */}
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Assistant
        </div>
        {!active || active.messages.length === 0 ? (
          <div className="rounded-xl border border-border bg-surface-2 p-4 text-sm text-foreground/80">
            Conversation started. Ask your audit question below.
          </div>
        ) : (
          <div className="space-y-3">
            {active.messages.map((m) => (
              <Message key={m.id} message={m} />
            ))}
          </div>
        )}
      </div>

      {/* Query input (docked at bottom) */}
      <ChatInput onSend={(q) => send?.(q)} onStop={() => stop?.()} streaming={running} />
    </aside>
  );
}
