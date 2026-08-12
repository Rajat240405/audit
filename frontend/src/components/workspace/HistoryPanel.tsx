import { History, MessagesSquare, Pencil, Plus, Trash2 } from "lucide-react";
import { useSessionStore } from "@/store/useSessionStore";
import { formatDate } from "@/utils/formatters";
import { cn } from "@/utils/cn";

/**
 * History — the list of saved SESSIONS. Each session is labelled by its
 * first query. Clicking one opens it: the sidebar shows that session's full
 * conversation and the canvas loads the last answer.
 * Each entry supports RENAME (edit the name) and DELETE.
 */
export function HistoryPanel({ onOpen }: { onOpen?: () => void }) {
  const sessions = useSessionStore((s) => s.sessions);
  const activeId = useSessionStore((s) => s.activeSessionId);
  const setActive = useSessionStore((s) => s.setActive);
  const createSession = useSessionStore((s) => s.createSession);
  const renameSession = useSessionStore((s) => s.renameSession);
  const deleteSession = useSessionStore((s) => s.deleteSession);

  const open = (id: string) => {
    setActive(id);
    onOpen?.();
  };

  const rename = (id: string, current: string) => {
    const name = window.prompt("Session name", current);
    if (name && name.trim()) renameSession(id, name.trim());
  };

  const remove = (id: string) => {
    if (!window.confirm("Delete this session? This cannot be undone.")) return;
    deleteSession(id);
  };

  const newSession = () => {
    const id = createSession();
    setActive(id);
  };

  const sorted = [...sessions]
    .filter((s) => s.messages.length > 0) // hide empty sessions (e.g. fresh startup)
    .sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-3 py-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-muted">
          <History className="h-3.5 w-3.5" />
          History
        </div>
        <button
          onClick={newSession}
          className="flex items-center gap-1 rounded-md border border-border bg-surface-2 px-2 py-1 text-[11px] font-semibold text-foreground/80 hover:bg-surface"
        >
          <Plus className="h-3 w-3" />
          New session
        </button>
      </div>

      <div className="flex-1 space-y-1 overflow-y-auto px-2 pb-2">
        {sorted.length === 0 ? (
          <p className="px-2 py-6 text-center text-xs text-muted">
            No sessions yet. Start a conversation and it will appear here.
          </p>
        ) : (
          sorted.map((s) => {
            const firstMsg = s.messages.find((m) => m.role === "user");
            const lastAssistant = [...s.messages].reverse().find((m) => m.role === "assistant");
            const label = firstMsg?.content || s.title || "Untitled session";
            const qCount = s.messages.filter((m) => m.role === "user").length;
            return (
              <div
                key={s.id}
                role="button"
                tabIndex={0}
                onClick={() => open(s.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    open(s.id);
                  }
                }}
                title="Open this session"
                className={cn(
                  "group flex cursor-pointer items-start justify-between gap-2 rounded-md border border-border px-2.5 py-2 hover:bg-surface-2",
                  activeId === s.id && "border-accent/40 bg-accent/10"
                )}
              >
                <div className="min-w-0">
                  <p className="line-clamp-2 text-xs font-medium">{label}</p>
                  <p className="mt-0.5 flex items-center gap-1 text-[10px] text-muted">
                    <MessagesSquare className="h-3 w-3" />
                    {qCount} quer{qCount === 1 ? "y" : "ies"}
                    {lastAssistant?.content ? " · answered" : ""}
                  </p>
                  <p className="text-[10px] text-muted">{formatDate(s.updatedAt)}</p>
                </div>
                <div
                  className="flex shrink-0 items-center gap-0.5"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    className="rounded p-1 text-muted opacity-0 hover:bg-surface hover:text-accent group-hover:opacity-100"
                    onClick={() => rename(s.id, label)}
                    title="Rename session"
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </button>
                  <button
                    className="rounded p-1 text-muted opacity-0 hover:bg-surface hover:text-danger group-hover:opacity-100"
                    onClick={() => remove(s.id)}
                    title="Delete session"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
