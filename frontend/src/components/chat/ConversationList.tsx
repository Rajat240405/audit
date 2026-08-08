import { useSessionStore } from "@/store/useSessionStore";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Message } from "./Message";

/** Renders the messages of the active session (read-only transcript). */
export function ConversationList() {
  const sessions = useSessionStore((s) => s.sessions);
  const activeId = useSessionStore((s) => s.activeSessionId);
  const active = sessions.find((s) => s.id === activeId);

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto flex max-w-3xl flex-col gap-4 px-4 py-4">
        {!active || active.messages.length === 0 ? (
          <EmptyState />
        ) : (
          active.messages.map((m) => <Message key={m.id} message={m} />)
        )}
      </div>
    </ScrollArea>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-3 py-16 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-accent/15 text-accent text-lg font-bold">
        IA
      </div>
      <h2 className="text-lg font-semibold">Scientific Audit Workstation</h2>
      <p className="max-w-sm text-sm text-muted">
        Query real Lok Sabha Q&amp;A documents through Hybrid RAG or GraphRAG.
        Evidence, pipeline traces and drafting tools live in the right panel.
      </p>
    </div>
  );
}
