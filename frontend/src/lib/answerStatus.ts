/**
 * Answer lifecycle helpers shared by the run registry and the session store.
 *
 * An assistant message is written while its answer streams and is settled
 * when the run ends (see MessageStatus in types/api.ts). Whatever text existed
 * is kept and labelled, so a partial answer is never presented as finished.
 */
import type { Session } from "@/types";

export const NOTICE = {
  stopped: "> ⏹ Generation was stopped before the answer finished.",
  connectionClosed: "> ⚠️ The connection closed before the answer finished.",
  reloaded: "> ⚠️ The page was reloaded or closed before this answer finished.",
} as const;

/** Append a labelled notice after any partial text (or stand alone if none). */
export function withNotice(partial: string, notice: string): string {
  return partial.trim() ? `${partial}\n\n---\n\n${notice}` : notice;
}

/**
 * Turn persisted storage into a safe starting state.
 *
 * A run cannot survive a page load, so every message still marked `streaming`
 * is recovered as `interrupted` with its partial text kept and labelled. The
 * active session is restored only if it still exists. Nothing is created here:
 * a reload never adds a session.
 */
export function recoverPersistedSessions(persisted: unknown): {
  sessions: Session[];
  activeSessionId: string | null;
} {
  const p = (persisted ?? {}) as { sessions?: unknown; activeSessionId?: unknown };
  const raw = Array.isArray(p.sessions) ? (p.sessions as Session[]) : [];
  const sessions = raw.map((s) => ({
    ...s,
    messages: (s.messages ?? []).map((m) =>
      m.status === "streaming"
        ? {
            ...m,
            status: "interrupted" as const,
            content: withNotice(m.content ?? "", NOTICE.reloaded),
          }
        : m
    ),
  }));
  const wanted = typeof p.activeSessionId === "string" ? p.activeSessionId : null;
  const activeSessionId = wanted && sessions.some((s) => s.id === wanted) ? wanted : null;
  return { sessions, activeSessionId };
}
