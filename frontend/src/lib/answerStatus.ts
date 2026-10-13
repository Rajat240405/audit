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
  /** The server no longer has the run (it restarted, or the run expired). */
  serverLost: "> ⚠️ The server no longer has this answer (it restarted or the run expired). The text above is what was saved.",
  /** The server reports the run failed. Its detail is never shown. */
  serverFailed: "> ⚠️ Generation failed on the server.",
} as const;

/** Append a labelled notice after any partial text (or stand alone if none). */
export function withNotice(partial: string, notice: string): string {
  return partial.trim() ? `${partial}\n\n---\n\n${notice}` : notice;
}

/**
 * Turn persisted storage into a safe starting state.
 *
 * A message still marked `streaming` with NO server run id cannot be resumed
 * (it predates server-owned runs), so it is recovered as `interrupted` with its
 * partial text kept and labelled.
 *
 * A message with a run id is different: the generation lives on the server. It
 * stays `streaming` here, and the run registry asks the server for its real
 * status when the chat is opened (or on load, for finished runs). A reload never
 * marks a still-running answer as interrupted.
 *
 * The active session is NOT taken from storage: it is per tab (see
 * useSessionStore). Nothing is created here.
 */
export function recoverPersistedSessions(persisted: unknown): {
  sessions: Session[];
  deletedSessionIds: string[];
  deletedMessageIds: string[];
} {
  const p = (persisted ?? {}) as {
    sessions?: unknown;
    deletedSessionIds?: unknown;
    deletedMessageIds?: unknown;
  };
  const raw = Array.isArray(p.sessions) ? (p.sessions as Session[]) : [];
  const strings = (v: unknown) => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
  const sessions = raw.map((s) => ({
    ...s,
    messages: (s.messages ?? []).map((m) =>
      m.status === "streaming" && !m.runId
        ? {
            ...m,
            status: "interrupted" as const,
            content: withNotice(m.content ?? "", NOTICE.reloaded),
          }
        : m
    ),
  }));
  return {
    sessions,
    deletedSessionIds: strings(p.deletedSessionIds),
    deletedMessageIds: strings(p.deletedMessageIds),
  };
}
