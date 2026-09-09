/**
 * Session state transitions.
 *
 * A session's state is spread across four stores by design:
 *
 *   useSessionStore   conversation (messages, active id)   — persisted
 *   useDraftStore     canvas content + sources + grounding — per session
 *   useActivityStore  live model activity + error          — per session
 *   usePipelineStore  retrieval stage trace                — per session
 *
 * Creating a session previously only touched useSessionStore, so the canvas,
 * sources (which carry the GraphRAG provenance the Graph tab renders) and
 * activity of the PREVIOUS session survived into the new one. Restoring a
 * history item did clear/replace them, which is why the bug only showed on
 * the "open old session -> New Session" path.
 *
 * Both transitions live here so the two "+" call sites (Sidebar and
 * HistoryPanel) can never drift apart again.
 *
 * Global state is deliberately untouched: provider/model selection lives in
 * useAppStore, saved knowledge is server-side, and neither is referenced here.
 */
import { useActivityStore } from "@/store/useActivityStore";
import { useDraftStore } from "@/store/useDraftStore";
import { usePipelineStore } from "@/store/usePipelineStore";
import { useSessionStore } from "@/store/useSessionStore";

/** Clear every per-session store. Does NOT touch app/provider settings. */
export function clearSessionScopedState(): void {
  const draft = useDraftStore.getState();
  draft.reset();
  // reset() intentionally leaves the binding alone (it is also used mid-stream),
  // so unbind explicitly: a new session owns no message yet.
  draft.bindMessage(null, null);

  // Activity: back to idle, not "retrieving" — nothing is running yet.
  useActivityStore.setState({
    phase: "idle",
    sources: [],
    reasoning: "",
    answerChars: 0,
    model: null,
    question: null,
    startedAt: null,
    error: null,
  });

  usePipelineStore.getState().reset();
}

/**
 * Start a genuinely clean session: new conversation + empty canvas, sources,
 * graph and activity. Returns the new session id.
 */
export function startNewSession(): string {
  const sessions = useSessionStore.getState();
  const id = sessions.createSession();
  sessions.setActive(id);
  clearSessionScopedState();
  return id;
}

/**
 * Open an existing session and restore its canvas from the last assistant
 * message. Per-session state is cleared first so nothing leaks across from the
 * session being left.
 */
export function restoreSession(id: string): void {
  const sessions = useSessionStore.getState();
  sessions.setActive(id);
  clearSessionScopedState();

  const session = useSessionStore.getState().sessions.find((s) => s.id === id);
  const last = session
    ? [...session.messages].reverse().find((m) => m.role === "assistant")
    : undefined;
  if (!last) return;

  const draft = useDraftStore.getState();
  draft.setDraft(last.content || "", last.sources ?? []);
  draft.bindMessage(session?.id ?? null, last.id ?? null);
}
