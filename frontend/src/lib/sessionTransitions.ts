/**
 * Session state transitions.
 *
 * A session's state is spread across stores by design:
 *
 *   useSessionStore   conversation (messages, active id)   — persisted
 *   useDraftStore     canvas content + sources + grounding — projection
 *   useActivityStore  live model activity + error          — projection
 *   usePipelineStore  retrieval stage trace                — projection
 *   useChatRunStore   the in-flight answer, if any         — lib/chatRun.ts
 *
 * The three projection stores show ONE chat at a time. A running answer keeps
 * writing to its own message after the user switches away (lib/chatRun.ts).
 * Switching re-binds those stores to whichever chat is opened, and re-attaches
 * the live run if that chat is the one still generating.
 *
 * Both "+" call sites (Sidebar and HistoryPanel) and the history "open" path go
 * through here, so the transitions cannot drift apart.
 *
 * Global state is deliberately untouched: provider/model selection lives in
 * useAppStore, saved knowledge is server-side, and neither is referenced here.
 */
import { useActivityStore } from "@/store/useActivityStore";
import { useDraftStore } from "@/store/useDraftStore";
import { usePipelineStore } from "@/store/usePipelineStore";
import { useSessionStore } from "@/store/useSessionStore";
import { clearAnswerScopedPanels } from "@/lib/answerScope";
import {
  attachActiveRunIfAny,
  attachStoredRun,
  getActiveRun,
  reconcileStoredRuns,
  stopActiveRun,
} from "@/lib/chatRun";

export { clearAnswerScopedPanels } from "@/lib/answerScope";

/** Clear every per-session projection. Does NOT touch app/provider settings. */
export function clearSessionScopedState(): void {
  const draft = useDraftStore.getState();
  draft.reset();
  // reset() intentionally leaves the binding alone, so unbind explicitly: the
  // canvas is not showing any message until the next bind.
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

  // Answer-specific transient panels: Cross-Verified Facts, its inline
  // investigations, the Edit-with-AI comparison, and any in-flight edit.
  clearAnswerScopedPanels();
}

/**
 * Start a genuinely clean session: new conversation + empty canvas, sources,
 * graph and activity. A run still generating in another chat keeps going.
 * Returns the new session id.
 */
export function startNewSession(): string {
  const sessions = useSessionStore.getState();
  const id = sessions.createSession();
  sessions.setActive(id);
  clearSessionScopedState();
  return id;
}

/**
 * Open an existing session. If its answer is still generating in this tab, the
 * live state is re-attached. If it is generating on the server (started in
 * another tab, or before a reload), this tab follows that run: the text produced
 * so far is shown at once, and new tokens arrive live. Otherwise the canvas
 * loads the last answer. Per-session state is cleared first, so nothing leaks.
 */
export function restoreSession(id: string): void {
  const sessions = useSessionStore.getState();
  sessions.setActive(id);
  clearSessionScopedState();

  if (attachActiveRunIfAny(id)) return;

  const session = useSessionStore.getState().sessions.find((s) => s.id === id);
  const last = session
    ? [...session.messages].reverse().find((m) => m.role === "assistant")
    : undefined;
  if (!last) return;
  if (last.status === "streaming" && last.runId && attachStoredRun(id, last.id)) return;

  const draft = useDraftStore.getState();
  draft.setDraft(last.content || "", last.sources ?? []);
  draft.bindMessage(session?.id ?? null, last.id ?? null);
}

/**
 * Delete a session. If it is still generating, that run is stopped first. If
 * the canvas shows it, the canvas is cleared.
 */
export function deleteSessionSafely(id: string): void {
  if (getActiveRun()?.sessionId === id) stopActiveRun();
  const wasBound = useDraftStore.getState().activeSessionId === id;
  useSessionStore.getState().deleteSession(id);
  if (wasBound) clearSessionScopedState();
}

let loadRestored = false;

/**
 * Once per page load: re-open the chat that THIS TAB had open before a reload,
 * if it still exists (a new tab has none, so it starts fresh). Then settle any
 * answer that was still streaming, in any chat. Never creates a session.
 */
export function restoreActiveSessionOnLoad(): void {
  if (loadRestored) return;
  loadRestored = true;
  const id = useSessionStore.getState().activeSessionId;
  if (id && !getActiveRun()) restoreSession(id);
  reconcileStoredRuns();
}
