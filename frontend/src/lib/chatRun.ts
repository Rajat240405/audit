/**
 * Chat run registry — the single owner of an in-flight answer.
 *
 * POLICY (see docs/frontend-session-isolation.md):
 *
 *  1. One run at a time. Sending is refused while any run is active, in any
 *     session. A second send (double click, Enter twice) is refused
 *     synchronously, so overlapping requests cannot start.
 *  2. A run belongs to the session and assistant message it was started in.
 *     Its text, sources, trace, metadata, status, errors and completion are
 *     written to THAT message in the session store, whichever chat is on
 *     screen. Switching chats does not stop the run.
 *  3. The live canvas, activity panel and RAG-pipeline view are a PROJECTION
 *     of the run, shown only while the canvas is bound to the run's message.
 *     Switching back to a running chat re-projects the accumulated state and
 *     the stream continues from there. Nothing from one chat is ever written
 *     into the stores while another chat is displayed.
 *  4. The generation belongs to the SERVER (see generation_runs.py). The run id
 *     is stored on the message, so a reload, another tab, or a dropped
 *     connection can attach to the same run: the server replays every frame
 *     produced so far and then follows it live. Attaching never starts a
 *     second inference.
 *  5. Only Stop cancels a run (an explicit request to the server, plus a local
 *     abort). Closing, reloading or navigating a tab does not: the run goes on.
 *  6. The server decides how a run ends (GET /api/chat/runs/{id}). A stream that
 *     drops while the server still reports "running" is resumed, never shown as
 *     interrupted. "interrupted" means the server lost the run or it is
 *     incomplete. Partial text is kept and labelled.
 *  7. Every late callback from a closed run, or from an earlier attempt of the
 *     same run, is ignored. A replay therefore cannot duplicate tokens.
 */
import {
  attachRunStream,
  cancelRun,
  fetchRunStatus,
  streamChat,
  verifyAnswer,
  type RunStatus,
} from "@/api/chat";
import type { StreamHandlers } from "@/api/client";
import { useAppStore } from "@/store/useAppStore";
import {
  useDraftStore,
  deepVerifyBlocksNewQuery,
  type DeepVerifyStatus,
} from "@/store/useDraftStore";
import { useActivityStore, type ActivityPhase } from "@/store/useActivityStore";
import { usePipelineStore } from "@/store/usePipelineStore";
import { useSessionStore } from "@/store/useSessionStore";
import { useToastStore } from "@/store/useToastStore";
import { create } from "zustand";
import { showsHybridStages } from "@/lib/retrievalMode";
import { expandOrgFilter } from "@/lib/sourceFilter";
import { clearAnswerScopedPanels } from "@/lib/answerScope";
import { NOTICE, withNotice } from "@/lib/answerStatus";
import { toGroundingClaims } from "@/lib/serverGrounding";
import { graphStages, hybridStages } from "@/utils/formatters";
import type {
  ChatMessage,
  GenerationMeta,
  GroundingClaim,
  KnowledgeMatchCard,
  PipelineStage,
  RetrievalMode,
  RetrievalTrace,
  SourceItem,
} from "@/types";

/** How often streaming text is written to the session store. */
export const PERSIST_INTERVAL_MS = 800;

export interface RunKnowledge {
  matches: KnowledgeMatchCard[];
  tier: string | null;
  fresh: { content: string; sources: SourceItem[] } | null;
}

/** Everything the run has produced so far, independent of what is on screen. */
export interface LiveAnswer {
  /** Raw streamed text. */
  text: string;
  /** Meta or final arrived (or the run closed): `content` is the settled text. */
  committed: boolean;
  content: string;
  /** The run has closed (done, stopped, interrupted or error). */
  finished: boolean;
  sources: SourceItem[];
  trace: RetrievalTrace | null;
  grounding: GroundingClaim[] | null;
  /** The answer text `grounding` was verified against (server rows only). */
  groundingFor: string | null;
  meta: GenerationMeta | null;
  knowledge: RunKnowledge | null;
  reasoning: string;
  answerChars: number;
  phase: ActivityPhase;
  model: string | null;
  error: string | null;
  startedAt: number;
  stages: PipelineStage[];
  stageMessage: string;
}

export interface ChatRun {
  /** The server's generation id (also stored on the message). */
  runId: string;
  sessionId: string;
  messageId: string;
  question: string;
  /** Aborts the CURRENT stream attempt only. Replaced for each attempt. */
  abort: AbortController;
  live: LiveAnswer;
  /** Pristine copy of `live`, restored before each replay from frame 0. */
  initial: LiveAnswer;
  persistTimer: ReturnType<typeof setTimeout> | null;
  /** Set once the run has closed. Late callbacks are then ignored. */
  closed: boolean;
  /** Stream attempt number. Events from an older attempt are ignored. */
  attempt: number;
  /** The server has acknowledged this run (it sent at least one frame). */
  acked: boolean;
  /** A replay (not the original generation): its completion is never re-verified. */
  replay: boolean;
  /** Registered as this tab's active run (the canvas and Stop act on it). */
  foreground: boolean;
  /** The terminal status has been replayed once, so its full text is known. */
  replayedTerminal: boolean;
  /** Consecutive failed status checks (drives the retry back-off). */
  retries: number;
  resumeTimer: ReturnType<typeof setTimeout> | null;
}

interface ChatRunState {
  active: ChatRun | null;
}

/** The one active run (or null). Components subscribe to this. */
export const useChatRunStore = create<ChatRunState>(() => ({ active: null }));

export function getActiveRun(): ChatRun | null {
  return useChatRunStore.getState().active;
}

function newId(): string {
  return Math.random().toString(36).slice(2, 10);
}

/**
 * A run id the server accepts (32 hex characters). Generated with
 * getRandomValues, which also works on plain-http origins where randomUUID does not.
 */
function newRunId(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

const RESUME_MIN_MS = 1000;
const RESUME_MAX_MS = 15000;

function backoffMs(retries: number): number {
  return Math.min(RESUME_MAX_MS, RESUME_MIN_MS * 2 ** Math.min(retries, 4));
}

function cloneLive(live: LiveAnswer): LiveAnswer {
  return JSON.parse(JSON.stringify(live)) as LiveAnswer;
}

/** Fresh projection state for a run, before any frame has arrived. */
function newLive(retrievalMode: RetrievalMode, now: number): LiveAnswer {
  const defs = showsHybridStages(retrievalMode) ? hybridStages() : graphStages();
  return {
    text: "",
    committed: false,
    content: "",
    finished: false,
    sources: [],
    trace: null,
    grounding: null,
    groundingFor: null,
    meta: null,
    knowledge: null,
    reasoning: "",
    answerChars: 0,
    phase: "retrieving",
    model: null,
    error: null,
    startedAt: now,
    stages: defs.map((d) => ({ key: d.key, label: d.label, status: "pending" as const })),
    stageMessage: "",
  };
}

function makeRun(init: {
  runId: string;
  sessionId: string;
  messageId: string;
  question: string;
  live: LiveAnswer;
  replay: boolean;
  foreground: boolean;
}): ChatRun {
  return {
    ...init,
    abort: new AbortController(),
    initial: cloneLive(init.live),
    persistTimer: null,
    closed: false,
    attempt: 0,
    acked: false,
    replayedTerminal: false,
    retries: 0,
    resumeTimer: null,
  };
}

function messageContent(sessionId: string, messageId: string): string | undefined {
  const s = useSessionStore.getState().sessions.find((x) => x.id === sessionId);
  return s?.messages.find((m) => m.id === messageId)?.content;
}

/** True only while the canvas is bound to this run's message. */
export function isForeground(run: ChatRun): boolean {
  const d = useDraftStore.getState();
  return d.activeSessionId === run.sessionId && d.activeMessageId === run.messageId;
}

function persist(run: ChatRun, patch: Partial<ChatMessage>): void {
  useSessionStore.getState().updateMessage(run.sessionId, run.messageId, patch);
}

function markStage(stages: PipelineStage[], key: string, status: PipelineStage["status"]): PipelineStage[] {
  return stages.map((st) => (st.key === key ? { ...st, status } : st));
}

/**
 * Write the run's state into the foreground stores. Only ever called for the
 * run that the canvas is bound to. Every field is derived from `run.live`, so
 * the projection is the same whether it is a live update or a re-attach.
 */
function writeStores(run: ChatRun): void {
  const L = run.live;
  useDraftStore.setState({
    isStreaming: !L.committed,
    streamingText: L.committed ? "" : L.text,
    content: L.committed ? L.content : "",
    sources: L.sources,
    trace: L.trace,
    // Server rows only. Empty until /api/verify has answered for this run.
    grounding: L.grounding ?? [],
    groundingFor: L.groundingFor,
    lastMeta: L.meta as unknown as Record<string, unknown> | null,
    knowledge: L.knowledge,
  });
  useActivityStore.setState({
    phase: L.phase,
    sources: L.sources,
    reasoning: L.reasoning,
    answerChars: L.answerChars,
    model: L.model,
    question: run.question,
    error: L.error,
    startedAt: L.startedAt,
  });
  usePipelineStore.setState({
    stages: L.stages,
    running: !L.finished,
    message: L.stageMessage,
  });
}

/** Project the run onto the stores, but only if it is the foreground run. */
function project(run: ChatRun): void {
  if (run.closed || !isForeground(run)) return;
  writeStores(run);
}

/** Write the partial text now (also used on page hide). Never after close. */
function flushPartial(run: ChatRun): void {
  if (run.closed) return;
  const L = run.live;
  persist(run, {
    content: L.committed ? L.content : L.text,
    status: "streaming",
  });
}

function schedulePersist(run: ChatRun): void {
  if (run.closed || run.persistTimer) return;
  run.persistTimer = setTimeout(() => {
    run.persistTimer = null;
    flushPartial(run);
  }, PERSIST_INTERVAL_MS);
}

/**
 * How a run ends. `lost` is the server reporting the run unknown (it restarted or
 * the run expired); it is stored as `interrupted`.
 */
type Outcome = "done" | "stopped" | "interrupted" | "error" | "lost";

/**
 * Set on `pagehide` and cleared on `pageshow`. While the page is leaving, the
 * browser aborts the open stream, and that drop is not a generation error. It
 * is recorded as an interrupted answer with the reload notice instead.
 */
let pageLeaving = false;

/**
 * Close the run exactly once: settle the message, release the registry slot,
 * and project the final state if it is still on screen.
 */
function finalize(run: ChatRun, outcome: Outcome): void {
  if (run.closed) return;
  run.closed = true;
  if (run.persistTimer) {
    clearTimeout(run.persistTimer);
    run.persistTimer = null;
  }
  if (run.resumeTimer) {
    clearTimeout(run.resumeTimer);
    run.resumeTimer = null;
  }
  const L = run.live;
  const base = L.committed ? L.content : L.text;
  let content: string;
  switch (outcome) {
    case "stopped":
      content = withNotice(base, NOTICE.stopped);
      L.phase = "done";
      break;
    case "interrupted": {
      const notice = pageLeaving ? NOTICE.reloaded : NOTICE.connectionClosed;
      content = withNotice(base, notice);
      L.error = notice.replace(/^> ⚠️ /, "");
      L.phase = "error";
      break;
    }
    case "error":
      content = withNotice(base, `> ⚠️ ${L.error ?? "Generation failed."}`);
      break;
    case "lost":
      content = withNotice(base, NOTICE.serverLost);
      L.error = NOTICE.serverLost.replace(/^> ⚠️ /, "");
      L.phase = "error";
      break;
    default:
      content = base;
      if (L.phase !== "error") L.phase = "done";
  }
  L.content = content;
  L.committed = true;
  L.finished = true;
  if (outcome === "done" && L.knowledge) {
    // Snapshot the FRESH answer: the comparison panel needs a stable "fresh"
    // side even after the user picks the saved answer.
    L.knowledge = { ...L.knowledge, fresh: { content, sources: L.sources } };
  }
  L.stageMessage = "";
  L.stages = L.stages.map((st) => (st.status === "running" ? { ...st, status: "done" } : st));

  persist(run, {
    content,
    status: outcome === "lost" ? "interrupted" : outcome,
    sources: L.sources,
    trace: L.trace ?? undefined,
    meta: L.meta ?? undefined,
  });

  const reg = useChatRunStore.getState();
  if (reg.active === run) useChatRunStore.setState({ active: null });

  if (isForeground(run)) writeStores(run);
}

/**
 * Post-generation verification (grounding / judge / rewrite). It runs after a
 * completed answer. Results go to the answer's message. The canvas and the
 * toasts are updated only when that answer is still on screen. A background
 * result is stored only if the stored text is still the text that was verified.
 */
function startVerification(run: ChatRun, content: string, sources: SourceItem[]): void {
  const sid = run.sessionId;
  const mid = run.messageId;
  const isDeep = useAppStore.getState().mode === "deep";
  if (isDeep && isForeground(run)) {
    useDraftStore.getState().beginDeepVerify(sid, mid);
  }
  let verifyStatus: DeepVerifyStatus = "done";
  verifyAnswer(content, sources, isDeep ? "full" : "light")
    .then((res) => {
      if (res.error) {
        if (isForeground(run)) {
          useToastStore.getState().pushSticky("error", `Verification failed: ${res.error}`);
        }
        return;
      }
      if (!isForeground(run)) {
        if (
          res.text &&
          res.text !== content &&
          messageContent(sid, mid) === content
        ) {
          useSessionStore.getState().updateMessage(sid, mid, { content: res.text });
        }
        return;
      }
      const draft = useDraftStore.getState();
      const userEdited = draft.content !== content;
      if (userEdited) {
        // The user changed the draft during verify — don't overwrite.
        useToastStore.getState().pushSticky(
          "info",
          "Verification produced revisions, but you edited the draft — review the grounding report instead."
        );
        // Rows describe the text that was verified (`content`), which the
        // user has since changed, so the panel shows them as stale.
        const rows = toGroundingClaims(res.grounding || []);
        run.live.grounding = rows;
        run.live.groundingFor = content;
        useDraftStore.getState().setGrounding(rows, content);
        return;
      }
      if (res.text && res.text !== draft.content) {
        draft.applyEdit(res.text);
        useToastStore.getState().pushSticky(
          "success",
          "Answer verified & cleaned (unsupported claims removed)"
        );
      }
      // Rows describe the text that was VERIFIED (`content`). If the judge
      // rewrote the canvas above, `content` no longer matches it, so the panel
      // shows the rows as stale and the user can re-run Cross-Verify.
      const rows = toGroundingClaims(res.grounding || []);
      run.live.grounding = rows;
      run.live.groundingFor = content;
      useDraftStore.getState().setGrounding(rows, content);
      useSessionStore.getState().updateMessage(sid, mid, {
        content: useDraftStore.getState().content,
      });
    })
    .catch((err) => {
      console.warn("[verify] error:", err);
      verifyStatus = "error";
      if (isForeground(run)) {
        useToastStore.getState().pushSticky("error", "Verification failed unexpectedly");
      }
    })
    .finally(() => {
      if (isDeep && isForeground(run)) {
        useDraftStore.getState().settleDeepVerify(sid, mid, verifyStatus);
      }
    });
}

function buildHandlers(run: ChatRun, attempt: number): StreamHandlers {
  const L = run.live;
  // Only the current attempt of a live run may write. Events from an earlier
  // attempt (aborted, replaced by a replay) are dropped.
  const current = () => !run.closed && run.attempt === attempt;
  return {
    onStatus: (stage, message, done, _count, substage) => {
      if (!current()) return;
      run.acked = true;
      L.stages = markStage(L.stages, stage, done ? "done" : "running");
      // Substage events (embed/dense/bm25/rrf/rerank) drive the pipeline view
      // only. They must not replace the single phase label.
      if (!substage) L.stageMessage = message;
      // Entering the generate stage means the model is now thinking.
      if ((stage === "generate" || stage === "generating") && !done && !L.finished) {
        L.phase = "thinking";
      }
      project(run);
    },
    onSources: (sources) => {
      if (!current()) return;
      run.acked = true;
      const typed = sources as SourceItem[];
      L.sources = typed;
      persist(run, { sources: typed });
      project(run);
    },
    onTrace: (trace) => {
      if (!current()) return;
      run.acked = true;
      L.trace = trace as RetrievalTrace;
      persist(run, { trace: trace as RetrievalTrace });
      project(run);
    },
    onReasoning: (text) => {
      if (!current()) return;
      run.acked = true;
      if (L.phase !== "thinking") L.phase = "thinking";
      L.reasoning += text;
      project(run);
    },
    onPhase: (phase, model) => {
      if (!current()) return;
      run.acked = true;
      L.phase = phase as ActivityPhase;
      if (model) L.model = model;
      project(run);
    },
    onTokens: (text) => {
      if (!current()) return;
      run.acked = true;
      L.text += text;
      if (L.committed) L.content = L.text;
      if (L.phase === "thinking") L.phase = "generating";
      L.answerChars += text.length;
      project(run);
      schedulePersist(run);
    },
    onMeta: (meta) => {
      if (!current()) return;
      run.acked = true;
      const m = meta as { model?: string };
      if (m.model) L.model = m.model;
      L.meta = meta as GenerationMeta;
      if (!L.committed) {
        L.committed = true;
        L.content = L.text;
      }
      persist(run, { content: L.content, meta: L.meta });
      project(run);
    },
    onGrounding: (grounding) => {
      if (!current()) return;
      run.acked = true;
      // Server-verified grounding overrides the client-side heuristic.
      L.grounding = toGroundingClaims(
        grounding as Array<{ text: string; found: boolean; source?: string; note?: string; kind?: string }>
      );
      L.groundingFor = L.content;
      project(run);
    },
    onKnowledge: ({ matches, tier }) => {
      if (!current()) return;
      run.acked = true;
      const cards = matches as KnowledgeMatchCard[];
      if (!cards.length) return; // weak/ambiguous → nothing to compare
      L.knowledge = { matches: cards, tier, fresh: null };
      project(run);
    },
    onFinal: (text, droppedCount, _dropped, judgeRewritten) => {
      if (!current()) return;
      run.acked = true;
      if (!L.committed) {
        L.committed = true;
        L.content = L.text;
      }
      if (judgeRewritten && text && text !== L.content) {
        // The judge rewrote the answer to remove unsupported claims.
        L.content = text;
      }
      if (!L.content && text) L.content = text;
      if (droppedCount > 0 || judgeRewritten) {
        console.info(`[grounding] ${droppedCount} flagged, judge-rewrite=${judgeRewritten}`);
      }
      persist(run, { content: L.content });
      project(run);
    },
    onError: (message) => {
      if (!current()) return;
      // The page is leaving: the browser dropped the stream. onDone(false)
      // then closes the run as interrupted, not as a generation error.
      if (pageLeaving) return;
      // The stream's `done` (or the connection close) finalises the run.
      L.error = message;
      L.phase = "error";
      project(run);
    },
    onDone: (complete) => {
      if (!current()) return;
      if (complete) {
        if (L.error) {
          finalize(run, "error");
          return;
        }
        finalize(run, "done");
        // A replay of an answer that finished elsewhere is not re-verified: the
        // verification result (if any) was stored by the original tab.
        if (!run.replay && L.content && L.sources.length > 0) {
          startVerification(run, L.content, L.sources);
        }
        return;
      }
      // The stream ended without the server's `done`. The page is leaving, so
      // the server run simply continues; the next page load reconnects to it.
      if (pageLeaving) return;
      // Otherwise ask the server what happened. A dropped connection is not an
      // interruption while the run is still generating.
      scheduleSettle(run, run.retries === 0 ? 0 : backoffMs(run.retries));
    },
  };
}

/** Start one stream attempt that sends the question (the original generation). */
function openPost(run: ChatRun, question: string, app: ReturnType<typeof useAppStore.getState>): void {
  run.attempt += 1;
  const attempt = run.attempt;
  streamChat({
    message: question,
    mode: app.mode,
    // Model-aware clamp at the send site: only send an effort the SELECTED
    // model declares. `null` means the catalog hasn't resolved yet, so the
    // stored pick is sent as-is (the backend enforces the ladder anyway);
    // `[]` means the model has no ladder, so nothing is sent.
    thinkingEffort:
      app.modelEfforts === null
        ? app.thinkingEffort
        : app.modelEfforts.includes(app.thinkingEffort)
          ? app.thinkingEffort
          : undefined,
    retrievalMode: app.retrievalMode,
    draftStyle: app.draftStyle,
    // Tree-rule expansion: ministry -> flat org list (see lib/sourceFilter.ts)
    orgs: expandOrgFilter(app.sourceFilter.ministry, app.sourceFilter.orgs),
    docCategories: app.sourceFilter.docCategories,
    runId: run.runId,
    signal: run.abort.signal,
    handlers: buildHandlers(run, attempt),
  });
}

/** Start one stream attempt that replays the run from frame 0, then follows it live. */
function openReplay(run: ChatRun): void {
  run.abort.abort();
  run.abort = new AbortController();
  run.attempt += 1;
  const attempt = run.attempt;
  // Rebuild the projection from the server's ordered frames, never on top of
  // what this tab already had: that keeps the text exactly once and in order.
  Object.assign(run.live, cloneLive(run.initial));
  attachRunStream(run.runId, buildHandlers(run, attempt), run.abort.signal);
}

function scheduleSettle(run: ChatRun, delayMs: number): void {
  if (run.closed) return;
  if (run.resumeTimer) clearTimeout(run.resumeTimer);
  run.resumeTimer = setTimeout(() => {
    run.resumeTimer = null;
    void settleNow(run);
  }, delayMs);
}

/**
 * Decide what a run that has stopped streaming should do, from the server's
 * status (the only authority on how a run ends).
 */
async function settleNow(run: ChatRun): Promise<void> {
  if (run.closed || pageLeaving) return;
  let st: RunStatus | null | undefined;
  try {
    st = await fetchRunStatus(run.runId);
  } catch {
    st = undefined; // network or server error: unknown for now
  }
  if (run.closed) return;
  if (st === undefined) {
    // Unknown for now. Keep the answer streaming and try again later.
    run.retries += 1;
    scheduleSettle(run, backoffMs(run.retries));
    return;
  }
  run.retries = 0;
  if (st === null) {
    // The server does not know this run.
    if (!run.acked) {
      // It never started: a refused request, not a lost answer.
      if (!run.live.error) run.live.error = "The answer could not be started.";
      finalize(run, "error");
    } else {
      finalize(run, "lost");
    }
    return;
  }
  if (st.status === "running") {
    // Still generating on the server. A background run waits for its chat to be
    // opened; a foreground run reconnects and follows it live.
    if (run.foreground) openReplay(run);
    return;
  }
  if (!run.replayedTerminal) {
    // The run has ended. Replay its frames once to get the full text and the
    // real end (done, or the stream closes without done).
    run.replayedTerminal = true;
    openReplay(run);
    return;
  }
  switch (st.status) {
    case "cancelled":
      finalize(run, "stopped");
      break;
    case "failed":
      run.live.error = NOTICE.serverFailed.replace(/^> ⚠️ /, "");
      finalize(run, "error");
      break;
    default:
      // "done" without its done frame, or "incomplete": keep the text, labelled.
      finalize(run, "interrupted");
  }
}

/**
 * Start a run for `question` in the active session (created if needed). Returns
 * null when refused: another run is active, or the Deep "Wait" gate is set.
 */
export function startChatRun(question: string): ChatRun | null {
  if (getActiveRun()) {
    useToastStore.getState().pushSticky(
      "info",
      "An answer is still generating. Wait for it to finish, or press Stop."
    );
    return null;
  }
  if (deepVerifyBlocksNewQuery(useDraftStore.getState().deepVerify)) {
    useToastStore.getState().pushSticky(
      "info",
      "Deep verification is still running. Wait for it to finish, or choose “Start anyway” to continue now."
    );
    return null;
  }

  const app = useAppStore.getState();
  const sessions = useSessionStore.getState();
  let sessionId = sessions.activeSessionId;
  if (!sessionId) sessionId = sessions.createSession();

  const now = Date.now();
  sessions.addMessage(sessionId, {
    id: newId(),
    role: "user",
    content: question,
    createdAt: now,
  });
  const assistantId = newId();
  const runId = newRunId();
  // The run id is stored with the message before the first byte is sent, so a
  // reload at any point can find and follow the run.
  sessions.addMessage(sessionId, {
    id: assistantId,
    role: "assistant",
    content: "",
    status: "streaming",
    createdAt: now,
    runId,
  });

  const run = makeRun({
    runId,
    sessionId,
    messageId: assistantId,
    question,
    live: newLive(app.retrievalMode, now),
    replay: false,
    foreground: true,
  });

  // Register first: from here on, a second send is refused synchronously.
  useChatRunStore.setState({ active: run });

  // A genuinely new query: drop the previous answer's transient panels, then
  // bind the canvas to this answer and project the (empty) run onto it.
  clearAnswerScopedPanels();
  useDraftStore.setState({ selectedEvidence: null });
  useDraftStore.getState().bindMessage(sessionId, assistantId);
  writeStores(run);

  openPost(run, question, app);
  return run;
}

/**
 * Stop the active run (from any chat). This is the ONLY way a generation is
 * cancelled: the server is asked to cancel it, and the partial text is kept,
 * labelled "stopped". The run is closed before the abort, so no late event can
 * apply.
 */
export function stopActiveRun(): boolean {
  const run = getActiveRun();
  if (!run || run.closed) return false;
  finalize(run, "stopped");
  run.abort.abort();
  void cancelRun(run.runId);
  return true;
}

/**
 * Bind the canvas to `sessionId`'s active run, if there is one. Used when a
 * chat is opened, so the live state of a running answer is shown again.
 */
export function attachActiveRunIfAny(sessionId: string): boolean {
  const run = getActiveRun();
  if (!run || run.closed || run.sessionId !== sessionId) return false;
  useDraftStore.getState().bindMessage(run.sessionId, run.messageId);
  writeStores(run);
  return true;
}

/**
 * Open a chat whose answer was started earlier (in this tab before a reload, in
 * another tab, or a closed tab). If the server still has the run, this tab
 * follows it: the replay shows every token produced so far, then the live tail.
 * Returns false when the message is not an answer that can be followed, or when
 * this tab is already generating another answer.
 */
export function attachStoredRun(sessionId: string, messageId: string): boolean {
  const current = getActiveRun();
  if (current && !current.closed) {
    return current.sessionId === sessionId && current.messageId === messageId;
  }
  const session = useSessionStore.getState().sessions.find((x) => x.id === sessionId);
  const msg = session?.messages.find((m) => m.id === messageId);
  if (!session || !msg || msg.status !== "streaming" || !msg.runId) return false;
  const question =
    [...session.messages]
      .slice(0, session.messages.indexOf(msg))
      .reverse()
      .find((m) => m.role === "user")?.content ?? "";
  const run = makeRun({
    runId: msg.runId,
    sessionId,
    messageId,
    question,
    live: newLive(useAppStore.getState().retrievalMode, msg.createdAt),
    replay: true,
    foreground: true,
  });
  useChatRunStore.setState({ active: run });
  useDraftStore.getState().bindMessage(sessionId, messageId);
  writeStores(run);
  scheduleSettle(run, 0);
  return true;
}

/**
 * On load: settle every answer that was still streaming when the page was last
 * open, and that no tab is following now. Finished runs are replayed so their
 * text reaches the chat. Running runs are left alone: they are followed when
 * their chat is opened. Nothing is started.
 */
export function reconcileStoredRuns(): void {
  const active = getActiveRun();
  for (const session of useSessionStore.getState().sessions) {
    for (const msg of session.messages) {
      if (msg.status !== "streaming" || !msg.runId) continue;
      if (active && active.messageId === msg.id) continue;
      const question =
        [...session.messages]
          .slice(0, session.messages.indexOf(msg))
          .reverse()
          .find((m) => m.role === "user")?.content ?? "";
      const run = makeRun({
        runId: msg.runId,
        sessionId: session.id,
        messageId: msg.id,
        question,
        live: newLive(useAppStore.getState().retrievalMode, msg.createdAt),
        replay: true,
        foreground: false,
      });
      scheduleSettle(run, 0);
    }
  }
}

/** Page is closing or hidden: keep the partial text so a reload can show it. */
export function flushActiveRunForUnload(): void {
  const run = getActiveRun();
  if (run) flushPartial(run);
}

if (typeof window !== "undefined") {
  window.addEventListener("pagehide", () => {
    pageLeaving = true;
    flushActiveRunForUnload();
  });
  window.addEventListener("pageshow", (e: PageTransitionEvent) => {
    pageLeaving = false;
    // Restored from the back/forward cache: the stream was closed while the page
    // was away. Follow the run again.
    if (e.persisted) {
      const run = getActiveRun();
      if (run && !run.closed) scheduleSettle(run, 0);
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushActiveRunForUnload();
  });
}
