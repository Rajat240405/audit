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
 *  4. Stop aborts the run and keeps the partial text, labelled "stopped".
 *     A stream that ends without the backend's explicit `done` event is kept
 *     as "interrupted" (truncated), never shown as a finished answer.
 *  5. Partial text is persisted while it streams (throttled). A reload cannot
 *     resume a run, so a message left as "streaming" is recovered as
 *     "interrupted" (lib/answerStatus.ts).
 *  6. Every late callback from a run that is closed is ignored.
 */
import { streamChat, verifyAnswer } from "@/api/chat";
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
import { buildGroundingReport } from "@/services/grounding";
import { graphStages, hybridStages } from "@/utils/formatters";
import type {
  ChatMessage,
  GenerationMeta,
  GroundingClaim,
  KnowledgeMatchCard,
  PipelineStage,
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
  runId: string;
  sessionId: string;
  messageId: string;
  question: string;
  abort: AbortController;
  live: LiveAnswer;
  persistTimer: ReturnType<typeof setTimeout> | null;
  /** Set once the run has closed. Late callbacks are then ignored. */
  closed: boolean;
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
    grounding:
      L.grounding ?? (L.committed ? buildGroundingReport(L.content, L.sources) : []),
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

type Outcome = "done" | "stopped" | "interrupted" | "error";

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
    status: outcome,
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
        if (res.grounding && res.grounding.length) {
          useDraftStore.setState({
            grounding: res.grounding.map((g) => ({ text: g.text, found: g.found, source: g.source })),
          });
        }
        return;
      }
      if (res.text && res.text !== draft.content) {
        draft.applyEdit(res.text);
        useToastStore.getState().pushSticky(
          "success",
          "✓ Answer verified & cleaned (unsupported claims removed)"
        );
      }
      if (res.grounding && res.grounding.length) {
        useDraftStore.setState({
          grounding: res.grounding.map((g) => ({ text: g.text, found: g.found, source: g.source })),
        });
      }
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

function buildHandlers(run: ChatRun): StreamHandlers {
  const L = run.live;
  return {
    onStatus: (stage, message, done, _count, substage) => {
      if (run.closed) return;
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
      if (run.closed) return;
      const typed = sources as SourceItem[];
      L.sources = typed;
      persist(run, { sources: typed });
      project(run);
    },
    onTrace: (trace) => {
      if (run.closed) return;
      L.trace = trace as RetrievalTrace;
      persist(run, { trace: trace as RetrievalTrace });
      project(run);
    },
    onReasoning: (text) => {
      if (run.closed) return;
      if (L.phase !== "thinking") L.phase = "thinking";
      L.reasoning += text;
      project(run);
    },
    onPhase: (phase, model) => {
      if (run.closed) return;
      L.phase = phase as ActivityPhase;
      if (model) L.model = model;
      project(run);
    },
    onTokens: (text) => {
      if (run.closed) return;
      L.text += text;
      if (L.committed) L.content = L.text;
      if (L.phase === "thinking") L.phase = "generating";
      L.answerChars += text.length;
      project(run);
      schedulePersist(run);
    },
    onMeta: (meta) => {
      if (run.closed) return;
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
      if (run.closed) return;
      // Server-verified grounding overrides the client-side heuristic.
      L.grounding = (grounding as Array<{ text: string; found: boolean; source?: string }>).map(
        (g) => ({ text: g.text, found: g.found, source: g.source })
      );
      project(run);
    },
    onKnowledge: ({ matches, tier }) => {
      if (run.closed) return;
      const cards = matches as KnowledgeMatchCard[];
      if (!cards.length) return; // weak/ambiguous → nothing to compare
      L.knowledge = { matches: cards, tier, fresh: null };
      project(run);
    },
    onFinal: (text, droppedCount, _dropped, judgeRewritten) => {
      if (run.closed) return;
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
      if (run.closed) return;
      // The page is leaving: the browser dropped the stream. onDone(false)
      // then closes the run as interrupted, not as a generation error.
      if (pageLeaving) return;
      // The stream's `done` (or the connection close) finalises the run.
      L.error = message;
      L.phase = "error";
      project(run);
    },
    onDone: (complete) => {
      if (run.closed) return;
      if (L.error) {
        finalize(run, "error");
        return;
      }
      if (!complete) {
        // The stream ended without the backend's explicit `done`, so the answer
        // may be truncated. Keep it, labelled. Do not verify it.
        finalize(run, "interrupted");
        return;
      }
      finalize(run, "done");
      if (L.content && L.sources.length > 0) {
        startVerification(run, L.content, L.sources);
      }
    },
  };
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
  sessions.addMessage(sessionId, {
    id: assistantId,
    role: "assistant",
    content: "",
    status: "streaming",
    createdAt: now,
  });

  const defs = showsHybridStages(app.retrievalMode) ? hybridStages() : graphStages();
  const run: ChatRun = {
    runId: newId(),
    sessionId,
    messageId: assistantId,
    question,
    abort: new AbortController(),
    persistTimer: null,
    closed: false,
    live: {
      text: "",
      committed: false,
      content: "",
      finished: false,
      sources: [],
      trace: null,
      grounding: null,
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
    },
  };

  // Register first: from here on, a second send is refused synchronously.
  useChatRunStore.setState({ active: run });

  // A genuinely new query: drop the previous answer's transient panels, then
  // bind the canvas to this answer and project the (empty) run onto it.
  clearAnswerScopedPanels();
  useDraftStore.setState({ selectedEvidence: null });
  useDraftStore.getState().bindMessage(sessionId, assistantId);
  writeStores(run);

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
    signal: run.abort.signal,
    handlers: buildHandlers(run),
  });
  return run;
}

/**
 * Stop the active run (from any chat). Its partial text is kept, labelled
 * "stopped". The run is closed before the abort, so no late event can apply.
 */
export function stopActiveRun(): boolean {
  const run = getActiveRun();
  if (!run || run.closed) return false;
  finalize(run, "stopped");
  run.abort.abort();
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
  window.addEventListener("pageshow", () => {
    pageLeaving = false;
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushActiveRunForUnload();
  });
}
