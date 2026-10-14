/**
 * Chat run registry: streaming isolation between chats, Stop / retry, refused
 * overlapping sends, truncation, reload recovery, and New Session. The network
 * layer is mocked; the handlers are driven directly.
 */
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { attachRunStream, fetchRunStatus, streamChat } from "@/api/chat";
import type { StreamHandlers } from "@/api/client";
import {
  attachActiveRunIfAny,
  flushActiveRunForUnload,
  PERSIST_INTERVAL_MS,
  startChatRun,
  stopActiveRun,
  useChatRunStore,
} from "@/lib/chatRun";
import {
  deleteSessionSafely,
  restoreActiveSessionOnLoad,
  restoreSession,
  startNewSession,
} from "@/lib/sessionTransitions";
import { NOTICE } from "@/lib/answerStatus";
import { TAB_ACTIVE_KEY, useSessionStore } from "@/store/useSessionStore";
import { useDraftStore } from "@/store/useDraftStore";
import { useActivityStore } from "@/store/useActivityStore";
import { usePipelineStore } from "@/store/usePipelineStore";
import type { ChatMessage, Session } from "@/types";
import { META, resetAppState, src } from "@/test/harness";

vi.mock("@/api/chat", () => ({
  streamChat: vi.fn(),
  streamEdit: vi.fn(),
  attachRunStream: vi.fn(),
  fetchRunStatus: vi.fn(),
  cancelRun: vi.fn(async () => undefined),
}));

type Handlers = Required<StreamHandlers>;

/** Handlers and abort signal of the most recent streamChat call. */
function lastStream(): { h: Handlers; signal: AbortSignal } {
  const calls = vi.mocked(streamChat).mock.calls;
  const opts = calls[calls.length - 1][0];
  return { h: opts.handlers as Handlers, signal: opts.signal as AbortSignal };
}

/** Start a run and return its handlers. Fails loudly if the run is refused. */
function start(question: string) {
  const run = startChatRun(question);
  if (!run) throw new Error(`run for "${question}" was refused`);
  return { run, ...lastStream() };
}

function sessionOf(id: string): Session {
  const s = useSessionStore.getState().sessions.find((x) => x.id === id);
  if (!s) throw new Error(`no session ${id}`);
  return s;
}

function assistantOf(sessionId: string): ChatMessage {
  const m = [...sessionOf(sessionId).messages].reverse().find((x) => x.role === "assistant");
  if (!m) throw new Error("no assistant message");
  return m;
}

/** Every foreground store is empty: nothing from another chat is on screen. */
function expectBlankCanvas(): void {
  const d = useDraftStore.getState();
  expect(d.content).toBe("");
  expect(d.streamingText).toBe("");
  expect(d.isStreaming).toBe(false);
  expect(d.sources).toEqual([]);
  expect(d.trace).toBeNull();
  expect(d.knowledge).toBeNull();
  const a = useActivityStore.getState();
  expect(a.phase).toBe("idle");
  expect(a.reasoning).toBe("");
  expect(a.sources).toEqual([]);
  expect(a.answerChars).toBe(0);
  expect(a.error).toBeNull();
  const p = usePipelineStore.getState();
  expect(p.stages).toEqual([]);
  expect(p.message).toBe("");
  expect(p.running).toBe(false);
}

const TRACE = {
  embed_query_ms: 1,
  dense_search_ms: 1,
  bm25_search_ms: 1,
  rrf_fusion_ms: 1,
  rerank_ms: 1,
  retrieval_total_ms: 5,
};

beforeEach(() => {
  resetAppState();
  vi.clearAllMocks();
  // Server defaults: the run has ended as "incomplete". A replay of it ends at once.
  vi.mocked(fetchRunStatus).mockResolvedValue({
    run_id: "run",
    status: "incomplete",
    frames: 0,
    cancel_requested: false,
  });
  vi.mocked(attachRunStream).mockImplementation((_id, h) => h.onDone?.(false));
});

afterEach(() => {
  vi.useRealTimers();
});

describe("switching chats while an answer streams", () => {
  it("never shows the old chat's tokens, status, sources, errors or completion in the new chat", () => {
    const a = startNewSession();
    const { h } = start("Question A");
    h.onStatus("retrieve", "Searching A corpus", false);
    h.onSources([src("A-1")], false);
    h.onPhase("thinking", "model-A");
    h.onReasoning("A private reasoning");
    h.onTokens("A secret answer");

    const b = startNewSession();
    expect(b).not.toBe(a);
    expectBlankCanvas();

    // The old run keeps streaming while chat B is on screen.
    h.onStatus("generate", "A status", false);
    h.onSources([src("A-2")], false);
    h.onTrace(TRACE);
    h.onTokens(" more");
    h.onReasoning(" more thoughts");
    h.onMeta(META);
    h.onFinal("A secret answer more");
    h.onDone(true);

    expectBlankCanvas();
    expect(sessionOf(b).messages).toHaveLength(0);

    // ...and the completed answer is stored on A's own message.
    const stored = assistantOf(a);
    expect(stored.content).toBe("A secret answer more");
    expect(stored.status).toBe("done");
    expect(stored.sources?.map((s) => s.doc_id)).toEqual(["A-2"]);
    expect(stored.trace).toEqual(TRACE);
  });

  it("does not show a late error from the old chat in the new chat", () => {
    startNewSession();
    const { h } = start("Question A");
    h.onTokens("partial");
    startNewSession();

    h.onError("A backend failed");
    h.onDone(true);

    expectBlankCanvas();
    expect(useActivityStore.getState().error).toBeNull();
  });

  it("re-attaches the live text when the running chat is opened again, and continues it", () => {
    const a = startNewSession();
    const { h } = start("Question A");
    h.onTokens("Hel");
    startNewSession();
    h.onTokens("lo");

    restoreSession(a);
    const d = useDraftStore.getState();
    expect(d.activeSessionId).toBe(a);
    expect(d.isStreaming).toBe(true);
    expect(d.streamingText).toBe("Hello");

    h.onTokens(" world");
    expect(useDraftStore.getState().streamingText).toBe("Hello world");

    h.onMeta(META);
    h.onFinal("Hello world");
    h.onDone(true);
    expect(useDraftStore.getState().content).toBe("Hello world");
    expect(useDraftStore.getState().isStreaming).toBe(false);
    expect(assistantOf(a).content).toBe("Hello world");
  });

  it("Stop from another chat stops the run and labels its partial answer", () => {
    const a = startNewSession();
    const { h, signal } = start("Question A");
    h.onTokens("Half");
    startNewSession();

    expect(stopActiveRun()).toBe(true);
    expect(signal.aborted).toBe(true);
    expect(assistantOf(a).status).toBe("stopped");
    expect(assistantOf(a).content).toBe(`Half\n\n---\n\n${NOTICE.stopped}`);
    expectBlankCanvas();
  });
});

describe("stop, retry and overlapping requests", () => {
  it("keeps the partial answer labelled, ignores late events, and lets a new send run", () => {
    const a = startNewSession();
    const { run, h, signal } = start("Question");
    h.onTokens("Partial");

    expect(stopActiveRun()).toBe(true);
    expect(signal.aborted).toBe(true);
    expect(useChatRunStore.getState().active).toBeNull();
    const stopped = `Partial\n\n---\n\n${NOTICE.stopped}`;
    expect(assistantOf(a).content).toBe(stopped);
    expect(assistantOf(a).status).toBe("stopped");

    // Late callbacks from the aborted stream change nothing, not even the run.
    h.onTokens(" late");
    h.onMeta(META);
    h.onFinal("Partial late");
    h.onDone(true);
    expect(run.live.text).toBe("Partial");
    expect(run.live.content).toBe(stopped);
    expect(assistantOf(a).content).toBe(stopped);

    // Retry: a new run gets its own handlers and is not affected by the old one.
    const retry = start("Question again");
    h.onTokens("OLD");
    expect(useDraftStore.getState().streamingText).toBe("");
    retry.h.onTokens("New");
    expect(useDraftStore.getState().streamingText).toBe("New");
    expect(retry.run.live.text).toBe("New");
  });

  it("refuses a second send while an answer is generating, in the same chat or another", () => {
    const a = startNewSession();
    start("Question 1");
    expect(startChatRun("Question 2")).toBeNull();

    const b = startNewSession();
    expect(startChatRun("Question 3")).toBeNull();

    expect(vi.mocked(streamChat)).toHaveBeenCalledTimes(1);
    expect(sessionOf(a).messages).toHaveLength(2);
    expect(sessionOf(b).messages).toHaveLength(0);
  });

  it("deleting the chat that is generating stops its run and clears the canvas", () => {
    const a = startNewSession();
    const { signal } = start("Question");
    deleteSessionSafely(a);
    expect(signal.aborted).toBe(true);
    expect(useChatRunStore.getState().active).toBeNull();
    expect(useSessionStore.getState().sessions.some((s) => s.id === a)).toBe(false);
    expectBlankCanvas();
  });
});

describe("unfinished and failed answers", () => {
  it("keeps a stream that ended without `done` as interrupted, not finished, and the run is not finished", async () => {
    const a = startNewSession();
    const { h } = start("Question");
    h.onSources([src("A-1")], false);
    h.onTokens("Half an answer");
    h.onDone(false);

    // The client asks the server what happened. The run ended as incomplete, and
    // the replay carries the text the server holds.
    vi.mocked(attachRunStream).mockImplementation((_id, rh) => {
      rh.onTokens?.("Half an answer");
      rh.onDone?.(false);
    });
    await vi.waitFor(() => expect(assistantOf(a).status).toBe("interrupted"));
    expect(assistantOf(a).content).toBe(`Half an answer\n\n---\n\n${NOTICE.connectionClosed}`);
    expect(useChatRunStore.getState().active).toBeNull();
  });

  it("a connection dropped while the page is leaving keeps the answer streaming on the server", async () => {
    // The server keeps generating. The replay stays open in this test.
    vi.mocked(attachRunStream).mockImplementation(() => undefined);
    vi.mocked(fetchRunStatus).mockResolvedValue({ run_id: "run", status: "running", frames: 3, cancel_requested: false });
    const a = startNewSession();
    const { h } = start("Question");
    h.onTokens("Half an answer");
    // The browser tears the page down: pagehide fires, then the stream drops.
    window.dispatchEvent(new Event("pagehide"));
    h.onError("Failed to fetch");
    h.onDone(false);
    await new Promise((r) => setTimeout(r, 20));

    // Not interrupted, not finished, and no status check while the page is leaving.
    expect(assistantOf(a).status).toBe("streaming");
    expect(assistantOf(a).content).toBe("Half an answer");
    expect(fetchRunStatus).not.toHaveBeenCalled();
    expect(useChatRunStore.getState().active).not.toBeNull();

    // The page is restored from the back/forward cache: follow the run again.
    const shown = new Event("pageshow") as PageTransitionEvent;
    Object.defineProperty(shown, "persisted", { value: true });
    window.dispatchEvent(shown);
    await vi.waitFor(() => expect(fetchRunStatus).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(attachRunStream).toHaveBeenCalledTimes(1));
    expect(assistantOf(a).status).toBe("streaming");
  });

  it("keeps partial text on error, labelled as an error", () => {
    const a = startNewSession();
    const { h } = start("Question");
    h.onSources([src("A-1")], false);
    h.onTokens("Half");
    h.onError("Backend failed");
    h.onDone(true);

    const m = assistantOf(a);
    expect(m.status).toBe("error");
    expect(m.content).toBe("Half\n\n---\n\n> ⚠️ Backend failed");
    expect(useActivityStore.getState().phase).toBe("error");
  });

  it("a stream with no answer text shows only the error, not a blank finished answer", () => {
    const a = startNewSession();
    const { h } = start("Question");
    h.onError("Nothing came back");
    h.onDone(true);
    expect(assistantOf(a).content).toBe("> ⚠️ Nothing came back");
    expect(assistantOf(a).status).toBe("error");
  });
});

describe("persistence while streaming", () => {
  it("writes the partial text to the chat while it streams, labelled streaming", () => {
    vi.useFakeTimers();
    const a = startNewSession();
    const { h } = start("Question");
    h.onTokens("abc");
    expect(assistantOf(a).content).toBe("");

    vi.advanceTimersByTime(PERSIST_INTERVAL_MS);
    expect(assistantOf(a).content).toBe("abc");
    expect(assistantOf(a).status).toBe("streaming");

    h.onTokens("def");
    flushActiveRunForUnload();
    expect(assistantOf(a).content).toBe("abcdef");
  });

  it("events from a running chat never write into the chat that is on screen", () => {
    const a = startNewSession();
    const { h } = start("Question A");
    h.onTokens("A-live");
    // Chat B has a finished answer that is on screen.
    const b = startNewSession();
    useSessionStore.getState().addMessage(b, { id: "bq", role: "user", content: "QB", createdAt: 5 });
    useSessionStore.getState().addMessage(b, { id: "ba", role: "assistant", content: "B answer", createdAt: 6, status: "done" });
    restoreSession(b);

    h.onSources([src("A-1")], false);
    h.onTokens(" more");
    h.onMeta(META);
    h.onFinal("A-live more");
    h.onDone(true);

    expect(assistantOf(b).content).toBe("B answer");
    expect(assistantOf(b).sources).toBeUndefined();
    expect(useDraftStore.getState().content).toBe("B answer");
    expect(assistantOf(a).content).toBe("A-live more");
  });

  it("re-attaches a run only to its own chat", () => {
    const a = startNewSession();
    start("Question");
    expect(attachActiveRunIfAny(a)).toBe(true);
    expect(attachActiveRunIfAny("some-other-chat")).toBe(false);
  });
});

describe("reload and startup", () => {
  it("recovers a message left streaming as interrupted, keeps the active chat, creates nothing", async () => {
    const persisted = {
      state: {
        sessions: [
          {
            id: "s1",
            title: "Question",
            pinned: false,
            createdAt: 1,
            updatedAt: 2,
            messages: [
              { id: "u1", role: "user", content: "Question", createdAt: 1 },
              { id: "a1", role: "assistant", content: "Partial text", status: "streaming", createdAt: 2 },
            ],
          },
        ],
        activeSessionId: "s1",
      },
      version: 0,
    };
    localStorage.setItem("incois-sessions", JSON.stringify(persisted));
    // This tab had s1 open before the reload (per-tab, in sessionStorage).
    sessionStorage.setItem(TAB_ACTIVE_KEY, "s1");
    await useSessionStore.persist.rehydrate();

    const st = useSessionStore.getState();
    expect(st.sessions.map((s) => s.id)).toEqual(["s1"]);
    expect(st.activeSessionId).toBe("s1");
    const m = st.sessions[0].messages[1];
    expect(m.status).toBe("interrupted");
    expect(m.content).toBe(`Partial text\n\n---\n\n${NOTICE.reloaded}`);

    restoreActiveSessionOnLoad();
    expect(useSessionStore.getState().sessions).toHaveLength(1);
    expect(useDraftStore.getState().activeMessageId).toBe("a1");
    expect(useDraftStore.getState().content).toBe(`Partial text\n\n---\n\n${NOTICE.reloaded}`);
    expect(useDraftStore.getState().isStreaming).toBe(false);
  });

  it("after a reload the chat re-attaches to the server run instead of marking it interrupted", async () => {
    // The server still reports the run as running, and the replay stays open.
    vi.mocked(attachRunStream).mockImplementation(() => undefined);
    vi.mocked(fetchRunStatus).mockResolvedValue({ run_id: "run", status: "running", frames: 2, cancel_requested: false });
    const a = startNewSession();
    start("Question");
    expect(useChatRunStore.getState().active).not.toBeNull();
    // Reload is simulated by dropping the in-memory run. The server keeps it.
    useChatRunStore.setState({ active: null });
    restoreSession(a);
    await vi.waitFor(() => expect(attachRunStream).toHaveBeenCalledTimes(1));
    expect(useChatRunStore.getState().active).not.toBeNull();
    expect(useDraftStore.getState().isStreaming).toBe(true);
    expect(assistantOf(a).status).toBe("streaming");
    expect(assistantOf(a).runId).toBeTruthy();
  });
});

describe("New Session", () => {
  it("creates an empty chat with a clean canvas and keeps the old chat in History", () => {
    const a = startNewSession();
    const { h } = start("Question");
    h.onSources([src("A-1")], false);
    h.onTokens("Answer");
    h.onMeta(META);
    h.onDone(true);

    const b = startNewSession();
    expect(b).not.toBe(a);
    expect(useSessionStore.getState().activeSessionId).toBe(b);
    expect(sessionOf(b).messages).toEqual([]);
    expectBlankCanvas();
    expect(sessionOf(a).messages).toHaveLength(2);
  });

  it("does not stop a run that is still generating in the previous chat", () => {
    startNewSession();
    const { signal } = start("Question");
    startNewSession();
    expect(signal.aborted).toBe(false);
    expect(useChatRunStore.getState().active).not.toBeNull();
  });
});

describe("replay of a server run", () => {
  it("a replay after a dropped stream rebuilds the text exactly once, and late events from the old attempt are ignored", async () => {
    const a = startNewSession();
    const first = start("Question");
    first.h.onTokens("Hel");
    // The connection drops. The server still reports the run as running.
    vi.mocked(fetchRunStatus).mockResolvedValue({ run_id: "run", status: "running", frames: 1, cancel_requested: false });
    vi.mocked(attachRunStream).mockImplementation((_id, rh) => {
      // The server replays the frames it holds, in order, then follows live.
      rh.onTokens?.("Hel");
      rh.onTokens?.("lo");
      rh.onDone?.(true);
    });
    first.h.onDone(false);
    await vi.waitFor(() => expect(assistantOf(a).status).toBe("done"));
    expect(assistantOf(a).content).toBe("Hello");

    // A late token from the first connection must not apply to the finished answer.
    first.h.onTokens(" late");
    expect(assistantOf(a).content).toBe("Hello");
    expect(useDraftStore.getState().content).toBe("Hello");
  });
});
