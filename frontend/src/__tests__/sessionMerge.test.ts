/**
 * Cross-tab persistence. Two tabs share localStorage; each keeps an in-memory
 * copy that can be stale. These tests reproduce the production failures that a
 * stale write caused (a finished answer overwritten by an interrupted copy, a
 * new chat erased, a deleted chat coming back) and check they cannot recur.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { mergeMessage, mergeSessionLists } from "@/lib/sessionMerge";
import { TAB_ACTIVE_KEY, useSessionStore } from "@/store/useSessionStore";
import type { ChatMessage, Session } from "@/types";

const STORE_KEY = "incois-sessions";

function msg(over: Partial<ChatMessage> & { id: string; role: "user" | "assistant" }): ChatMessage {
  return { content: "", createdAt: 1, ...over };
}

function session(over: Partial<Session> & { id: string }): Session {
  return { title: "T", pinned: false, createdAt: 1, updatedAt: 1, messages: [], ...over };
}

function storedSessions(): Session[] {
  const raw = localStorage.getItem(STORE_KEY);
  return raw ? (JSON.parse(raw).state.sessions as Session[]) : [];
}

describe("mergeMessage", () => {
  it("a settled answer always beats a streaming copy, so a late partial write cannot erase it", () => {
    const done = msg({ id: "a", role: "assistant", status: "done", content: "x".repeat(4439), rev: 10 });
    const stale = msg({ id: "a", role: "assistant", status: "streaming", content: "x".repeat(4000), rev: 99 });
    expect(mergeMessage(done, stale)).toBe(done);
    expect(mergeMessage(stale, done)).toBe(done);
  });

  it("a stale interrupted copy does not overwrite a finished answer", () => {
    const done = msg({ id: "a", role: "assistant", status: "done", content: "full answer", rev: 10 });
    // Made from an older streaming copy (a reload marks it interrupted without a new write).
    const interrupted = msg({ id: "a", role: "assistant", status: "interrupted", content: "> notice", rev: 5 });
    expect(mergeMessage(done, interrupted).content).toBe("full answer");
  });

  it("two streaming copies of one run keep the longer text (the more complete replay)", () => {
    const short = msg({ id: "a", role: "assistant", status: "streaming", content: "Sentence 01", rev: 5 });
    const long = msg({ id: "a", role: "assistant", status: "streaming", content: "Sentence 01 Sentence 02", rev: 3 });
    expect(mergeMessage(short, long).content).toBe("Sentence 01 Sentence 02");
    expect(mergeMessage(long, short).content).toBe("Sentence 01 Sentence 02");
  });

  it("two settled copies: the newer write wins, so an edit is kept", () => {
    const original = msg({ id: "a", role: "assistant", status: "done", content: "long original text", rev: 10 });
    const edited = msg({ id: "a", role: "assistant", status: "done", content: "short", rev: 20 });
    expect(mergeMessage(original, edited).content).toBe("short");
    expect(mergeMessage(edited, original).content).toBe("short");
  });
});

describe("mergeSessionLists", () => {
  it("keeps messages that exist in only one copy, in chronological order", () => {
    const local = { sessions: [session({ id: "s", messages: [msg({ id: "u1", role: "user", createdAt: 1 })] })] };
    const remote = {
      sessions: [session({ id: "s", messages: [msg({ id: "u2", role: "user", createdAt: 3, content: "other tab" })] })],
    };
    const out = mergeSessionLists(local, remote);
    expect(out.sessions[0].messages.map((m) => m.id)).toEqual(["u1", "u2"]);
  });

  it("orders a user message before the answer when both share a timestamp", () => {
    const local = { sessions: [session({ id: "s", messages: [msg({ id: "a", role: "assistant", createdAt: 5 })] })] };
    const remote = { sessions: [session({ id: "s", messages: [msg({ id: "u", role: "user", createdAt: 5 })] })] };
    const out = mergeSessionLists(local, remote);
    expect(out.sessions[0].messages.map((m) => m.role)).toEqual(["user", "assistant"]);
  });

  it("keeps sessions created in another tab, newest first", () => {
    const local = { sessions: [session({ id: "old", createdAt: 1 })] };
    const remote = { sessions: [session({ id: "FAST two", createdAt: 9 })] };
    const out = mergeSessionLists(local, remote);
    expect(out.sessions.map((s) => s.id)).toEqual(["FAST two", "old"]);
  });

  it("does not revive a deleted chat from a stale copy", () => {
    const stale = { sessions: [session({ id: "deleted" })] };
    const fresh = { sessions: [], deletedSessionIds: ["deleted"] };
    expect(mergeSessionLists(stale, fresh).sessions).toEqual([]);
    expect(mergeSessionLists(fresh, stale).sessions).toEqual([]);
  });

  it("does not bring back a cleared message from a stale copy", () => {
    const stale = { sessions: [session({ id: "s", messages: [msg({ id: "m", role: "user" })] })] };
    const cleared = { sessions: [session({ id: "s", messages: [] })], deletedMessageIds: ["m"] };
    expect(mergeSessionLists(stale, cleared).sessions[0].messages).toEqual([]);
  });
});

describe("useSessionStore across tabs", () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    useSessionStore.setState({ sessions: [], activeSessionId: null, deletedSessionIds: [], deletedMessageIds: [] });
  });

  it("a stale whole-store write cannot overwrite a finished answer another tab stored", () => {
    // Tab A finished the answer and stored it in full.
    const full = "1. Overview and Mandate of INCOIS\n\n" + "Sentence text. ".repeat(300);
    localStorage.setItem(
      STORE_KEY,
      JSON.stringify({
        state: {
          sessions: [
            session({
              id: "chat-a",
              messages: [
                msg({ id: "u", role: "user", content: "Q", createdAt: 1 }),
                msg({ id: "a", role: "assistant", content: full, status: "done", createdAt: 1, rev: 500 }),
              ],
            }),
          ],
        },
        version: 0,
      })
    );
    // Tab B's in-memory copy is stale: it only saw the partial text and was
    // then marked interrupted by a reload. Its write must not win.
    useSessionStore.setState({
      sessions: [
        session({
          id: "chat-a",
          messages: [
            msg({ id: "u", role: "user", content: "Q", createdAt: 1 }),
            msg({ id: "a", role: "assistant", content: "> ⚠️ reloaded", status: "interrupted", createdAt: 1, rev: 40 }),
          ],
        }),
      ],
    });

    const stored = storedSessions().find((s) => s.id === "chat-a")!;
    expect(stored.messages[1].status).toBe("done");
    expect(stored.messages[1].content).toBe(full);
  });

  it("a chat created in another tab is not erased by this tab's next write", () => {
    localStorage.setItem(
      STORE_KEY,
      JSON.stringify({
        state: { sessions: [session({ id: "fast-two", title: "FAST two", createdAt: 9 })] },
        version: 0,
      })
    );
    // This tab has never seen "FAST two" and writes a different chat.
    useSessionStore.setState({ sessions: [session({ id: "fast-three", title: "FAST three", createdAt: 10 })] });
    const ids = storedSessions().map((s) => s.id);
    expect(ids).toContain("fast-two");
    expect(ids).toContain("fast-three");
  });

  it("a chat deleted in this tab stays deleted after a stale copy from another tab is written", () => {
    useSessionStore.setState({ sessions: [session({ id: "gone", createdAt: 1 }), session({ id: "keep", createdAt: 2 })] });
    useSessionStore.getState().deleteSession("gone");
    // The other tab still has "gone" in memory and writes it back.
    useSessionStore.setState({ sessions: [session({ id: "gone", createdAt: 1 }), session({ id: "keep", createdAt: 2 })] });
    expect(storedSessions().map((s) => s.id)).toEqual(["keep"]);
  });

  it("another tab's write reaches this tab's History through the storage event", () => {
    useSessionStore.setState({ sessions: [session({ id: "mine", createdAt: 1 })] });
    const theirs = {
      state: { sessions: [session({ id: "theirs", createdAt: 5 })], deletedSessionIds: [], deletedMessageIds: [] },
      version: 0,
    };
    window.dispatchEvent(new StorageEvent("storage", { key: STORE_KEY, newValue: JSON.stringify(theirs) }));
    const ids = useSessionStore.getState().sessions.map((s) => s.id);
    expect(ids).toEqual(["theirs", "mine"]);
  });

  it("the active chat is per tab: a new tab does not inherit another tab's chat", async () => {
    // Tab A has chat-a open. Only sessionStorage holds that, and it is per tab.
    useSessionStore.setState({ sessions: [session({ id: "chat-a", createdAt: 1 })] });
    useSessionStore.getState().setActive("chat-a");
    expect(sessionStorage.getItem(TAB_ACTIVE_KEY)).toBe("chat-a");
    // The shared store must not carry the active chat.
    expect(JSON.parse(localStorage.getItem(STORE_KEY)!).state).not.toHaveProperty("activeSessionId");

    // A brand-new tab: same localStorage, empty sessionStorage.
    sessionStorage.clear();
    await useSessionStore.persist.rehydrate();
    expect(useSessionStore.getState().activeSessionId).toBeNull();
    expect(useSessionStore.getState().sessions.map((s) => s.id)).toEqual(["chat-a"]);
  });

  it("a reload of the same tab restores that tab's chat", async () => {
    useSessionStore.setState({ sessions: [session({ id: "chat-a", createdAt: 1 })] });
    useSessionStore.getState().setActive("chat-a");
    // Reload: sessionStorage survives it, memory does not.
    useSessionStore.setState({ sessions: [], activeSessionId: null });
    await useSessionStore.persist.rehydrate();
    expect(useSessionStore.getState().activeSessionId).toBe("chat-a");
  });
});
