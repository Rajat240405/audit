/**
 * AI edits are bound to the answer they started from: a switch cancels them,
 * and a staged result is applied only to that same answer.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { streamEdit } from "@/api/chat";
import type { StreamHandlers } from "@/api/client";
import { useEditDraft } from "@/hooks/useEditDraft";
import { restoreSession, startNewSession } from "@/lib/sessionTransitions";
import { useDraftStore } from "@/store/useDraftStore";
import { useEditStore } from "@/store/useEditStore";
import { useSessionStore } from "@/store/useSessionStore";
import { resetAppState } from "@/test/harness";

vi.mock("@/api/chat", () => ({
  streamChat: vi.fn(),
  streamEdit: vi.fn(),
}));

function lastEdit(): { h: Required<StreamHandlers>; signal: AbortSignal } {
  const calls = vi.mocked(streamEdit).mock.calls;
  const opts = calls[calls.length - 1][0];
  return { h: opts.handlers as Required<StreamHandlers>, signal: opts.signal as AbortSignal };
}

/** Two saved answers, A (m1) and B (m2); the canvas is opened on A's answer. */
function setupAnswers() {
  const a = startNewSession();
  useSessionStore.getState().addMessage(a, { id: "m1", role: "user", content: "Q", createdAt: 1 });
  useSessionStore.getState().addMessage(a, { id: "m1a", role: "assistant", content: "Doc A", createdAt: 2 });
  const b = startNewSession();
  useSessionStore.getState().addMessage(b, { id: "m2", role: "user", content: "Q2", createdAt: 3 });
  useSessionStore.getState().addMessage(b, { id: "m2a", role: "assistant", content: "Doc B", createdAt: 4 });
  restoreSession(a);
  return { a, b };
}

function messageContent(sessionId: string, messageId: string): string | undefined {
  return useSessionStore
    .getState()
    .sessions.find((s) => s.id === sessionId)
    ?.messages.find((m) => m.id === messageId)?.content;
}

beforeEach(() => {
  resetAppState();
  vi.clearAllMocks();
});

describe("AI edit binding", () => {
  it("is cancelled when the user switches chats, and its late result is never staged", () => {
    const { a } = setupAnswers();
    const { result } = renderHook(() => useEditDraft());
    act(() => result.current.edit("shorten"));
    const { h, signal } = lastEdit();
    h.onTokens("Short A");

    startNewSession();
    expect(signal.aborted).toBe(true);
    expect(useEditStore.getState().editing).toBe(false);

    h.onDone(true);
    expect(useEditStore.getState().pendingEdit).toBeNull();
    expect(messageContent(a, "m1a")).toBe("Doc A");
  });

  it("a staged edit is discarded, not applied, if the canvas now shows a different answer", () => {
    const { a, b } = setupAnswers();
    const { result } = renderHook(() => useEditDraft());
    act(() => result.current.edit("shorten"));
    const { h } = lastEdit();
    h.onTokens("Short A");
    h.onDone(true);
    expect(useEditStore.getState().pendingEdit).toMatchObject({ sessionId: a, messageId: "m1a" });

    // Bind the canvas to B's answer without going through the transition.
    useDraftStore.getState().bindMessage(b, "m2a");
    act(() => result.current.accept());

    expect(useEditStore.getState().pendingEdit).toBeNull();
    expect(messageContent(a, "m1a")).toBe("Doc A");
    expect(messageContent(b, "m2a")).toBe("Doc B");
  });

  it("accepting on the same answer applies the edit to that answer's message", () => {
    const { a } = setupAnswers();
    const { result } = renderHook(() => useEditDraft());
    act(() => result.current.edit("shorten"));
    const { h } = lastEdit();
    h.onTokens("Short A");
    h.onDone(true);

    act(() => result.current.accept());
    expect(useDraftStore.getState().content).toBe("Short A");
    expect(messageContent(a, "m1a")).toBe("Short A");
    expect(useEditStore.getState().pendingEdit).toBeNull();
  });
});
