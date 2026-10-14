/**
 * Component behaviour for the two places a stream can leak into the UI: the
 * message card (live text under the wrong message) and the input box (a send
 * refused while another answer generates must not lose the typed text).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ChatInput } from "@/components/chat/ChatInput";
import { Message } from "@/components/chat/Message";
import type { ChatRun } from "@/lib/chatRun";
import type { ChatMessage } from "@/types";
import { useChatRunStore } from "@/lib/chatRun";
import { useDraftStore } from "@/store/useDraftStore";
import { useSessionStore } from "@/store/useSessionStore";
import { resetAppState } from "@/test/harness";

vi.mock("@/api/chat", () => ({
  streamChat: vi.fn(),
  streamEdit: vi.fn(),
}));

/** Minimal run stand-in: ChatInput only reads sessionId. */
function fakeRun(sessionId: string): ChatRun {
  return { sessionId, messageId: "x" } as unknown as ChatRun;
}

beforeEach(() => {
  resetAppState();
  useSessionStore.setState({
    sessions: [
      { id: "s1", title: "A", pinned: false, createdAt: 1, updatedAt: 1, messages: [] },
      { id: "s2", title: "B", pinned: false, createdAt: 1, updatedAt: 1, messages: [] },
    ],
    activeSessionId: "s2",
  });
});

describe("ChatInput while an answer is generating", () => {
  it("keeps the typed text and explains why when another chat is generating", () => {
    useChatRunStore.setState({ active: fakeRun("s1") });
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} onStop={vi.fn()} />);
    const box = screen.getByPlaceholderText(/Enter audit query/) as HTMLTextAreaElement;

    fireEvent.change(box, { target: { value: "my next question" } });
    fireEvent.keyDown(box, { key: "Enter" });

    expect(onSend).not.toHaveBeenCalled();
    expect(box.value).toBe("my next question");
    expect(screen.getByRole("status").textContent).toMatch(/another chat/);
  });

  it("offers Stop for the other chat's answer, and does not allow clearing the chat that is generating", () => {
    useChatRunStore.setState({ active: fakeRun("s1") });
    render(<ChatInput onSend={vi.fn()} onStop={vi.fn()} />);
    expect(screen.getByLabelText("Stop the answer in another chat")).toBeTruthy();
    // The clear control belongs to the chat on screen (s2), which is not generating.
    expect((screen.getByText("clear") as HTMLButtonElement).disabled).toBe(false);
  });

  it("disables clear for the chat that is generating on screen", () => {
    useChatRunStore.setState({ active: fakeRun("s2") });
    render(<ChatInput onSend={vi.fn()} onStop={vi.fn()} />);
    expect((screen.getByText("clear") as HTMLButtonElement).disabled).toBe(true);
  });

  it("sends normally when no answer is generating", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} onStop={vi.fn()} />);
    const box = screen.getByPlaceholderText(/Enter audit query/) as HTMLTextAreaElement;
    fireEvent.change(box, { target: { value: "hello" } });
    fireEvent.keyDown(box, { key: "Enter" });
    expect(onSend).toHaveBeenCalledWith("hello");
    expect(box.value).toBe("");
  });
});

describe("message card", () => {
  it("never renders another chat's live text under a message", () => {
    // The canvas is bound to m2 in s2 and is streaming secret text ...
    useDraftStore.getState().bindMessage("s2", "m2");
    useDraftStore.setState({ isStreaming: true, streamingText: "SECRET-LIVE-TEXT" });
    // ... so the card for m1 (an unrelated message) must not show it.
    const { unmount } = render(
      <Message message={{ id: "m1", role: "assistant", content: "", createdAt: 1 }} />
    );
    expect(screen.queryByText("SECRET-LIVE-TEXT")).toBeNull();
    unmount();

    // The bound message does show the live text.
    render(<Message message={{ id: "m2", role: "assistant", content: "", createdAt: 2, status: "streaming" }} />);
    expect(screen.getByText("SECRET-LIVE-TEXT")).toBeTruthy();
  });

  it("does not offer Save to Knowledge for a stopped or interrupted answer", () => {
    // The question must exist in the store, or Save is disabled for that reason
    // alone and this test would prove nothing.
    const stopped: ChatMessage = {
      id: "a",
      role: "assistant",
      content: "Partial\n\n---\n\n> ⏹ stopped",
      createdAt: 2,
      status: "stopped",
    };
    useSessionStore.getState().addMessage("s2", { id: "q", role: "user", content: "Q", createdAt: 1 });
    useSessionStore.getState().addMessage("s2", stopped);
    const { unmount } = render(<Message message={stopped} />);
    expect((screen.getByTestId("save-knowledge-btn") as HTMLButtonElement).disabled).toBe(true);
    unmount();

    // Control: the same settled answer can be saved, so the disabled state above
    // is caused by the status and not by anything else.
    const done: ChatMessage = { ...stopped, id: "b", status: "done", content: "Full answer" };
    useSessionStore.getState().addMessage("s2", done);
    render(<Message message={done} />);
    expect((screen.getByTestId("save-knowledge-btn") as HTMLButtonElement).disabled).toBe(false);
  });
});
