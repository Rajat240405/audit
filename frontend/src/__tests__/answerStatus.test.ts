import { describe, expect, it } from "vitest";
import { NOTICE, recoverPersistedSessions, withNotice } from "@/lib/answerStatus";

describe("recoverPersistedSessions", () => {
  it("a streaming message WITHOUT a server run cannot resume: it becomes interrupted, keeping its text", () => {
    const out = recoverPersistedSessions({
      sessions: [
        {
          id: "s1",
          messages: [
            { id: "u", role: "user", content: "Q" },
            { id: "a", role: "assistant", content: "half", status: "streaming" },
            { id: "b", role: "assistant", content: "done", status: "done" },
          ],
        },
      ],
    });
    expect(out.sessions[0].messages[1]).toMatchObject({ status: "interrupted" });
    expect(out.sessions[0].messages[1].content).toBe(withNotice("half", NOTICE.reloaded));
    expect(out.sessions[0].messages[2]).toMatchObject({ status: "done", content: "done" });
  });

  it("a streaming message WITH a server run stays streaming: the server decides how it ends", () => {
    const out = recoverPersistedSessions({
      sessions: [
        {
          id: "s1",
          messages: [
            { id: "a", role: "assistant", content: "half", status: "streaming", runId: "run-abcdef12" },
          ],
        },
      ],
    });
    expect(out.sessions[0].messages[0]).toMatchObject({
      status: "streaming",
      content: "half",
      runId: "run-abcdef12",
    });
  });

  it("keeps the tombstones of deleted chats and cleared messages", () => {
    const out = recoverPersistedSessions({
      sessions: [],
      deletedSessionIds: ["gone-1"],
      deletedMessageIds: ["m-1", 7],
    });
    expect(out.deletedSessionIds).toEqual(["gone-1"]);
    expect(out.deletedMessageIds).toEqual(["m-1"]);
  });

  it("handles empty or unknown storage without throwing, and never creates a chat", () => {
    expect(recoverPersistedSessions(undefined)).toEqual({
      sessions: [],
      deletedSessionIds: [],
      deletedMessageIds: [],
    });
    expect(recoverPersistedSessions({ sessions: "bad" })).toEqual({
      sessions: [],
      deletedSessionIds: [],
      deletedMessageIds: [],
    });
  });
});

describe("withNotice", () => {
  it("appends after partial text and stands alone when there is none", () => {
    expect(withNotice("text", "> n")).toBe("text\n\n---\n\n> n");
    expect(withNotice("   ", "> n")).toBe("> n");
  });
});
