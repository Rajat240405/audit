/**
 * The SSE client reports whether the backend finished the stream. This is what
 * lets the run registry tell a complete answer from a truncated one.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { consumeSSE, type StreamHandlers } from "@/api/client";

function sseResponse(events: unknown[]): Response {
  const enc = new TextEncoder();
  const body = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(enc.encode(body));
      c.close();
    },
  });
  return new Response(stream, { status: 200 });
}

function run(events: unknown[], extra: StreamHandlers = {}) {
  vi.stubGlobal("fetch", vi.fn(async () => sseResponse(events)));
  const onDone = vi.fn();
  const onError = vi.fn();
  const onTokens = vi.fn();
  consumeSSE("/api/x", {}, { onDone, onError, onTokens, ...extra });
  return { onDone, onError, onTokens };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("consumeSSE completion flag", () => {
  it("reports complete=true when the backend sends its explicit done event", async () => {
    const { onDone, onTokens } = run([{ type: "tokens", text: "hi" }, { type: "done" }]);
    await vi.waitFor(() => expect(onDone).toHaveBeenCalledTimes(1));
    expect(onDone).toHaveBeenCalledWith(true);
    expect(onTokens).toHaveBeenCalledWith("hi");
  });

  it("reports complete=false when the stream ends without done (possible truncation)", async () => {
    const { onDone } = run([{ type: "tokens", text: "partial" }]);
    await vi.waitFor(() => expect(onDone).toHaveBeenCalledTimes(1));
    expect(onDone).toHaveBeenCalledWith(false);
  });

  it("delivers error then the done event, and completes exactly once", async () => {
    const order: string[] = [];
    const { onDone } = run([{ type: "error", message: "boom" }, { type: "done" }], {
      onError: () => order.push("error"),
    });
    await vi.waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(order).toEqual(["error"]);
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(onDone).toHaveBeenCalledWith(true);
  });

  it("reports a network failure as an error and an incomplete stream", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("network down"))));
    const onDone = vi.fn();
    const onError = vi.fn();
    consumeSSE("/api/x", {}, { onDone, onError });
    await vi.waitFor(() => expect(onDone).toHaveBeenCalledTimes(1));
    expect(onError).toHaveBeenCalledWith("network down");
    expect(onDone).toHaveBeenCalledWith(false);
  });

  it("does not call onDone when the caller aborts", async () => {
    const abort = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: RequestInit) => {
        await new Promise((_, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        });
        return sseResponse([]);
      })
    );
    const onDone = vi.fn();
    consumeSSE("/api/x", {}, { onDone }, abort.signal);
    abort.abort();
    await new Promise((r) => setTimeout(r, 0));
    expect(onDone).not.toHaveBeenCalled();
  });
});
