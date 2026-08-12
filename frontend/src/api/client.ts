// Base HTTP client + SSE reader for the FastAPI backend.
// All browser code talks to relative /api URLs (Vite dev server proxies to
// the backend; production builds are served by FastAPI itself).

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(`HTTP ${res.status}: ${detail}`, res.status);
  }
  return (await res.json()) as T;
}

export interface StreamHandlers {
  onStatus?: (stage: string, message: string, done: boolean, count?: number) => void;
  onSources?: (sources: unknown[], isGraph: boolean) => void;
  onTrace?: (trace: unknown) => void;
  onTokens?: (text: string) => void;
  onReasoning?: (text: string) => void;
  onPhase?: (phase: string, model?: string) => void;
  onMeta?: (meta: unknown) => void;
  onGrounding?: (grounding: unknown) => void;
  onFinal?: (text: string, droppedCount: number, dropped: string[], judgeRewritten: boolean) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
}

/**
 * Consume an SSE stream from the backend. Returns an AbortController so the
 * caller can cancel (Stop button).
 */
export function consumeSSE(
  path: string,
  body: unknown,
  handlers: StreamHandlers,
  signal?: AbortSignal
): void {
  fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  })
    .then(async (res) => {
      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => "");
        handlers.onError?.(`HTTP ${res.status}: ${text.slice(0, 200)}`);
        handlers.onDone?.();
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      // SSE event framing (M6): consecutive `data:` lines accumulate until a
      // blank line, then the whole event is parsed ONCE. This handles
      // multi-line JSON payloads instead of silently dropping them.
      let eventData: string[] = [];

      const dispatchEvent = () => {
        if (eventData.length === 0) return;
        const payload = eventData.join("\n").trim();
        eventData = [];
        if (!payload) return;
        let ev: Record<string, unknown>;
        try {
          ev = JSON.parse(payload);
        } catch {
          return;
        }
        switch (ev.type) {
          case "status":
            handlers.onStatus?.(
              String(ev.stage ?? ""),
              String(ev.message ?? ""),
              Boolean(ev.done),
              typeof ev.count === "number" ? ev.count : undefined
            );
            break;
          case "sources":
            handlers.onSources?.(
              Array.isArray(ev.sources) ? ev.sources : [],
              Boolean(ev.is_graph)
            );
            break;
          case "trace":
            handlers.onTrace?.(ev.trace);
            break;
          case "tokens":
            handlers.onTokens?.(String(ev.text ?? ""));
            break;
          case "reasoning":
            handlers.onReasoning?.(String(ev.text ?? ""));
            break;
          case "phase":
            handlers.onPhase?.(String(ev.phase ?? ""), typeof ev.model === "string" ? ev.model : undefined);
            break;
          case "meta":
            handlers.onMeta?.(ev.meta);
            break;
          case "grounding":
            handlers.onGrounding?.(ev.grounding);
            break;
          case "final":
            handlers.onFinal?.(
              String(ev.text ?? ""),
              typeof ev.citation_dropped_count === "number" ? ev.citation_dropped_count : 0,
              Array.isArray(ev.citation_dropped) ? ev.citation_dropped : [],
              Boolean(ev.judge_rewritten)
            );
            break;
          case "error":
            handlers.onError?.(String(ev.message ?? "Unknown error"));
            break;
          case "done":
            handlers.onDone?.();
            break;
        }
      };

      // Feed one line into the SSE framer: blank line ends the event.
      const processLine = (line: string) => {
        const trimmed = line.trim();
        if (!trimmed) {
          dispatchEvent();
          return;
        }
        if (!trimmed.startsWith("data:")) return;
        const payload = trimmed.slice(5).trim();
        if (payload) eventData.push(payload);
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) processLine(line);
      }
      if (buffer.trim()) processLine(buffer);
      dispatchEvent(); // flush any trailing event (no final blank line)
      handlers.onDone?.();
    })
    .catch((err) => {
      if ((err as Error).name === "AbortError") return;
      handlers.onError?.(err instanceof Error ? err.message : String(err));
      handlers.onDone?.();
    });
}
