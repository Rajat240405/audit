import { consumeSSE, type StreamHandlers } from "./client";
import type {
  ChatMessage,
  DraftStyle,
  ExecutionMode,
  RetrievalMode,
  RetrievalTrace,
  SourceItem,
} from "@/types";

export interface ChatStreamOptions {
  message: string;
  mode: ExecutionMode;
  retrievalMode: RetrievalMode;
  draftStyle?: DraftStyle | string;
  topK?: number;
  signal?: AbortSignal;
  handlers: StreamHandlers;
}

/** Open the SSE stream for a question -> retrieval -> streaming answer. */
export function streamChat(opts: ChatStreamOptions): void {
  consumeSSE(
    "/api/chat/stream",
    {
      message: opts.message,
      mode: opts.mode,
      retrieval_mode: opts.retrievalMode,
      draft_style: opts.draftStyle && opts.draftStyle !== "default" ? opts.draftStyle : undefined,
      top_k: opts.topK ?? 5,
    },
    opts.handlers,
    opts.signal
  );
}

export interface EditStreamOptions {
  document: string;
  instruction: string;
  draftStyle?: string;
  signal?: AbortSignal;
  handlers: StreamHandlers;
}

/** Stream an AI edit of the current draft. */
export function streamEdit(opts: EditStreamOptions): void {
  consumeSSE(
    "/api/edit",
    {
      document: opts.document,
      instruction: opts.instruction,
      draft_style: opts.draftStyle,
    },
    opts.handlers,
    opts.signal
  );
}

export async function exportDocument(format: "md" | "txt" | "docx", title: string, content: string): Promise<Blob> {
  const res = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format, title, content }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `HTTP ${res.status}`);
  }
  return res.blob();
}

/** Convert a ChatMessage into a SourceItem list (for older messages). */
export function sourcesOf(msg: ChatMessage): SourceItem[] {
  return msg.sources ?? [];
}

export function traceOf(msg: ChatMessage): RetrievalTrace | undefined {
  return msg.trace;
}
