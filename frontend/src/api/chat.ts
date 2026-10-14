import { apiFetch, consumeRunEvents, consumeSSE, type StreamHandlers } from "./client";
import type {
  ChatMessage,
  DraftStyle,
  ExecutionMode,
  RetrievalMode,
  RetrievalTrace,
  SourceItem,
  ThinkingEffort,
  InvestigationResult,
} from "@/types";

export interface ChatStreamOptions {
  message: string;
  mode: ExecutionMode;
  /** Effort to request, or undefined when the selected model declares no
   * ladder — then the field is omitted rather than sending a value the model
   * cannot honour (the backend also enforces its own ladder). */
  thinkingEffort?: ThinkingEffort;
  retrievalMode: RetrievalMode;
  draftStyle?: DraftStyle | string;
  docTypes?: string[];
  orgs?: string[];
  docCategories?: string[];
  topK?: number;
  /** Client-chosen id of this generation (the server run). See api/chat.ts runs. */
  runId?: string;
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
      reasoning_effort: opts.thinkingEffort,
      retrieval_mode: opts.retrievalMode,
      draft_style: opts.draftStyle && opts.draftStyle !== "default" ? opts.draftStyle : undefined,
      doc_types: opts.docTypes && opts.docTypes.length ? opts.docTypes : undefined,
      orgs: opts.orgs && opts.orgs.length ? opts.orgs : undefined,
      doc_categories: opts.docCategories && opts.docCategories.length ? opts.docCategories : undefined,
      top_k: opts.topK ?? 5,
      run_id: opts.runId,
    },
    opts.handlers,
    opts.signal
  );
}

/** Server-side status of a generation run. `null`: the server does not know it. */
export type RunState = "running" | "done" | "cancelled" | "failed" | "incomplete";

export interface RunStatus {
  run_id: string;
  status: RunState;
  frames: number;
  cancel_requested: boolean;
}

/**
 * Ask the server for a run's status. Resolves `null` when the server has no such
 * run (restarted, or expired). Rejects on network or server errors, which the
 * caller treats as "unknown for now", never as "interrupted".
 */
export async function fetchRunStatus(runId: string): Promise<RunStatus | null> {
  const res = await fetch(`/api/chat/runs/${encodeURIComponent(runId)}`, {
    headers: { Accept: "application/json" },
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as RunStatus;
}

/**
 * Follow a run from the start: every frame produced so far, then the live tail.
 * Starts no generation. Used when a chat is opened, after a reload, or from
 * another tab.
 */
export function attachRunStream(runId: string, handlers: StreamHandlers, signal?: AbortSignal): void {
  consumeRunEvents(runId, 0, handlers, signal);
}

/**
 * Explicit Stop: ask the server to cancel the run. Only this (and not a closed or
 * reloaded tab) cancels a generation. Resolves whether or not the run existed.
 */
export async function cancelRun(runId: string): Promise<void> {
  await fetch(`/api/chat/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" }).catch(() => undefined);
}

export interface EditStreamOptions {
  document: string;
  instruction: string;
  draftStyle?: string;
  signal?: AbortSignal;
  handlers: StreamHandlers;
}

/** Post-generation verification (non-blocking, runs after the stream closes).
 *  Returns the grounding report + a judge-rewritten answer with unsupported
 *  claims removed. */
export async function verifyAnswer(
  answer: string,
  sources: SourceItem[],
  depth?: "light" | "full"
): Promise<{
  text: string;
  /** Sentence rows (kind "sentence") and entity/figure rows (kind "claim"). */
  grounding: Array<{ text: string; found: boolean; source?: string; note?: string; kind?: "claim" | "sentence" }>;
  judge_rewritten: boolean;
  judge_removed_count: number;
  error?: string;
}> {
  return apiFetch("/api/verify", {
    method: "POST",
    body: JSON.stringify({ answer, sources, depth }),
  });
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

/** Targeted investigation of ONE suspicious fact. Report-only. */
export async function investigateClaim(opts: {
  claim: string;
  claimId: string;
  messageId: string;
  sources: SourceItem[];
}): Promise<InvestigationResult> {
  return apiFetch("/api/investigate", {
    method: "POST",
    body: JSON.stringify({
      claim: opts.claim,
      claim_id: opts.claimId,
      message_id: opts.messageId,
      sources: opts.sources,
    }),
  });
}
