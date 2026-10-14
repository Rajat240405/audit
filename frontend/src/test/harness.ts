/**
 * Shared test harness: resets every store the chat, session and edit flows
 * touch, and stops any run a previous test left active, so tests are isolated.
 */
import { useSessionStore } from "@/store/useSessionStore";
import { useDraftStore } from "@/store/useDraftStore";
import { useActivityStore } from "@/store/useActivityStore";
import { usePipelineStore } from "@/store/usePipelineStore";
import { cancelInFlightEdit, useEditStore } from "@/store/useEditStore";
import { getActiveRun, stopActiveRun, useChatRunStore } from "@/lib/chatRun";
import type { GenerationMeta, SourceItem } from "@/types";

export function resetAppState(): void {
  // Only a real run (one with live state) is stopped; component tests may
  // install a minimal stand-in in the registry.
  if (getActiveRun()?.live) stopActiveRun();
  cancelInFlightEdit();
  localStorage.clear();
  useSessionStore.setState({ sessions: [], activeSessionId: null, searchQuery: "" });
  useDraftStore.getState().reset();
  useDraftStore.getState().bindMessage(null, null);
  useDraftStore.setState({ knowledge: null });
  useActivityStore.setState({
    phase: "idle",
    sources: [],
    reasoning: "",
    answerChars: 0,
    model: null,
    question: null,
    startedAt: null,
    error: null,
  });
  usePipelineStore.getState().reset();
  useChatRunStore.setState({ active: null });
  useEditStore.setState({ editing: false, editingLabel: null, pendingEdit: null });
}

export const META: GenerationMeta = {
  provider: "test",
  model: "test-model",
  profile: "fast",
  retrieved_documents: 1,
  retrieved_chunks: 1,
  response_time_ms: 1,
  is_fallback: false,
};

export function src(docId: string): SourceItem {
  return {
    doc_id: docId,
    ministry: "Ministry",
    subject: "Subject",
    score: 1,
    question: "q",
    answer: "a",
    dense_score: null,
    bm25_score: null,
    rrf_score: null,
    rerank_score: null,
  };
}
