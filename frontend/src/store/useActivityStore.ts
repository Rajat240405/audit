import { create } from "zustand";
import type { SourceItem } from "@/types";

/**
 * Model Activity — live view of what the backend is doing while an answer
 * is generated: which documents the model received, its chain-of-thought
 * (qwen3 reasoning tokens), and the moment the visible answer starts.
 */
export type ActivityPhase = "idle" | "retrieving" | "thinking" | "generating" | "done" | "error";

interface ActivityState {
  open: boolean;
  phase: ActivityPhase;
  /** Sources the model received (from the SSE "sources" event). */
  sources: SourceItem[];
  /** Accumulated reasoning text (qwen3's thinking, streamed live). */
  reasoning: string;
  /** Accumulated visible answer characters (live progress). */
  answerChars: number;
  model: string | null;
  question: string | null;
  startedAt: number | null;
  error: string | null;

  openPanel: () => void;
  closePanel: () => void;
  toggle: () => void;
  reset: () => void;
  setPhase: (phase: ActivityPhase, model?: string) => void;
  setSources: (sources: SourceItem[]) => void;
  appendReasoning: (text: string) => void;
  appendAnswer: (chars: number) => void;
  setModel: (model: string) => void;
  setQuestion: (q: string) => void;
  setError: (msg: string) => void;
}

export const useActivityStore = create<ActivityState>((set) => ({
  open: false,
  phase: "idle",
  sources: [],
  reasoning: "",
  answerChars: 0,
  model: null,
  question: null,
  startedAt: null,
  error: null,

  openPanel: () => set({ open: true }),
  closePanel: () => set({ open: false }),
  toggle: () => set((s) => ({ open: !s.open })),

  reset: () =>
    set({
      phase: "retrieving",
      sources: [],
      reasoning: "",
      answerChars: 0,
      model: null,
      error: null,
      startedAt: Date.now(),
    }),

  setPhase: (phase, model) =>
    set((s) => ({
      phase,
      model: model ?? s.model,
    })),

  setSources: (sources) => set({ sources }),
  appendReasoning: (text) => set((s) => ({ reasoning: s.reasoning + text })),
  appendAnswer: (chars) => set((s) => ({ answerChars: s.answerChars + chars })),
  setModel: (model) => set({ model }),
  setQuestion: (q) => set({ question: q }),
  setError: (msg) => set({ phase: "error", error: msg }),
}));
