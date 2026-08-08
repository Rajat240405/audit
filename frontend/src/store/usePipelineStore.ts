import { create } from "zustand";
import type { PipelineStage } from "@/types";

interface PipelineState {
  stages: PipelineStage[];
  running: boolean;
  message: string;

  setStages: (stages: Array<{ key: string; label: string }>) => void;
  markStage: (key: string, status: PipelineStage["status"], count?: number) => void;
  setMessage: (msg: string) => void;
  start: () => void;
  finish: () => void;
  reset: () => void;
}

export const usePipelineStore = create<PipelineState>((set) => ({
  stages: [],
  running: false,
  message: "",

  setStages: (defs) =>
    set({
      stages: defs.map((d) => ({ key: d.key, label: d.label, status: "pending" })),
    }),

  markStage: (key, status, count) =>
    set((s) => ({
      stages: s.stages.map((st) =>
        st.key === key ? { ...st, status, count: count ?? st.count } : st
      ),
    })),

  setMessage: (msg) => set({ message: msg }),
  start: () => set({ running: true, message: "" }),
  finish: () => set({ running: false }),
  reset: () => set({ stages: [], running: false, message: "" }),
}));
