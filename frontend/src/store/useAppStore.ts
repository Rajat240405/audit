import { create } from "zustand";
import type { DraftStyle, ExecutionMode, RetrievalMode } from "@/types";

interface AppState {
  provider: string;
  modelFamily: string;
  model: string;
  mode: ExecutionMode;
  retrievalMode: RetrievalMode;
  draftStyle: DraftStyle;
  gpu: string;
  backendOnline: boolean | null; // null = unknown
  // header / global flags
  settingsOpen: boolean;
  buildModalOpen: boolean;

  setProvider: (p: string) => void;
  setModelFamily: (f: string) => void;
  setModel: (m: string) => void;
  setMode: (m: ExecutionMode) => void;
  setRetrievalMode: (m: RetrievalMode) => void;
  setDraftStyle: (s: DraftStyle) => void;
  setGpu: (g: string) => void;
  setBackendOnline: (v: boolean) => void;
  setSettingsOpen: (v: boolean) => void;
  setBuildModalOpen: (v: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
  provider: "ollama",
  modelFamily: "qwen2.5",
  model: "qwen2.5:7b",
  mode: "fast",
  retrievalMode: "hybrid",
  draftStyle: "default",
  gpu: "CPU",
  backendOnline: null,
  settingsOpen: false,
  buildModalOpen: false,

  setProvider: (p) => set({ provider: p }),
  setModelFamily: (f) => set({ modelFamily: f }),
  setModel: (m) => set({ model: m }),
  setMode: (m) => set({ mode: m }),
  setRetrievalMode: (m) => set({ retrievalMode: m }),
  setDraftStyle: (s) => set({ draftStyle: s }),
  setGpu: (g) => set({ gpu: g }),
  setBackendOnline: (v) => set({ backendOnline: v }),
  setSettingsOpen: (v) => set({ settingsOpen: v }),
  setBuildModalOpen: (v) => set({ buildModalOpen: v }),
}));
