import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { DraftStyle, ExecutionMode, RetrievalMode, ThinkingEffort } from "@/types";

interface AppState {
  provider: string;
  modelFamily: string;
  model: string;
  mode: ExecutionMode;
  thinkingEffort: ThinkingEffort;
  /** Effort ladder declared by the ACTIVE model family, synced from
   * /api/models. `null` = not resolved yet (models still loading); `[]` = the
   * model documents no effort ladder (selector unavailable). Keeping it here
   * lets the request path clamp the outgoing effort too, so a value left over
   * from a previous model can never leak onto the wire. */
  modelEfforts: ThinkingEffort[] | null;
  retrievalMode: RetrievalMode;
  draftStyle: DraftStyle;
  /** Source filter (ministry-tree + doc categories) for retrieval.
   *  - ministry: "all" | ministry slug | "sansad" (top-level special source)
   *  - orgs:     selected org slugs under the ministry; [] = all orgs of it
   *  - docCategories: selected doc categories; [] = all
   *  Empty orgs+ministry="all" = no filter (retrieve everything). */
  sourceFilter: SourceFilterState;
  gpu: string;
  backendOnline: boolean | null; // null = unknown
  // header / global flags
  settingsOpen: boolean;
  buildModalOpen: boolean;

  setProvider: (p: string) => void;
  setModelFamily: (f: string) => void;
  setModel: (m: string) => void;
  setMode: (m: ExecutionMode) => void;
  setThinkingEffort: (e: ThinkingEffort) => void;
  setModelEfforts: (e: ThinkingEffort[] | null) => void;
  setRetrievalMode: (m: RetrievalMode) => void;
  setDraftStyle: (s: DraftStyle) => void;
  setSourceFilter: (f: SourceFilterState) => void;
  setGpu: (g: string) => void;
  setBackendOnline: (v: boolean) => void;
  setSettingsOpen: (v: boolean) => void;
  setBuildModalOpen: (v: boolean) => void;
}

/** Source filter selection (see AppState.sourceFilter). */
export interface SourceFilterState {
  ministry: string;
  orgs: string[];
  docCategories: string[];
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
  provider: "ollama",
  modelFamily: "qwen2.5",
  model: "qwen2.5:7b",
  mode: "fast",
  thinkingEffort: "medium",
  modelEfforts: null,
  retrievalMode: "auto",
  // Parliamentary is the default register for a fresh install. It is the only
  // field persisted (see persist() below), so a user's later explicit choice is
  // respected across reloads and never force-reset to Parliamentary.
  draftStyle: "parliamentary",
  sourceFilter: { ministry: "all", orgs: [], docCategories: [] },
  gpu: "CPU",
  backendOnline: null,
  settingsOpen: false,
  buildModalOpen: false,

  setProvider: (p) => set({ provider: p }),
  setModelFamily: (f) => set({ modelFamily: f }),
  setModel: (m) => set({ model: m }),
  setMode: (m) => set({ mode: m }),
  setThinkingEffort: (e) => set({ thinkingEffort: e }),
  setModelEfforts: (e) => set({ modelEfforts: e }),
  setRetrievalMode: (m) => set({ retrievalMode: m }),
  setDraftStyle: (s) => set({ draftStyle: s }),
  setSourceFilter: (f) => set({ sourceFilter: f }),
  setGpu: (g) => set({ gpu: g }),
  setBackendOnline: (v) => set({ backendOnline: v }),
  setSettingsOpen: (v) => set({ settingsOpen: v }),
  setBuildModalOpen: (v) => set({ buildModalOpen: v }),
    }),
    {
      name: "incois-app",
      // Persist ONLY the draft style. Everything else (provider, model, mode,
      // retrieval mode, source filter) stays session-scoped as before.
      partialize: (s) => ({ draftStyle: s.draftStyle }),
    }
  )
);
