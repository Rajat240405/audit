import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { GroundingClaim, RetrievalTrace, SourceItem } from "@/types";
import { buildGroundingReport } from "@/services/grounding";

interface DraftState {
  /** The active (latest) version content of the current answer. */
  content: string;
  /** Raw streamed text currently being generated (before commit). */
  streamingText: string;
  isStreaming: boolean;
  /** Sources + trace attached to the current answer. */
  sources: SourceItem[];
  trace: RetrievalTrace | null;
  grounding: GroundingClaim[];
  /** Metadata from the last committed generation (for the metrics panel). */
  lastMeta: Record<string, unknown> | null;
  /** Version history. */
  versions: Array<{ id: string; label: string; description: string; content: string; createdAt: number }>;
  activeVersion: string | null;
  /** Evidence selection (audit transparency: which sentence is highlighted). */
  selectedEvidence: SourceItem | null;

  startStream: () => void;
  appendToken: (text: string) => void;
  commitStream: (meta?: { sources?: SourceItem[]; trace?: RetrievalTrace | null }) => void;
  cancelStream: () => void;
  setDraft: (content: string, sources?: SourceItem[], trace?: RetrievalTrace | null) => void;
  setSources: (sources: SourceItem[]) => void;
  setTrace: (trace: RetrievalTrace | null) => void;
  selectEvidence: (src: SourceItem | null) => void;
  restoreVersion: (id: string) => void;
  applyEdit: (newContent: string, description?: string) => void;
  setLastMeta: (meta: Record<string, unknown>) => void;
  reset: () => void;
}

function makeId(): string {
  return Math.random().toString(36).slice(2, 10);
}

export const useDraftStore = create<DraftState>()(
  persist(
    (set) => ({
  content: "",
  streamingText: "",
  isStreaming: false,
  sources: [],
  trace: null,
  grounding: [],
  lastMeta: null,
  versions: [],
  activeVersion: null,
  selectedEvidence: null,

  startStream: () => set({ isStreaming: true, streamingText: "", content: "" }),

  appendToken: (text) =>
    set((s) => ({ streamingText: s.streamingText + text })),

  commitStream: (meta) =>
    set((s) => {
      const content = s.streamingText;
      const sources = meta?.sources ?? s.sources;
      const trace = meta?.trace ?? s.trace;
      const version = {
        id: makeId(),
        label: `Version ${s.versions.length + 1}`,
        description: "Generated answer",
        content,
        createdAt: Date.now(),
      };
      return {
        content,
        streamingText: "",
        isStreaming: false,
        sources,
        trace,
        grounding: buildGroundingReport(content, sources),
        versions: [...s.versions, version],
        activeVersion: version.id,
        selectedEvidence: null,
      };
    }),

  cancelStream: () =>
    set((s) => {
      const content = s.streamingText || s.content;
      return {
        streamingText: "",
        isStreaming: false,
        content,
        versions:
          content && content !== s.content
            ? [...s.versions, { id: makeId(), label: `Version ${s.versions.length + 1}`, description: "Partial/stopped generation", content, createdAt: Date.now() }]
            : s.versions,
      };
    }),

  setDraft: (content, sources, trace) =>
    set((s) => ({
      content,
      sources: sources ?? s.sources,
      trace: trace ?? s.trace,
      grounding: buildGroundingReport(content, sources ?? s.sources),
    })),

  setSources: (sources) => set({ sources }),

  setTrace: (trace) => set({ trace }),

  selectEvidence: (src) => set({ selectedEvidence: src }),

  restoreVersion: (id) =>
    set((s) => {
      const v = s.versions.find((x) => x.id === id);
      if (!v) return {};
      // restore the full answer onto the canvas and re-verify it against the
      // currently attached sources
      return {
        content: v.content,
        activeVersion: id,
        grounding: buildGroundingReport(v.content, s.sources),
      };
    }),

  applyEdit: (newContent, description) =>
    set((s) => {
      const latest = s.versions[s.versions.length - 1];
      if (latest && latest.content === newContent) {
        // content unchanged — update description only, no new version
        return { content: newContent, activeVersion: latest.id };
      }
      return {
        content: newContent,
        grounding: buildGroundingReport(newContent, s.sources),
        versions: [
          ...s.versions,
          { id: makeId(), label: `Version ${s.versions.length + 1}`, description: description || "AI edit", content: newContent, createdAt: Date.now() },
        ],
        activeVersion: null,
      };
    }),

  setLastMeta: (meta) => set({ lastMeta: meta }),

  reset: () =>
    set({
      content: "",
      streamingText: "",
      isStreaming: false,
      sources: [],
      trace: null,
      grounding: [],
      lastMeta: null,
      versions: [],
      activeVersion: null,
      selectedEvidence: null,
    }),
  }),
  {
      name: "incois-draft",
      // persist versions + the latest draft content + sources so work survives
      // refresh; skip transient streaming state.
      partialize: (state) => ({
        versions: state.versions,
        activeVersion: state.activeVersion,
        content: state.content,
        sources: state.sources,
        trace: state.trace,
        grounding: state.grounding,
        lastMeta: state.lastMeta,
      }),
      // Every reload snapshots the current draft into History automatically
      // (no manual "Save version" button). Dedup: if the draft is unchanged
      // from the latest snapshot, don't create a duplicate entry.
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        const content = state.content ?? "";
        if (!content.trim()) return;
        const versions = state.versions ?? [];
        const latest = versions[versions.length - 1];
        if (!latest || latest.content !== content) {
          useDraftStore.setState({
            versions: [
              ...versions,
              {
                id: makeId(),
                label: `Version ${versions.length + 1}`,
                description: `Session open (${new Date().toLocaleTimeString()})`,
                content,
                createdAt: Date.now(),
              },
            ],
          });
        }
      },
    }
  )
);
