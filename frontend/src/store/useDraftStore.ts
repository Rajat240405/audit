import { create } from "zustand";
import type { GroundingClaim, RetrievalTrace, SourceItem } from "@/types";
import { buildGroundingReport } from "@/services/grounding";

/**
 * Draft workspace store — NO version history, NO persistence.
 *
 * Behaviour (per product decision):
 *  - App startup: canvas is EMPTY (nothing persisted here).
 *  - Editing an answer (AI edit / verify / direct canvas edit) replaces the
 *    content IN-PLACE — no "new version" is ever created.
 *  - The conversation lives in useSessionStore (persisted). Clicking a
 *    session in History loads its last answer onto the canvas.
 */
interface DraftState {
  /** Current canvas content (the active answer). */
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
  /** Evidence selection (audit transparency: which sentence is highlighted). */
  selectedEvidence: SourceItem | null;
  /** Which chat message the canvas currently mirrors (for in-place sync). */
  activeSessionId: string | null;
  activeMessageId: string | null;

  startStream: () => void;
  appendToken: (text: string) => void;
  commitStream: (meta?: { sources?: SourceItem[]; trace?: RetrievalTrace | null }) => void;
  cancelStream: () => void;
  /** Load a full answer onto the canvas (session open / restore). */
  setDraft: (content: string, sources?: SourceItem[], trace?: RetrievalTrace | null) => void;
  /** Direct user edit of the canvas text (no AI). Replaces in place. */
  setContent: (content: string) => void;
  /** AI edit / verify result — replaces content in place (no version). */
  applyEdit: (newContent: string) => void;
  setSources: (sources: SourceItem[]) => void;
  setTrace: (trace: RetrievalTrace | null) => void;
  selectEvidence: (src: SourceItem | null) => void;
  setLastMeta: (meta: Record<string, unknown>) => void;
  /** Bind the canvas to a chat message (edits then update that message). */
  bindMessage: (sessionId: string | null, messageId: string | null) => void;
  reset: () => void;
}

export const useDraftStore = create<DraftState>((set) => ({
  content: "",
  streamingText: "",
  isStreaming: false,
  sources: [],
  trace: null,
  grounding: [],
  lastMeta: null,
  selectedEvidence: null,
  activeSessionId: null,
  activeMessageId: null,

  startStream: () => set({ isStreaming: true, streamingText: "", content: "" }),

  appendToken: (text) => set((s) => ({ streamingText: s.streamingText + text })),

  commitStream: (meta) =>
    set((s) => {
      const content = s.streamingText;
      const sources = meta?.sources ?? s.sources;
      const trace = meta?.trace ?? s.trace;
      return {
        content,
        streamingText: "",
        isStreaming: false,
        sources,
        trace,
        grounding: buildGroundingReport(content, sources),
        selectedEvidence: null,
      };
    }),

  cancelStream: () =>
    set((s) => ({
      streamingText: "",
      isStreaming: false,
      content: s.streamingText || s.content,
    })),

  setDraft: (content, sources, trace) =>
    set((s) => ({
      content,
      sources: sources ?? s.sources,
      trace: trace ?? s.trace,
      grounding: buildGroundingReport(content, sources ?? s.sources),
    })),

  setContent: (content) =>
    set((s) => ({
      content,
      grounding: buildGroundingReport(content, s.sources),
    })),

  applyEdit: (newContent) =>
    set((s) => ({
      content: newContent,
      grounding: buildGroundingReport(newContent, s.sources),
    })),

  setSources: (sources) => set({ sources }),
  setTrace: (trace) => set({ trace }),
  selectEvidence: (src) => set({ selectedEvidence: src }),
  setLastMeta: (meta) => set({ lastMeta: meta }),
  bindMessage: (sessionId, messageId) =>
    set({ activeSessionId: sessionId, activeMessageId: messageId }),

  reset: () =>
    set({
      content: "",
      streamingText: "",
      isStreaming: false,
      sources: [],
      trace: null,
      grounding: [],
      lastMeta: null,
      selectedEvidence: null,
    }),
}));
