import { create } from "zustand";
import type {
  KnowledgeMatchCard,
  RetrievalTrace,
  SourceItem,
} from "@/types";

/**
 * Draft workspace store — NO version history, NO persistence.
 *
 * Behaviour (per product decision):
 *  - App startup: canvas is EMPTY (nothing persisted here).
 *  - Editing an answer (AI edit / direct canvas edit) replaces the
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
  /** Metadata from the last committed generation (for the metrics panel). */
  lastMeta: Record<string, unknown> | null;
  /** Evidence selection (audit transparency: which sentence is highlighted). */
  selectedEvidence: SourceItem | null;
  /** Which chat message the canvas currently mirrors (for in-place sync). */
  activeSessionId: string | null;
  activeMessageId: string | null;
  /** Saved-knowledge matches for the CURRENT answer — ANSWER-SCOPED.
   *  `fresh` snapshots the freshly generated answer at stream completion so
   *  the comparison panel can offer "Use This" on both sides even after the
   *  user has already picked the saved one. */
  knowledge: {
    matches: KnowledgeMatchCard[];
    tier: string | null;
    fresh: { content: string; sources: SourceItem[] } | null;
  } | null;

  startStream: () => void;
  appendToken: (text: string) => void;
  commitStream: (meta?: { sources?: SourceItem[]; trace?: RetrievalTrace | null }) => void;
  cancelStream: () => void;
  /** Load a full answer onto the canvas (session open / restore). */
  setDraft: (content: string, sources?: SourceItem[], trace?: RetrievalTrace | null) => void;
  /** Direct user edit of the canvas text (no AI). Replaces in place. */
  setContent: (content: string) => void;
  /** AI edit result — replaces content in place (no version). */
  applyEdit: (newContent: string) => void;
  setSources: (sources: SourceItem[]) => void;
  setTrace: (trace: RetrievalTrace | null) => void;
  selectEvidence: (src: SourceItem | null) => void;
  setLastMeta: (meta: Record<string, unknown>) => void;
  /** Bind the canvas to a chat message (edits then update that message). */
  bindMessage: (sessionId: string | null, messageId: string | null) => void;
  /** SSE `knowledge` event: remember the saved matches for this answer. */
  setKnowledge: (matches: KnowledgeMatchCard[], tier: string | null) => void;
  /** Capture the fresh RAG answer once the stream completes (comparison panel). */
  snapshotKnowledgeFresh: () => void;
  /** "Use This" on the comparison panel: the picked answer becomes the draft.
   *  Deliberately does NOT re-save anything to the knowledge base. */
  useKnowledgeAnswer: (answer: string, sources: SourceItem[]) => void;
  /** Clear every ANSWER-SPECIFIC transient panel (saved-knowledge comparison).
   *  Called when a new query starts so Query-N UI cannot attach to N+1. */
  clearAnswerScopedUi: () => void;
  reset: () => void;
}

export const useDraftStore = create<DraftState>((set) => ({
  content: "",
  streamingText: "",
  isStreaming: false,
  sources: [],
  trace: null,
  lastMeta: null,
  selectedEvidence: null,
  activeSessionId: null,
  activeMessageId: null,
  knowledge: null,

  // `sources` is cleared too: it carries the GraphRAG traversal provenance the
  // Graph tab renders, so leaving the previous answer's sources in place would
  // show a stale graph for the duration of the new query.
  startStream: () =>
    set({
      isStreaming: true, streamingText: "", content: "", sources: [],
      // answer-scoped panels belong to the PREVIOUS answer
      knowledge: null,
    }),

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
    })),

  setContent: (content) => set({ content }),

  applyEdit: (newContent) => set({ content: newContent }),

  setSources: (sources) => set({ sources }),
  setTrace: (trace) => set({ trace }),
  selectEvidence: (src) => set({ selectedEvidence: src }),
  setLastMeta: (meta) => set({ lastMeta: meta }),

  setKnowledge: (matches, tier) =>
    set({ knowledge: { matches, tier, fresh: null } }),

  snapshotKnowledgeFresh: () =>
    set((s) => {
      if (!s.knowledge) return {}; // nothing to snapshot — no comparison shown
      return {
        knowledge: {
          ...s.knowledge,
          fresh: { content: s.content, sources: s.sources },
        },
      };
    }),

  useKnowledgeAnswer: (answer, sources) =>
    set(() => ({
      content: answer,
      sources,
      selectedEvidence: null,
      // Choosing an answer CLOSES the comparison: the panel disappears and
      // the state is cleared so the next query starts clean. (The panel used
      // to linger so the other side could still be picked — now re-asking
      // the query is the way back.) Nothing is re-saved to Knowledge.
      knowledge: null,
    })),

  clearAnswerScopedUi: () => set({ knowledge: null }),

  bindMessage: (sessionId, messageId) =>
    set({ activeSessionId: sessionId, activeMessageId: messageId }),

  reset: () =>
    set({
      content: "",
      streamingText: "",
      isStreaming: false,
      sources: [],
      trace: null,
      lastMeta: null,
      selectedEvidence: null,
    }),
}));
