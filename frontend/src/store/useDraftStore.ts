import { create } from "zustand";
import type { GroundingClaim, RetrievalTrace, SourceItem , InvestigationResult } from "@/types";
import { buildGroundingReport } from "@/services/grounding";

/**
 * Deep-mode verification progress.
 *
 * A Deep answer streams first and is verified afterwards: `useChatStream`
 * dispatches `verifyAnswer(..., "full")` once the stream closes. Until now that
 * work was invisible — the user saw a finished answer with no indication a
 * judge pass could still rewrite it, and no way to say "hold on".
 *
 * This state is ANSWER-SCOPED and identity-bound for exactly the same reason
 * `investigations` is: the verify call has no AbortSignal, so a completion from
 * Query N can land while Query N+1 owns the canvas. `settleDeepVerify` therefore
 * only writes when the stored identity still matches, so a late result can never
 * clear or overwrite a newer query's banner.
 */
export type DeepVerifyStatus = "idle" | "running" | "done" | "error";
/** "unset" keeps the historical non-blocking behaviour: the user is offered a
 *  choice but is never held up unless they explicitly ask to wait. */
export type DeepVerifyChoice = "unset" | "wait" | "anyway";

export interface DeepVerifyState {
  status: DeepVerifyStatus;
  choice: DeepVerifyChoice;
  sessionId: string | null;
  messageId: string | null;
}

export const IDLE_DEEP_VERIFY: DeepVerifyState = {
  status: "idle",
  choice: "unset",
  sessionId: null,
  messageId: null,
};

/**
 * Should a new query be held back right now?
 *
 * Only an explicit "Wait" blocks. Light verification never enters this state at
 * all, and Deep verification with no choice made stays non-blocking, so the
 * default behaviour is unchanged.
 */
export function deepVerifyBlocksNewQuery(dv: DeepVerifyState): boolean {
  return dv.status === "running" && dv.choice === "wait";
}

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
  /** Cross-Verified Facts panel visibility — ANSWER-SCOPED (lifted from
   *  DraftCanvas local state so a new query can reset it). */
  factsOpen: boolean;
  /** claim_id -> investigation result / in-flight marker. Answer-scoped. */
  investigations: Record<string, InvestigationResult>;
  investigating: Record<string, boolean>;
  /** Deep-mode post-generation verification progress. Answer-scoped. */
  deepVerify: DeepVerifyState;

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
  setFactsOpen: (open: boolean) => void;
  setInvestigating: (claimId: string, busy: boolean) => void;
  setInvestigation: (claimId: string, result: InvestigationResult) => void;
  /** Mark a Deep answer's verification as in flight, bound to its identity. */
  beginDeepVerify: (sessionId: string, messageId: string) => void;
  /** Record the user's Wait / Start-anyway decision for the running pass. */
  setDeepVerifyChoice: (choice: DeepVerifyChoice) => void;
  /** Finish the running pass. IDENTITY-GATED: a result belonging to a
   *  superseded answer is dropped instead of touching the current banner. */
  settleDeepVerify: (
    sessionId: string,
    messageId: string,
    status: DeepVerifyStatus
  ) => void;
  /** Clear every ANSWER-SPECIFIC transient panel (facts, investigations).
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
  grounding: [],
  lastMeta: null,
  selectedEvidence: null,
  activeSessionId: null,
  activeMessageId: null,
  factsOpen: false,
  investigations: {},
  investigating: {},
  deepVerify: IDLE_DEEP_VERIFY,

  // `sources` is cleared too: it carries the GraphRAG traversal provenance the
  // Graph tab renders, so leaving the previous answer's sources in place would
  // show a stale graph for the duration of the new query.
  startStream: () =>
    set({
      isStreaming: true, streamingText: "", content: "", sources: [],
      // answer-scoped panels belong to the PREVIOUS answer
      factsOpen: false, investigations: {}, investigating: {},
      // A new query supersedes any in-flight verification banner. The old pass
      // keeps running (it has no AbortSignal) but its identity no longer
      // matches, so settleDeepVerify will drop its result.
      deepVerify: IDLE_DEEP_VERIFY,
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
  setFactsOpen: (open) => set({ factsOpen: open }),

  setInvestigating: (claimId, busy) =>
    set((s) => ({ investigating: { ...s.investigating, [claimId]: busy } })),

  setInvestigation: (claimId, result) =>
    set((s) => ({
      investigations: { ...s.investigations, [claimId]: result },
      investigating: { ...s.investigating, [claimId]: false },
    })),

  setDeepVerifyChoice: (choice) =>
    set((s) => ({ deepVerify: { ...s.deepVerify, choice } })),

  beginDeepVerify: (sessionId, messageId) =>
    set({
      deepVerify: {
        status: "running",
        // A fresh pass never inherits the previous answer's Wait decision.
        choice: "unset",
        sessionId,
        messageId,
      },
    }),

  settleDeepVerify: (sessionId, messageId, status) =>
    set((s) => {
      // IDENTITY GUARD — mirrors the stale-result guard in useChatStream. A
      // newer answer owns the banner now, so this completion is dropped
      // entirely: no status write, no banner change.
      if (
        s.deepVerify.sessionId !== sessionId ||
        s.deepVerify.messageId !== messageId
      ) {
        return s;
      }
      return { deepVerify: { ...s.deepVerify, status } };
    }),

  clearAnswerScopedUi: () =>
    set({
      factsOpen: false,
      investigations: {},
      investigating: {},
      deepVerify: IDLE_DEEP_VERIFY,
    }),

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
      factsOpen: false,
      investigations: {},
      investigating: {},
      deepVerify: IDLE_DEEP_VERIFY,
    }),
}));
