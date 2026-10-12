import { create } from "zustand";

/**
 * Shared AI-edit state so ANY component (Toolbar chips, DraftCanvas buttons)
 * can trigger an edit and all loading indicators stay in sync.
 *
 * Accept/Reject: when an edit finishes, the result is staged as `pendingEdit`
 * (original + revised). The user reviews it and explicitly Accepts (apply) or
 * Rejects (discard) — never silently clobbers the draft.
 *
 * ORIGIN: an edit belongs to the answer (session + message) it was started
 * from. It is only applied while the canvas is still bound to that message.
 * An edit in flight is aborted when the user switches session or answer.
 */
export interface EditOrigin {
  sessionId: string | null;
  messageId: string | null;
}

export interface PendingEdit extends EditOrigin {
  original: string;
  revised: string;
  label: string;
}

interface EditState {
  editing: boolean;
  editingLabel: string | null;
  pendingEdit: PendingEdit | null;
  start: (label: string) => void;
  stage: (original: string, revised: string, label: string, origin?: EditOrigin) => void;
  accept: () => void;
  reject: () => void;
  finish: () => void;
  fail: () => void;
}

/** The single in-flight AI edit request (module scope so transitions can abort it). */
let inFlight: AbortController | null = null;

/** Register a new edit request, aborting any previous one. */
export function beginEditRequest(): AbortController {
  inFlight?.abort();
  const c = new AbortController();
  inFlight = c;
  return c;
}

/** Forget a finished request (only if it is still the current one). */
export function endEditRequest(c: AbortController): void {
  if (inFlight === c) inFlight = null;
}

/**
 * Abort the in-flight edit (session switch, new query, Toolbar cancel). Its
 * late callbacks are ignored by the hook, which checks `signal.aborted`.
 */
export function cancelInFlightEdit(): void {
  const c = inFlight;
  inFlight = null;
  c?.abort();
  useEditStore.setState({ editing: false, editingLabel: null });
}

export const useEditStore = create<EditState>((set, get) => ({
  editing: false,
  editingLabel: null,
  pendingEdit: null,
  start: (label) => set({ editing: true, editingLabel: label, pendingEdit: null }),
  stage: (original, revised, label, origin) =>
    set({
      editing: false,
      editingLabel: null,
      pendingEdit: {
        original,
        revised,
        label,
        sessionId: origin?.sessionId ?? null,
        messageId: origin?.messageId ?? null,
      },
    }),
  accept: () => {
    const p = get().pendingEdit;
    if (p) {
      // apply the revision via the draft store (the hook wires this)
      set({ pendingEdit: null });
    }
  },
  reject: () => set({ pendingEdit: null }),
  finish: () => set({ editing: false, editingLabel: null }),
  fail: () => set({ editing: false, editingLabel: null, pendingEdit: null }),
}));
