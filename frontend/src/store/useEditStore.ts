import { create } from "zustand";

/**
 * Shared AI-edit state so ANY component (Toolbar chips, DraftCanvas buttons)
 * can trigger an edit and all loading indicators stay in sync.
 *
 * Accept/Reject: when an edit finishes, the result is staged as `pendingEdit`
 * (original + revised). The user reviews it and explicitly Accepts (apply) or
 * Rejects (discard) — never silently clobbers the draft.
 */
interface PendingEdit {
  original: string;
  revised: string;
  label: string;
}

interface EditState {
  editing: boolean;
  editingLabel: string | null;
  pendingEdit: PendingEdit | null;
  start: (label: string) => void;
  stage: (original: string, revised: string, label: string) => void;
  accept: () => void;
  reject: () => void;
  finish: () => void;
  fail: () => void;
}

export const useEditStore = create<EditState>((set, get) => ({
  editing: false,
  editingLabel: null,
  pendingEdit: null,
  start: (label) => set({ editing: true, editingLabel: label, pendingEdit: null }),
  stage: (original, revised, label) =>
    set({ editing: false, editingLabel: null, pendingEdit: { original, revised, label } }),
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
