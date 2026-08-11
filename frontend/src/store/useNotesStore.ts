import { create } from "zustand";
import { persist } from "zustand/middleware";

/**
 * Notes workspace store — persisted to localStorage so notes survive refresh.
 * Stores the notes as HTML (from the rich-text contentEditable editor:
 * bold/italic/headings/lists are real formatting, not markdown markers).
 */
interface NotesState {
  /** Markdown source of the current notes. */
  notes: string;
  setNotes: (text: string) => void;
  append: (text: string) => void;
  clear: () => void;
}

export const useNotesStore = create<NotesState>()(
  persist(
    (set) => ({
      notes: "",
      setNotes: (text) => set({ notes: text }),
      append: (text) => set((s) => ({ notes: s.notes + (s.notes ? "\n\n" : "") + text })),
      clear: () => set({ notes: "" }),
    }),
    { name: "incois-notes" }
  )
);
