import { create } from "zustand";
import type { SourceItem } from "@/types";

/**
 * Which source document is currently open in the full-document viewer.
 * Opened from the Sources tab (EvidenceCard), so the same reader can be
 * reached from every evidence view.
 */
interface DocViewerState {
  source: SourceItem | null;
  openDoc: (source: SourceItem) => void;
  close: () => void;
}

export const useDocViewerStore = create<DocViewerState>((set) => ({
  source: null,
  openDoc: (source) => set({ source }),
  close: () => set({ source: null }),
}));
