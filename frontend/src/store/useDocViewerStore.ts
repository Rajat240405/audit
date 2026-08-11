import { create } from "zustand";
import type { SourceItem } from "@/types";

/**
 * Which source document is currently open in the full-document viewer.
 * Shared by the Sources tab (EvidenceCard) and Cross-Verify Facts
 * (ClaimsList) so either can open the same reader.
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
