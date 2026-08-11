import { create } from "zustand";

/**
 * Lets the Sidebar's query input trigger the chat stream owned by the
 * Workspace without prop-drilling through the layout tree, and lets the
 * Model Activity panel switch the workspace tab ("Go to canvas").
 *
 * All of these are OPTIONAL bridges — the Workspace keeps its own local tab
 * state, so a stale copy of this store can never break tab switching.
 */
interface ChatActionsState {
  send: ((q: string) => void) | null;
  stop: (() => void) | null;
  running: boolean;
  tab: string;
  setTab: (tab: string) => void;
}

export const useChatActionsStore = create<ChatActionsState>((set) => ({
  send: null,
  stop: null,
  running: false,
  tab: "draft",
  setTab: (tab) => set({ tab }),
}));
