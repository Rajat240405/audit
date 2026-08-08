import { create } from "zustand";

/**
 * Lets the Sidebar's query input trigger the chat stream owned by the
 * Workspace without prop-drilling through the layout tree.
 */
interface ChatActionsState {
  send: ((q: string) => void) | null;
  stop: (() => void) | null;
  running: boolean;
}

export const useChatActionsStore = create<ChatActionsState>(() => ({
  send: null,
  stop: null,
  running: false,
}));
