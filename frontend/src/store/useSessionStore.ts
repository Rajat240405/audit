import { create } from "zustand";
import type { ChatMessage, Session } from "@/types";

interface SessionState {
  sessions: Session[];
  activeSessionId: string | null;
  searchQuery: string;

  createSession: () => string;
  deleteSession: (id: string) => void;
  renameSession: (id: string, title: string) => void;
  togglePin: (id: string) => void;
  setActive: (id: string) => void;
  addMessage: (sessionId: string, message: ChatMessage) => void;
  updateMessage: (sessionId: string, messageId: string, patch: Partial<ChatMessage>) => void;
  clearMessages: (sessionId: string) => void;
  setSearchQuery: (q: string) => void;
}

function makeId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function sessionTitle(firstMessage?: string): string {
  if (!firstMessage) return "New session";
  const t = firstMessage.replace(/\s+/g, " ").trim();
  return t.length > 48 ? `${t.slice(0, 48)}…` : t;
}

export const useSessionStore = create<SessionState>((set) => ({
  sessions: [],
  activeSessionId: null,
  searchQuery: "",

  createSession: () => {
    const id = makeId();
    const now = Date.now();
    set((s) => ({
      sessions: [{ id, title: "New session", pinned: false, createdAt: now, updatedAt: now, messages: [] }, ...s.sessions],
      activeSessionId: id,
    }));
    return id;
  },

  deleteSession: (id) =>
    set((s) => ({
      sessions: s.sessions.filter((x) => x.id !== id),
      activeSessionId: s.activeSessionId === id ? null : s.activeSessionId,
    })),

  renameSession: (id, title) =>
    set((s) => ({
      sessions: s.sessions.map((x) => (x.id === id ? { ...x, title } : x)),
    })),

  togglePin: (id) =>
    set((s) => ({
      sessions: s.sessions.map((x) => (x.id === id ? { ...x, pinned: !x.pinned } : x)),
    })),

  setActive: (id) => set({ activeSessionId: id }),

  addMessage: (sessionId, message) =>
    set((s) => ({
      sessions: s.sessions.map((x) =>
        x.id === sessionId
          ? {
              ...x,
              messages: [...x.messages, message],
              updatedAt: message.createdAt,
              title: x.messages.length === 0 ? sessionTitle(message.content) : x.title,
            }
          : x
      ),
    })),

  updateMessage: (sessionId, messageId, patch) =>
    set((s) => ({
      sessions: s.sessions.map((x) =>
        x.id === sessionId
          ? {
              ...x,
              messages: x.messages.map((m) => (m.id === messageId ? { ...m, ...patch } : m)),
            }
          : x
      ),
    })),

  clearMessages: (sessionId) =>
    set((s) => ({
      sessions: s.sessions.map((x) => (x.id === sessionId ? { ...x, messages: [] } : x)),
    })),

  setSearchQuery: (q) => set({ searchQuery: q }),
}));
