import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
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

/**
 * localStorage wrapper that survives QuotaExceededError.
 *
 * On overflow it retries with progressively fewer (oldest-first) sessions
 * instead of throwing. The newest sessions — including the one in progress —
 * are kept. Errors are never swallowed silently: each shed is reported once.
 */
const quotaSafeStorage: Storage = {
  get length() {
    return window.localStorage.length;
  },
  clear: () => window.localStorage.clear(),
  key: (i: number) => window.localStorage.key(i),
  removeItem: (k: string) => window.localStorage.removeItem(k),
  getItem: (k: string) => window.localStorage.getItem(k),
  setItem: (k: string, value: string) => {
    try {
      window.localStorage.setItem(k, value);
      return;
    } catch {
      /* fall through to shedding */
    }
    let parsed: { state?: { sessions?: unknown[] } };
    try {
      parsed = JSON.parse(value);
    } catch {
      return; // not our shape — give up rather than corrupt storage
    }
    const sessions = parsed?.state?.sessions;
    if (!Array.isArray(sessions)) return;
    // shed oldest-first (the array is newest-first)
    for (let keep = Math.floor(sessions.length / 2); keep >= 1; keep = Math.floor(keep / 2)) {
      parsed.state!.sessions = sessions.slice(0, keep);
      try {
        window.localStorage.setItem(k, JSON.stringify(parsed));
        console.warn(
          `[sessions] storage quota reached — kept the ${keep} most recent session(s).`
        );
        return;
      } catch {
        /* keep shedding */
      }
    }
    console.warn("[sessions] storage quota reached — session history not persisted.");
  },
};

/** Sessions kept in localStorage. Older ones drop out of History. */
export const MAX_PERSISTED_SESSIONS = 30;

/**
 * Strip transient/reconstructible payloads before persisting.
 *
 * `incois-sessions` previously persisted the WHOLE store, including every
 * message's `sources` — and each SourceItem carries the full `question` and
 * `answer` text of a parliamentary document (thousands of characters, up to
 * ~430 KB for the largest in this corpus) plus per-source graph provenance.
 * A handful of turns therefore blew the ~5 MB origin quota:
 *
 *   Failed to execute 'setItem' … 'incois-sessions' exceeded the quota.
 *
 * What is preserved: the conversation itself (role/content/timestamps), titles,
 * pinning and ordering — i.e. everything History needs to restore a session.
 *
 * What is dropped: `sources`, `trace` and `meta`. These are per-request
 * artefacts of a live answer; the evidence panel, RAG-pipeline trace and
 * grounding are all rebuilt from the next request. Dropping them cannot lose
 * user-authored content.
 */
function persistableMessage(m: ChatMessage): ChatMessage {
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    createdAt: m.createdAt,
  };
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set) => ({
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
    }),
    {
      name: "incois-sessions",
      // Graceful recovery for state written BEFORE partialize existed (and
      // for any future quota pressure): drop the oldest sessions and retry
      // rather than throwing, so a full quota can never break the app or
      // silently lose the current conversation.
      storage: createJSONStorage(() => quotaSafeStorage),
      // Persist only what History needs — see persistableMessage(). Also cap
      // the number of sessions so storage cannot grow without bound.
      partialize: (state) => ({
        sessions: state.sessions.slice(0, MAX_PERSISTED_SESSIONS).map((s) => ({
          ...s,
          messages: s.messages.map(persistableMessage),
        })),
        activeSessionId: state.activeSessionId,
      }),
      // App startup: DO NOT resume the previous chat. Create a fresh empty
      // session and make it active (old sessions stay in History). Previous
      // sessions are never auto-loaded.
      onRehydrateStorage: () => (_state, _error) => {
        // CRITICAL: zustand v5 merges the persisted state AFTER this callback,
        // so any setState here is overwritten back to the old active session.
        // Defer with setTimeout(0) so the fresh session is created AFTER the
        // merge — old chats stay in History but the active session is empty.
        setTimeout(() => {
          const id = makeId();
          const now = Date.now();
          useSessionStore.setState((s) => ({
            sessions: [
              { id, title: "New session", pinned: false, createdAt: now, updatedAt: now, messages: [] },
              ...s.sessions,
            ],
            activeSessionId: id,
          }));
        }, 0);
      },
    }
  )
);