import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { ChatMessage, Session } from "@/types";
import { recoverPersistedSessions } from "@/lib/answerStatus";
import { mergeSessionLists, nextRev } from "@/lib/sessionMerge";

/**
 * Conversation state.
 *
 *  - `sessions` (History) is SHARED: persisted to localStorage `incois-sessions`
 *    and kept in step between tabs. Every write is merged with the copy that is
 *    already stored, so a tab with an older view cannot erase another tab's
 *    messages or a finished answer (lib/sessionMerge.ts).
 *  - `activeSessionId` is PER TAB: stored in sessionStorage, which a new tab
 *    does not inherit. A new tab therefore starts with no open chat, a reload of
 *    the same tab reopens its chat, and two tabs can show different chats.
 */

export const TAB_ACTIVE_KEY = "incois-tab-active-session";
const STORE_KEY = "incois-sessions";

interface SessionState {
  sessions: Session[];
  activeSessionId: string | null;
  /** Tombstones: deleted chats and cleared messages, so stale copies cannot revive them. */
  deletedSessionIds: string[];
  deletedMessageIds: string[];
  searchQuery: string;
  createSession: () => string;
  deleteSession: (id: string) => void;
  renameSession: (id: string, title: string) => void;
  togglePin: (id: string) => void;
  setActive: (id: string | null) => void;
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

/** The chat this tab has open. Per tab: sessionStorage is not shared between tabs. */
export function readTabActive(): string | null {
  try {
    return window.sessionStorage.getItem(TAB_ACTIVE_KEY);
  } catch {
    return null;
  }
}

function writeTabActive(id: string | null): void {
  try {
    if (id) window.sessionStorage.setItem(TAB_ACTIVE_KEY, id);
    else window.sessionStorage.removeItem(TAB_ACTIVE_KEY);
  } catch {
    /* storage unavailable: the tab simply starts without a remembered chat */
  }
}

/**
 * localStorage wrapper that merges with the stored copy and survives
 * QuotaExceededError.
 *
 * On a write, the incoming state is merged with what another tab stored, so the
 * result never drops messages that another tab has written. On overflow it
 * retries with progressively fewer (oldest-first) sessions instead of throwing.
 * Errors are never swallowed silently: each shed is reported once.
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
    let parsed: {
      state?: { sessions?: unknown[] } & Record<string, unknown>;
      version?: number;
    };
    try {
      parsed = JSON.parse(value);
    } catch {
      parsed = {};
    }
    const incoming = parsed?.state;
    if (incoming && Array.isArray(incoming.sessions)) {
      let stored: { sessions: Session[]; deletedSessionIds?: string[]; deletedMessageIds?: string[] } = {
        sessions: [],
      };
      try {
        const cur = window.localStorage.getItem(k);
        const curState = cur ? JSON.parse(cur)?.state : null;
        if (curState && Array.isArray(curState.sessions)) stored = curState;
      } catch {
        /* unreadable stored copy: write this tab's view */
      }
      const merged = mergeSessionLists(
        incoming as unknown as { sessions: Session[]; deletedSessionIds?: string[]; deletedMessageIds?: string[] },
        stored,
        MAX_PERSISTED_SESSIONS
      );
      parsed.state = {
        ...incoming,
        sessions: merged.sessions,
        deletedSessionIds: merged.deletedSessionIds,
        deletedMessageIds: merged.deletedMessageIds,
      };
      value = JSON.stringify(parsed);
    }
    try {
      window.localStorage.setItem(k, value);
      return;
    } catch {
      /* fall through to shedding */
    }
    const sessions = parsed?.state?.sessions;
    if (!Array.isArray(sessions)) return;
    // shed oldest-first (the array is newest-first)
    for (let keep = Math.floor(sessions.length / 2); keep >= 1; keep = Math.floor(keep / 2)) {
      parsed.state!.sessions = sessions.slice(0, keep);
      try {
        window.localStorage.setItem(k, JSON.stringify(parsed));
        console.warn(`[sessions] storage quota reached — kept the ${keep} most recent session(s).`);
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
 * A handful of turns therefore blew the ~5 MB origin quota.
 *
 * What is preserved: the conversation itself (role/content/timestamps/status),
 * titles, pinning and ordering, plus the server run id and write revision that
 * let a reload or another tab reconnect to a running answer.
 *
 * What is dropped: `sources`, `trace` and `meta`. These are rebuilt from the
 * next request or from the run replay.
 */
function persistableMessage(m: ChatMessage): ChatMessage {
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    createdAt: m.createdAt,
    ...(m.status ? { status: m.status } : {}),
    ...(m.runId ? { runId: m.runId } : {}),
    ...(m.rev ? { rev: m.rev } : {}),
  };
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set) => ({
      sessions: [],
      activeSessionId: null,
      deletedSessionIds: [],
      deletedMessageIds: [],
      searchQuery: "",

      createSession: () => {
        const id = makeId();
        const now = Date.now();
        set((s) => ({
          sessions: [
            { id, title: "New session", pinned: false, createdAt: now, updatedAt: now, messages: [], rev: nextRev() },
            ...s.sessions,
          ],
          activeSessionId: id,
        }));
        writeTabActive(id);
        return id;
      },

      deleteSession: (id) => {
        set((s) => ({
          sessions: s.sessions.filter((x) => x.id !== id),
          activeSessionId: s.activeSessionId === id ? null : s.activeSessionId,
          deletedSessionIds: [...s.deletedSessionIds.filter((x) => x !== id), id].slice(-200),
        }));
        if (readTabActive() === id) writeTabActive(null);
      },

      renameSession: (id, title) =>
        set((s) => ({
          sessions: s.sessions.map((x) => (x.id === id ? { ...x, title, rev: nextRev() } : x)),
        })),

      togglePin: (id) =>
        set((s) => ({
          sessions: s.sessions.map((x) => (x.id === id ? { ...x, pinned: !x.pinned, rev: nextRev() } : x)),
        })),

      setActive: (id) => {
        set({ activeSessionId: id });
        writeTabActive(id);
      },

      addMessage: (sessionId, message) =>
        set((s) => ({
          sessions: s.sessions.map((x) =>
            x.id === sessionId
              ? {
                  ...x,
                  messages: [...x.messages, { ...message, rev: message.rev ?? nextRev() }],
                  updatedAt: message.createdAt,
                  title: x.messages.length === 0 ? sessionTitle(message.content) : x.title,
                  rev: nextRev(),
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
                  messages: x.messages.map((m) =>
                    m.id === messageId ? { ...m, ...patch, rev: nextRev() } : m
                  ),
                }
              : x
          ),
        })),

      clearMessages: (sessionId) =>
        set((s) => {
          const gone = s.sessions.find((x) => x.id === sessionId)?.messages.map((m) => m.id) ?? [];
          return {
            sessions: s.sessions.map((x) => (x.id === sessionId ? { ...x, messages: [] } : x)),
            deletedMessageIds: [...s.deletedMessageIds, ...gone].slice(-2000),
          };
        }),

      setSearchQuery: (q) => set({ searchQuery: q }),
    }),
    {
      name: STORE_KEY,
      // Graceful recovery for quota pressure: drop the oldest sessions and retry
      // rather than throwing, so a full quota can never break the app.
      storage: createJSONStorage(() => quotaSafeStorage),
      // Persist only what History needs — see persistableMessage(). The active
      // chat is NOT persisted here: it is per tab (sessionStorage).
      partialize: (state) => ({
        sessions: state.sessions.slice(0, MAX_PERSISTED_SESSIONS).map((s) => ({
          ...s,
          messages: s.messages.map(persistableMessage),
        })),
        deletedSessionIds: state.deletedSessionIds,
        deletedMessageIds: state.deletedMessageIds,
      }),
      // Startup / reload: restore this tab's chat if it still exists. A new tab
      // has none, so it starts with a fresh chat. No session is created here.
      // A message still marked `streaming` is kept as streaming when it has a
      // server run id (the server decides its fate); otherwise it is interrupted.
      merge: (persisted, current) => {
        const loaded = recoverPersistedSessions(persisted);
        const tab = readTabActive();
        return {
          ...current,
          sessions: loaded.sessions,
          deletedSessionIds: loaded.deletedSessionIds,
          deletedMessageIds: loaded.deletedMessageIds,
          activeSessionId: tab && loaded.sessions.some((s) => s.id === tab) ? tab : null,
        };
      },
    }
  )
);

/**
 * Keep this tab's History in step with other tabs. Another tab's write arrives
 * as a `storage` event carrying the merged state; it is merged in here the same
 * way, so a tab never drops its own unsaved view.
 */
if (typeof window !== "undefined") {
  window.addEventListener("storage", (e: StorageEvent) => {
    if (e.key !== STORE_KEY || !e.newValue) return;
    let remote: { sessions?: Session[]; deletedSessionIds?: string[]; deletedMessageIds?: string[] };
    try {
      remote = JSON.parse(e.newValue)?.state;
    } catch {
      return;
    }
    if (!remote || !Array.isArray(remote.sessions)) return;
    const st = useSessionStore.getState();
    const merged = mergeSessionLists(
      {
        sessions: st.sessions,
        deletedSessionIds: st.deletedSessionIds,
        deletedMessageIds: st.deletedMessageIds,
      },
      remote as { sessions: Session[]; deletedSessionIds?: string[]; deletedMessageIds?: string[] },
      MAX_PERSISTED_SESSIONS
    );
    const activeGone = !!st.activeSessionId && !merged.sessions.some((s) => s.id === st.activeSessionId);
    useSessionStore.setState({
      sessions: merged.sessions,
      deletedSessionIds: merged.deletedSessionIds,
      deletedMessageIds: merged.deletedMessageIds,
      ...(activeGone ? { activeSessionId: null } : {}),
    });
    if (activeGone) writeTabActive(null);
  });
}
