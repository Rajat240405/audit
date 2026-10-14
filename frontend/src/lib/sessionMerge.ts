/**
 * Merging conversation state between browser tabs.
 *
 * Every tab of the app shares one persisted copy of the conversations
 * (localStorage `incois-sessions`) but keeps its own in-memory copy. Each write
 * from a tab therefore carries a view that may be stale. These rules make a
 * stale view unable to erase what another tab has already written:
 *
 *  - Messages are merged by id. A message that exists in only one copy is kept.
 *  - A settled message (done / stopped / error / interrupted) always beats a
 *    message that is still streaming. A late partial write from a tab that is
 *    still generating can never overwrite a finished answer.
 *  - Two settled copies: the newer write wins (rev). This is how edits win
 *    over an older copy.
 *  - Two streaming copies of the same run: the longer text wins. Both come from
 *    the same ordered event stream, so the longer one is the more complete.
 *  - Sessions are merged by id, with the same rev rule for their metadata.
 *  - Deleted sessions and messages are kept as tombstones so a stale copy cannot
 *    bring them back.
 */
import type { ChatMessage, Session } from "@/types";

const SETTLED: ReadonlySet<string> = new Set(["done", "stopped", "error", "interrupted"]);

let lastRev = 0;

/** A strictly increasing revision number for this tab's writes. */
export function nextRev(): number {
  const now = Date.now();
  lastRev = now > lastRev ? now : lastRev + 1;
  return lastRev;
}

/** True when the message will not change any more (or never had a status). */
export function isSettled(m: ChatMessage): boolean {
  return !m.status || SETTLED.has(m.status);
}

export function mergeMessage(a: ChatMessage, b: ChatMessage): ChatMessage {
  const sa = isSettled(a);
  const sb = isSettled(b);
  if (sa !== sb) return sa ? a : b;
  if (sa) return (b.rev ?? 0) > (a.rev ?? 0) ? b : a;
  return (b.content ?? "").length > (a.content ?? "").length ? b : a;
}

const ROLE_ORDER = (m: ChatMessage) => (m.role === "user" ? 0 : 1);

function sortMessages(list: ChatMessage[]): ChatMessage[] {
  return list
    .map((m, i) => ({ m, i }))
    .sort((x, y) => {
      if (x.m.createdAt !== y.m.createdAt) return x.m.createdAt - y.m.createdAt;
      if (ROLE_ORDER(x.m) !== ROLE_ORDER(y.m)) return ROLE_ORDER(x.m) - ROLE_ORDER(y.m);
      return x.i - y.i;
    })
    .map((x) => x.m);
}

export function mergeSessionMessages(
  local: ChatMessage[],
  remote: ChatMessage[],
  deadMessages: ReadonlySet<string>
): ChatMessage[] {
  const byId = new Map<string, ChatMessage>();
  for (const m of local) if (!deadMessages.has(m.id)) byId.set(m.id, m);
  for (const m of remote) {
    if (deadMessages.has(m.id)) continue;
    const have = byId.get(m.id);
    byId.set(m.id, have ? mergeMessage(have, m) : m);
  }
  return sortMessages([...byId.values()]);
}

function mergeSession(local: Session, remote: Session, deadMessages: ReadonlySet<string>): Session {
  const newer = (remote.rev ?? 0) > (local.rev ?? 0) ? remote : local;
  return {
    ...local,
    title: newer.title,
    pinned: newer.pinned,
    updatedAt: Math.max(local.updatedAt, remote.updatedAt),
    rev: Math.max(local.rev ?? 0, remote.rev ?? 0) || undefined,
    messages: mergeSessionMessages(local.messages, remote.messages, deadMessages),
  };
}

export interface MergeInput {
  sessions: Session[];
  deletedSessionIds?: string[];
  deletedMessageIds?: string[];
}

/**
 * Union of two views of the conversation list. Returns sessions newest first
 * (the order History shows) with tombstoned sessions and messages removed.
 */
export function mergeSessionLists(
  local: MergeInput,
  remote: MergeInput,
  limit = Number.POSITIVE_INFINITY
): { sessions: Session[]; deletedSessionIds: string[]; deletedMessageIds: string[] } {
  const deadSessions = new Set([...(local.deletedSessionIds ?? []), ...(remote.deletedSessionIds ?? [])]);
  const deadMessages = new Set([...(local.deletedMessageIds ?? []), ...(remote.deletedMessageIds ?? [])]);
  const byId = new Map<string, Session>();
  for (const s of local.sessions) if (!deadSessions.has(s.id)) byId.set(s.id, s);
  for (const s of remote.sessions) {
    if (deadSessions.has(s.id)) continue;
    const have = byId.get(s.id);
    byId.set(s.id, have ? mergeSession(have, s, deadMessages) : { ...s, messages: mergeSessionMessages([], s.messages, deadMessages) });
  }
  const sessions = [...byId.values()]
    .map((s) => ({ ...s, messages: mergeSessionMessages(s.messages, [], deadMessages) }))
    .sort((a, b) => b.createdAt - a.createdAt)
    .slice(0, limit);
  return {
    sessions,
    deletedSessionIds: [...deadSessions].slice(-200),
    deletedMessageIds: [...deadMessages].slice(-2000),
  };
}
