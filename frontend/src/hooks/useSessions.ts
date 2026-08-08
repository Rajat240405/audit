import { useMemo } from "react";
import { useSessionStore } from "@/store/useSessionStore";
import type { Session } from "@/types";

/** Derived selector: sessions filtered by search, pinned first. */
export function useFilteredSessions(): Session[] {
  const sessions = useSessionStore((s) => s.sessions);
  const search = useSessionStore((s) => s.searchQuery);

  return useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = q
      ? sessions.filter(
          (s) => s.title.toLowerCase().includes(q) || s.messages.some((m) => m.content.toLowerCase().includes(q))
        )
      : sessions;
    return [...filtered].sort((a, b) => {
      if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
      return b.updatedAt - a.updatedAt;
    });
  }, [sessions, search]);
}
