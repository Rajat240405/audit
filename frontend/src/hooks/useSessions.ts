/**
 * Session state and the UI behaviours that depend on it.
 *
 * Three concerns, deliberately kept separate:
 *
 * 1. :func:`useSession` — the one source of truth for "who is signed in, and do
 *    we know yet?". It reports one of THREE states, because "we have not asked
 *    the server yet" and "the server said no" are different situations. Merging
 *    them is what let a signed-out visitor see protected content: an unknown
 *    session was treated as a known-good one.
 *
 * 2. :func:`useSignOut` — end the session in one click: revoke it server-side,
 *    drop every cached response, then navigate. The redirect is explicit; it
 *    does not wait for some unrelated request to fail and notice.
 *
 * 3. :func:`useSessionCacheBridge` — when a session ends underneath the app,
 *    discard cached data so a later sign-in on this tab cannot inherit it.
 *
 * PHASE 3 NOTE: this file also carries `useSessionGuard` in the Phase 4 tree,
 * which listens for the 401 event `apiFetch` dispatches. Phase 3's `apiFetch`
 * never dispatches it (the API is not gated yet), so that hook is omitted here
 * rather than left as dead code. Phase 4's version of this file is a superset —
 * it replaces this one wholesale when you move on.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";

import { AuthApiError, signOut } from "@/api/auth";
import { fetchSession } from "@/api/auth";
import type { SessionState, SessionUser } from "@/api/auth";
import { useToastStore } from "@/store/useToastStore";

/** Cache key every session reader shares, so the lookup happens once. */
export const SESSION_KEY = ["session"] as const;

/** What an absent session looks like in the cache. */
export const SIGNED_OUT: SessionState = { authenticated: false, user: null };

/**
 * `loading`          — the lookup is in flight; we do not know who you are yet.
 * `authenticated`    — the server confirmed a valid session.
 * `unauthenticated`  — the server said no (401), or the lookup failed outright.
 *
 * Only `authenticated` may render protected content.
 */
export type SessionStatus = "loading" | "authenticated" | "unauthenticated";

export interface SessionView {
  status: SessionStatus;
  /** Convenience for `status === "loading"`. */
  isLoading: boolean;
  authenticated: boolean;
  user: SessionUser | null;
  isAdmin: boolean;
}

/**
 * Drop every cached response, then re-seed the session as signed out.
 *
 * The re-seed matters: it leaves the cache in a definite state, so nothing
 * refetches to "find out" whether we are signed in, and no frame renders with
 * the previous account's data.
 *
 * Cached responses are user-scoped (saved knowledge, chat runs), so they must
 * not survive a session boundary — otherwise the next account to sign in on
 * this tab can read them.
 */
export function clearSessionState(queryClient: QueryClient): void {
  // Everything except the session key, which is re-seeded just below. Removing
  // it too would make the observer refetch mid-teardown.
  queryClient.removeQueries({
    predicate: (query) => query.queryKey[0] !== SESSION_KEY[0],
  });
  queryClient.setQueryData(SESSION_KEY, SIGNED_OUT);
}

/**
 * Whether the session is known-good, unknown, or known-absent.
 *
 * `retry` is off and `refetchOnWindowFocus` is off: this query drives access
 * control, so it must resolve on what the server actually said, not on a
 * background refresh the user did not ask for.
 */
export function useSession(): SessionView {
  const { data, isPending } = useQuery<SessionState>({
    queryKey: SESSION_KEY,
    // `signal` lets a login/logout abort an in-flight lookup.
    queryFn: ({ signal }) => fetchSession(signal),
    staleTime: 60_000,
    // A session can be ended in another tab. Re-check when the window regains
    // focus rather than trusting the cached answer indefinitely.
    refetchOnWindowFocus: true,
    retry: false,
  });

  // Fails closed, in both failure directions:
  //   * still loading, no answer yet                   -> loading
  //   * lookup failed with no cached answer            -> unauthenticated
  //   * lookup failed, but we hold a good prior answer -> that answer stands
  //   * server said 401                                -> unauthenticated
  const authenticated = Boolean(data?.authenticated);

  const status: SessionStatus = isPending
    ? "loading"
    : authenticated
      ? "authenticated"
      : "unauthenticated";

  return {
    status,
    isLoading: isPending,
    authenticated,
    user: authenticated ? (data?.user ?? null) : null,
    // Phase 3 has no role model — `users.role` and `is_admin` arrive with
    // Phase 4, which replaces this file with a superset version. Kept in the
    // shape so Header's useIsAdmin() keeps working when it is dropped in.
    isAdmin: false,
  };
}

/**
 * Discard cached responses when a session ends underneath the app.
 *
 * A session can lapse between calls — the cookie expires, or another tab signs
 * out — so this watches the state itself rather than waiting for a request to
 * fail. Mount once. Renders nothing.
 */
export function useSessionCacheBridge(): void {
  const queryClient = useQueryClient();
  const { status } = useSession();
  const wasAuthenticated = useRef(false);

  useEffect(() => {
    if (status === "authenticated") {
      wasAuthenticated.current = true;
      return;
    }
    if (status === "unauthenticated" && wasAuthenticated.current) {
      wasAuthenticated.current = false;
      clearSessionState(queryClient);
    }
  }, [status, queryClient]);
}

/**
 * Re-read the session from the server and write it into the cache.
 *
 * Used after a successful login. The cookie is set by then, but the cache still
 * holds whatever it held before — after a logout that is "signed out", so a
 * bare `navigate()` would render ProtectedRoute against a stale answer and
 * bounce the user straight back to sign-in. That is the bug this fixes.
 *
 * In-flight lookups are cancelled first: one that started before the login
 * could otherwise resolve afterwards and overwrite the newer state with
 * "signed out".
 */
export async function primeSession(queryClient: QueryClient): Promise<SessionState> {
  await queryClient.cancelQueries({ queryKey: SESSION_KEY });
  return queryClient.fetchQuery<SessionState>({
    queryKey: SESSION_KEY,
    queryFn: ({ signal }) => fetchSession(signal),
    staleTime: 0, // always read the server; never serve the stale answer
    retry: false,
  });
}

/**
 * Re-check the session when this tab becomes visible again.
 *
 * `refetchOnWindowFocus` covers refocusing the window, but switching between
 * tabs does not always produce a focus event — and that is exactly when a
 * session is most likely to have been ended elsewhere. Invalidate (not just
 * refetch) so the check happens even inside `staleTime`.
 */
export function useSessionRevalidation(): void {
  const queryClient = useQueryClient();

  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.visibilityState !== "visible") return;
      void queryClient.invalidateQueries({ queryKey: SESSION_KEY });
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [queryClient]);
}

export interface UseSignOut {
  /** End the session. Resolves true if signed out, false if it failed. */
  signOut: () => Promise<boolean>;
  /** True while the request is in flight, so the control can disable itself. */
  pending: boolean;
}

/**
 * Sign out in one click.
 *
 * The order is deliberate:
 *
 *   1. Ask the server to revoke the session. Until that succeeds the cookie is
 *      still valid, so clearing local state first would be a lie — and if the
 *      call fails the user must be told, not silently left half-signed-out.
 *   2. Clear the cache. Synchronous, so no frame renders with stale account
 *      data; ProtectedRoute reads the same cache and would bounce us anyway.
 *   3. Navigate, replacing the history entry so Back cannot return to a
 *      protected page.
 */
export function useSignOut(): UseSignOut {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const pushToast = useToastStore((s) => s.push);
  const [pending, setPending] = useState(false);
  const busy = useRef(false);

  // StrictMode mounts, unmounts and remounts, so the flag is (re)set in the
  // effect body — setting it only in the cleanup would leave it stuck false.
  useEffect(() => {
    busy.current = false;
  }, []);

  const run = useCallback(async (): Promise<boolean> => {
    if (busy.current) return false; // ignore double-clicks
    busy.current = true;
    setPending(true);

    try {
      await signOut();
    } catch (err) {
      // A 401 here means the server has no session for us either way — the
      // intended outcome, not a failure. This never treats 401 as proof of
      // authentication; it is proof of the opposite.
      const alreadyGone = err instanceof AuthApiError && err.status === 401;
      if (!alreadyGone) {
        busy.current = false;
        setPending(false);
        pushToast(
          "error",
          "Could not sign you out. Your session may still be active — please try again."
        );
        return false;
      }
    }

    // Cancel first, so a lookup already in flight cannot land after this and
    // overwrite "signed out" with the session we just ended.
    await queryClient.cancelQueries({ queryKey: SESSION_KEY });
    clearSessionState(queryClient);
    setPending(false);
    // No await: navigate is synchronous from the caller's point of view.
    navigate("/signin", { replace: true });
    return true;
  }, [navigate, queryClient, pushToast]);

  return { signOut: run, pending };
}
