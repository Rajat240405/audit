/**
 * Regression tests for the two authentication-lifecycle bugs.
 *
 *   1. Login did not take effect until a hard refresh. `SignIn` called
 *      `navigate("/dashboard")` straight after `signIn()`. The cookie was set,
 *      but the cached session still said "signed out" (all the more so right
 *      after a logout, which writes exactly that). ProtectedRoute read the
 *      stale answer and bounced the user back to /signin. A hard refresh
 *      emptied the cache, which is why it appeared to fix things.
 *
 *   2. A new tab did not restore the session. The session cookie is
 *      per-browser-profile, not per-tab, and the server proves it accepts the
 *      same cookie from a brand-new client (see the backend test module). The
 *      UI's problem was that it never re-checked: `refetchOnWindowFocus` was
 *      off, so a backgrounded tab kept showing authenticated content after the
 *      session had ended elsewhere.
 *
 * `fetch` is the only thing mocked. The real App, router, pages and hooks run.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import App from "@/App";
import { SESSION_KEY } from "@/hooks/useAuthSession";
import { useToastStore } from "@/store/useToastStore";

// ── a stand-in auth backend whose session state can be flipped ──────────────

interface MockResponse {
  status?: number;
  body?: unknown;
}

const USER = {
  id: "u1",
  email: "boss@incois.gov.in",
  email_verified: true,
  is_active: true,
};

const LOGGED_IN = { authenticated: true, user: USER };
const NOT_AUTHENTICATED = { code: "not_authenticated", message: "you are not signed in" };

const POLICY = {
  min_length: 12,
  max_length: 256,
  require_lowercase: true,
  require_uppercase: true,
  require_digit: true,
  require_symbol: false,
  rejects_email_substring: true,
  rejects_common_passwords: true,
};

/**
 * One server, shared by every client in a test. `signedIn` stands in for the
 * browser cookie jar: two independently-rendered apps that share it behave the
 * way two tabs do.
 */
function makeServer() {
  const state = { signedIn: false, loginFails: false, logoutFails: false };
  const calls: Array<{ url: string; method: string }> = [];

  const handler = (url: string, init?: RequestInit): MockResponse => {
    const method = init?.method ?? "GET";
    calls.push({ url, method });

    if (url.includes("/api/auth/login")) {
      if (state.loginFails) {
        return {
          status: 401,
          body: { detail: { code: "invalid_credentials", message: "the email or password is incorrect" } },
        };
      }
      state.signedIn = true; // server-side: the cookie is issued
      return { body: { status: "signed_in", user_id: USER.id, email: USER.email, expires_in_seconds: 28800 } };
    }
    if (url.includes("/api/auth/logout")) {
      if (state.logoutFails) {
        return { status: 500, body: { detail: { code: "server_error", message: "boom" } } };
      }
      state.signedIn = false; // server-side revocation
      return { body: { status: "signed_out", revoked: true } };
    }
    if (url.includes("/api/auth/me")) {
      if (!state.signedIn) return { status: 401, body: { detail: NOT_AUTHENTICATED } };
      return { body: LOGGED_IN };
    }
    if (url.includes("/api/auth/password-policy")) return { body: POLICY };
    return { body: {} };
  };

  return {
    state,
    calls,
    handler,
    logins: () => calls.filter((c) => c.url.includes("/api/auth/login")),
    logouts: () => calls.filter((c) => c.url.includes("/api/auth/logout")),
  };
}

type Server = ReturnType<typeof makeServer>;

function install(server: Server): void {
  global.fetch = vi.fn(async (input: unknown, init?: RequestInit) => {
    const result = server.handler(String(input), init);
    const status = result.status ?? 200;
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: "",
      json: async () => result.body ?? {},
      headers: { get: () => null },
    } as unknown as Response;
  }) as unknown as typeof fetch;
}

/** A fresh QueryClient is what a page load looks like: nothing cached. */
function renderAppAt(path: string) {
  window.history.pushState({}, "", path);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
  return { ...view, client };
}

const chrome = () => document.querySelector("[data-tour^='header-']");

/** Type credentials and submit the sign-in form. */
async function submitLogin(email = "boss@incois.gov.in", password = "Tide-Quiet-77-K9") {
  // findBy* (not getBy*): the auth pages are lazy-loaded, so the form is not
  // on screen for the first frame.
  fireEvent.change(await screen.findByLabelText(/email/i), { target: { value: email } });
  fireEvent.change(await screen.findByLabelText(/password/i), { target: { value: password } });
  await waitFor(() => {
    expect(screen.getByRole("button", { name: /^sign in$/i })).toBeTruthy();
  });
  fireEvent.click(screen.getByRole("button", { name: /^sign in$/i }));
}

// ── tests ──────────────────────────────────────────────────────────────────

describe("login lifecycle", () => {
  beforeEach(() => useToastStore.setState({ toasts: [] }));
  afterEach(() => {
    window.history.pushState({}, "", "/");
    useToastStore.setState({ toasts: [] });
  });

  it("1. a successful login updates auth state and navigates with NO hard refresh", async () => {
    const server = makeServer();
    install(server);
    renderAppAt("/signin");

    await submitLogin();

    // Lands on the dashboard, and the cache agrees with the server.
    await waitFor(() => {
      expect(window.location.pathname).toBe("/dashboard");
    });
    // The shell is lazy-loaded, so the chrome arrives a frame later.
    await waitFor(() => {
      expect(chrome()).not.toBeNull();
    });
    expect(server.logins()).toHaveLength(1);
    // The canonical cache was actually refreshed — not left saying "signed out".
    expect(server.calls.some((c) => c.url.includes("/api/auth/me"))).toBe(true);
  });

  it("1b. login works after a logout in the SAME session (the stale-cache case)", async () => {
    const server = makeServer();
    server.state.signedIn = true;
    install(server);

    // Signed in, then sign out — this writes "signed out" into the cache.
    const first = renderAppAt("/workspace");
    const button = await waitFor(() => {
      const el = document.querySelector<HTMLElement>('[data-testid="session-signout"]');
      expect(el).not.toBeNull();
      return el as HTMLElement;
    });
    fireEvent.click(button);
    await waitFor(() => expect(window.location.pathname).toBe("/signin"));
    first.unmount();

    // Sign in again WITHOUT reloading. Previously the stale cached "signed
    // out" made ProtectedRoute bounce this straight back to /signin.
    renderAppAt("/signin");
    await submitLogin();

    await waitFor(() => {
      expect(window.location.pathname).toBe("/dashboard");
    });
    await waitFor(() => {
      expect(chrome()).not.toBeNull();
    });
  });

  it("2. reloading while authenticated restores the session", async () => {
    const server = makeServer();
    server.state.signedIn = true; // cookie still valid
    install(server);

    // Fresh QueryClient == a reload: nothing cached, so /me must be asked.
    renderAppAt("/workspace");

    await waitFor(() => {
      expect(chrome()).not.toBeNull();
    });
    expect(window.location.pathname).toBe("/workspace");
    expect(screen.getByTestId("session-signout")).toBeTruthy();
  });

  it("3. a second app instance restores the same session from /api/auth/me", async () => {
    const server = makeServer();
    server.state.signedIn = true;
    install(server);

    // Tab 1.
    const tab1 = renderAppAt("/workspace");
    await waitFor(() => expect(chrome()).not.toBeNull());

    // Tab 2 — a brand-new QueryClient sharing only the server (the cookie jar).
    const tab2 = renderAppAt("/workspace");
    await waitFor(() => {
      expect(document.querySelectorAll('[data-testid="session-signout"]').length).toBeGreaterThan(0);
    });
    expect(window.location.pathname).toBe("/workspace");

    // Tab 2 never logged in: it restored purely from /api/auth/me.
    expect(server.logins()).toHaveLength(0);
    tab1.unmount();
    tab2.unmount();
  });

  it("4. logout clears cached queries and redirects immediately", async () => {
    const server = makeServer();
    server.state.signedIn = true;
    install(server);
    const { client } = renderAppAt("/workspace");

    fireEvent.click(await waitFor(() => {
      const el = document.querySelector<HTMLElement>('[data-testid="session-signout"]');
      expect(el).not.toBeNull();
      return el as HTMLElement;
    }));

    await waitFor(() => expect(window.location.pathname).toBe("/signin"));
    expect(client.getQueryData(SESSION_KEY)).toEqual({ authenticated: false, user: null });
    expect(chrome()).toBeNull();
  });

  it("5. a revoked or expired session does not restore authentication", async () => {
    const server = makeServer();
    server.state.signedIn = false; // revoked server-side
    install(server);

    renderAppAt("/workspace");

    await waitFor(() => expect(window.location.pathname).toBe("/signin"));
    expect(chrome()).toBeNull();
    expect(screen.queryByTestId("session-signout")).toBeNull();
  });

  it("6. a failed login shows the error and does not stick in a loading state", async () => {
    const server = makeServer();
    server.state.loginFails = true;
    install(server);
    renderAppAt("/signin");

    await submitLogin();

    // The server's own safe message, not a generic one.
    await waitFor(() => {
      expect(screen.getByText(/the email or password is incorrect/i)).toBeTruthy();
    });
    // Not stuck: the button is back to "Sign in", not "Signing in…".
    expect(screen.getByRole("button", { name: /^sign in$/i })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /signing in/i })).toBeNull();
    // Still on the sign-in screen.
    expect(window.location.pathname).toBe("/signin");
  });
});

describe("cross-tab session revalidation", () => {
  afterEach(() => {
    window.history.pushState({}, "", "/");
  });

  it("a tab that comes back into view re-checks the server", async () => {
    const server = makeServer();
    server.state.signedIn = true;
    install(server);
    renderAppAt("/workspace");

    await waitFor(() => expect(chrome()).not.toBeNull());
    const before = server.calls.filter((c) => c.url.includes("/api/auth/me")).length;
    expect(before).toBeGreaterThan(0);

    // The session ends elsewhere (another tab signed out).
    server.state.signedIn = false;

    // This tab becomes visible again.
    Object.defineProperty(document, "visibilityState", {
      value: "visible",
      configurable: true,
    });
    fireEvent(document, new Event("visibilitychange"));

    // It notices, instead of continuing to show authenticated content.
    await waitFor(() => expect(window.location.pathname).toBe("/signin"));
    expect(chrome()).toBeNull();
  });

  it("a still-valid session survives a visibility re-check", async () => {
    const server = makeServer();
    server.state.signedIn = true;
    install(server);
    renderAppAt("/workspace");

    await waitFor(() => expect(chrome()).not.toBeNull());

    Object.defineProperty(document, "visibilityState", {
      value: "visible",
      configurable: true,
    });
    fireEvent(document, new Event("visibilitychange"));

    // Re-checked, still valid, still here.
    await waitFor(() => {
      expect(server.calls.filter((c) => c.url.includes("/api/auth/me")).length).toBeGreaterThan(1);
    });
    expect(window.location.pathname).toBe("/workspace");
    expect(chrome()).not.toBeNull();
  });
});
