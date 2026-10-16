/**
 * Regression tests for the two authentication bugs.
 *
 *   1. Sign-out took two clicks. The handler cleared the cached session and
 *      stopped there — it never navigated. The first click only swapped the
 *      header's log-out icon for a sign-in link; the second click was following
 *      that link. Nothing was waiting on the response, so no second request was
 *      involved.
 *
 *   2. In-app routes rendered for signed-out visitors. There was no route
 *      guard at all, so opening /workspace from the address bar (or reloading
 *      it) painted a working-looking screen whose every API call answered 401.
 *
 * `fetch` is the only thing mocked: the real App, router, pages and hooks run.
 */
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import App from "@/App";
import { SESSION_KEY, useSignOut } from "@/hooks/useAuthSession";
import { useToastStore } from "@/store/useToastStore";

// ── a stand-in auth backend whose session state can be flipped ──────────────

interface MockResponse {
  status?: number;
  body?: unknown;
  headers?: Record<string, string>;
}

const LOGGED_IN = {
  authenticated: true,
  user: { id: "u1", email: "boss@incois.gov.in", email_verified: true, is_active: true, role: "user", is_admin: false },
};

const NOT_AUTHENTICATED = {
  code: "not_authenticated",
  message: "you are not signed in",
};

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
 * The important part is `state.signedIn`: a successful logout flips it, so the
 * app has to cope with the server genuinely revoking the session rather than
 * just being told to forget it locally.
 */
function makeServer() {
  const state = { signedIn: true, logoutFails: false };
  const calls: Array<{ url: string; method: string }> = [];

  const handler = (url: string, init?: RequestInit): MockResponse => {
    const method = init?.method ?? "GET";
    calls.push({ url, method });

    if (url.includes("/api/auth/logout")) {
      if (state.logoutFails) {
        return { status: 500, body: { detail: { code: "server_error", message: "boom" } } };
      }
      // Server-side revocation: /me will now answer 401.
      state.signedIn = false;
      return { body: { status: "signed_out", revoked: true } };
    }
    if (url.includes("/api/auth/me")) {
      if (!state.signedIn) return { status: 401, body: { detail: NOT_AUTHENTICATED } };
      return { body: LOGGED_IN };
    }
    if (url.includes("/api/auth/password-policy")) return { body: POLICY };
    return { body: {} };
  };

  const logouts = () => calls.filter((c) => c.url.includes("/api/auth/logout"));
  return { state, calls, handler, logouts };
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
      headers: { get: (key: string) => result.headers?.[key] ?? null },
    } as unknown as Response;
  }) as unknown as typeof fetch;
}

/**
 * Render the app at a path with a fresh QueryClient — a fresh client is what a
 * page reload looks like: nothing cached, so the session has to be re-checked.
 */
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

/** Present only inside the dashboard shell — the app's protected chrome. */
const chrome = () => document.querySelector("[data-tour^='header-']");

async function waitForAppReady(): Promise<HTMLElement> {
  return waitFor(() => {
    const btn = document.querySelector<HTMLElement>('[data-testid="session-signout"]');
    expect(btn).not.toBeNull();
    return btn as HTMLElement;
  });
}

// ── tests ──────────────────────────────────────────────────────────────────

describe("sign out", () => {
  beforeEach(() => {
    useToastStore.setState({ toasts: [] });
  });

  afterEach(() => {
    window.history.pushState({}, "", "/");
    useToastStore.setState({ toasts: [] });
  });

  it("lands on /signin after a SINGLE click", async () => {
    const server = makeServer();
    install(server);
    renderAppAt("/workspace");

    const button = await waitForAppReady();

    fireEvent.click(button); // one click, and only one

    await waitFor(() => {
      expect(window.location.pathname).toBe("/signin");
    });

    // Exactly one request: the bug's second click never reached the server, it
    // was following a link. If navigation came from a second interaction, the
    // endpoint would have been hit twice.
    expect(server.logouts()).toHaveLength(1);
    // Protected chrome is gone.
    expect(chrome()).toBeNull();
  });

  it("clears the cached session so no screen can render from stale state", async () => {
    const server = makeServer();
    install(server);
    const { client } = renderAppAt("/workspace");

    fireEvent.click(await waitForAppReady());

    await waitFor(() => {
      expect(client.getQueryData(SESSION_KEY)).toEqual({ authenticated: false, user: null });
    });
  });

  it("blocks direct navigation to /workspace after signing out", async () => {
    const server = makeServer();
    install(server);
    renderAppAt("/workspace");

    fireEvent.click(await waitForAppReady());
    await waitFor(() => expect(window.location.pathname).toBe("/signin"));

    // Sign out again from a clean mount, then walk straight to /workspace as if
    // it had been typed into the address bar.
    renderAppAt("/workspace");

    await waitFor(() => expect(window.location.pathname).toBe("/signin"));
    expect(chrome()).toBeNull();
    expect(document.querySelector('[data-testid="session-signout"]')).toBeNull();
  });

  it("surfaces an error and stays put when the server refuses the logout", async () => {
    const server = makeServer();
    server.state.logoutFails = true;
    install(server);
    renderAppAt("/workspace");

    fireEvent.click(await waitForAppReady());

    // The user is told, rather than left in an ambiguous state.
    await waitFor(() => {
      expect(screen.getByText(/could not sign you out/i)).toBeTruthy();
    });

    // Not silently signed out: still on the protected page, and the session is
    // still valid server-side.
    expect(window.location.pathname).toBe("/workspace");
    expect(server.state.signedIn).toBe(true);
  });
});

/**
 * The sign-out handler tested on its own, with no ProtectedRoute in the tree.
 *
 * This is the test that pins Bug 1 specifically. The full-app test above would
 * also pass if only the route guard were fixed — with the session cache cleared,
 * the guard redirects on the next render, so the symptom disappears even if the
 * handler never navigates. Relying on that alone would leave sign-out depending
 * on a re-render rather than doing what the user asked, so the handler is held
 * to its own behaviour here.
 */
describe("useSignOut, independent of the route guard", () => {
  function LocationProbe() {
    const location = useLocation();
    return <span data-testid="loc">{location.pathname}</span>;
  }

  function SignOutProbe() {
    const { signOut } = useSignOut();
    const [result, setResult] = useState("pending");
    return (
      <button
        data-testid="probe"
        onClick={async () => setResult((await signOut()) ? "ok" : "failed")}
      >
        {result}
      </button>
    );
  }

  function renderProbe() {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <MemoryRouter initialEntries={["/workspace"]}>
        <QueryClientProvider client={client}>
          <LocationProbe />
          <SignOutProbe />
        </QueryClientProvider>
      </MemoryRouter>,
    );
  }

  it("navigates to /signin by itself, not via a re-render of the guard", async () => {
    const server = makeServer();
    install(server);
    renderProbe();

    expect(screen.getByTestId("loc").textContent).toBe("/workspace");

    fireEvent.click(screen.getByTestId("probe"));

    await waitFor(() => {
      expect(screen.getByTestId("loc").textContent).toBe("/signin");
    });
    await waitFor(() => {
      expect(screen.getByTestId("probe").textContent).toBe("ok");
    });
    expect(server.logouts()).toHaveLength(1);
  });

  it("reports failure and does not navigate when the server refuses", async () => {
    const server = makeServer();
    server.state.logoutFails = true;
    install(server);
    renderProbe();

    fireEvent.click(screen.getByTestId("probe"));

    await waitFor(() => {
      expect(screen.getByTestId("probe").textContent).toBe("failed");
    });
    // No navigation: the session is still live server-side.
    expect(screen.getByTestId("loc").textContent).toBe("/workspace");
  });
});

describe("protected routes", () => {
  afterEach(() => {
    window.history.pushState({}, "", "/");
  });

  it("shows a loading state, then redirects, when /workspace is reloaded while signed out", async () => {
    const server = makeServer();
    server.state.signedIn = false; // the session is gone
    install(server);

    // Fresh QueryClient == a reload: nothing cached, so the app must ask.
    renderAppAt("/workspace");

    // While it is asking: a loading state, never protected content.
    expect(screen.getByTestId("auth-loading")).toBeTruthy();
    expect(chrome()).toBeNull();

    await waitFor(() => expect(window.location.pathname).toBe("/signin"));
    expect(chrome()).toBeNull();
  });

  it("never paints protected content for a signed-out visitor", async () => {
    const server = makeServer();
    server.state.signedIn = false;
    install(server);

    renderAppAt("/workspace");

    await waitFor(() => expect(window.location.pathname).toBe("/signin"));
    expect(chrome()).toBeNull();
    expect(screen.queryByTestId("session-signout")).toBeNull();
  });

  it("redirects the catch-all route too, not just the named ones", async () => {
    const server = makeServer();
    server.state.signedIn = false;
    install(server);

    renderAppAt("/some/unknown/path");

    await waitFor(() => expect(window.location.pathname).toBe("/signin"));
    expect(chrome()).toBeNull();
  });

  it("lets a signed-in user open /workspace normally", async () => {
    const server = makeServer();
    install(server);
    renderAppAt("/workspace");

    await waitFor(() => {
      expect(chrome()).not.toBeNull();
    });

    // No bounce: still where the user asked to be.
    expect(window.location.pathname).toBe("/workspace");
    expect(screen.getByTestId("session-signout")).toBeTruthy();
  });

  it("replaces the history entry so Back cannot return to the protected page", async () => {
    const server = makeServer();
    server.state.signedIn = false;
    install(server);

    window.history.pushState({}, "", "/signin");
    window.history.pushState({}, "", "/workspace");

    renderAppAt("/workspace");
    await waitFor(() => expect(window.location.pathname).toBe("/signin"));

    // The /workspace entry was replaced, so going back leaves it behind.
    window.history.back();
    await waitFor(() => {
      expect(window.location.pathname).not.toBe("/workspace");
    });
  });
});
