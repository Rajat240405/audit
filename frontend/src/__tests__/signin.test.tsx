/**
 * Phase 3: dedicated sign-in, sign-up and password-reset screens.
 *
 * The most important thing these tests pin down is the layout split — an auth
 * screen must render with no dashboard chrome around it. Every `AuthLayout`
 * route is checked against a list of things that only exist inside the
 * dashboard shell.
 *
 * Like the Phase 2 signup tests, `fetch` is the only thing mocked: the real
 * @/api/auth module and the real pages run.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "@/App";
import { SignIn } from "@/pages/SignIn";
import { ForgotPassword } from "@/pages/ForgotPassword";
import { AuthApiError, signOut } from "@/api/auth";

// ── helpers ────────────────────────────────────────────────────────────────

interface MockResponse {
  status?: number;
  body?: unknown;
  headers?: Record<string, string>;
}

type Handler = (url: string, payload: Record<string, unknown>) => MockResponse | null;

function installFetch(handler: Handler) {
  const calls: Array<{ url: string; payload: Record<string, unknown> }> = [];
  global.fetch = vi.fn(async (input: unknown, init?: RequestInit) => {
    const url = String(input);
    const payload = init?.body
      ? (JSON.parse(String(init.body)) as Record<string, unknown>)
      : {};
    calls.push({ url, payload });
    const result = handler(url, payload) ?? {};
    const status = result.status ?? 200;
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: "",
      json: async () => result.body ?? {},
      headers: { get: (key: string) => result.headers?.[key] ?? null },
    } as unknown as Response;
  }) as unknown as typeof fetch;
  return calls;
}

const LOGGED_IN = {
  authenticated: true,
  user: { id: "u1", email: "boss@incois.gov.in", email_verified: true, is_active: true },
};
const LOGGED_OUT = { authenticated: false, user: null };

/** Default handler: /me says signed out; everything else succeeds quietly. */
function baseHandler(url: string): MockResponse | null {
  if (url.includes("/api/auth/me")) return { body: LOGGED_OUT };
  if (url.includes("/api/auth/login"))
    return {
      body: { status: "logged_in", user_id: "u1", email: "boss@incois.gov.in", expires_in_seconds: 28800 },
    };
  if (url.includes("/api/auth/logout")) return { body: { status: "signed_out", revoked: true } };
  if (url.includes("/api/auth/password-policy"))
    return {
      body: {
        min_length: 12,
        max_length: 256,
        require_lowercase: true,
        require_uppercase: true,
        require_digit: true,
        require_symbol: false,
        rejects_email_substring: true,
        rejects_common_passwords: true,
      },
    };
  if (url.includes("/api/auth/password/forgot"))
    return {
      body: {
        status: "code_sent",
        email: "boss@incois.gov.in",
        expires_in_seconds: 600,
        resend_available_in_seconds: 60,
      },
    };
  if (url.includes("/api/auth/password/reset"))
    return { body: { status: "password_reset", email: "boss@incois.gov.in", sessions_revoked: 2 } };
  return { body: {} };
}

/**
 * As `baseHandler`, but `/api/auth/me` reports a valid session.
 *
 * Needed by tests that render in-app routes: those are gated now, so a handler
 * that reports "signed out" would bounce them to /signin before the thing under
 * test can render.
 */
function signedInHandler(url: string): MockResponse | null {
  if (url.includes("/api/auth/me")) return { body: LOGGED_IN };
  return baseHandler(url);
}

/** Render the whole app at a given path. */
function renderAppAt(path: string) {
  window.history.pushState({}, "", path);
  // QueryClientProvider lives in main.tsx, not App.tsx, so tests must supply it.
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}

function renderStandalone(node: React.ReactNode) {
  // SignIn now reads the session cache after a successful login, so it needs a
  // QueryClientProvider. main.tsx supplies one for the real app; standalone
  // renders have to supply their own.
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={client}>{node}</QueryClientProvider>
    </MemoryRouter>,
  );
}

/**
 * Things that exist ONLY inside the dashboard shell. If any of these appears
 * on an auth screen, the layout split has regressed.
 */
const DASHBOARD_CHROME = [
  "Build Graph", // header: graph build button
  "Model activity", // header: activity button title
  "My Saved Knowledge", // header: saved-knowledge button title
  "New session", // sidebar: new chat session
];

function expectNoDashboardChrome() {
  for (const marker of DASHBOARD_CHROME) {
    expect(screen.queryByText(marker, { exact: false })).toBeNull();
  }
  // No tour anchors, no header region, no sidebar.
  expect(document.querySelector("[data-tour^='header-']")).toBeNull();
  expect(document.querySelector("header")).toBeNull();
  expect(document.querySelector("aside")).toBeNull();
}

beforeEach(() => {
  // Sign-in and forgot-password both consider it acceptable for the policy
  // call to fail, so log the rejection noise rather than failing on it.
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ═══════════════════════════════════════════════════════════════════════════
// Layout isolation — the critical UI requirement
// ═══════════════════════════════════════════════════════════════════════════

describe("auth pages render independently of the dashboard layout", () => {
  it("/ renders sign in with no dashboard chrome", async () => {
    installFetch(baseHandler);
    renderAppAt("/");

    expect(await screen.findByRole("button", { name: /^sign in$/i })).toBeTruthy();
    expect(screen.getByLabelText("Email", { selector: "input" })).toBeTruthy();
    expect(screen.getByLabelText("Password", { selector: "input" })).toBeTruthy();
    expectNoDashboardChrome();
  });

  it("/signin renders sign in with no dashboard chrome", async () => {
    installFetch(baseHandler);
    renderAppAt("/signin");

    expect(await screen.findByRole("button", { name: /^sign in$/i })).toBeTruthy();
    expectNoDashboardChrome();
  });

  it("/signup renders the signup screen with no dashboard chrome", async () => {
    installFetch(baseHandler);
    renderAppAt("/signup");

    expect(await screen.findByRole("button", { name: /send verification code/i })).toBeTruthy();
    expectNoDashboardChrome();
  });

  it("/forgot-password renders the reset screen with no dashboard chrome", async () => {
    installFetch(baseHandler);
    renderAppAt("/forgot-password");

    expect(await screen.findByRole("button", { name: /send reset code/i })).toBeTruthy();
    expectNoDashboardChrome();
  });

  it("the dashboard route DOES get its chrome (the split is real, not a global removal)", async () => {
    // Signed in: /dashboard is gated, so a signed-out handler would redirect
    // to /signin and this test would be checking the redirect, not the chrome.
    installFetch(signedInHandler);
    renderAppAt("/dashboard");

    // The header renders once the lazy chunk resolves.
    await waitFor(() => {
      expect(document.querySelector("[data-tour^='header-']")).not.toBeNull();
    });
    expect(screen.getByText("Build Graph")).toBeTruthy();
    expect(document.querySelector("header")).not.toBeNull();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Sign in
// ═══════════════════════════════════════════════════════════════════════════

describe("sign in", () => {
  it("posts the email and password to /api/auth/login", async () => {
    const calls = installFetch(baseHandler);
    renderStandalone(<SignIn />);

    fireEvent.change(screen.getByLabelText("Email", { selector: "input" }), {
      target: { value: "boss@incois.gov.in" },
    });
    fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
      target: { value: "Tide-Quiet-77-K9" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^sign in$/i }));

    await waitFor(() => {
      const login = calls.find((c) => c.url.includes("/api/auth/login"));
      expect(login).toBeTruthy();
      expect(login!.payload).toEqual({
        email: "boss@incois.gov.in",
        password: "Tide-Quiet-77-K9",
      });
    });
  });

  it("shows the server's message when the credentials are rejected", async () => {
    installFetch((url) => {
      if (url.includes("/api/auth/login"))
        return {
          status: 401,
          body: {
            detail: {
              code: "invalid_credentials",
              message: "That email and password don't match.",
            },
          },
        };
      return baseHandler(url);
    });
    renderStandalone(<SignIn />);

    fireEvent.change(screen.getByLabelText("Email", { selector: "input" }), {
      target: { value: "boss@incois.gov.in" },
    });
    fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
      target: { value: "wrong" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^sign in$/i }));

    expect(
      await screen.findByText("That email and password don't match."),
    ).toBeTruthy();
    // Still on the sign-in form, retryable.
    expect(screen.getByRole("button", { name: /^sign in$/i })).toBeTruthy();
  });

  it("cannot be submitted while either field is empty", async () => {
    const calls = installFetch(baseHandler);
    renderStandalone(<SignIn />);

    const submit = screen.getByRole("button", { name: /^sign in$/i });
    expect((submit as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(screen.getByLabelText("Email", { selector: "input" }), {
      target: { value: "boss@incois.gov.in" },
    });
    expect((submit as HTMLButtonElement).disabled).toBe(true); // no password yet

    fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
      target: { value: "Tide-Quiet-77-K9" },
    });
    expect((submit as HTMLButtonElement).disabled).toBe(false);

    expect(calls.filter((c) => c.url.includes("/api/auth/login")).length).toBe(0);
  });

  it("links to create an account and to password reset", async () => {
    installFetch(baseHandler);
    renderStandalone(<SignIn />);

    expect(screen.getByRole("link", { name: /create an account/i })).toHaveProperty(
      "getAttribute('href')" in screen ? ("href" as never) : ("href" as never),
    );
    const createAccount = screen
      .getAllByRole("link")
      .find((a) => /create an account/i.test(a.textContent ?? ""));
    expect(createAccount?.getAttribute("href")).toBe("/signup");

    const forgot = screen
      .getAllByRole("link")
      .find((a) => /forgot your password/i.test(a.textContent ?? ""));
    expect(forgot?.getAttribute("href")).toBe("/forgot-password");
  });

  it("sends no request until the form is submitted", async () => {
    const calls = installFetch(baseHandler);
    renderStandalone(<SignIn />);
    await waitFor(() => expect(calls.length).toBeGreaterThanOrEqual(0));
    expect(calls.filter((c) => c.url.includes("/api/auth/login")).length).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Forgot password
// ═══════════════════════════════════════════════════════════════════════════

describe("forgot password", () => {
  it("walks request → code → new password, calling each endpoint", async () => {
    const calls = installFetch(baseHandler);
    renderStandalone(<ForgotPassword />);

    // 1. request a code
    fireEvent.change(screen.getByLabelText("Email", { selector: "input" }), {
      target: { value: "boss@incois.gov.in" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send reset code/i }));

    expect(await screen.findByLabelText("Reset code", { selector: "input" })).toBeTruthy();

    // 2. enter the code
    fireEvent.change(screen.getByLabelText("Reset code", { selector: "input" }), {
      target: { value: "481902" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^continue$/i }));

    // 3. set a new password
    fireEvent.change(await screen.findByLabelText("New password", { selector: "input" }), {
      target: { value: "Tide-Quiet-77-K9" },
    });
    fireEvent.change(screen.getByLabelText("Confirm new password", { selector: "input" }), {
      target: { value: "Tide-Quiet-77-K9" },
    });
    fireEvent.click(screen.getByRole("button", { name: /set new password/i }));

    await waitFor(() => {
      const reset = calls.find((c) => c.url.includes("/api/auth/password/reset"));
      expect(reset).toBeTruthy();
      expect(reset!.payload).toEqual({
        email: "boss@incois.gov.in",
        otp: "481902",
        new_password: "Tide-Quiet-77-K9",
      });
    });

    // Confirmation, and a route back to sign in.
    expect(await screen.findByText(/go to sign in/i)).toBeTruthy();
    expect(await screen.findByText(/Password updated for boss@incois.gov.in/)).toBeTruthy();
  });

  it("reports the same thing whether or not the address has an account", async () => {
    // The server answers identically either way, so the screen must too —
    // it must not become an account-enumeration oracle.
    installFetch((url) => {
      if (url.includes("/api/auth/password/forgot"))
        return {
          body: {
            status: "code_sent",
            email: "nobody@incois.gov.in",
            expires_in_seconds: 600,
            resend_available_in_seconds: 60,
          },
        };
      return baseHandler(url);
    });
    renderStandalone(<ForgotPassword />);

    fireEvent.change(screen.getByLabelText("Email", { selector: "input" }), {
      target: { value: "nobody@incois.gov.in" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send reset code/i }));

    expect(
      await screen.findByText(/If nobody@incois.gov.in has an account, a reset code is on its way/),
    ).toBeTruthy();
    // Still advances, so there is no observable difference.
    expect(screen.getByLabelText("Reset code", { selector: "input" })).toBeTruthy();
  });

  it("surfaces a wrong-code rejection and stays on the code step", async () => {
    installFetch((url) => {
      if (url.includes("/api/auth/password/reset"))
        return {
          status: 400,
          body: {
            detail: { code: "invalid_otp", message: "That code has expired." },
          },
        };
      return baseHandler(url);
    });
    renderStandalone(<ForgotPassword />);

    fireEvent.change(screen.getByLabelText("Email", { selector: "input" }), {
      target: { value: "boss@incois.gov.in" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send reset code/i }));

    fireEvent.change(await screen.findByLabelText("Reset code", { selector: "input" }), {
      target: { value: "000000" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^continue$/i }));

    fireEvent.change(await screen.findByLabelText("New password", { selector: "input" }), {
      target: { value: "Tide-Quiet-77-K9" },
    });
    fireEvent.change(screen.getByLabelText("Confirm new password", { selector: "input" }), {
      target: { value: "Tide-Quiet-77-K9" },
    });
    fireEvent.click(screen.getByRole("button", { name: /set new password/i }));

    expect(await screen.findByText("That code has expired.")).toBeTruthy();
  });

  it("blocks submission until the two passwords match", async () => {
    installFetch(baseHandler);
    renderStandalone(<ForgotPassword />);

    fireEvent.change(screen.getByLabelText("Email", { selector: "input" }), {
      target: { value: "boss@incois.gov.in" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send reset code/i }));

    fireEvent.change(await screen.findByLabelText("Reset code", { selector: "input" }), {
      target: { value: "481902" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^continue$/i }));

    fireEvent.change(await screen.findByLabelText("New password", { selector: "input" }), {
      target: { value: "Tide-Quiet-77-K9" },
    });
    fireEvent.change(screen.getByLabelText("Confirm new password", { selector: "input" }), {
      target: { value: "Tide-Quiet-77-K8" },
    });

    expect(screen.getByText("The two passwords do not match.")).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: /set new password/i }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("links back to sign in", async () => {
    installFetch(baseHandler);
    renderStandalone(<ForgotPassword />);

    const back = screen
      .getAllByRole("link")
      .find((a) => /back to sign in/i.test(a.textContent ?? ""));
    expect(back?.getAttribute("href")).toBe("/signin");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Sign out
// ═══════════════════════════════════════════════════════════════════════════

describe("sign out", () => {
  it("calls POST /api/auth/logout", async () => {
    const calls = installFetch(baseHandler);
    await signOut();
    expect(calls.some((c) => c.url.includes("/api/auth/logout"))).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Session lookup
// ═══════════════════════════════════════════════════════════════════════════

describe("session lookup", () => {
  it("treats a 401 as signed out instead of throwing", async () => {
    installFetch(() => ({ status: 401, body: { detail: { code: "not_authenticated" } } }));
    const { fetchSession } = await import("@/api/auth");
    await expect(fetchSession()).resolves.toEqual({ authenticated: false, user: null });
  });

  it("surfaces the signed-in user when there is a session", async () => {
    installFetch(() => ({ body: LOGGED_IN }));
    const { fetchSession } = await import("@/api/auth");
    const state = await fetchSession();
    expect(state.authenticated).toBe(true);
    expect(state.user?.email).toBe("boss@incois.gov.in");
  });

  it("throws (rather than reporting signed out) when the server is broken", async () => {
    installFetch(() => ({ status: 500, body: {} }));
    const { fetchSession } = await import("@/api/auth");
    await expect(fetchSession()).rejects.toBeInstanceOf(AuthApiError);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Phase boundary — Phase 3 signs you in, it does not lock anything down
// ═══════════════════════════════════════════════════════════════════════════

describe("phase boundary", () => {
  // Inverted: this used to assert the opposite ("gating is Phase 4"), which
  // pinned the bug — a signed-out visitor could open an in-app route directly
  // and see a working-looking screen. In-app routes are gated now.
  it("in-app routes redirect to /signin while signed out", async () => {
    installFetch(baseHandler);
    renderAppAt("/dashboard");

    await waitFor(() => {
      expect(window.location.pathname).toBe("/signin");
    });
    // No dashboard chrome may reach a signed-out visitor.
    expect(document.querySelector("[data-tour^='header-']")).toBeNull();
  });

  it("the sign-in screen offers no role, permission or admin control", async () => {
    installFetch(baseHandler);
    renderStandalone(<SignIn />);

    await waitFor(() => expect(screen.getByRole("button", { name: /^sign in$/i })).toBeTruthy());
    expect(screen.queryByText(/role/i)).toBeNull();
    expect(screen.queryByText(/permission/i)).toBeNull();
    expect(screen.queryByText(/admin/i)).toBeNull();
  });
});
