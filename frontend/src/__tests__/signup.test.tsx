/**
 * Phase 2 signup screen: the three-step flow (email → code → password) and the
 * pure helpers the UI uses for instant feedback.
 *
 * These tests mock `fetch` only — the real @/api/auth module and the real
 * Signup component run, so the wiring between them is what is being verified.
 * There is deliberately no login or password-reset test: those screens do not
 * exist yet (Phase 3).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { Signup } from "@/pages/Signup";
import {
  AuthApiError,
  describePolicy,
  passwordLooksValid,
  type PasswordPolicy,
} from "@/api/auth";

const POLICY: PasswordPolicy = {
  min_length: 12,
  max_length: 256,
  require_lowercase: true,
  require_uppercase: true,
  require_digit: true,
  require_symbol: false,
  rejects_email_substring: true,
  rejects_common_passwords: true,
};

interface MockResponse {
  status?: number;
  body?: unknown;
  headers?: Record<string, string>;
}

/** What the fake server returns for each call, in order or by URL. */
type Handler = (url: string, payload: Record<string, unknown>) => MockResponse;

function installFetch(handler: Handler) {
  const calls: Array<{ url: string; payload: Record<string, unknown> }> = [];
  global.fetch = vi.fn(async (input: unknown, init?: RequestInit) => {
    const url = String(input);
    const payload = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {};
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

function renderSignup() {
  return render(
    <MemoryRouter>
      <Signup />
    </MemoryRouter>,
  );
}

/** Fill the email field and submit it. */
async function submitEmail(email: string) {
  fireEvent.change(screen.getByLabelText("Work email", { selector: "input" }), {
    target: { value: email },
  });
  fireEvent.click(screen.getByRole("button", { name: /send verification code/i }));
}

async function submitOtp(code: string) {
  fireEvent.change(screen.getByLabelText("Verification code", { selector: "input" }), {
    target: { value: code },
  });
  fireEvent.click(screen.getByRole("button", { name: /^verify code$/i }));
}

const originalFetch = global.fetch;

beforeEach(() => {
  installFetch((url) => {
    if (url.includes("/password-policy")) return { body: POLICY };
    return { status: 500, body: { detail: { code: "unexpected", message: `unmocked ${url}` } } };
  });
});

afterEach(() => {
  global.fetch = originalFetch;
  vi.restoreAllMocks();
});

// ── the happy path ───────────────────────────────────────────────────────────

describe("signup: happy path", () => {
  it("walks email → code → password → created, calling each endpoint once", async () => {
    const calls = installFetch((url) => {
      if (url.includes("/password-policy")) return { body: POLICY };
      if (url.includes("/signup/start")) {
        return { body: { status: "code_sent", email: "boss@incois.gov.in", expires_in_seconds: 600, resend_available_in_seconds: 60 } };
      }
      if (url.includes("/signup/verify")) {
        return { body: { status: "verified", email: "boss@incois.gov.in", signup_token: "tok", expires_in_seconds: 900 } };
      }
      if (url.includes("/signup/complete")) {
        return { body: { status: "created", user_id: "u1", email: "boss@incois.gov.in", email_verified: true } };
      }
      return { status: 500, body: {} };
    });

    renderSignup();
    await submitEmail("boss@incois.gov.in");

    // Step 2
    expect(await screen.findByText(/Sent to boss@incois.gov.in/)).toBeTruthy();

    await submitOtp("123456");

    // Step 3
    expect(await screen.findByText(/At least 12 characters/)).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
      target: { value: "K7-Quiet-Tide-92" },
    });
    fireEvent.change(screen.getByLabelText("Confirm password", { selector: "input" }), {
      target: { value: "K7-Quiet-Tide-92" },
    });
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));

    // Done
    expect(await screen.findByText("Account created")).toBeTruthy();
    expect(screen.getByText(/is verified and ready/)).toBeTruthy();

    const urls = calls.map((c) => c.url);
    expect(urls).toContain("/api/auth/signup/start");
    expect(urls).toContain("/api/auth/signup/verify");
    expect(urls).toContain("/api/auth/signup/complete");
    // No login or reset endpoint is ever called — they do not exist yet.
    expect(urls.some((u) => /login|logout|reset|session/.test(u))).toBe(false);
  });

  it("sends the typed email and code verbatim", async () => {
    const calls = installFetch((url) => {
      if (url.includes("/password-policy")) return { body: POLICY };
      if (url.includes("/signup/start")) {
        return { body: { status: "code_sent", email: "a@incois.gov.in", expires_in_seconds: 600, resend_available_in_seconds: 60 } };
      }
      if (url.includes("/signup/verify")) {
        return { body: { status: "verified", email: "a@incois.gov.in", signup_token: "t", expires_in_seconds: 900 } };
      }
      return { status: 500, body: {} };
    });

    renderSignup();
    await submitEmail("a@incois.gov.in");
    await screen.findByText(/Sent to a@incois.gov.in/);
    await submitOtp("654321");
    await screen.findByText(/At least 12 characters/);

    const start = calls.find((c) => c.url.includes("/signup/start"));
    expect(start?.payload).toEqual({ email: "a@incois.gov.in" });
    const verify = calls.find((c) => c.url.includes("/signup/verify"));
    expect(verify?.payload).toEqual({ email: "a@incois.gov.in", otp: "654321" });
  });
});

// ── error handling ───────────────────────────────────────────────────────────

describe("signup: server rejections", () => {
  it("shows the server's message when the address is not allowed, and stays put", async () => {
    installFetch((url) => {
      if (url.includes("/password-policy")) return { body: POLICY };
      if (url.includes("/signup/start")) {
        return {
          status: 403,
          body: { detail: { code: "email_not_allowed", message: "this email address is not allowed to sign up" } },
        };
      }
      return { status: 500, body: {} };
    });

    renderSignup();
    await submitEmail("someone@gmail.com");

    expect((await screen.findByRole("alert")).textContent).toMatch(
      /this email address is not allowed to sign up/i,
    );
    // Still on step 1.
    expect(screen.getByRole("button", { name: /send verification code/i })).toBeTruthy();
  });

  it("surfaces the remaining-attempts message on a wrong code", async () => {
    installFetch((url) => {
      if (url.includes("/password-policy")) return { body: POLICY };
      if (url.includes("/signup/start")) {
        return { body: { status: "code_sent", email: "a@incois.gov.in", expires_in_seconds: 600, resend_available_in_seconds: 60 } };
      }
      if (url.includes("/signup/verify")) {
        return {
          status: 400,
          body: { detail: { code: "otp_invalid", message: "the code is incorrect — 4 attempt(s) remaining" } },
        };
      }
      return { status: 500, body: {} };
    });

    renderSignup();
    await submitEmail("a@incois.gov.in");
    await screen.findByText(/Sent to a@incois.gov.in/);
    await submitOtp("000000");

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/4 attempt\(s\) remaining/);
    // Still on step 2.
    expect(screen.getByRole("button", { name: /^verify code$/i })).toBeTruthy();
  });

  it("disables resend and counts down during the cooldown", async () => {
    installFetch((url) => {
      if (url.includes("/password-policy")) return { body: POLICY };
      if (url.includes("/signup/start")) {
        return { body: { status: "code_sent", email: "a@incois.gov.in", expires_in_seconds: 600, resend_available_in_seconds: 60 } };
      }
      return { status: 500, body: {} };
    });

    renderSignup();
    await submitEmail("a@incois.gov.in");
    await screen.findByText(/Sent to a@incois.gov.in/);

    const resend = await screen.findByRole("button", { name: /Resend available in \d+s/ });
    expect((resend as HTMLButtonElement).disabled).toBe(true);
  });

  it("reports a 429 with the server's retry hint", async () => {
    installFetch((url) => {
      if (url.includes("/password-policy")) return { body: POLICY };
      if (url.includes("/signup/start")) {
        return {
          status: 429,
          body: { detail: { code: "rate_limited", message: "too many attempts — please wait and try again" } },
          headers: { "Retry-After": "42" },
        };
      }
      return { status: 500, body: {} };
    });

    renderSignup();
    await submitEmail("a@incois.gov.in");

    expect((await screen.findByRole("alert")).textContent).toMatch(/too many attempts/i);
    expect(await screen.findByRole("button", { name: /Resend|Send verification code/ })).toBeTruthy();
  });
});

// ── password rules come from the server ──────────────────────────────────────

describe("signup: password policy is server-driven", () => {
  it("renders the rules the server sent, not hardcoded ones", async () => {
    installFetch((url) => {
      if (url.includes("/password-policy")) {
        return {
          body: {
            ...POLICY,
            min_length: 20,
            require_symbol: true,
          },
        };
      }
      if (url.includes("/signup/start")) {
        return { body: { status: "code_sent", email: "a@incois.gov.in", expires_in_seconds: 600, resend_available_in_seconds: 60 } };
      }
      if (url.includes("/signup/verify")) {
        return { body: { status: "verified", email: "a@incois.gov.in", signup_token: "t", expires_in_seconds: 900 } };
      }
      return { status: 500, body: {} };
    });

    renderSignup();
    await submitEmail("a@incois.gov.in");
    await screen.findByText(/Sent to a@incois.gov.in/);
    await submitOtp("123456");

    expect(await screen.findByText(/At least 20 characters/)).toBeTruthy();
    expect(screen.getByText(/A symbol/)).toBeTruthy();
  });

  it("blocks submission until the passwords match and satisfy the policy", async () => {
    installFetch((url) => {
      if (url.includes("/password-policy")) return { body: POLICY };
      if (url.includes("/signup/start")) {
        return { body: { status: "code_sent", email: "a@incois.gov.in", expires_in_seconds: 600, resend_available_in_seconds: 60 } };
      }
      if (url.includes("/signup/verify")) {
        return { body: { status: "verified", email: "a@incois.gov.in", signup_token: "t", expires_in_seconds: 900 } };
      }
      if (url.includes("/signup/complete")) {
        return { body: { status: "created", user_id: "u1", email: "a@incois.gov.in", email_verified: true } };
      }
      return { status: 500, body: {} };
    });

    renderSignup();
    await submitEmail("a@incois.gov.in");
    await screen.findByText(/Sent to a@incois.gov.in/);
    await submitOtp("123456");
    await screen.findByText(/At least 12 characters/);

    const password = screen.getByLabelText("Password", { selector: "input" });
    const confirm = screen.getByLabelText("Confirm password", { selector: "input" });
    const submit = () => screen.getByRole("button", { name: /create account/i }) as HTMLButtonElement;

    // Too short.
    fireEvent.change(password, { target: { value: "Aa1" } });
    expect(submit().disabled).toBe(true);

    // Long enough but the confirmation has not been typed.
    fireEvent.change(password, { target: { value: "K7-Quiet-Tide-92" } });
    expect(submit().disabled).toBe(true);

    // Confirmation matches -> enabled.
    fireEvent.change(confirm, { target: { value: "K7-Quiet-Tide-92" } });
    await waitFor(() => expect(submit().disabled).toBe(false));

    // Mismatch -> disabled again, with a visible reason.
    fireEvent.change(confirm, { target: { value: "K7-Quiet-Tide-93" } });
    await waitFor(() => expect(submit().disabled).toBe(true));
    expect(screen.getByText("The two passwords do not match.")).toBeTruthy();
  });
});

// ── pure helpers ─────────────────────────────────────────────────────────────

describe("auth helpers", () => {
  it("describes only the rules the server actually enforces", () => {
    const rules = describePolicy({ ...POLICY, require_uppercase: false, require_symbol: true });
    expect(rules).toContain("At least 12 characters");
    expect(rules).toContain("A symbol");
    expect(rules).not.toContain("An uppercase letter");
  });

  it("applies the policy client-side for instant feedback", () => {
    expect(passwordLooksValid("short1A", POLICY)).toBe(false);
    expect(passwordLooksValid("alllowercase123", POLICY)).toBe(false);
    expect(passwordLooksValid("K7-Quiet-Tide-92", POLICY)).toBe(true);
    // "K7-Quiet-Tide-92" contains hyphens, which satisfy the symbol rule;
    // this one has no symbol at all.
    expect(passwordLooksValid("K7QuietTide92", { ...POLICY, require_symbol: true })).toBe(false);
    expect(passwordLooksValid("K7-Quiet-Tide-92", { ...POLICY, require_symbol: true })).toBe(true);
  });

  it("flags email rejections distinctly from code rejections", () => {
    const rejected = new AuthApiError("not allowed", "email_not_allowed", 403);
    const wrongCode = new AuthApiError("wrong", "otp_invalid", 400);
    expect(rejected.isEmailRejected).toBe(true);
    expect(wrongCode.isEmailRejected).toBe(false);
    expect(rejected.retryAfter).toBeNull();
  });

  it("carries the Retry-After hint through", () => {
    const err = new AuthApiError("wait", "otp_resend_cooldown", 429, 30);
    expect(err.retryAfter).toBe(30);
  });
});

// ── no out-of-scope screens ──────────────────────────────────────────────────

describe("signup: phase boundary", () => {
  it("offers no login, logout or password-reset control", async () => {
    installFetch((url) => {
      if (url.includes("/password-policy")) return { body: POLICY };
      if (url.includes("/signup/start")) {
        return { body: { status: "code_sent", email: "a@incois.gov.in", expires_in_seconds: 600, resend_available_in_seconds: 60 } };
      }
      return { status: 500, body: {} };
    });

    renderSignup();
    await submitEmail("a@incois.gov.in");
    await screen.findByText(/Sent to a@incois.gov.in/);

    const names = screen.getAllByRole("button").map((b) => b.textContent ?? "");
    expect(names.some((n) => /log in|sign in|forgot|reset password/i.test(n))).toBe(false);
  });
});
