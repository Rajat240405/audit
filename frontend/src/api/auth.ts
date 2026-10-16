// Phase 2 — signup and email OTP verification.
//
// Deliberately absent (later phases):
//   * login / logout            (Phase 3)
//   * password reset / recovery (Phase 3)
//   * sessions, roles, guards   (Phase 4)
//
// So there is nothing here to call on app start-up: an unauthenticated visitor
// is the normal state until Phase 4 protects routes.
//
// These endpoints return a STRUCTURED error body — {"detail": {"code": ...,
// "message": ...}} — so they use their own fetch rather than the shared
// apiFetch, which flattens `detail` into a message string. The precedent for a
// module doing its own fetch is uploadDocument() in ./model.ts.

export type AuthErrorCode =
  | "email_required"
  | "email_invalid"
  | "email_not_allowed"
  | "otp_invalid"
  | "otp_expired"
  | "otp_attempts_exceeded"
  | "otp_already_used"
  | "otp_not_found"
  | "otp_resend_cooldown"
  | "token_invalid"
  | "token_expired"
  | "password_invalid"
  | "account_exists"
  | "rate_limited"
  | "mailer_failed"
  | string;

/** An auth failure carrying the server's machine-readable code. */
export class AuthApiError extends Error {
  code: AuthErrorCode;
  status: number;
  /** Seconds to wait, when the server sent Retry-After (429s). */
  retryAfter: number | null;

  constructor(message: string, code: AuthErrorCode, status: number, retryAfter: number | null = null) {
    super(message);
    this.name = "AuthApiError";
    this.code = code;
    this.status = status;
    this.retryAfter = retryAfter;
  }

  /** True when the server rejected the address itself, not the code. */
  get isEmailRejected(): boolean {
    return this.code === "email_invalid" || this.code === "email_not_allowed";
  }
}

export interface PasswordPolicy {
  min_length: number;
  max_length: number;
  require_lowercase: boolean;
  require_uppercase: boolean;
  require_digit: boolean;
  require_symbol: boolean;
  rejects_email_substring: boolean;
  rejects_common_passwords: boolean;
}

export interface SignupStartResult {
  status: string;
  email: string;
  expires_in_seconds: number;
  resend_available_in_seconds: number;
  /**
   * Present ONLY when the backend runs with AUTH_DEV_RETURN_OTP. The server
   * refuses to enable that switch in production, so this is never populated
   * against a real deployment — it exists purely for local development.
   */
  dev_otp?: string;
}

export interface SignupVerifyResult {
  status: string;
  email: string;
  signup_token: string;
  expires_in_seconds: number;
}

export interface SignupCompleteResult {
  status: string;
  user_id: string;
  email: string;
  email_verified: boolean;
}

async function authRequest<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    // The session lives in an HttpOnly cookie, so every auth call must carry
    // credentials. `same-origin` is the browser default; stating it stops a
    // later refactor from silently dropping the cookie — which presents as
    // "login works, but a new tab is signed out".
    credentials: "same-origin",
  });

  const payload = await res.json().catch(() => null);

  if (!res.ok) {
    const detail = payload?.detail;
    // The backend sends {detail: {code, message}}; tolerate a plain string too.
    const message =
      (typeof detail === "object" && detail !== null && "message" in detail
        ? String(detail.message)
        : null) ??
      (typeof detail === "string" ? detail : null) ??
      res.statusText;
    const code =
      typeof detail === "object" && detail !== null && "code" in detail
        ? String(detail.code)
        : "unknown";
    const rawRetry = res.headers.get("Retry-After");
    const retryAfter = rawRetry !== null ? Number.parseInt(rawRetry, 10) : null;
    throw new AuthApiError(
      message,
      code,
      res.status,
      Number.isFinite(retryAfter as number) ? retryAfter : null,
    );
  }

  return payload as T;
}

/** Step 1 — ask the server to email a verification code. */
export function startSignup(email: string): Promise<SignupStartResult> {
  return authRequest<SignupStartResult>("/api/auth/signup/start", { email });
}

/** Step 1b — send a fresh code (subject to the server's resend cooldown). */
export function resendSignupOtp(email: string): Promise<SignupStartResult> {
  return authRequest<SignupStartResult>("/api/auth/signup/resend", { email });
}

/** Step 2 — verify the code; returns a short-lived token to set a password. */
export function verifySignupOtp(email: string, otp: string): Promise<SignupVerifyResult> {
  return authRequest<SignupVerifyResult>("/api/auth/signup/verify", { email, otp });
}

/** Step 3 — set the password and create the verified account. */
export function completeSignup(signupToken: string, password: string): Promise<SignupCompleteResult> {
  return authRequest<SignupCompleteResult>("/api/auth/signup/complete", {
    signup_token: signupToken,
    password,
  });
}

/** The authoritative password rules, rendered by the UI rather than duplicated. */
export async function fetchPasswordPolicy(): Promise<PasswordPolicy> {
  const res = await fetch("/api/auth/password-policy");
  if (!res.ok) {
    throw new AuthApiError("Could not load the password rules", "policy_unavailable", res.status);
  }
  return (await res.json()) as PasswordPolicy;
}

/**
 * Client-side mirror of the server's policy, used only to give instant feedback
 * and to disable the submit button. The SERVER is the real gate — this is a
 * convenience, never a security control.
 */
export function describePolicy(policy: PasswordPolicy): string[] {
  const rules: string[] = [`At least ${policy.min_length} characters`];
  if (policy.require_lowercase) rules.push("A lowercase letter");
  if (policy.require_uppercase) rules.push("An uppercase letter");
  if (policy.require_digit) rules.push("A number");
  if (policy.require_symbol) rules.push("A symbol");
  if (policy.rejects_email_substring) rules.push("Must not contain your email address");
  if (policy.rejects_common_passwords) rules.push("Not a commonly used password");
  return rules;
}

/** Cheap pre-check so the user is not told by a round-trip. Server still validates. */
export function passwordLooksValid(password: string, policy: PasswordPolicy): boolean {
  if (password.length < policy.min_length || password.length > policy.max_length) return false;
  if (policy.require_lowercase && !/[a-z]/.test(password)) return false;
  if (policy.require_uppercase && !/[A-Z]/.test(password)) return false;
  if (policy.require_digit && !/[0-9]/.test(password)) return false;
  if (policy.require_symbol && !/[^A-Za-z0-9]/.test(password)) return false;
  return true;
}


// ═══════════════════════════════════════════════════════════════════════════
// Phase 3 — sign in, sign out, session lookup, password reset
//
// The session lives in an HttpOnly cookie set by the server. Browser code
// never reads or stores the session token — it just asks the server who it is.
// ═══════════════════════════════════════════════════════════════════════════

export interface SessionUser {
  id: string;
  email: string;
  email_verified: boolean;
  is_active: boolean;
}

export interface SessionState {
  authenticated: boolean;
  user: SessionUser | null;
}

export interface LoginResult {
  status: string;
  user_id: string;
  email: string;
  expires_in_seconds: number;
}

export interface LogoutResult {
  status: string;
  revoked: boolean;
}

export interface PasswordForgotResult {
  status: string;
  email: string;
  expires_in_seconds: number;
  resend_available_in_seconds: number;
  /** Dev backends only (AUTH_DEV_RETURN_OTP); never in production. */
  dev_otp?: string;
}

export interface PasswordResetResult {
  status: string;
  email: string;
  sessions_revoked: number;
}

const SIGNED_OUT: SessionState = { authenticated: false, user: null };

/** Sign in. Sets the session cookie via the response. */
export async function signIn(email: string, password: string): Promise<LoginResult> {
  return authRequest<LoginResult>("/api/auth/login", { email, password });
}

/** Sign out. Clears the cookie server-side, whether or not a session existed. */
export async function signOut(): Promise<LogoutResult> {
  return authRequest<LogoutResult>("/api/auth/logout", {});
}

/**
 * Who am I? Returns `authenticated: false` rather than throwing when there is
 * no valid session, so callers can render without a try/catch. Any other
 * error (server down, 500) still throws.
 */
export async function fetchSession(signal?: AbortSignal): Promise<SessionState> {
  const res = await fetch("/api/auth/me", {
    headers: { Accept: "application/json" },
    // The session is the cookie. Restating `same-origin` keeps the requirement
    // visible at the point that matters most: the call that decides whether you
    // are signed in.
    credentials: "same-origin",
    // Lets a login/logout cancel an in-flight lookup, so its (now stale)
    // answer can never overwrite the newer state.
    signal,
  });
  if (res.status === 401) return SIGNED_OUT;
  if (!res.ok) {
    throw new AuthApiError("Could not load the session", "session_unavailable", res.status);
  }
  const body = (await res.json()) as { authenticated?: boolean; user?: SessionUser };
  return { authenticated: Boolean(body.authenticated), user: body.user ?? null };
}

/**
 * Ask for a password-reset code. The response is identical whether or not the
 * address has an account, so this cannot be used to discover who is registered.
 */
export async function requestPasswordReset(email: string): Promise<PasswordForgotResult> {
  return authRequest<PasswordForgotResult>("/api/auth/password/forgot", { email });
}

/** Verify the reset code and set a new password. Signs out every device. */
export async function resetPassword(
  email: string,
  otp: string,
  newPassword: string,
): Promise<PasswordResetResult> {
  return authRequest<PasswordResetResult>("/api/auth/password/reset", {
    email,
    otp,
    new_password: newPassword,
  });
}
