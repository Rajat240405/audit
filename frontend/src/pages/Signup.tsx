// Phase 3 — create an account (Phase 2 signup flow, re-shelled).
//
// The OTP flow itself is unchanged from Phase 2: enter email → enter the
// emailed code → set a password. What changed in Phase 3 is the presentation:
// this screen now renders standalone via AuthLayout, so no dashboard navbar,
// sidebar, toolbar or graph control appears around it, and it cross-links to
// sign in.
//
// The account is still only created at the very end, after the code is
// verified, so an abandoned signup leaves nothing behind.
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AuthAlert, AuthLayout, AuthLink } from "@/components/auth/AuthLayout";
import {
  AuthApiError,
  completeSignup,
  describePolicy,
  fetchPasswordPolicy,
  passwordLooksValid,
  resendSignupOtp,
  startSignup,
  verifySignupOtp,
  type PasswordPolicy,
} from "@/api/auth";

type Step = "email" | "otp" | "password" | "done";

const STEP_ORDER: Step[] = ["email", "otp", "password"];
const STEP_LABELS: Record<string, string> = {
  email: "Your email",
  otp: "Verify code",
  password: "Set password",
};

/** A conservative fallback so the UI still renders rules if the policy call fails. */
const FALLBACK_POLICY: PasswordPolicy = {
  min_length: 12,
  max_length: 256,
  require_lowercase: true,
  require_uppercase: true,
  require_digit: true,
  require_symbol: false,
  rejects_email_substring: true,
  rejects_common_passwords: true,
};

export function Signup() {
  const navigate = useNavigate();

  const [step, setStep] = useState<Step>("email");
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [signupToken, setSignupToken] = useState("");
  const [verifiedEmail, setVerifiedEmail] = useState("");

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [devOtp, setDevOtp] = useState<string | null>(null);

  const [policy, setPolicy] = useState<PasswordPolicy>(FALLBACK_POLICY);
  const [cooldown, setCooldown] = useState(0);

  // The real rules come from the server so the UI can never drift from them.
  useEffect(() => {
    let active = true;
    fetchPasswordPolicy()
      .then((p) => {
        if (active) setPolicy(p);
      })
      .catch(() => {
        /* keep the fallback rules; the server still enforces the real ones */
      });
    return () => {
      active = false;
    };
  }, []);

  // Resend cooldown countdown — cosmetic; the server enforces its own.
  useEffect(() => {
    if (cooldown <= 0) return;
    const id = setTimeout(() => setCooldown((c) => Math.max(0, c - 1)), 1000);
    return () => clearTimeout(id);
  }, [cooldown]);

  const policyRules = useMemo(() => describePolicy(policy), [policy]);
  const passwordOk = passwordLooksValid(password, policy);
  const passwordsMatch = password.length > 0 && password === confirmPassword;
  const stepIndex = STEP_ORDER.indexOf(step);

  const run = useCallback(async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await fn();
    } catch (err) {
      setError(
        err instanceof AuthApiError ? err.message : "Something went wrong. Please try again.",
      );
      if (err instanceof AuthApiError && err.retryAfter !== null) {
        setCooldown(err.retryAfter);
      }
    } finally {
      setBusy(false);
    }
  }, []);

  const handleStart = () =>
    run(async () => {
      const result = await startSignup(email);
      setVerifiedEmail(result.email);
      // Only ever populated on a dev backend (AUTH_DEV_RETURN_OTP, which the
      // server refuses to enable in production).
      setDevOtp(result.dev_otp ?? null);
      setCooldown(result.resend_available_in_seconds);
      setOtp("");
      setStep("otp");
      setNotice(`We sent a verification code to ${result.email}.`);
    });

  const handleResend = () =>
    run(async () => {
      const result = await resendSignupOtp(email);
      setDevOtp(result.dev_otp ?? null);
      setCooldown(result.resend_available_in_seconds);
      setOtp("");
      setNotice(`A new code is on its way to ${result.email}.`);
    });

  const handleVerify = () =>
    run(async () => {
      const result = await verifySignupOtp(email, otp);
      setSignupToken(result.signup_token);
      setStep("password");
      setNotice("Email verified. Choose a password to finish.");
    });

  const handleComplete = () =>
    run(async () => {
      if (password !== confirmPassword) {
        setError("The two passwords do not match.");
        return;
      }
      const result = await completeSignup(signupToken, password);
      setVerifiedEmail(result.email);
      setStep("done");
      setNotice(null);
    });

  return (
    <AuthLayout
      title={step === "done" ? "Account created" : "Create your account"}
      subtitle={
        step === "done"
          ? "Your email is verified and your password is set."
          : "Verify your email with a one-time code, then choose a password."
      }
      alert={
        <>
          <AuthAlert kind="error">{error}</AuthAlert>
          <AuthAlert kind="success">{notice}</AuthAlert>
        </>
      }
      footer={
        step === "done" ? null : (
          <>
            Already have an account? <AuthLink to="/signin">Sign in</AuthLink>
          </>
        )
      }
    >
      {step === "done" ? (
        <div className="space-y-4" data-testid="signup-done">
          <div className="rounded-md border border-success/30 bg-success/10 px-3 py-3 text-xs text-success">
            <p>
              <strong className="text-foreground">{verifiedEmail}</strong> is verified and
              ready to use.
            </p>
          </div>
          <Button className="w-full" onClick={() => navigate("/signin")}>
            Continue to sign in
          </Button>
          <p className="text-center text-[11px] text-muted">
            <Link
              to="/signin"
              className="font-medium text-accent underline-offset-2 hover:underline"
            >
              Go to sign in
            </Link>
          </p>
        </div>
      ) : (
        <>
          <ol className="mb-5 flex items-center gap-2 text-[11px]" aria-label="Signup progress">
            {STEP_ORDER.map((name, i) => (
              <li key={name} className="flex flex-1 items-center gap-1.5">
                <span
                  className={
                    i < stepIndex
                      ? "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-success/40 bg-success/15 text-[10px] text-success"
                      : i === stepIndex
                        ? "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-accent bg-accent text-[10px] text-accent-foreground"
                        : "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-border text-[10px] text-muted"
                  }
                  aria-current={i === stepIndex ? "step" : undefined}
                >
                  {i < stepIndex ? "✓" : i + 1}
                </span>
                <span
                  className={i === stepIndex ? "truncate text-foreground" : "truncate text-muted"}
                >
                  {STEP_LABELS[name]}
                </span>
              </li>
            ))}
          </ol>

          {step === "email" && (
            <form
              className="space-y-4"
              onSubmit={(e) => {
                e.preventDefault();
                if (!busy && email.trim()) handleStart();
              }}
            >
              <label className="block space-y-1.5">
                <span className="text-xs font-medium">Work email</span>
                <Input
                  type="email"
                  name="email"
                  autoComplete="email"
                  autoFocus
                  required
                  placeholder="you@incois.gov.in"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={busy}
                />
              </label>
              <Button type="submit" className="w-full" disabled={busy || !email.trim()}>
                {busy ? "Sending…" : "Send verification code"}
              </Button>
            </form>
          )}

          {step === "otp" && (
            <form
              className="space-y-4"
              onSubmit={(e) => {
                e.preventDefault();
                if (!busy && otp.trim()) handleVerify();
              }}
            >
              <label className="block space-y-1.5">
                <span className="text-xs font-medium">Verification code</span>
                <Input
                  type="text"
                  name="otp"
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  autoFocus
                  required
                  maxLength={8}
                  placeholder="6-digit code"
                  value={otp}
                  onChange={(e) => setOtp(e.target.value.replace(/\D/g, ""))}
                  disabled={busy}
                />
              </label>
              <p className="text-[11px] text-muted">
                Sent to {verifiedEmail || email}. The code expires and can only be used once.
              </p>

              {devOtp && (
                <p className="rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-[11px] text-warning">
                  Development mode — the server returned the code directly:{" "}
                  <strong>{devOtp}</strong>
                </p>
              )}

              <Button type="submit" className="w-full" disabled={busy || !otp.trim()}>
                {busy ? "Verifying…" : "Verify code"}
              </Button>

              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="flex-1"
                  disabled={busy || cooldown > 0}
                  onClick={handleResend}
                >
                  {cooldown > 0 ? `Resend available in ${cooldown}s` : "Resend code"}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="flex-1"
                  disabled={busy}
                  onClick={() => {
                    setStep("email");
                    setError(null);
                    setNotice(null);
                    setDevOtp(null);
                    setCooldown(0);
                  }}
                >
                  Change email
                </Button>
              </div>
            </form>
          )}

          {step === "password" && (
            <form
              className="space-y-4"
              onSubmit={(e) => {
                e.preventDefault();
                if (!busy && passwordOk && passwordsMatch) handleComplete();
              }}
            >
              <label className="block space-y-1.5">
                <span className="text-xs font-medium">Password</span>
                <Input
                  type="password"
                  name="password"
                  autoComplete="new-password"
                  autoFocus
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={busy}
                />
              </label>
              <label className="block space-y-1.5">
                <span className="text-xs font-medium">Confirm password</span>
                <Input
                  type="password"
                  name="confirmPassword"
                  autoComplete="new-password"
                  required
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  disabled={busy}
                />
              </label>

              {confirmPassword.length > 0 && !passwordsMatch && (
                <p className="text-[11px] text-danger">The two passwords do not match.</p>
              )}

              <ul className="space-y-0.5 text-[11px] text-muted">
                {policyRules.map((rule) => (
                  <li key={rule}>• {rule}</li>
                ))}
              </ul>

              <Button
                type="submit"
                className="w-full"
                disabled={busy || !passwordOk || !passwordsMatch}
              >
                {busy ? "Creating account…" : "Create account"}
              </Button>
            </form>
          )}
        </>
      )}
    </AuthLayout>
  );
}

export default Signup;
