// Phase 3 — forgot password, via emailed one-time code.
//
// Two steps: request a code, then set a new password. Rendered standalone.
//
// The request step always reports the same thing ("if that address exists, a
// code is on its way") because the server replies identically either way — so
// this screen cannot be used to discover who has an account.
import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AuthAlert, AuthLayout, AuthLink } from "@/components/auth/AuthLayout";
import {
  AuthApiError,
  fetchPasswordPolicy,
  passwordLooksValid,
  requestPasswordReset,
  resetPassword,
  type PasswordPolicy,
} from "@/api/auth";

const STEP_LABELS = ["Your email", "Reset code", "New password"];

/** Conservative fallback so the UI still renders rules if the policy call fails. */
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

export function ForgotPassword() {
  const navigate = useNavigate();

  const [step, setStep] = useState(0);
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [policy, setPolicy] = useState<PasswordPolicy>(FALLBACK_POLICY);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

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

  const passwordOk = passwordLooksValid(password, policy);
  const passwordsMatch = password.length > 0 && password === confirmPassword;

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
    } finally {
      setBusy(false);
    }
  }, []);

  const handleRequest = () =>
    run(async () => {
      const result = await requestPasswordReset(email);
      setNotice(`If ${email.trim()} has an account, a reset code is on its way.`);
      if (result.dev_otp) {
        setNotice(
          `If ${email.trim()} has an account, a reset code is on its way. (Development mode — code ${result.dev_otp})`,
        );
      }
      setCode("");
      setStep(1);
    });

  const handleVerify = () =>
    run(async () => {
      // The code is checked by the server together with the new password, so
      // this step only advances the form.
      if (!code.trim()) {
        setError("Enter the code from the email.");
        return;
      }
      setStep(2);
      setNotice(null);
    });

  const handleReset = () =>
    run(async () => {
      if (password !== confirmPassword) {
        setError("The two passwords do not match.");
        return;
      }
      const result = await resetPassword(email, code, password);
      setNotice(
        `Password updated for ${result.email}. ${
          result.sessions_revoked > 0
            ? `${result.sessions_revoked} session(s) were signed out.`
            : "No other devices were signed in."
        }`,
      );
      setStep(3);
    });

  return (
    <AuthLayout
      title="Reset your password"
      subtitle={
        step === 3
          ? "Your password has been changed."
          : "Enter your email and we'll send a one-time code."
      }
      alert={
        <>
          <AuthAlert kind="error">{error}</AuthAlert>
          <AuthAlert kind="success">{notice}</AuthAlert>
        </>
      }
      footer={
        step === 3 ? (
          <button
            type="button"
            className="font-medium text-accent underline-offset-2 hover:underline"
            onClick={() => navigate("/signin")}
          >
            Back to sign in
          </button>
        ) : (
          <>
            Remembered it? <AuthLink to="/signin">Back to sign in</AuthLink>
          </>
        )
      }
    >
      {step < 3 && (
        <ol className="mb-5 flex items-center gap-2 text-[11px]" aria-label="Reset progress">
          {STEP_LABELS.map((label, i) => (
            <li key={label} className="flex flex-1 items-center gap-1.5">
              <span
                className={
                  i < step
                    ? "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-success/40 bg-success/15 text-[10px] text-success"
                    : i === step
                      ? "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-accent bg-accent text-[10px] text-accent-foreground"
                      : "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-border text-[10px] text-muted"
                }
                aria-current={i === step ? "step" : undefined}
              >
                {i < step ? "✓" : i + 1}
              </span>
              <span className={i === step ? "truncate text-foreground" : "truncate text-muted"}>
                {label}
              </span>
            </li>
          ))}
        </ol>
      )}

      {step === 0 && (
        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (!busy && email.trim()) handleRequest();
          }}
        >
          <label className="block space-y-1.5">
            <span className="text-xs font-medium">Email</span>
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
            {busy ? "Sending…" : "Send reset code"}
          </Button>
        </form>
      )}

      {step === 1 && (
        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (!busy) handleVerify();
          }}
        >
          <label className="block space-y-1.5">
            <span className="text-xs font-medium">Reset code</span>
            <Input
              type="text"
              name="otp"
              inputMode="numeric"
              autoComplete="one-time-code"
              autoFocus
              required
              maxLength={8}
              placeholder="6-digit code"
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
              disabled={busy}
            />
          </label>
          <p className="text-[11px] text-muted">
            Sent to {email.trim()}. The code expires and can be used once.
          </p>
          <Button type="submit" className="w-full" disabled={busy || !code.trim()}>
            Continue
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="w-full"
            disabled={busy}
            onClick={handleRequest}
          >
            Resend code
          </Button>
        </form>
      )}

      {step === 2 && (
        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (!busy && passwordOk && passwordsMatch) handleReset();
          }}
        >
          <label className="block space-y-1.5">
            <span className="text-xs font-medium">New password</span>
            <Input
              type="password"
              name="newPassword"
              autoComplete="new-password"
              autoFocus
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={busy}
            />
          </label>
          <label className="block space-y-1.5">
            <span className="text-xs font-medium">Confirm new password</span>
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
          <p className="text-[11px] text-muted">
            At least {policy.min_length} characters, with upper and lower case and a number.
            Setting a new password signs out every other device.
          </p>
          <Button type="submit" className="w-full" disabled={busy || !passwordOk || !passwordsMatch}>
            {busy ? "Updating…" : "Set new password"}
          </Button>
        </form>
      )}

      {step === 3 && (
        <div className="space-y-4">
          <p className="text-xs text-muted">
            You can now{" "}
            <Link to="/signin" className="font-medium text-accent underline-offset-2 hover:underline">
              sign in
            </Link>{" "}
            with your new password.
          </p>
          <Button className="w-full" onClick={() => navigate("/signin")}>
            Go to sign in
          </Button>
        </div>
      )}
    </AuthLayout>
  );
}

export default ForgotPassword;
