// Phase 3 — sign in.
//
// Rendered standalone (AuthLayout), never inside the dashboard shell. The
// default entry point: the app opens here at "/".
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AuthAlert, AuthLayout, AuthLink } from "@/components/auth/AuthLayout";
import { useQueryClient } from "@tanstack/react-query";
import { AuthApiError, signIn } from "@/api/auth";
import { primeSession } from "@/hooks/useAuthSession";

export function SignIn() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = !busy && email.trim().length > 0 && password.length > 0;

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    try {
      await signIn(email, password);
      // The cookie is set now, but the cached session still holds whatever it
      // held before — after a logout that is "signed out". Re-read it from the
      // server BEFORE navigating, or ProtectedRoute renders against the stale
      // answer and bounces the user straight back here. (This is why signing
      // in used to appear to do nothing until a hard refresh.)
      await primeSession(queryClient);
      navigate("/dashboard");
    } catch (err) {
      // The server already returns one generic message for every credential
      // failure, so there is nothing to disambiguate here (deliberately).
      // The form keeps what was typed: only the error and the busy flag move.
      setError(
        err instanceof AuthApiError ? err.message : "Sign in failed. Please try again.",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthLayout
      title="Sign in"
      subtitle="Sign in to continue to INCOIS Audit Pro."
      alert={<AuthAlert kind="error">{error}</AuthAlert>}
      footer={
        <>
          Don&apos;t have an account? <AuthLink to="/signup">Create an account</AuthLink>
        </>
      }
    >
      <form className="space-y-4" onSubmit={handleSubmit} noValidate>
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

        <label className="block space-y-1.5">
          <span className="text-xs font-medium">Password</span>
          <Input
            type="password"
            name="password"
            autoComplete="current-password"
            required
            placeholder="Your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={busy}
          />
        </label>

        <div className="text-right">
          <AuthLink to="/forgot-password">Forgot your password?</AuthLink>
        </div>

        <Button type="submit" className="w-full" disabled={!canSubmit}>
          {busy ? "Signing in…" : "Sign in"}
        </Button>
      </form>
    </AuthLayout>
  );
}

export default SignIn;
