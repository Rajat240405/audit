/**
 * Gate for every in-application route.
 *
 * Rendered as a layout route, so it wraps Dashboard, Workspace and the
 * catch-all — a nested or newly added in-app route cannot sidestep it by
 * simply not knowing about it.
 *
 * Three rules, in order:
 *
 *   1. While the session lookup is in flight, show a loading state. Never
 *      protected content: "we have not asked yet" is not "yes".
 *   2. Signed out -> redirect to sign-in, `replace`d so the Back button cannot
 *      return here.
 *   3. Otherwise render the child route.
 *
 * This is presentation-layer routing only. It keeps signed-out visitors from
 * looking at a screen that cannot work; it is not what stops them reading
 * data. The backend enforces that, and every protected endpoint returns 401
 * without a valid session.
 */

import { Navigate, Outlet } from "react-router-dom";
import { useSession } from "@/hooks/useAuthSession";

export function ProtectedRoute() {
  const { status } = useSession();

  if (status === "loading") {
    return (
      <div
        data-testid="auth-loading"
        role="status"
        aria-live="polite"
        className="flex h-screen w-full items-center justify-center text-xs text-muted"
      >
        Checking your session…
      </div>
    );
  }

  if (status === "unauthenticated") {
    // `replace` so the protected URL is not left in history: Back would
    // otherwise land the user on a page they are no longer allowed to see.
    return <Navigate to="/signin" replace />;
  }

  return <Outlet />;
}

export default ProtectedRoute;
