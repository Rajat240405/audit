import { BrowserRouter, Route, Routes } from "react-router-dom";
import { lazy, Suspense } from "react";

const Dashboard = lazy(() => import("@/pages/Dashboard").then((m) => ({ default: m.Dashboard })));
const Workspace = lazy(() => import("@/pages/Workspace").then((m) => ({ default: m.Workspace })));
const Signup = lazy(() => import("@/pages/Signup").then((m) => ({ default: m.Signup })));
const SignIn = lazy(() => import("@/pages/SignIn").then((m) => ({ default: m.SignIn })));
const ForgotPassword = lazy(() =>
  import("@/pages/ForgotPassword").then((m) => ({ default: m.ForgotPassword }))
);

import { AppShell } from "@/components/layout/AppShell";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { useSessionCacheBridge, useSessionRevalidation } from "@/hooks/useAuthSession";

function Loader() {
  return (
    <div className="flex h-screen w-full items-center justify-center text-xs text-muted">
      Loading…
    </div>
  );
}

/**
 * Phase 3 entry behaviour:
 *
 *   /                 -> sign in (the default entry point)
 *   /signin           -> sign in
 *   /signup           -> create account
 *   /forgot-password  -> password reset by emailed code
 *   /dashboard        -> the existing Dashboard
 *   /workspace        -> the existing Workspace
 *
 * The three auth routes sit OUTSIDE `AppShell`, so they render with no
 * dashboard chrome: no Header, no Sidebar, no toolbar, no Build-Graph button,
 * no chat interface, no product tour. `AppShell` (MainLayout + global
 * overlays) wraps only the in-app routes via `<Outlet />`.
 *
 * In-app routes are gated by `ProtectedRoute`: without a valid session they
 * redirect to /signin instead of rendering. Note this is a UI convenience —
 * Phase 3 does not gate the API itself, so the enforcement only begins with
 * Phase 4. What this DOES fix is a signed-out visitor being shown a screen that
 * cannot work (every call behind it answers 401).
 */
export function App() {
  return (
    <BrowserRouter>
      <SessionBridge />
      <Suspense fallback={<Loader />}>
        <Routes>
          {/* Standalone auth routes — deliberately no layout wrapper. */}
          <Route path="/" element={<SignIn />} />
          <Route path="/signin" element={<SignIn />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />

          {/* In-app routes — these get the dashboard chrome.
              ProtectedRoute wraps the group rather than each child, so the
              catch-all and any future in-app route are gated too. */}
          <Route element={<ProtectedRoute />}>
            <Route element={<AppShell />}>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/workspace" element={<Workspace />} />
              <Route path="*" element={<Dashboard />} />
            </Route>
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;

/**
 * Tiny component so the bridge hook can sit inside the router.
 * Hooks cannot be called conditionally, so it is its own component.
 */
function SessionBridge() {
  // A session that ends between calls must drop cached data, so the next
  // account to sign in on this tab cannot read the previous one's.
  useSessionCacheBridge();
  // Sessions are shared by every tab on this origin (the cookie is per-profile,
  // not per-tab). Re-check the server when this tab comes back into view, so a
  // logout in another tab is noticed instead of being shown stale content.
  useSessionRevalidation();
  return null;
}
