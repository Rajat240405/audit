/**
 * Chrome for the in-application routes (Dashboard, Workspace).
 *
 * This is the OTHER half of the split introduced in Phase 3: `MainLayout` plus
 * the global overlays and the product tour now live here, wrapped around an
 * `<Outlet />`, so they mount only for in-app routes — never for the
 * sign-in / sign-up / forgot-password screens, which render standalone.
 */
import { Suspense } from "react";
import { Outlet } from "react-router-dom";
import { MainLayout } from "./MainLayout";
import { BuildGraphModal } from "@/components/graph/BuildGraphModal";
import { ProductTour } from "@/components/onboarding/ProductTour";

// Lazy, matching App's route-level code splitting.
import { lazy } from "react";
const Settings = lazy(() =>
  import("@/pages/Settings").then((m) => ({ default: m.Settings }))
);
const MyKnowledgeModal = lazy(() =>
  import("@/components/knowledge/MyKnowledgeModal").then((m) => ({
    default: m.MyKnowledgeModal,
  }))
);

function Loader() {
  return (
    <div className="flex h-full w-full items-center justify-center text-xs text-muted">
      Loading…
    </div>
  );
}

export function AppShell() {
  return (
    <>
      <MainLayout>
        <Suspense fallback={<Loader />}>
          <Outlet />
        </Suspense>
      </MainLayout>
      <Settings />
      <MyKnowledgeModal />
      <BuildGraphModal />
      {/* Product tour — auto-starts on first run; scoped to the app shell so it
          never pops up over the sign-in screen. */}
      <ProductTour />
    </>
  );
}
