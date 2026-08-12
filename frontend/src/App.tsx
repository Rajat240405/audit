import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { MainLayout } from "@/components/layout/MainLayout";
import { BuildGraphModal } from "@/components/graph/BuildGraphModal";
import { useBackendStatus } from "@/hooks/useBackendStatus";

// Route-level code splitting (M7): each page loads on demand, so the initial
// bundle is much smaller. Suspense fallback is a lightweight loader.
const Workspace = lazy(() =>
  import("@/pages/Workspace").then((m) => ({ default: m.Workspace }))
);
const Dashboard = lazy(() =>
  import("@/pages/Dashboard").then((m) => ({ default: m.Dashboard }))
);
const Settings = lazy(() =>
  import("@/pages/Settings").then((m) => ({ default: m.Settings }))
);

function Loader() {
  return (
    <div className="flex h-full w-full items-center justify-center text-xs text-muted">
      Loading…
    </div>
  );
}

// Note: QueryClientProvider lives in main.tsx so useQuery hooks (e.g.
// useBackendStatus) are always inside a provider context.
export function App() {
  useBackendStatus();
  return (
    <BrowserRouter>
      <MainLayout>
        <Suspense fallback={<Loader />}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/workspace" element={<Workspace />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </Suspense>
      </MainLayout>
      <Settings />
      <BuildGraphModal />
    </BrowserRouter>
  );
}
