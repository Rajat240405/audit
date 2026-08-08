import { BrowserRouter, Route, Routes } from "react-router-dom";
import { MainLayout } from "@/components/layout/MainLayout";
import { Workspace } from "@/pages/Workspace";
import { Dashboard } from "@/pages/Dashboard";
import { Settings } from "@/pages/Settings";
import { BuildGraphModal } from "@/components/graph/BuildGraphModal";
import { useBackendStatus } from "@/hooks/useBackendStatus";

// Note: QueryClientProvider lives in main.tsx so useQuery hooks (e.g.
// useBackendStatus) are always inside a provider context.
export function App() {
  useBackendStatus();
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/workspace" element={<Workspace />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </MainLayout>
      <Settings />
      <BuildGraphModal />
    </BrowserRouter>
  );
}
