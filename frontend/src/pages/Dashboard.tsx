import { Navigate } from "react-router-dom";

/** The workstation is document-centric; the dashboard redirects to the workspace. */
export function Dashboard() {
  return <Navigate to="/workspace" replace />;
}
