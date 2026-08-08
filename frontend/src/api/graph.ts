import { apiFetch } from "./client";
import type { GraphBuildStatus } from "@/types";

export async function fetchGraphBuildStatus(): Promise<GraphBuildStatus> {
  return apiFetch<GraphBuildStatus>("/api/graph/build-status");
}

/** Trigger a GraphRAG build via the legacy graph CLI path (fire-and-forget). */
export async function startGraphBuild(): Promise<void> {
  // Placeholder: the production GraphRAG build runs through `graphrag build`
  // on the backend; a dedicated endpoint can be wired here later.
  return Promise.resolve();
}
