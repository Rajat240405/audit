import { apiFetch } from "./client";
import type { GraphBuildStatus } from "@/types";

/**
 * GraphRAG build progress.
 *
 * The build is started on the backend/HPC (`python -m src.graphrag.cli build`,
 * usually detached via nohup) — NOT from the UI. Progress is published by the
 * build process into the GraphRAG checkpoint file and read back by this
 * endpoint, so the backend is always the source of truth and the reported
 * state survives a browser refresh or an app restart.
 */
export async function fetchGraphBuildStatus(): Promise<GraphBuildStatus> {
  return apiFetch<GraphBuildStatus>("/api/graph/build-status");
}
