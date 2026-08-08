import type { RetrievalMode } from "@/types";

export async function fetchRetrievalModes(): Promise<RetrievalMode[]> {
  return ["hybrid", "graph"];
}

export async function setRetrievalMode(mode: RetrievalMode): Promise<void> {
  // Mode is sent per-request; kept for API symmetry / future persistence.
  void mode;
}
