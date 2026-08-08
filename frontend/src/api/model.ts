import { apiFetch } from "./client";
import type { ExecutionMode, ServerStatus } from "@/types";

export interface ProviderInfo {
  name: string;
  label?: string;
}

export interface ModelFamily {
  id: string;
  display_name: string;
  model_name: string;
  context_window: number;
  thinking_capable: boolean;
  recommended_execution_mode?: string;
}

export async function fetchProviders(): Promise<ProviderInfo[]> {
  const data = await apiFetch<unknown>("/api/providers");
  // The backend may return {providers: [...]} or a bare list — normalize.
  if (Array.isArray(data)) return data as ProviderInfo[];
  return ((data as { providers?: ProviderInfo[] })?.providers ?? []) as ProviderInfo[];
}

export async function fetchModels(provider: string): Promise<ModelFamily[]> {
  const data = await apiFetch<unknown>(`/api/models?provider=${encodeURIComponent(provider)}`);
  if (Array.isArray(data)) return data as ModelFamily[];
  return ((data as { models?: ModelFamily[] })?.models ?? []) as ModelFamily[];
}

export async function setProvider(provider: string, model: string, apiKey?: string): Promise<void> {
  await apiFetch("/api/provider", {
    method: "POST",
    body: JSON.stringify({ provider, model, api_key: apiKey ?? null }),
  });
}

export async function fetchStatus(): Promise<ServerStatus> {
  const data = await apiFetch<ServerStatus>("/api/status");
  return data;
}

export async function setExecutionMode(mode: ExecutionMode): Promise<void> {
  // Mode is passed per-request; kept for API symmetry.
  void mode;
}
