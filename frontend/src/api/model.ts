import { apiFetch } from "./client";
import type { ExecutionMode, ServerStatus, SourceItem } from "@/types";

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

export interface IngestStatus {
  running: boolean;
  pending: number;
  last: {
    at: string;
    ok: number;
    failed: number;
    records: number;
    message: string;
  } | null;
  inbox: string;
}

export async function fetchIngestStatus(): Promise<IngestStatus> {
  return apiFetch<IngestStatus>("/api/ingest/status");
}

export async function triggerIngest(): Promise<{ status: string }> {
  return apiFetch("/api/ingest", {
    method: "POST",
    body: JSON.stringify({ source: "inbox" }),
  });
}

export async function uploadDocument(file: File): Promise<{ status: string; file: string; size: number }> {
  const res = await fetch(`/api/upload?filename=${encodeURIComponent(file.name)}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: file,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Upload failed (HTTP ${res.status})`);
  }
  return res.json();
}

export async function saveKnowledge(payload: {
  question: string;
  answer: string;
  sources?: SourceItem[];
}): Promise<{ status: string; file: string }> {
  return apiFetch("/api/save-knowledge", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function lookupKnowledge(q: string): Promise<{
  found: boolean;
  answer?: string;
  sources?: SourceItem[];
  question?: string;
  matched?: string;
}> {
  return apiFetch(`/api/knowledge-lookup?q=${encodeURIComponent(q)}`);
}

export interface SourceOrg {
  slug: string;
  name: string;
  count: number;
  categories: string[];
}

export interface SourceMinistry {
  name: string;
  count: number;
  orgs: SourceOrg[];
}

export interface SourceCatalogue {
  tree: Record<string, SourceMinistry>;
  types: Array<{ type: string; count: number }>;
  categories: Array<{ category: string; count: number }>;
  total: number;
}

export async function fetchSources(): Promise<SourceCatalogue> {
  return apiFetch<SourceCatalogue>("/api/sources");
}
