import { apiFetch } from "./client";
import type { ExecutionMode, ServerStatus, SourceItem } from "@/types";

export interface ProviderInfo {
  name: string;
  label?: string;
  active?: boolean;
}

export interface ModelFamily {
  id: string;
  display_name: string;
  model_name: string;
  context_window: number;
  thinking_capable: boolean;
  recommended_execution_mode?: string;
  provider?: string;
  think_mode?: string;
  /** true when the entry was discovered from the running server (always true
   * for the discovery-driven /api/models endpoint) */
  served?: boolean;
  /** where the metadata came from: catalog | server | fallback (assumed) */
  metadata_source?: string;
  /** tri-state capability: true/false known, null = unknown (dynamic model —
   * never claimed thinking-capable; no thinking control is sent on the wire) */
  thinking_supported?: boolean | null;
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

export interface IngestFileVerdict {
  name: string;
  verdict: "new" | "duplicate" | "failed" | "skipped_duplicate_pdf" | string;
  records?: number;
  message?: string;
}

export interface IngestStatus {
  running: boolean;
  pending: number;
  /** Phase 3: targeted uploads staged into the hierarchy, awaiting ingest */
  pending_uploads?: number;
  staged_uploads?: string[];
  last: {
    at: string;
    ok: number;
    failed: number;
    records: number;
    message: string;
    /** additive Phase-3 fields (present on runs that processed uploads) */
    received?: number;
    new_documents?: number;
    duplicates?: number;
    failed_documents?: number;
    records_added?: number;
    records_embedded?: number;
    files?: IngestFileVerdict[];
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

// ── Phase 3: hierarchical ingest targets (Ministry → Org → Document type) ──
// EVERYTHING below is server-discovered (GET /api/ingest/targets reads
// config/sources.yaml + data/ tree) — the frontend hardcodes no orgs/types.

export interface IngestCategoryTarget {
  document_type: string;
  label: string;
  category_dir: string;
  path: string;
  exists: boolean;
  files: number;
  file_names: string[];
  truncated: boolean;
}

export interface IngestOrgTarget {
  slug: string;
  label: string;
  dir: string | null;
  categories: IngestCategoryTarget[];
}

export interface IngestSourceTarget {
  name: string;
  label: string;
  description?: string;
  hierarchical: boolean;
  discovered?: boolean;
  ministry?: string | null;
  /** whether documents may be uploaded into this source from the UI */
  upload?: boolean;
  orgs: IngestOrgTarget[];
}

export interface IngestTargets {
  version: number;
  category_map: Record<string, string>;
  document_types: string[];
  data_root: string;
  sources: IngestSourceTarget[];
}

export async function fetchIngestTargets(): Promise<IngestTargets> {
  return apiFetch<IngestTargets>("/api/ingest/targets");
}

export interface TargetedUploadResult {
  status: string;
  file: string;
  size: number;
  target: { source: string; org: string | null; document_type: string | null; path: string };
  pending_uploads?: number;
  message?: string;
}

export async function uploadToTarget(
  file: File,
  target: { source: string; org?: string; document_type?: string },
): Promise<TargetedUploadResult> {
  const params = new URLSearchParams({ filename: file.name, source: target.source });
  if (target.org) params.set("org", target.org);
  if (target.document_type) params.set("document_type", target.document_type);
  const res = await fetch(`/api/ingest/upload?${params.toString()}`, {
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
  // label is config-driven (sources.yaml `presentation.categories`); older
  // backends omit it — fall back to CATEGORY_LABELS/slug (see SourceFilter).
  categories: Array<{ category: string; count: number; label?: string }>;
  total: number;
}

export async function fetchSources(): Promise<SourceCatalogue> {
  return apiFetch<SourceCatalogue>("/api/sources");
}
