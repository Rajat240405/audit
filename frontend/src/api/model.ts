import { apiFetch } from "./client";
import type {
  ExecutionMode,
  KnowledgeLookupResult,
  KnowledgeRecord,
  ServerStatus,
  SourceItem,
  ThinkingEffort,
} from "@/types";

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
  /** Reasoning-effort ladder THIS model documents (Deep mode only). Empty or
   * absent = the model has no effort control: the selector must be shown as
   * unavailable rather than offering a ladder the model cannot honour. */
  reasoning_efforts?: ThinkingEffort[];
  /** Where that effort is sent: chat_template_kwargs | request_field | none.
   * Informational — the backend decides the wire; the UI only renders the
   * ladder. */
  effort_wire?: string;
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

/**
 * Save a NEW contribution to the shared knowledge base. Every call creates an
 * independent record (never overwrites another saver's), keyed by a
 * server-generated UUID. `savedBy` is the display name the user typed in the
 * save modal — v1 identity is manual.
 */
export async function saveKnowledge(payload: {
  question: string;
  answer: string;
  sources?: SourceItem[];
  savedBy?: string;
}): Promise<{ status: string; knowledge_id: string; file: string; saved_by: string }> {
  return apiFetch("/api/save-knowledge", {
    method: "POST",
    body: JSON.stringify({
      question: payload.question,
      answer: payload.answer,
      sources: payload.sources,
      saved_by: payload.savedBy,
    }),
  });
}

/** Look up saved answers relevant to a question (top 5, ranked by similarity). */
export async function lookupKnowledge(q: string): Promise<KnowledgeLookupResult> {
  return apiFetch(`/api/knowledge-lookup?q=${encodeURIComponent(q)}`);
}

/** The current user's own saved contributions. */
export async function myKnowledge(
  owner: string
): Promise<{ owner_name: string; owner_id: string; count: number; records: KnowledgeRecord[] }> {
  return apiFetch(`/api/knowledge/mine?owner=${encodeURIComponent(owner)}`);
}

/** Edit one of my contributions — keeps the id, bumps the version. */
export async function updateKnowledge(
  knowledgeId: string,
  payload: { question?: string; answer?: string; savedBy?: string }
): Promise<{ status: string; record: KnowledgeRecord }> {
  return apiFetch(`/api/knowledge/${encodeURIComponent(knowledgeId)}`, {
    method: "PATCH",
    body: JSON.stringify({
      question: payload.question,
      answer: payload.answer,
      saved_by: payload.savedBy,
    }),
  });
}

/** Archive (soft-delete) one of my contributions. */
export async function deleteKnowledge(
  knowledgeId: string,
  owner: string
): Promise<{ status: string; knowledge_id: string; lifecycle: string }> {
  return apiFetch(
    `/api/knowledge/${encodeURIComponent(knowledgeId)}?owner=${encodeURIComponent(owner)}`,
    { method: "DELETE" }
  );
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
  /** org slug -> category -> count, for org-scoped category counts. Older
   *  backends omit it; the UI then falls back to the global counts. */
  org_category_counts?: Record<string, Record<string, number>>;
  total: number;
}

export async function fetchSources(): Promise<SourceCatalogue> {
  return apiFetch<SourceCatalogue>("/api/sources");
}
