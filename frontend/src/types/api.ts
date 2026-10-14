// Shared domain types for the INCOIS Audit Pro workstation.

export type ExecutionMode = "fast" | "deep";
/**
 * Reasoning depth — Deep mode only. The union covers every ladder the catalog
 * declares; which subset a model actually offers comes from that model's
 * `reasoning_efforts` capability (GET /api/models), never from a hardcoded
 * list. Note the ladders are NOT the same set: `xhigh` is a Qwen3-style
 * template effort and `high` is the top-level OpenAI-compatible effort — each
 * is only ever offered/sent for models that declare it.
 */
export type ThinkingEffort = "low" | "medium" | "high" | "xhigh";
export type RetrievalMode = "auto" | "hybrid" | "graph" | "hybrid_and_graph";
export type DraftStyle =
  | "default"
  | "professional"
  | "parliamentary"
  | "concise"
  | "detailed";

/** One entity matched from the query text (GraphRAG anchor). */
export interface GraphAnchor {
  key: string;
  label: string;
  name: string;
  via: string;
}

/**
 * Graph-traversal provenance attached to a GraphRAG result.
 * Present only for retrieval_mode="graph"; `null`/absent for Hybrid RAG.
 */
export interface GraphProvenance {
  anchors: GraphAnchor[];
  via: string;
  fact_keys: string[];
}

export interface SourceItem {
  doc_id: string;
  ministry: string;
  subject: string;
  date?: string | null;
  score: number;
  question: string;
  answer: string;
  dense_score: number | null;
  bm25_score: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
  /** GraphRAG traversal provenance — only on graph-mode results. */
  graph?: GraphProvenance | null;
  /**
   * Matched retrieval units inside this document (display provenance). Absent
   * on older payloads, graph results and fallbacks. Entries are capped on the
   * server; `matched_chunk_total` is the real count.
   */
  matched_chunks?: MatchedChunk[];
  matched_chunk_total?: number;
  /**
   * Basis of each summary score: "parent" = the parent row's own score;
   * "best_chunk" = the best score among its matched units. Null/absent means
   * no score. Scores from different systems are never comparable.
   */
  score_sources?: Partial<Record<"dense" | "bm25" | "rrf", ScoreBasis | null>>;
}

export type ScoreBasis = "parent" | "best_chunk";

/** One retrieval unit that matched inside a document. */
export interface MatchedChunk {
  chunk_id: string;
  parent_doc_id: string;
  /** e.g. "annexure", "question", "answer", "v2_table", "v2_figure", "parent". */
  unit_type: string;
  /** Stored chunk text. Null for the parent row itself. */
  text: string | null;
  page?: number | null;
  dense_score: number | null;
  bm25_score: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
  /** True for the one unit the reranker scored for this document. */
  reranked: boolean;
}

export interface RetrievalTrace {
  embed_query_ms: number;
  dense_search_ms: number;
  bm25_search_ms: number;
  rrf_fusion_ms: number;
  rerank_ms: number;
  retrieval_total_ms: number;
}

export interface GenerationMeta {
  provider: string;
  model: string;
  profile: string;
  retrieved_documents: number;
  retrieved_chunks: number;
  response_time_ms: number;
  is_fallback: boolean;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  sources_used?: string[];
}

export interface PipelineStage {
  key: string;
  label: string;
  status: "pending" | "running" | "done" | "error";
  count?: number;
}

export interface Session {
  id: string;
  title: string;
  pinned: boolean;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  /** Write revision of the session metadata (title, pin). See lib/sessionMerge.ts. */
  rev?: number;
}

/**
 * Lifecycle of an assistant answer. `undefined` (legacy messages) means the
 * answer is complete.
 *  - streaming:   a live run is writing this message (persisted partial text)
 *  - done:        completed normally
 *  - stopped:     the user pressed Stop; the partial text is kept
 *  - error:       the backend reported an error; partial text is kept
 *  - interrupted: the stream or page ended before `done` (or a reload found
 *                 a message still marked streaming)
 */
export type MessageStatus = "streaming" | "done" | "stopped" | "error" | "interrupted";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: MessageStatus;
  sources?: SourceItem[];
  trace?: RetrievalTrace;
  meta?: GenerationMeta;
  createdAt: number;
  /** Server generation run that produces this answer (see generation_runs.py). */
  runId?: string;
  /** Write revision; the newer settled copy wins a merge. See lib/sessionMerge.ts. */
  rev?: number;
}

export interface ServerStatus {
  provider: string;
  model_family: string;
  model: string;
  mode: ExecutionMode;
  retrieval_mode: RetrievalMode;
  gpu: string;
  /** additive (served-model discovery) — optional extras from the backend */
  enabled_providers?: string[];
  provider_base_url?: string;
  served_model?: string;
  model_display_name?: string;
  /** where capabilities came from: catalog | server | fallback */
  model_metadata_source?: string;
  /** tri-state thinking capability: true/false known, null/absent = unknown */
  thinking_supported?: boolean | null;
  /** model NATIVE context (capability), null = not documented */
  native_context_tokens?: number | null;
  /** vLLM serving limit (--max-model-len) if the server reports it, else null */
  serving_context_tokens?: number | null;
  /** application safety ceiling (RAG_MAX_CONTEXT_TOKENS or catalogue default) */
  app_context_limit_tokens?: number;
  /** effective runtime context = min(native?, serving?, ceiling) */
  effective_context_tokens?: number;
}

export type GraphBuildState =
  | "idle"
  | "running"
  | "completed"
  | "failed"
  | "interrupted";

export interface GraphBuildStatus {
  state: GraphBuildState;
  running: boolean;
  mode?: string | null;
  documents_processed: number;
  total: number;
  percent: number;
  added: number;
  reconciled: number;
  skipped_unchanged: number;
  failed: number;
  documents_done_total?: number;
  current_doc?: string | null;
  last_doc?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  elapsed_seconds?: number | null;
  error?: string | null;
  last_updated: number | null;
  checkpoint_exists: boolean;
  path?: string;
}

// ── Saved Knowledge (shared curated Q&A) ────────────────────────────────────

/** Citation-identity source attached to a saved answer (no full text). */
export interface KnowledgeSource {
  doc_id: string;
  subject: string;
  ministry: string;
  document_type: string;
  score: number | null;
}

/**
 * One display card in a knowledge lookup result. Records whose question AND
 * answer are byte-identical (same content_hash) from different savers are
 * collapsed into ONE card listing every contributor; the underlying records
 * stay independent server-side.
 */
export interface KnowledgeMatchCard {
  content_hash: string;
  question: string;
  question_normalized: string;
  answer: string;
  sources: KnowledgeSource[];
  knowledge_ids: string[];
  contributors: string[];
  created_at: string;
  updated_at: string;
  score: number;
  tier: string;
}

export interface KnowledgeLookupResult {
  matches: KnowledgeMatchCard[];
  tier: string | null;
  ambiguous: boolean;
  found: boolean;
  diagnostics?: Record<string, unknown>;
  // Legacy single-hit keys — still populated by the backend for older clients.
  answer?: string;
  question?: string;
  sources?: KnowledgeSource[];
  matched?: string;
  score?: number;
  saved_by?: string;
}

/** One saved contribution, as returned by /api/knowledge/mine. */
export interface KnowledgeRecord {
  schema_version: number;
  knowledge_id: string;
  content_hash: string;
  question: string;
  question_normalized: string;
  answer: string;
  sources: KnowledgeSource[];
  owner_id: string;
  owner_name: string;
  created_at: string;
  updated_at: string;
  version: number;
  lifecycle: string;
  legacy?: boolean;
}

// SSE event types emitted by the backend /api/chat/stream endpoint
export type StreamEvent =
  | { type: "status"; stage: string; message: string; done: boolean; count?: number }
  | { type: "sources"; sources: SourceItem[]; is_graph: boolean }
  | { type: "trace"; trace: RetrievalTrace }
  | { type: "tokens"; text: string }
  | { type: "reasoning"; text: string }
  | {
      type: "phase";
      phase: "retrieving" | "thinking" | "generating" | "done" | "error";
      model?: string;
    }
  | { type: "meta"; meta: GenerationMeta }
  | { type: "final"; text: string }
  | { type: "error"; message: string }
  /** Saved-knowledge matches offered ALONGSIDE the fresh RAG answer — the
   *  saved answer never replaces generation, so old clients that ignore this
   *  event keep working unchanged. */
  | { type: "knowledge"; matches: KnowledgeMatchCard[]; tier: string | null; ambiguous: boolean }
  | { type: "done" };
