// Shared domain types for the INCOIS Audit Pro workstation.

export type ExecutionMode = "fast" | "deep";
export type RetrievalMode = "hybrid" | "graph";
export type DraftStyle =
  | "formal"
  | "concise"
  | "executive"
  | "scientific"
  | "government"
  | "default";

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
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
  trace?: RetrievalTrace;
  meta?: GenerationMeta;
  createdAt: number;
}

export interface GroundingClaim {
  text: string;
  found: boolean;
  source?: string;
}

export interface ServerStatus {
  provider: string;
  model_family: string;
  model: string;
  mode: ExecutionMode;
  retrieval_mode: RetrievalMode;
  gpu: string;
}

export interface GraphBuildStatus {
  running: boolean;
  documents_processed: number;
  failed: number;
  total: number;
  last_updated: number | null;
  checkpoint_exists: boolean;
  path?: string;
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
  | { type: "grounding"; grounding: GroundingClaim[] }
  | {
      type: "final";
      text: string;
      citation_dropped_count: number;
      citation_dropped: string[];
    }
  | { type: "error"; message: string }
  | { type: "done" };
