import type { RetrievalTrace, SourceItem } from "@/types";

export function formatMs(ms: number | undefined | null): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function formatNumber(n: number | undefined | null): string {
  if (n == null) return "—";
  return n.toLocaleString("en-IN");
}

export function formatDate(ts: number): string {
  return new Date(ts).toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function maskKey(key: string | undefined | null): string {
  if (!key) return "";
  if (key.length <= 4) return "****";
  return `…${key.slice(-4)}`;
}

export function confidenceLabel(score: number | undefined | null): string {
  if (score == null) return "N/A";
  if (score >= 0.7) return "High";
  if (score >= 0.4) return "Medium";
  return "Low";
}

/** Predict which component contributed most to a hit (for Evidence badges). */
export function dominantComponent(s: SourceItem): string {
  const candidates: Array<[string, number | null]> = [
    ["BM25", s.bm25_score],
    ["Dense", s.dense_score],
    ["Rerank", s.rerank_score],
  ];
  let best = "Rerank";
  let bestVal = -Infinity;
  for (const [label, val] of candidates) {
    if (val != null && val > bestVal) {
      bestVal = val;
      best = label;
    }
  }
  return best;
}

/** Build a PipelineStage list for the given retrieval mode. */
export function hybridStages(): Array<{ key: string; label: string }> {
  return [
    { key: "embed", label: "Embed query" },
    { key: "dense", label: "Semantic search (dense)" },
    { key: "bm25", label: "BM25 search" },
    { key: "rrf", label: "RRF fusion" },
    { key: "rerank", label: "Reranking" },
    { key: "generate", label: "Generating" },
  ];
}

export function graphStages(): Array<{ key: string; label: string }> {
  return [
    { key: "entities", label: "Entity detection" },
    { key: "traversal", label: "Graph traversal" },
    { key: "expansion", label: "Neighbour expansion" },
    { key: "evidence", label: "Evidence selection" },
    { key: "generate", label: "Generating" },
  ];
}

export function traceToRows(trace: RetrievalTrace): Array<[string, number]> {
  return [
    ["Embed query", trace.embed_query_ms],
    ["Dense (FAISS)", trace.dense_search_ms],
    ["BM25", trace.bm25_search_ms],
    ["RRF fusion", trace.rrf_fusion_ms],
    ["Rerank", trace.rerank_ms],
    ["Total", trace.retrieval_total_ms],
  ];
}
