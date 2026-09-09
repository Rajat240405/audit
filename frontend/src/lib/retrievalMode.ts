/**
 * Retrieval-mode helpers.
 *
 * The mode is now four-valued (`auto | hybrid | graph | hybrid_and_graph`), so
 * the old `mode === "graph"` checks are wrong in two directions:
 *
 *   * `hybrid_and_graph` also produces graph evidence, and
 *   * `auto` is decided by the backend agent, so the REQUESTED mode says
 *     nothing about what actually ran.
 *
 * For anything describing a completed answer, prefer the evidence itself
 * (`hasGraphResults(sources)` / the SSE `is_graph` flag) over the requested
 * mode. These helpers exist for the cases that legitimately need to reason
 * about the request before the response arrives.
 */
import type { RetrievalMode } from "@/types";

/** Modes that always include a graph branch (excludes `auto`). */
export function requestsGraph(mode: RetrievalMode): boolean {
  return mode === "graph" || mode === "hybrid_and_graph";
}

/** Modes that always include a hybrid branch (excludes `auto`). */
export function requestsHybrid(mode: RetrievalMode): boolean {
  return mode === "hybrid" || mode === "hybrid_and_graph";
}

/** True when the backend agent decides the route. */
export function isAuto(mode: RetrievalMode): boolean {
  return mode === "auto";
}

/**
 * True when the request definitely runs a document pipeline whose per-stage
 * trace (embed/dense/bm25/rrf/rerank) is meaningful. `auto` is unknown until
 * the backend reports its route, so it is treated as hybrid-shaped — the
 * common case, and the stage list is replaced by real backend events anyway.
 */
export function showsHybridStages(mode: RetrievalMode): boolean {
  return mode !== "graph";
}

/** Short human label, reused by panels/badges. */
export function retrievalModeLabel(mode: RetrievalMode): string {
  switch (mode) {
    case "auto":
      return "Auto";
    case "graph":
      return "GraphRAG";
    case "hybrid_and_graph":
      return "Hybrid + Graph";
    default:
      return "Hybrid RAG";
  }
}
