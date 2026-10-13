/**
 * Presentation rules for matched-chunk provenance (display only).
 *
 * The LLM never sees these helpers' output. The model reads `answer` (the
 * assembled parent evidence). These functions only decide what the source
 * viewer shows as the matched section and how each score is labelled.
 */
import type { MatchedChunk, ScoreBasis, SourceItem } from "@/types";

/** Matched units that carry stored text (the parent-row entry has none). */
export function matchedTextUnits(source: Pick<SourceItem, "matched_chunks">): MatchedChunk[] {
  return (source.matched_chunks ?? []).filter(
    (m): m is MatchedChunk & { text: string } =>
      typeof m.text === "string" && m.text.trim().length > 0,
  );
}

/**
 * True when there is matched-section text to show as the primary view. False
 * means the caller keeps the existing display (full answer as retrieved).
 */
export function hasMatchedSections(source: Pick<SourceItem, "matched_chunks">): boolean {
  return matchedTextUnits(source).length > 0;
}

/** How many matched units exist in total (may exceed what is in `matched_chunks`). */
export function matchedTotal(source: Pick<SourceItem, "matched_chunks" | "matched_chunk_total">): number {
  return source.matched_chunk_total ?? source.matched_chunks?.length ?? 0;
}

/** Human label for a unit type. Unknown types fall back to a neutral label. */
export function unitTypeLabel(unitType: string): string {
  switch (unitType) {
    case "annexure":
      return "Annexure section";
    case "question":
      return "Question";
    case "answer":
      return "Answer section";
    case "v2_table":
      return "Table";
    case "v2_figure":
      return "Figure";
    default:
      return "Section";
  }
}

/**
 * Label for where a summary score came from. Returns null when the basis is
 * unknown, so the UI does not guess.
 */
export function scoreBasisLabel(basis: ScoreBasis | null | undefined): string | null {
  if (basis === "parent") return "parent document";
  if (basis === "best_chunk") return "best matched section";
  return null;
}

/**
 * Basis of the rerank score. The reranker scores one candidate per document,
 * which is the parent row or one of its matched units. Null when the source
 * has no rerank score or no matched-unit record to attribute it to.
 */
export function rerankBasis(source: Pick<SourceItem, "doc_id" | "matched_chunks" | "rerank_score">): ScoreBasis | null {
  if (source.rerank_score == null) return null;
  const reranked = (source.matched_chunks ?? []).find((m) => m.reranked);
  if (!reranked) return null;
  return reranked.chunk_id === source.doc_id ? "parent" : "best_chunk";
}

/**
 * True when the assembled parent evidence (`answer`) adds text beyond the
 * matched sections. False on the chunk-level path, where `answer` is the
 * matched answer chunk itself, so the viewer does not show it twice.
 */
export function parentEvidenceDiffers(source: Pick<SourceItem, "answer" | "matched_chunks">): boolean {
  const answer = source.answer.trim();
  return !matchedTextUnits(source).some((m) => (m.text ?? "").trim() === answer);
}
