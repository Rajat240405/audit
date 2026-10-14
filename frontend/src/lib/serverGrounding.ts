/**
 * The ONLY place the browser reads grounding results.
 *
 * Every verdict comes from the backend `/api/verify` response (the evidence-
 * grounded verifier in src/verification/semantic.py). This module copies the
 * rows into the store's shape and counts them for display. It never decides
 * that a claim is supported, contradicted or unverified, and it never changes a
 * verdict.
 */
import { verifyAnswer } from "@/api/chat";
import type { GroundingClaim, SourceItem } from "@/types";

const STATUSES = new Set(["verified", "rejected", "insufficient"]);

/**
 * Map server rows to store rows. Verdict fields pass through unchanged. Unknown
 * or missing values stay absent. A row without `kind` is a fact, which is the
 * safe default for legacy payloads.
 */
export function toGroundingClaims(rows: GroundingClaim[]): GroundingClaim[] {
  return (rows || []).map((r) => {
    const kind = r.kind === "term" ? "term" : "fact";
    const status =
      kind === "fact" && r.status && STATUSES.has(r.status) ? r.status : null;
    return {
      text: r.text,
      found: Boolean(r.found),
      status,
      kind,
      source: r.source || undefined,
      sources: Array.isArray(r.sources) ? r.sources : [],
      quotes: Array.isArray(r.quotes) ? r.quotes : [],
      sentence: r.sentence || null,
      method: r.method || undefined,
      note: r.note || undefined,
      rationale: r.rationale || undefined,
    };
  });
}

export interface GroundingSummary {
  /** Facts whose verdict is "verified" (Supported). */
  grounded: number;
  /** Facts whose verdict is "insufficient" (Needs review). */
  needsReview: number;
  /** Facts whose verdict is "rejected" (Conflicting evidence). */
  conflicting: number;
  /** All fact rows. */
  facts: number;
  /** Term presence rows, kept apart from the facts. */
  terms: number;
  termsFound: number;
}

/** Counts used by the panel header, the canvas score and the toast. */
export function summarizeGrounding(rows: GroundingClaim[]): GroundingSummary {
  const s: GroundingSummary = {
    grounded: 0, needsReview: 0, conflicting: 0, facts: 0, terms: 0, termsFound: 0,
  };
  for (const r of rows) {
    if (r.kind === "term") {
      s.terms += 1;
      if (r.found) s.termsFound += 1;
      continue;
    }
    s.facts += 1;
    if (r.status === "verified") s.grounded += 1;
    else if (r.status === "rejected") s.conflicting += 1;
    else s.needsReview += 1;
  }
  return s;
}

export type RefreshResult =
  | { ok: true; claims: GroundingClaim[] }
  | { ok: false; error: string };

/**
 * Ask the backend to verify `answer` against `sources` and return the grounding.
 *
 * The response's rewritten `text` is deliberately ignored here: a Cross-Verify
 * must never change the canvas. Only the automatic check applies a rewrite.
 */
export async function refreshGrounding(
  answer: string,
  sources: SourceItem[],
  depth: "light" | "full",
  verify: typeof verifyAnswer = verifyAnswer
): Promise<RefreshResult> {
  try {
    const res = await verify(answer, sources, depth);
    if (res.error) return { ok: false, error: res.error };
    return { ok: true, claims: toGroundingClaims(res.grounding || []) };
  } catch {
    return { ok: false, error: "Verification request failed" };
  }
}
