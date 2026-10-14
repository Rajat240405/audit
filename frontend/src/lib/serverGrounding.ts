/**
 * The ONLY place the browser reads grounding results.
 *
 * Every verdict comes from the backend `/api/verify` response (exact-containment
 * sentence checks plus the entity/figure claim checks and the LLM judge). This
 * module maps that response into the store's shape without changing any
 * `found` value. It never decides that a claim or sentence is supported.
 */
import { verifyAnswer } from "@/api/chat";
import type { GroundingClaim, SourceItem } from "@/types";

/** One row of the `/api/verify` `grounding` array. */
export interface ServerGroundingRow {
  text: string;
  found: boolean;
  source?: string;
  note?: string;
  kind?: string;
}

/** Map server rows to store rows. `found` passes through unchanged. */
export function toGroundingClaims(rows: ServerGroundingRow[]): GroundingClaim[] {
  return (rows || []).map((r) => ({
    text: r.text,
    found: r.found,
    source: r.source,
    note: r.note || undefined,
    kind: r.kind === "sentence" ? "sentence" : "claim",
  }));
}

export type RefreshResult =
  | { ok: true; claims: GroundingClaim[] }
  | { ok: false; error: string };

/**
 * Ask the backend to verify `answer` against `sources` and return the grounding.
 *
 * The response's rewritten `text` is deliberately ignored here: a Cross-Verify
 * must never change the canvas. Only the auto-verify path applies a rewrite.
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
