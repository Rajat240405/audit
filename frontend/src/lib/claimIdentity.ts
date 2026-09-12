/**
 * Stable identity for a grounding claim.
 *
 * The facts panel previously keyed rows on `text.slice(0,40) + arrayIndex`,
 * which breaks the moment grounding is re-issued or reordered — and index is
 * exactly the wrong thing to key an async investigation on.
 *
 * `claimId` is derived from the claim TEXT (order-independent) plus an
 * occurrence ordinal, so two genuinely identical claims still get distinct,
 * reproducible ids.
 */
import type { GroundingClaim } from "@/types";

/** FNV-1a — small, dependency-free, stable across runs. */
function hash(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(36);
}

/** Identity for one claim; `occurrence` disambiguates duplicates. */
export function claimId(text: string, occurrence = 0): string {
  const norm = (text || "").trim().replace(/\s+/g, " ").toLowerCase();
  return occurrence > 0 ? `${hash(norm)}-${occurrence}` : hash(norm);
}

/** Assign stable ids to a grounding report, handling duplicate texts. */
export function withClaimIds(
  claims: GroundingClaim[]
): Array<GroundingClaim & { claim_id: string }> {
  const seen = new Map<string, number>();
  return (claims || []).map((c) => {
    const base = claimId(c.text);
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    return { ...c, claim_id: claimId(c.text, n) };
  });
}
