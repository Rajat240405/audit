import { checkClaimsAgainstEvidence, extractClaims } from "@/utils/evidence";
import type { GroundingClaim, SourceItem } from "@/types";

/**
 * Produce a per-answer grounding report: every extracted claim/figure marked
 * verified (in sources) or unverified (not found verbatim).
 */
export function buildGroundingReport(
  answer: string,
  sources: SourceItem[]
): GroundingClaim[] {
  const claims = extractClaims(answer);
  return checkClaimsAgainstEvidence(claims, sources).map((h) => ({
    text: h.term,
    found: h.found,
    source: h.source,
  }));
}

export function groundingScore(report: GroundingClaim[]): number {
  if (report.length === 0) return 1; // nothing to verify
  const verified = report.filter((r) => r.found).length;
  return verified / report.length;
}
