import type { SourceItem } from "@/types";

/**
 * Client-side grounding check (audit transparency).
 * Extracts quoted claims / key figures from a draft and checks whether they
 * appear verbatim in the retrieved evidence. The authoritative grounding pass
 * lives in the backend; this is a lightweight local heuristic for display.
 */
const FIGURE_RE = /\b\d+(?:[.,]\d+)?\s?(?:%|mm|km|crore|lakh|MW|GW|sq\.? ?km|m|s|hrs?)\b/gi;

export interface GroundingHit {
  term: string;
  found: boolean;
  source?: string;
}

export function extractClaims(text: string, max = 8): string[] {
  const claims: string[] = [];
  const figures = text.match(FIGURE_RE) ?? [];
  for (const f of figures.slice(0, max)) {
    claims.push(f);
  }
  // also grab quoted phrases
  const quoted = text.match(/"(?:[^"\\]|\\.){6,80}"/g) ?? [];
  for (const q of quoted.slice(0, max)) {
    claims.push(q.replace(/"/g, ""));
  }
  return Array.from(new Set(claims)).slice(0, max);
}

export function checkClaimsAgainstEvidence(
  claims: string[],
  sources: SourceItem[]
): GroundingHit[] {
  const haystack = sources
    .map((s) => `${s.question} ${s.answer}`.toLowerCase())
    .join(" ");
  return claims.map((c) => {
    const found = haystack.includes(c.toLowerCase());
    const source = found
      ? sources.find((s) =>
          `${s.question} ${s.answer}`.toLowerCase().includes(c.toLowerCase())
        )?.doc_id
      : undefined;
    return { term: c, found, source };
  });
}
