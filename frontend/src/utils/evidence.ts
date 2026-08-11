import type { SourceItem } from "@/types";

/**
 * Client-side grounding check (audit transparency).
 * Extracts quoted claims / key figures from a draft and checks whether they
 * appear verbatim in the retrieved evidence. The authoritative grounding pass
 * lives in the backend; this is a lightweight local heuristic for display.
 */
const FIGURE_RE = /\b\d+(?:[.,]\d+)?\s?(?:%|mm|km|crore|lakh|MW|GW|sq\.? ?km|m|s|hrs?)\b/gi;
// "48 Doppler Weather Radars", "1,272 Automatic Rain Gauges", "32 Water Quality Buoys" —
// a number followed by a capitalized noun phrase. These are the list-number
// claims where models swap adjacent figures ("32 Water Quality Buoys" when the
// source says "2 Water Quality Buoys"); plain figure+unit regexes miss them.
const NUMBER_ENTITY_RE = /\b\d+(?:[.,]\d+)?\s+[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,3}/g;

export interface GroundingHit {
  term: string;
  found: boolean;
  source?: string;
}

const ACRONYM_RE = /\b[A-Z]{2,8}\b/g;
const NAMED_ABBR_RE = /\b[A-Z][A-Za-z&.\- ]{2,60}\s*\([A-Z]{2,10}\)/g;
const ACRONYM_STOPWORDS = new Set([
  "THE", "AND", "FOR", "NOT", "ARE", "WAS", "HAS", "ITS", "YOU", "OUR",
]);

// Equivalent surface forms of the same entity (parity with the backend
// _ALIAS_GROUPS) so "48 DWRs" matches a document that says
// "48 Doppler Weather Radars", and a table listing "AWS 675" matches "675 AWS".
const ENTITY_GROUPS: string[][] = [
  ["dwr", "doppler weather radar", "doppler weather radars"],
  ["aws", "automatic weather station", "automatic weather stations"],
  ["arg", "automatic rain gauge", "automatic rain gauges"],
  ["adcp", "acoustic doppler current profiler", "acoustic doppler current profilers"],
  ["gnss", "global navigation satellite system", "global navigation satellite systems"],
  ["incois", "indian national centre for ocean information services"],
  ["imd", "india meteorological department", "indian meteorological department"],
  ["isro", "indian space research organisation"],
  ["niot", "national institute of ocean technology"],
  ["cmlre", "centre for marine living resources and ecology", "center for marine living resources and ecology"],
  ["csir nio", "national institute of oceanography"],
  ["grse", "garden reach shipbuilders", "garden reach shipbuilders and engineers"],
  ["cwc", "central water commission"],
  ["ndma", "national disaster management authority"],
  ["mission mausam", "mission mausam scheme"],
];

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/(?<=\d),(?=\d)/g, "") // "1,272" -> "1272"
    .replace(/[^a-z0-9]+/g, " ") // "(AWS) 675" -> "aws 675"
    .replace(/\s+/g, " ")
    .trim();
}

function singularize(norm: string): string {
  if (norm.length > 4 && norm.endsWith("s") && !norm.endsWith("ss")) {
    return norm.slice(0, -1);
  }
  return norm;
}

/**
 * All surface forms that represent the same claim. Number+entity claims
 * ("48 DWRs", "675 AWS") expand the entity across its alias group and match
 * with the number BEFORE or AFTER the entity (tables often list "AWS 675").
 * A swapped figure ("32 Water Quality Buoys" vs the source's
 * "2 Water Quality Buoys") fails every candidate.
 */
function claimCandidates(claim: string): string[] {
  const c = normalizeText(claim);
  const out = new Set<string>([c]);
  const m = c.match(/^(\d+)\s+(.+)$/);
  if (m) {
    const num = m[1];
    const phrase = m[2];
    const phrases = new Set<string>([phrase, singularize(phrase)]);
    for (const group of ENTITY_GROUPS) {
      if (group.some((mem) => phrase.includes(mem) || mem.includes(phrase))) {
        for (const g of group) phrases.add(g);
      }
    }
    for (const p of phrases) {
      if (p) {
        out.add(`${num} ${p}`);
        out.add(`${p} ${num}`);
      }
    }
  }
  return [...out].filter(Boolean);
}

export function extractClaims(text: string, max = 16): string[] {
  const claims: string[] = [];
  // number-entity pairs FIRST — they are the claims plain figure regexes miss
  const numberEntities = text.match(NUMBER_ENTITY_RE) ?? [];
  for (const f of numberEntities.slice(0, 10)) claims.push(f);
  // figures with units ("92%", "Rs. 2,000 crore")
  const figures = text.match(FIGURE_RE) ?? [];
  for (const f of figures.slice(0, 10)) claims.push(f);
  // quoted phrases
  const quoted = text.match(/"(?:[^"\\]|\\.){6,80}"/g) ?? [];
  for (const q of quoted.slice(0, 8)) claims.push(q.replace(/"/g, ""));
  // "Name (ABBR)" patterns ("Krishi Advisory based on... (KALP)")
  const named = text.match(NAMED_ABBR_RE) ?? [];
  for (const n of named.slice(0, 8)) claims.push(n);
  // named abbreviations ("KALP", "Mausam SANKALP", "IMD")
  const acronyms = text.match(ACRONYM_RE) ?? [];
  for (const a of acronyms.slice(0, 8)) {
    if (ACRONYM_STOPWORDS.has(a)) continue;
    claims.push(a);
  }
  return Array.from(new Set(claims)).slice(0, max);
}

export function checkClaimsAgainstEvidence(
  claims: string[],
  sources: SourceItem[]
): GroundingHit[] {
  const haystacks = sources.map((s) => ({
    doc_id: s.doc_id,
    text: normalizeText(`${s.question} ${s.answer}`),
  }));
  return claims.map((c) => {
    const cands = claimCandidates(c);
    let found = false;
    let source: string | undefined;
    for (const h of haystacks) {
      if (cands.some((k) => k && h.text.includes(k))) {
        found = true;
        source = h.doc_id;
        break;
      }
    }
    return { term: c, found, source };
  });
}
