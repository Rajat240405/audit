import { fetchSources, type SourceCatalogue, type SourceMinistry } from "@/api/model";

/**
 * Shared source-catalogue cache + tree expansion for the source filter.
 *
 * The tree lives server-side (/api/sources, from org_tree.py). The UI fetches
 * it once; the header renders the ministry dropdown + org chips from it, and
 * useChatStream expands the selection into a flat org list before sending.
 *
 * TREE RULE: selecting a ministry includes ALL orgs under it; explicitly
 * selected orgs win over the ministry expansion.
 */

let _cache: SourceCatalogue | null = null;
let _inflight: Promise<SourceCatalogue> | null = null;

export function getCatalogue(): Promise<SourceCatalogue> {
  if (_cache) return Promise.resolve(_cache);
  if (!_inflight) {
    _inflight = fetchSources()
      .then((c) => {
        _cache = c;
        return c;
      })
      .finally(() => {
        _inflight = null;
      });
  }
  return _inflight;
}

export function currentTree(): Record<string, SourceMinistry> {
  return _cache?.tree ?? {};
}

/** Expand a (ministry, orgs) selection into the flat org list for the backend.
 *  [] = no org restriction. Falls back to explicit orgs when the tree hasn't
 *  loaded yet (never wrongly excludes). */
export function expandOrgFilter(ministry: string, orgs: string[]): string[] {
  if (orgs && orgs.length) return orgs;
  const tree = currentTree();
  if (ministry && ministry !== "all" && tree[ministry]) {
    const slugs = tree[ministry].orgs.map((o) => o.slug);
    if (slugs.length) return slugs;
  }
  return [];
}
