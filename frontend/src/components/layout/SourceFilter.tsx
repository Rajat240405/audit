import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Filter, Search, X } from "lucide-react";
import { getCatalogue } from "@/lib/sourceFilter";
import type { SourceCatalogue, SourceOrg } from "@/api/model";
import { useAppStore } from "@/store/useAppStore";
import { cn } from "@/utils/cn";
import { Input } from "@/components/ui/input";

/** Pretty display labels for doc categories (cadence axis).
 *  LEGACY FALLBACK ONLY — the backend catalogue (/api/sources) now carries
 *  config-driven labels (sources.yaml `presentation.categories`) on each
 *  category entry; those win. This map only covers older backends that omit
 *  the `label` field. Unknown categories fall back to the raw slug. */
const CATEGORY_LABELS: Record<string, string> = {
  parliamentary: "Parliamentary Questions",
  lok_sabha: "Lok Sabha",
  rajya_sabha: "Rajya Sabha",
  annual: "Annual Reports",
  monthly: "Monthly Reports",
  quarterly: "Quarterly Reports",
  scientific: "Scientific / Research",
  technical: "Technical Reports",
  general: "General Reports",
  budget: "Budget / Grants",
  policy: "Policy Documents",
  gazette: "Gazettes / Notices",
  news: "Newsletters / News",
  misc: "Misc",
};

function Row({
  label,
  count,
  checked,
  disabled,
  onToggle,
  sublabel,
}: {
  label: string;
  count: number;
  checked: boolean;
  disabled?: boolean;
  onToggle: () => void;
  sublabel?: string;
}) {
  return (
    <button
      onClick={() => !disabled && onToggle()}
      disabled={disabled}
      className={cn(
        "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[11px] transition-colors",
        checked ? "bg-accent/15 text-foreground" : "text-muted hover:bg-surface-2",
        disabled && "cursor-not-allowed opacity-45"
      )}
      title={disabled ? `${label} — not available for selected source(s)` : undefined}
    >
      <span
        className={cn(
          "flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-[4px] border",
          checked ? "border-accent bg-accent" : "border-muted"
        )}
      >
        {checked && <Check className="h-2.5 w-2.5 text-white" strokeWidth={3.5} />}
      </span>
      <span className="flex-1 truncate">{label}</span>
      {sublabel && <span className="truncate text-muted/70">{sublabel}</span>}
      <span className={cn("shrink-0 tabular-nums", checked ? "text-accent" : "text-muted/70")}>
        {count.toLocaleString("en-IN")}
      </span>
    </button>
  );
}

/** Source filter popover — Organizations (orgs) + Document Types (categories).
 *
 *  LIVE-COMMIT UX: every checkbox writes straight to the store, so the filter is
 *  applied the instant it is toggled — there is no Apply/confirm step. Clear
 *  resets to the unfiltered view immediately. "N selected" on the trigger
 *  reflects the live (applied) state. The filter only takes effect at query
 *  time (useChatStream), so toggling never triggers a network request and a
 *  burst of selections cannot cause a request storm.
 *
 *  HIERARCHY RULE (requirement):
 *  - Each org carries a `categories` list (from /api/sources → org_tree.py).
 *  - When ≥1 orgs are selected, active categories = UNION of those orgs' lists.
 *  - Categories outside the union are faded + non-selectable.
 *  - When no orgs selected ("All Sources"), all categories are active.
 *  - The mapping is fully backend-driven — no hardcoded org→category map here.
 */
export function SourceFilter() {
  const sourceFilter = useAppStore((s) => s.sourceFilter);
  const setSourceFilter = useAppStore((s) => s.setSourceFilter);
  const [catalogue, setCatalogue] = useState<SourceCatalogue | null>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const panelRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    getCatalogue().then(setCatalogue).catch(() => setCatalogue(null));
  }, []);

  // Click-outside closes the POPOVER only (never a tour — see ProductTour).
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const tree = catalogue?.tree ?? {};
  const categories = catalogue?.categories ?? [];

  // Live selection as Sets (single source of truth = the store).
  const selOrgs = useMemo(() => new Set(sourceFilter.orgs), [sourceFilter.orgs]);
  const selCats = useMemo(() => new Set(sourceFilter.docCategories), [sourceFilter.docCategories]);

  // Flatten orgs across ministries, sorted by count desc. The backend already
  // prunes orgs/ministries with no indexed records, so only supported sources
  // are ever listed here.
  const orgs: SourceOrg[] = useMemo(() => {
    const all: SourceOrg[] = [];
    for (const m of Object.values(tree)) all.push(...m.orgs);
    return all.sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
  }, [tree]);

  // org slug -> Set<category> from the backend catalogue (only source of truth).
  const orgCategoryMap = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const org of orgs) {
      map.set(org.slug, new Set(org.categories ?? []));
    }
    return map;
  }, [orgs]);

  // org -> category -> count from the backend; used to scope displayed counts.
  const orgCatCounts = catalogue?.org_category_counts ?? {};
  const scopedCount = (cat: string): number => {
    let n = 0;
    for (const o of selOrgs) n += orgCatCounts[o]?.[cat] ?? 0;
    return n;
  };

  // Active categories = union of categories for all SELECTED (applied) orgs.
  const activeCategorySet = useMemo((): Set<string> | null => {
    if (selOrgs.size === 0) return null; // null = all active
    const union = new Set<string>();
    for (const slug of selOrgs) {
      const cats = orgCategoryMap.get(slug);
      if (cats) cats.forEach((c) => union.add(c));
    }
    return union;
  }, [selOrgs, orgCategoryMap]);

  const cats = useMemo(
    () =>
      categories
        .filter((c) => c.count > 0)
        .sort((a, b) => b.count - a.count),
    [categories]
  );

  const q = query.trim().toLowerCase();
  const filteredOrgs = q ? orgs.filter((o) => o.name.toLowerCase().includes(q)) : orgs;
  const filteredCats = q
    ? cats.filter((c) => (c.label ?? CATEGORY_LABELS[c.category] ?? c.category).toLowerCase().includes(q))
    : cats;

  const appliedCount = sourceFilter.orgs.length + sourceFilter.docCategories.length;
  const allChecked = appliedCount === 0;

  const commit = (orgsSet: Set<string>, catsSet: Set<string>) =>
    setSourceFilter({
      ministry: "all",
      orgs: [...orgsSet],
      docCategories: [...catsSet],
    });

  const toggleOrg = (slug: string) => {
    const nextOrgs = new Set(selOrgs);
    if (nextOrgs.has(slug)) nextOrgs.delete(slug);
    else nextOrgs.add(slug);

    // Drop any selected categories that are no longer active after this toggle.
    const newActive =
      nextOrgs.size === 0
        ? null
        : (() => {
            const union = new Set<string>();
            for (const s of nextOrgs) {
              const c = orgCategoryMap.get(s);
              if (c) c.forEach((x) => union.add(x));
            }
            return union;
          })();

    const nextCats = new Set([...selCats].filter((c) => newActive === null || newActive.has(c)));
    commit(nextOrgs, nextCats);
  };

  const toggleCat = (cat: string) => {
    const nextCats = new Set(selCats);
    if (nextCats.has(cat)) nextCats.delete(cat);
    else nextCats.add(cat);
    commit(selOrgs, nextCats);
  };

  const clear = () => commit(new Set(), new Set());

  return (
    <div className="relative" ref={panelRef}>
      <button
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-semibold transition-colors",
          appliedCount > 0
            ? "border-accent bg-accent/15 text-accent"
            : "border-border bg-surface-2 text-muted hover:text-foreground"
        )}
        title="Filter sources by organization and document type"
      >
        <Filter className="h-3 w-3" />
        Sources{appliedCount > 0 ? ` (${appliedCount} selected)` : ""}
      </button>

      {open && (
        <div className="absolute right-0 top-full z-50 mt-1 flex w-80 flex-col overflow-hidden rounded-xl border border-border bg-surface shadow-xl">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-border px-3 py-2">
            <span className="text-[11px] font-semibold">Sources</span>
            <button onClick={() => setOpen(false)} className="text-muted hover:text-foreground" title="Close">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* Search */}
          <div className="border-b border-border px-3 py-2">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search sources..."
                className="h-7 pl-6 text-[11px]"
              />
            </div>
          </div>

          {/* Body */}
          <div className="max-h-[55vh] overflow-y-auto px-2 py-2">
            <Row
              label="All Sources"
              count={catalogue?.total ?? 0}
              checked={allChecked}
              onToggle={clear}
              sublabel={appliedCount > 0 ? "(clear filter)" : undefined}
            />

            <div className="px-2 pb-0.5 pt-2 text-[9px] font-bold uppercase tracking-wider text-muted/60">
              Organizations
            </div>
            {filteredOrgs.length === 0 && (
              <div className="px-2 py-1 text-[10px] text-muted/60">No organizations match "{query}"</div>
            )}
            {filteredOrgs.map((o) => (
              <Row
                key={o.slug}
                label={o.name}
                count={o.count}
                checked={selOrgs.has(o.slug)}
                disabled={o.count === 0}
                onToggle={() => toggleOrg(o.slug)}
              />
            ))}

            <div className="px-2 pb-0.5 pt-2 text-[9px] font-bold uppercase tracking-wider text-muted/60">
              Document Types
              {activeCategorySet !== null && (
                <span className="ml-1 font-normal normal-case text-muted/50">
                  (filtered by selected org{selOrgs.size > 1 ? "s" : ""})
                </span>
              )}
            </div>
            {filteredCats.length === 0 && (
              <div className="px-2 py-1 text-[10px] text-muted/60">No document types match "{query}"</div>
            )}
            {filteredCats.map((c) => {
              const isActive = activeCategorySet === null || activeCategorySet.has(c.category);
              return (
                <Row
                  key={c.category}
                  label={c.label ?? CATEGORY_LABELS[c.category] ?? c.category}
                  count={selOrgs.size ? scopedCount(c.category) : c.count}
                  checked={selCats.has(c.category)}
                  disabled={!isActive}
                  onToggle={() => toggleCat(c.category)}
                />
              );
            })}
          </div>

          {/* Footer — Clear only; selections apply live, no confirm step. */}
          <div className="flex items-center justify-end gap-2 border-t border-border px-3 py-2">
            <button
              onClick={clear}
              className="rounded-md px-2.5 py-1 text-[11px] font-medium text-muted hover:bg-surface-2 hover:text-foreground"
            >
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
