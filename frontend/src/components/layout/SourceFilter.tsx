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

interface Draft {
  orgs: Set<string>;
  cats: Set<string>;
}

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
 *  Draft-commit UX: checkboxes edit a local draft; Apply commits to the store,
 *  Clear resets. "N selected" badge on the trigger reflects the APPLIED state.
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
  const [draft, setDraft] = useState<Draft>({ orgs: new Set(), cats: new Set() });
  const panelRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    getCatalogue().then(setCatalogue).catch(() => setCatalogue(null));
  }, []);

  // Draft = applied state whenever the panel opens.
  useEffect(() => {
    if (open) {
      setDraft({ orgs: new Set(sourceFilter.orgs), cats: new Set(sourceFilter.docCategories) });
      setQuery("");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Click-outside to close.
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

  // Flatten orgs across ministries, sorted by count desc.
  const orgs: SourceOrg[] = useMemo(() => {
    const all: SourceOrg[] = [];
    for (const m of Object.values(tree)) all.push(...m.orgs);
    return all.sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
  }, [tree]);

  // Build a map: org slug -> Set<category> from backend catalogue.
  // This is the ONLY source of truth for the hierarchy — no hardcoding.
  const orgCategoryMap = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const org of orgs) {
      map.set(org.slug, new Set(org.categories ?? []));
    }
    return map;
  }, [orgs]);

  // Active categories = union of categories for all selected orgs.
  // If no orgs selected → all categories are active ("All Sources").
  const activeCategorySet = useMemo((): Set<string> | null => {
    if (draft.orgs.size === 0) return null; // null = all active
    const union = new Set<string>();
    for (const slug of draft.orgs) {
      const cats = orgCategoryMap.get(slug);
      if (cats) cats.forEach((c) => union.add(c));
    }
    return union;
  }, [draft.orgs, orgCategoryMap]);

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
  const allChecked = draft.orgs.size === 0 && draft.cats.size === 0;

  const toggleOrg = (slug: string) =>
    setDraft((d) => {
      const next = new Set(d.orgs);
      if (next.has(slug)) next.delete(slug);
      else next.add(slug);

      // Drop any selected categories that are no longer active after this toggle.
      // Recompute active set with the new org selection.
      const newActive = next.size === 0
        ? null
        : (() => {
            const union = new Set<string>();
            for (const s of next) {
              const cats = orgCategoryMap.get(s);
              if (cats) cats.forEach((c) => union.add(c));
            }
            return union;
          })();

      const newCats = new Set(
        [...d.cats].filter((c) => newActive === null || newActive.has(c))
      );

      return { orgs: next, cats: newCats };
    });

  const toggleCat = (cat: string) =>
    setDraft((d) => {
      const next = new Set(d.cats);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return { ...d, cats: next };
    });

  const apply = () => {
    setSourceFilter({
      ministry: "all",
      orgs: [...draft.orgs],
      docCategories: [...draft.cats],
    });
    setOpen(false);
  };

  const clear = () => setDraft({ orgs: new Set(), cats: new Set() });

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
              onToggle={() => clear()}
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
                checked={draft.orgs.has(o.slug)}
                disabled={o.count === 0}
                onToggle={() => toggleOrg(o.slug)}
              />
            ))}

            <div className="px-2 pb-0.5 pt-2 text-[9px] font-bold uppercase tracking-wider text-muted/60">
              Document Types
              {activeCategorySet !== null && (
                <span className="ml-1 font-normal normal-case text-muted/50">
                  (filtered by selected org{draft.orgs.size > 1 ? "s" : ""})
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
                  count={c.count}
                  checked={draft.cats.has(c.category)}
                  disabled={!isActive}
                  onToggle={() => toggleCat(c.category)}
                />
              );
            })}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between gap-2 border-t border-border px-3 py-2">
            <button
              onClick={clear}
              className="rounded-md px-2.5 py-1 text-[11px] font-medium text-muted hover:bg-surface-2 hover:text-foreground"
            >
              Clear
            </button>
            <button
              onClick={apply}
              className="rounded-md bg-accent px-4 py-1 text-[11px] font-semibold text-white hover:opacity-90"
            >
              Apply
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

