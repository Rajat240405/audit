import { CheckCircle2, X, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useDraftStore } from "@/store/useDraftStore";
import type { GroundingClaim } from "@/types";

/**
 * Persistent Cross-Verify Facts panel.
 *
 * Renders the CURRENT authoritative grounding state from `useDraftStore`. That
 * state is written by two producers and this panel deliberately owns neither:
 *
 *   * the server's `grounding` SSE event (useChatStream `onGrounding`), which
 *     overrides the client heuristic when it arrives, and
 *   * `buildGroundingReport()` re-run by the Cross-Verify button.
 *
 * So no second verification calculation is introduced — the panel is a view.
 *
 * It stays mounted until the user closes it explicitly; nothing auto-dismisses
 * it (the previous behaviour surfaced the claims only in a toast that expired).
 */
export function FactVerificationPanel({ onClose }: { onClose: () => void }) {
  const grounding = useDraftStore((s) => s.grounding);
  const selectEvidence = useDraftStore((s) => s.selectEvidence);
  const sources = useDraftStore((s) => s.sources);

  const verified = grounding.filter((g) => g.found);
  const unverified = grounding.filter((g) => !g.found);

  const openSource = (claim: GroundingClaim) => {
    if (!claim.source) return;
    const match = sources.find((s) => s.doc_id === claim.source);
    if (match) selectEvidence(match);
  };

  return (
    <div
      data-testid="fact-verification-panel"
      className="rounded-lg border border-border bg-surface shadow-sm"
    >
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <p className="text-xs font-bold uppercase tracking-wide">
            Cross-Verified Facts
          </p>
          <Badge variant="success">{verified.length} grounded</Badge>
          {unverified.length > 0 && (
            <Badge variant="danger">{unverified.length} unsupported</Badge>
          )}
        </div>
        <button
          data-testid="fact-verification-close"
          onClick={onClose}
          aria-label="Close fact verification"
          className="rounded p-1 text-muted hover:bg-surface-2 hover:text-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      {grounding.length === 0 ? (
        <p
          data-testid="fact-verification-empty"
          className="px-3 py-4 text-xs text-muted"
        >
          No verifiable claims were found in this draft.
        </p>
      ) : (
        <ScrollArea className="max-h-64">
          <ul className="divide-y divide-border">
            {grounding.map((claim, i) => (
              <li
                key={`${claim.text.slice(0, 40)}-${i}`}
                data-testid="fact-claim"
                data-found={claim.found ? "true" : "false"}
                className="flex items-start gap-2 px-3 py-2"
              >
                {claim.found ? (
                  <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-success" />
                ) : (
                  <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-danger" />
                )}
                <div className="min-w-0 flex-1">
                  <p className="text-xs leading-relaxed text-foreground/90">
                    {claim.text}
                  </p>
                  {claim.source && (
                    <button
                      onClick={() => openSource(claim)}
                      className="mt-0.5 text-[10px] text-accent underline underline-offset-2"
                    >
                      {claim.source}
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </ScrollArea>
      )}
    </div>
  );
}
