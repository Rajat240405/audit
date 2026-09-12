import { CheckCircle2, Loader2, Search, X, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { investigateClaim } from "@/api/chat";
import { withClaimIds } from "@/lib/claimIdentity";
import { useDraftStore } from "@/store/useDraftStore";
import type { GroundingClaim } from "@/types";

/**
 * Persistent Cross-Verified Facts panel.
 *
 * Renders the CURRENT authoritative grounding state from `useDraftStore` —
 * written either by the server's `grounding` SSE event or by the Cross-Verify
 * button. No second verification calculation lives here.
 *
 * Layout: the header (title, counts, close) is STICKY and the claim list is the
 * only scrolling region, so a long report never pushes the controls off-screen
 * and never scrolls the whole workspace.
 *
 * Each unsupported claim offers a targeted "Investigate" action. That call is
 * identity-scoped (claim_id + message_id) so a result arriving after the user
 * has fired another query is discarded rather than attached to the new answer.
 */
export function FactVerificationPanel({ onClose }: { onClose: () => void }) {
  const grounding = useDraftStore((s) => s.grounding);
  const selectEvidence = useDraftStore((s) => s.selectEvidence);
  const sources = useDraftStore((s) => s.sources);
  const investigations = useDraftStore((s) => s.investigations);
  const investigating = useDraftStore((s) => s.investigating);

  const claims = withClaimIds(grounding);
  const verified = claims.filter((g) => g.found);
  const unverified = claims.filter((g) => !g.found);

  const openSource = (claim: GroundingClaim) => {
    if (!claim.source) return;
    const match = sources.find((s) => s.doc_id === claim.source);
    if (match) selectEvidence(match);
  };

  const investigate = async (claimText: string, cid: string) => {
    const draft = useDraftStore.getState();
    // de-dupe rapid clicks on the same claim
    if (draft.investigating[cid]) return;

    // Identity captured at DISPATCH. The investigation has no AbortSignal, so
    // this is what makes a late completion safe.
    const messageId = draft.activeMessageId;
    const sessionId = draft.activeSessionId;
    if (!messageId) return;

    draft.setInvestigating(cid, true);
    try {
      const res = await investigateClaim({
        claim: claimText,
        claimId: cid,
        messageId,
        sources: draft.sources,
      });
      const now = useDraftStore.getState();
      // STALE GUARD — a newer answer owns the panel now: discard entirely.
      // No grounding mutation, no verdict change, no state write.
      if (now.activeMessageId !== messageId || now.activeSessionId !== sessionId) {
        return;
      }
      now.setInvestigation(cid, res);
    } catch {
      const now = useDraftStore.getState();
      if (now.activeMessageId !== messageId) return;
      now.setInvestigation(cid, {
        claim_id: cid,
        message_id: messageId,
        claim: claimText,
        status: "error",
        error: "Investigation failed",
      });
    }
  };

  return (
    <div
      data-testid="fact-verification-panel"
      className="flex max-h-[min(60vh,22rem)] flex-col rounded-lg border border-border bg-surface shadow-sm"
    >
      {/* sticky header — always reachable regardless of list length */}
      <div className="flex shrink-0 items-center justify-between border-b border-border px-3 py-2">
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

      {claims.length === 0 ? (
        <p
          data-testid="fact-verification-empty"
          className="px-3 py-4 text-xs text-muted"
        >
          No verifiable claims were found in this draft.
        </p>
      ) : (
        // ONLY this region scrolls. min-h-0 is required for a flex child to
        // shrink below its content height; without it the list would grow and
        // the page would scroll instead.
        <div
          data-testid="fact-list-scroll"
          className="min-h-0 flex-1 overflow-y-auto"
        >
          <ul className="divide-y divide-border">
            {claims.map((claim) => {
              const cid = claim.claim_id;
              const report = investigations[cid];
              const busy = Boolean(investigating[cid]);
              return (
                <li
                  key={cid}
                  data-testid="fact-claim"
                  data-claim-id={cid}
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
                    <div className="mt-0.5 flex flex-wrap items-center gap-2">
                      {claim.source && (
                        <button
                          onClick={() => openSource(claim)}
                          className="text-[10px] text-accent underline underline-offset-2"
                        >
                          {claim.source}
                        </button>
                      )}
                      {!claim.found && (
                        <button
                          data-testid={`investigate-${cid}`}
                          onClick={() => void investigate(claim.text, cid)}
                          disabled={busy}
                          className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-2 px-2 py-0.5 text-[10px] text-foreground/80 hover:bg-surface disabled:opacity-50"
                        >
                          {busy ? (
                            <>
                              <Loader2 className="h-2.5 w-2.5 animate-spin" />
                              Investigating…
                            </>
                          ) : (
                            <>
                              <Search className="h-2.5 w-2.5" />
                              Investigate
                            </>
                          )}
                        </button>
                      )}
                    </div>

                    {report && (
                      <div
                        data-testid={`investigation-${cid}`}
                        data-status={report.status}
                        className="mt-1.5 rounded border border-border bg-surface-2 p-2"
                      >
                        <div className="flex items-center gap-1.5">
                          <Badge
                            variant={
                              report.status === "supported"
                                ? "success"
                                : report.status === "error"
                                ? "warning"
                                : "danger"
                            }
                            className="text-[9px]"
                          >
                            {report.status}
                          </Badge>
                          <span className="text-[9px] uppercase tracking-wide text-muted">
                            investigation report
                          </span>
                        </div>
                        <p className="mt-1 text-[11px] leading-relaxed text-foreground/80">
                          {report.rationale || report.error}
                        </p>
                        {report.sources && report.sources.length > 0 && (
                          <p className="mt-1 text-[10px] text-muted">
                            Evidence: {report.sources.join(", ")}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
