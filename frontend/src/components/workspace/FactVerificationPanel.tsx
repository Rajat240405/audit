import { AlertTriangle, CheckCircle2, FileText, Info, Loader2, Quote, Search, X, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { investigateClaim } from "@/api/chat";
import { withClaimIds } from "@/lib/claimIdentity";
import { summarizeGrounding } from "@/lib/serverGrounding";
import { useDraftStore } from "@/store/useDraftStore";
import type { GroundingClaim, InvestigationResult } from "@/types";

/**
 * Persistent Cross-Verified Facts panel.
 *
 * Renders the verdict rows that `/api/verify` returned, held in `useDraftStore`.
 * It never extracts, judges or re-counts claims beyond summing the server's
 * verdicts. Each fact row shows one of three verdicts:
 *
 *   status "verified"     -> "Supported"            (Grounded)
 *   status "insufficient" -> "Needs review"         (Insufficient evidence)
 *   status "rejected"     -> "Conflicting evidence" (Contradicted)
 *
 * Term rows (entities, figures, acronyms) are presence checks in their own
 * section. They are not counted as facts.
 *
 * The header and the rows are sticky/scrollable as before: only the list
 * scrolls. If the draft changed after the last check, a stale notice is shown.
 * Each non-supported fact row offers a targeted "Investigate" action. That call
 * is identity-scoped, so a result arriving after the user has moved on is
 * discarded.
 */

const VERDICT = {
  verified: {
    label: "Supported",
    hint: "Grounded: the cited records support this claim.",
    badge: "success" as const,
    icon: CheckCircle2,
    iconClass: "text-success",
  },
  insufficient: {
    label: "Needs review",
    hint: "Insufficient evidence: the cited records do not establish this claim.",
    badge: "warning" as const,
    icon: AlertTriangle,
    iconClass: "text-warning",
  },
  rejected: {
    label: "Conflicting evidence",
    hint: "Contradicted: the cited records conflict with this claim.",
    badge: "danger" as const,
    icon: XCircle,
    iconClass: "text-danger",
  },
};

const INVESTIGATION_LABEL: Record<InvestigationResult["status"], { label: string; badge: "success" | "warning" | "danger" }> = {
  supported: { label: "Supported", badge: "success" },
  contradicted: { label: "Conflicting evidence", badge: "danger" },
  insufficient: { label: "Needs review", badge: "warning" },
  error: { label: "Investigation failed", badge: "warning" },
};

export function FactVerificationPanel({ onClose }: { onClose: () => void }) {
  const grounding = useDraftStore((s) => s.grounding);
  const groundingFor = useDraftStore((s) => s.groundingFor);
  const content = useDraftStore((s) => s.content);
  const selectEvidence = useDraftStore((s) => s.selectEvidence);
  const sources = useDraftStore((s) => s.sources);
  const investigations = useDraftStore((s) => s.investigations);
  const investigating = useDraftStore((s) => s.investigating);

  const groundingState: "none" | "current" | "stale" =
    groundingFor == null ? "none" : groundingFor === content ? "current" : "stale";

  const claims = withClaimIds(grounding ?? []);
  const factRows = claims.filter((g) => g.kind !== "term");
  const termRows = claims.filter((g) => g.kind === "term");
  const summary = summarizeGrounding(claims);

  const openSource = (sourceId: string) => {
    const match = sources.find((s) => s.doc_id === sourceId);
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

  const renderSourceChips = (claim: GroundingClaim) => {
    const ids = claim.sources && claim.sources.length > 0
      ? claim.sources
      : claim.source ? [claim.source] : [];
    return ids.map((id) => (
      <button
        key={id}
        onClick={() => openSource(id)}
        className="inline-flex items-center gap-1 rounded border border-border bg-surface-2 px-1.5 py-0.5 text-[10px] text-accent underline-offset-2 hover:underline"
      >
        <FileText className="h-2.5 w-2.5" aria-hidden="true" />
        {id}
      </button>
    ));
  };

  const renderFact = (claim: (typeof claims)[number]) => {
    const cid = claim.claim_id;
    const status = claim.status ?? "insufficient";
    const v = VERDICT[status];
    const Icon = v.icon;
    const report = investigations[cid];
    const busy = Boolean(investigating[cid]);
    const canInvestigate = status !== "verified";
    const explanation = claim.note || v.hint;
    return (
      <li
        key={cid}
        data-testid="fact-claim"
        data-claim-id={cid}
        data-kind="fact"
        data-status={status}
        data-found={claim.found ? "true" : "false"}
        className="flex items-start gap-2 px-3 py-2"
      >
        <Icon className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${v.iconClass}`} aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={v.badge} className="text-[9px]" title={v.hint}>
              {v.label}
            </Badge>
          </div>
          <p className="mt-1 text-xs leading-relaxed text-foreground/90">{claim.text}</p>
          {claim.sentence && claim.sentence !== claim.text && (
            <p className="mt-0.5 text-[10px] text-muted">From the answer: “{claim.sentence}”</p>
          )}
          <p className="mt-0.5 text-[10px] text-muted" data-testid="fact-explanation">
            {explanation}
          </p>
          {claim.rationale && claim.rationale !== claim.note && (
            <p className="mt-0.5 text-[10px] italic text-muted">{claim.rationale}</p>
          )}

          {claim.quotes && claim.quotes.length > 0 && (
            <ul className="mt-1 space-y-0.5" aria-label="Evidence quotes">
              {claim.quotes.map((q, i) => (
                <li key={`${q.source}-${i}`} className="flex items-start gap-1 text-[10px] text-foreground/75">
                  <Quote className="mt-px h-2.5 w-2.5 shrink-0 text-muted" aria-hidden="true" />
                  <span>
                    “{q.text}” <span className="text-muted">({q.source})</span>
                  </span>
                </li>
              ))}
            </ul>
          )}

          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            {renderSourceChips(claim)}
            {canInvestigate && (
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
                <Badge variant={INVESTIGATION_LABEL[report.status].badge} className="text-[9px]">
                  {INVESTIGATION_LABEL[report.status].label}
                </Badge>
                <span className="text-[9px] uppercase tracking-wide text-muted">
                  investigation report
                </span>
              </div>
              <p className="mt-1 text-[11px] leading-relaxed text-foreground/80">
                {report.rationale || report.error}
              </p>
              {report.quotes && report.quotes.length > 0 && (
                <ul className="mt-1 space-y-0.5">
                  {report.quotes.map((q, i) => (
                    <li key={`${q.source}-${i}`} className="text-[10px] text-foreground/75">
                      “{q.text}” <span className="text-muted">({q.source})</span>
                    </li>
                  ))}
                </ul>
              )}
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
  };

  const renderTerm = (claim: (typeof claims)[number]) => {
    const cid = claim.claim_id;
    return (
      <li
        key={cid}
        data-testid="term-claim"
        data-claim-id={cid}
        data-kind="term"
        data-found={claim.found ? "true" : "false"}
        className="flex items-start gap-2 px-3 py-2"
      >
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <p className="text-xs leading-relaxed text-foreground/90">{claim.text}</p>
          <p className="mt-0.5 text-[10px] text-muted">
            {claim.found ? "Mentioned in the cited records." : "Not found in the cited records."}
          </p>
          <div className="mt-0.5 flex flex-wrap items-center gap-1.5">{renderSourceChips(claim)}</div>
        </div>
      </li>
    );
  };

  return (
    <div
      data-testid="fact-verification-panel"
      className="flex max-h-[min(60vh,22rem)] flex-col rounded-lg border border-border bg-surface shadow-sm"
    >
      {/* sticky header — always reachable regardless of list length */}
      <div className="flex shrink-0 items-center justify-between border-b border-border px-3 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-xs font-bold uppercase tracking-wide">Cross-Verified Facts</p>
          {groundingState !== "none" && summary.facts > 0 && (
            <>
              <Badge variant="success" data-testid="fact-count-supported">
                {summary.grounded} supported
              </Badge>
              {summary.needsReview > 0 && (
                <Badge variant="warning" data-testid="fact-count-review">
                  {summary.needsReview} needs review
                </Badge>
              )}
              {summary.conflicting > 0 && (
                <Badge variant="danger" data-testid="fact-count-conflicting">
                  {summary.conflicting} conflicting
                </Badge>
              )}
            </>
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

      {groundingState === "stale" && (
        <p
          data-testid="fact-verification-stale"
          className="shrink-0 border-b border-border bg-surface-2 px-3 py-1.5 text-[10px] text-muted"
        >
          The draft changed after this check. Re-run Cross-Verify Facts to refresh.
        </p>
      )}

      {groundingState === "none" ? (
        <p data-testid="fact-verification-unchecked" className="px-3 py-4 text-xs text-muted">
          This draft has not been checked against its sources yet. Press Cross-Verify Facts to
          check it.
        </p>
      ) : claims.length === 0 ? (
        <p data-testid="fact-verification-empty" className="px-3 py-4 text-xs text-muted">
          No verifiable claims were found in this draft.
        </p>
      ) : (
        // ONLY this region scrolls. min-h-0 lets this flex child shrink below its
        // content height; without it the list would grow and the page would scroll.
        <div data-testid="fact-list-scroll" className="min-h-0 flex-1 overflow-y-auto">
          {factRows.length === 0 ? (
            <p data-testid="fact-verification-no-facts" className="px-3 pt-2 text-xs text-muted">
              No factual claims were found in this draft. Headings and fragments are not counted.
            </p>
          ) : (
            <section data-testid="fact-section-facts">
              <p className="px-3 pt-2 text-[10px] font-semibold uppercase tracking-wide text-muted">
                Factual claims
              </p>
              <ul className="divide-y divide-border">{factRows.map(renderFact)}</ul>
            </section>
          )}
          {termRows.length > 0 && (
            <section data-testid="fact-section-terms">
              <p className="px-3 pt-2 text-[10px] font-semibold uppercase tracking-wide text-muted">
                Terms &amp; figures (presence only)
              </p>
              <ul className="divide-y divide-border">{termRows.map(renderTerm)}</ul>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
