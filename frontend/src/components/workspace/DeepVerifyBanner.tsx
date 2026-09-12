import { Loader2, ShieldCheck, Hourglass } from "lucide-react";
import { useDraftStore } from "@/store/useDraftStore";

/**
 * Deep-mode "Verifying answer…" state.
 *
 * A Deep answer is streamed first and verified afterwards, so for a moment the
 * canvas shows an answer a judge pass could still rewrite. This banner makes
 * that window explicit and hands the user the choice:
 *
 *   Wait          — hold new queries until verification settles, so the result
 *                   they asked for cannot be superseded and discarded.
 *   Start anyway  — carry on immediately; verification keeps running in the
 *                   background and its result is applied if the answer is still
 *                   the live one.
 *
 * Safety: this component only READS `deepVerify` and writes `choice`. It never
 * touches grounding, content or sources, so it cannot interfere with the
 * identity-scoped stale-result guard in `useChatStream`. Rendering is driven by
 * `status === "running"` alone; `settleDeepVerify` is identity-gated, so a late
 * completion from a superseded answer can neither clear nor relight a banner
 * that now belongs to a newer query.
 *
 * Light verification never enters this state, so nothing here can block it.
 */
export function DeepVerifyBanner() {
  const deepVerify = useDraftStore((s) => s.deepVerify);
  const setChoice = useDraftStore((s) => s.setDeepVerifyChoice);

  if (deepVerify.status !== "running") return null;

  const waiting = deepVerify.choice === "wait";
  const startedAnyway = deepVerify.choice === "anyway";

  return (
    <div
      data-testid="deep-verify-banner"
      data-choice={deepVerify.choice}
      role="status"
      aria-live="polite"
      className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-accent/40 bg-accent/5 px-3 py-2"
    >
      <p className="flex items-center gap-2 text-xs font-semibold text-foreground/90">
        {waiting ? (
          <Hourglass className="h-3.5 w-3.5 shrink-0 animate-pulse text-accent" />
        ) : (
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-accent" />
        )}
        {waiting
          ? "Waiting for verification to finish…"
          : startedAnyway
          ? "Verifying answer… (continuing in the background)"
          : "Verifying answer…"}
      </p>

      <div className="flex items-center gap-2">
        {startedAnyway ? (
          <span
            data-testid="deep-verify-background-note"
            className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted"
          >
            <ShieldCheck className="h-3 w-3" />
            result applies if this answer is still current
          </span>
        ) : (
          <>
            <button
              data-testid="deep-verify-wait"
              onClick={() => setChoice("wait")}
              aria-pressed={waiting}
              disabled={waiting}
              className={
                waiting
                  ? "rounded-md border border-accent bg-accent/15 px-3 py-1.5 text-[11px] font-bold uppercase text-accent"
                  : "rounded-md border border-border bg-surface px-3 py-1.5 text-[11px] font-bold uppercase text-foreground shadow-sm hover:bg-surface-2"
              }
            >
              {waiting ? "Waiting…" : "Wait"}
            </button>
            <button
              data-testid="deep-verify-anyway"
              onClick={() => setChoice("anyway")}
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-[11px] font-bold uppercase text-foreground shadow-sm hover:bg-surface-2"
            >
              Start anyway
            </button>
          </>
        )}
      </div>
    </div>
  );
}
