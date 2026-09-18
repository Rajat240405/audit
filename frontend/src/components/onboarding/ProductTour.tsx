import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useTourStore } from "@/store/useTourStore";
import { useChatActionsStore } from "@/store/useChatActionsStore";
import { TOUR_STEPS, type TourPlacement, type TourStep } from "./tourSteps";
import { useTourTarget } from "./useTourTarget";

/** Fallback card height until the real one is measured. */
const CARD_H_ESTIMATE = 260;
const CARD_W = 384;
const GAP = 14;
const MARGIN = 16;

const OPPOSITE: Record<TourPlacement, TourPlacement> = {
  top: "bottom",
  bottom: "top",
  left: "right",
  right: "left",
};

interface Pos {
  top: number;
  left: number;
}

function posForSide(rect: DOMRect, side: TourPlacement, cardH: number): Pos {
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  switch (side) {
    case "bottom":
      return { top: rect.bottom + GAP, left: cx - CARD_W / 2 };
    case "top":
      return { top: rect.top - GAP - cardH, left: cx - CARD_W / 2 };
    case "right":
      return { top: cy - cardH / 2, left: rect.right + GAP };
    case "left":
      return { top: cy - cardH / 2, left: rect.left - GAP - CARD_W };
  }
}

function inBounds(p: Pos, cardH: number, vw: number, vh: number) {
  return (
    p.top >= MARGIN &&
    p.left >= MARGIN &&
    p.top + cardH <= vh - MARGIN &&
    p.left + CARD_W <= vw - MARGIN
  );
}

function clamp(p: Pos, cardH: number, vw: number, vh: number): Pos {
  return {
    top: Math.min(Math.max(p.top, MARGIN), Math.max(MARGIN, vh - cardH - MARGIN)),
    left: Math.min(Math.max(p.left, MARGIN), Math.max(MARGIN, vw - CARD_W - MARGIN)),
  };
}

/** Pick the first side that fits, preferring the step's requested side. */
function position(rect: DOMRect | null, preferred: TourPlacement, cardH: number): Pos {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  if (!rect) return clamp({ top: (vh - cardH) / 2, left: (vw - CARD_W) / 2 }, cardH, vw, vh);
  const order: TourPlacement[] = [
    preferred,
    OPPOSITE[preferred],
    "bottom",
    "top",
    "right",
    "left",
  ];
  for (const side of order) {
    const p = posForSide(rect, side, cardH);
    if (inBounds(p, cardH, vw, vh)) return p;
  }
  // Nothing fits (small viewport): keep it on screen.
  return clamp(posForSide(rect, preferred, cardH), cardH, vw, vh);
}

/** Viewport size as state, so the popover repositions on window resize. */
function useViewportSize() {
  const [size, setSize] = useState(() => ({
    w: window.innerWidth,
    h: window.innerHeight,
  }));
  useEffect(() => {
    const onResize = () => setSize({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return size;
}

/**
 * Product tour overlay.
 *
 * Rendered once, globally, from App.tsx — the same pattern the app already uses
 * for Settings and BuildGraphModal. It is a plain fixed-position overlay rather
 * than a Radix Dialog: a tour popover must anchor to an arbitrary element and
 * must not trap focus in the centre of the screen.
 *
 * Auto-starts once per browser (persisted ``hasSeenTour``) and is reopenable
 * from the header Guide button.
 *
 * DISMISSAL CONTRACT: once open, the tour stays open until the user explicitly
 * chooses "Skip tour" or "Finish" on the final step. Backdrop clicks, outside
 * clicks, Escape and ordinary interaction with the (still-interactive) app
 * underneath never close it.
 */
export function ProductTour() {
  const open = useTourStore((s) => s.open);
  const stepIndex = useTourStore((s) => s.stepIndex);

  const total = TOUR_STEPS.length;
  const step: TourStep = TOUR_STEPS[Math.min(stepIndex, total - 1)];
  const rect = useTourTarget(open ? step : null);

  const nextRef = useRef<HTMLButtonElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  /** Real card height — content length varies a lot between steps. */
  const [cardH, setCardH] = useState(CARD_H_ESTIMATE);
  /** Tab the user was on when the tour opened, restored when it closes. */
  const originTab = useRef<string | null>(null);
  useViewportSize();

  /* ---- first-run auto-start (StrictMode-safe: idempotent, no module flag) ---- */
  useEffect(() => {
    if (useTourStore.getState().hasSeenTour) return;
    // Let the shell paint and the lazy Workspace route mount before starting.
    const t = window.setTimeout(() => {
      const s = useTourStore.getState();
      if (!s.hasSeenTour && !s.open) s.start();
    }, 700);
    return () => window.clearTimeout(t);
  }, []);

  /* ---- remember where the user was, so closing the tour puts them back ---- */
  useEffect(() => {
    if (!open) return;
    if (originTab.current === null) {
      originTab.current = useChatActionsStore.getState().tab;
    }
    return () => {
      const back = originTab.current;
      originTab.current = null;
      if (back) useChatActionsStore.getState().setTab(back);
    };
  }, [open]);

  /* ---- keyboard navigation ----
     Arrows are a convenience only. They must never fire while the user is
     typing in, or navigating, an underlying control. Escape / Enter / Space no
     longer drive the tour at all — only Skip and Finish end it. */
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const ae = document.activeElement as HTMLElement | null;
      const inCard = cardRef.current?.contains(ae) ?? false;
      const idle = !ae || ae === document.body || ae === document.documentElement;
      if (!inCard && !idle) return; // let the app control handle the key
      const st = useTourStore.getState();
      if (e.key === "ArrowRight") {
        e.preventDefault();
        st.next(total);
      } else {
        e.preventDefault();
        st.back();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, total]);

  /* ---- measure the card BEFORE paint so placement is right on the first frame ---- */
  const measure = useCallback(() => {
    const el = cardRef.current;
    if (!el) return;
    const h = el.offsetHeight;
    if (h > 0) setCardH((prev) => (Math.abs(prev - h) < 1 ? prev : h));
  }, []);
  useLayoutEffect(measure, [measure, step.id, open]);
  useEffect(() => {
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [measure]);

  /* ---- move focus into the dialog on each step so keys work immediately ---- */
  useEffect(() => {
    if (open) nextRef.current?.focus();
  }, [open, step.id]);

  if (!open) return null;

  const s = useTourStore.getState();
  const pos = position(rect, step.placement ?? "bottom", cardH);
  const isLast = stepIndex >= total - 1;
  const isFirst = stepIndex === 0;
  const pct = ((stepIndex + 1) / total) * 100;

  const pad = 5;
  const spot = rect
    ? {
        top: rect.top - pad,
        left: rect.left - pad,
        width: rect.width + pad * 2,
        height: rect.height + pad * 2,
      }
    : null;

  return (
    <div data-testid="product-tour">
      {/* Purely visual scrim — pointer-events-none, so the app underneath stays
          fully interactive while the tour runs. Nothing here dismisses the tour;
          only Skip / Finish end it (no backdrop-click, no outside-click, no
          Escape). Dimming for targeted steps comes from the spotlight's
          box-shadow below. */}
      <div
        className={`pointer-events-none fixed inset-0 z-[9997] ${rect ? "" : "bg-black/55"}`}
        aria-hidden="true"
      />

      {spot && (
        <div
          className="pointer-events-none fixed z-[9998] rounded-lg border-2 border-accent shadow-[0_0_0_9999px_rgba(0,0,0,0.55)] transition-all duration-200 ease-out"
          style={spot}
          aria-hidden="true"
        />
      )}

      <div
        ref={cardRef}
        role="dialog"
        aria-modal="true"
        aria-label={`Product tour, step ${stepIndex + 1} of ${total}: ${step.title}`}
        data-testid="tour-card"
        data-step={step.id}
        className="fixed z-[9999] flex max-h-[70vh] w-[384px] flex-col overflow-hidden rounded-lg border border-border bg-surface shadow-xl shadow-black/25 transition-all duration-200 ease-out"
        style={{ top: pos.top, left: pos.left }}
      >
        <div className="h-1 w-full bg-muted/20">
          <div
            className="h-full bg-accent transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
          <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">
            Step {stepIndex + 1} of {total}
          </p>
          <h2 className="text-base font-semibold text-foreground">{step.title}</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted">{step.body}</p>
          {step.when && (
            <p className="mt-3 border-l-2 border-accent/50 pl-3 text-xs leading-relaxed text-muted/90">
              <span className="font-semibold text-foreground/80">When to use it: </span>
              {step.when}
            </p>
          )}
        </div>

        <div className="flex shrink-0 items-center justify-between border-t border-border px-4 py-2.5">
          <button
            type="button"
            onClick={() => s.dismiss()}
            className="rounded px-2 py-1 text-xs text-muted transition hover:text-foreground"
          >
            Skip tour
          </button>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => s.back()}
              disabled={isFirst}
              aria-label="Previous step"
              className="rounded border border-border px-2.5 py-1 text-xs text-muted transition hover:text-foreground disabled:opacity-40"
            >
              Back
            </button>
            <button
              ref={nextRef}
              type="button"
              onClick={() => s.next(total)}
              aria-label={isLast ? "Finish tour" : "Next step"}
              className="rounded bg-accent px-3 py-1 text-xs font-medium text-white transition hover:bg-accent/90"
            >
              {isLast ? "Finish" : "Next"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
