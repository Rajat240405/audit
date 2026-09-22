import { useEffect, useRef } from "react";

/**
 * Auto-scroll that respects the user.
 *
 * While the assistant thinks or streams an answer the UI used to force-scroll
 * to the bottom on every token, which made it impossible to read earlier
 * output during generation. These hooks keep auto-scrolling ONLY while the user
 * is already at the bottom; the moment they scroll up, their position is kept.
 * Scrolling back to the bottom re-enables auto-scroll.
 *
 * Two variants:
 *  - useStickyScroll        — the ref'd element IS the scroll container
 *                             (e.g. the reasoning pane in ModelActivityPanel).
 *  - useStickyAnchorScroll  — the ref'd element is an anchor inside a scroll
 *                             container found by walking up the DOM (e.g. the
 *                             streaming draft anchor in DraftCanvas, whose
 *                             scroller is whichever ancestor overflows).
 */

/** Nearest ancestor (or the element itself) that actually scrolls. */
function findScroller(start: HTMLElement): HTMLElement | null {
  let el: HTMLElement | null = start;
  while (el) {
    if (el.scrollHeight > el.clientHeight + 1) {
      const style = window.getComputedStyle(el);
      const oy = style.overflowY;
      if (oy === "auto" || oy === "scroll") return el;
    }
    el = el.parentElement;
  }
  return null;
}

/** Tolerance below which the user counts as "at the bottom". */
const EDGE_PX = 48;

/** Ref-is-the-scroller variant. Attach `onScroll` to the ref'd element. */
export function useStickyScroll<T extends HTMLElement>(deps: unknown[]) {
  const ref = useRef<T>(null);
  const stick = useRef(true);

  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < EDGE_PX;
  };

  useEffect(() => {
    const el = ref.current;
    if (!el || !stick.current) return;
    el.scrollTop = el.scrollHeight;
    // deps intentionally drive the scroll — see hook docs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { ref, onScroll, isStuck: () => stick.current };
}

/** Anchor variant: keeps the nearest scrollable ancestor pinned to the anchor
 *  while the user is at its bottom. */
export function useStickyAnchorScroll<T extends HTMLElement>(deps: unknown[]) {
  const ref = useRef<T>(null);
  const stick = useRef(true);

  // Re-resolved on EVERY render (no dep array): while a stream grows, the
  // nearest scrollable ancestor can APPEAR (content overflowing only after N
  // tokens) or move. A bind-once listener would have bound to nothing while
  // the canvas was short and then never track the real scroller.
  useEffect(() => {
    const anchor = ref.current;
    if (!anchor) return;
    const scroller = findScroller(anchor);
    if (!scroller) return;

    const onScroll = () => {
      stick.current =
        scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < EDGE_PX;
    };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => scroller.removeEventListener("scroll", onScroll);
  });

  useEffect(() => {
    const anchor = ref.current;
    if (!anchor || !stick.current) return;
    anchor.scrollIntoView({ block: "end" });
    // deps intentionally drive the scroll — see hook docs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { ref, isStuck: () => stick.current };
}
