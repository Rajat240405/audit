import { useEffect, useState } from "react";
import { useChatActionsStore } from "@/store/useChatActionsStore";
import type { TourStep } from "./tourSteps";

/**
 * Measurement cadence — fast enough that the highlight tracks scrolling, and
 * cheap enough (one querySelector + getBoundingClientRect) to leave running for
 * the lifetime of a step so that late-mounting targets are still picked up.
 */
const POLL_MS = 80;

/**
 * Resolve a step's target, honouring fallbacks.
 *
 * A step may list several ``data-tour`` values so that conditionally rendered
 * controls degrade to a stable alternative — "Mode & Thinking Effort" points at
 * the Thinking Effort pill when Deep mode is on and at the Mode selector when
 * it is not. The first match in the list wins.
 */
function findTarget(target: string | string[] | undefined): HTMLElement | null {
  if (!target) return null;
  const ids = Array.isArray(target) ? target : [target];
  for (const id of ids) {
    // CSS.escape guards ids that would otherwise need quoting in a selector.
    const el = document.querySelector<HTMLElement>(`[data-tour="${CSS.escape(id)}"]`);
    if (el) return el;
  }
  return null;
}

function sameRect(a: DOMRect | null, b: DOMRect | null): boolean {
  if (!a || !b) return a === b;
  return (
    a.top === b.top &&
    a.left === b.left &&
    a.width === b.width &&
    a.height === b.height
  );
}

/**
 * Resolves and tracks the DOM element a tour step points at.
 *
 * Switches to the step's workspace tab first, then polls for the target so
 * conditionally-rendered panels and lazily mounted routes get time to appear.
 * Returns ``null`` when the target is genuinely absent, which makes the overlay
 * fall back to a centred card rather than breaking.
 *
 * Tab switching goes through the ``useChatActionsStore`` bridge: while the
 * Workspace is mounted it has replaced the default ``setTab`` with one that
 * drives its own local tab state, so this really does switch the visible tab.
 * If the Workspace is not mounted the call only updates the store, the target
 * is never found, and the step renders centred — no crash, no dead end.
 */
export function useTourTarget(step: TourStep | null): DOMRect | null {
  const [rect, setRect] = useState<DOMRect | null>(null);
  // Stringify so the effect does not re-run on every render of an inline array.
  const targetKey = step?.target ? JSON.stringify(step.target) : "";
  const tab = step?.tab;

  useEffect(() => {
    setRect(null);
    if (!step) return;

    const target: string | string[] | undefined = targetKey
      ? (JSON.parse(targetKey) as string | string[])
      : undefined;

    if (tab) {
      useChatActionsStore.getState().setTab(tab);
    }

    let found: HTMLElement | null = null;

    const measure = () => {
      const el = findTarget(target);
      if (!el) {
        // Nothing to point at (yet). Keep null; the card renders centred.
        found = null;
        setRect((prev) => (prev === null ? prev : null));
        return;
      }
      // Scroll only on first acquisition so we never fight the user's scroll.
      if (el !== found) {
        found = el;
        el.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
      setRect((prev) => {
        const next = el.getBoundingClientRect();
        return sameRect(prev, next) ? prev : next;
      });
    };

    measure();
    const id = window.setInterval(measure, POLL_MS);
    return () => window.clearInterval(id);
  }, [step, targetKey, tab]);

  return rect;
}
