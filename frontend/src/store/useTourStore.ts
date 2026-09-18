import { create } from "zustand";
import { persist } from "zustand/middleware";

/**
 * Product-tour state.
 *
 * Only ``hasSeenTour`` is persisted (same middleware + naming convention as
 * ``useNotesStore`` / ``useSessionStore``), so the tour auto-starts exactly
 * once per browser and is never forced again on later page loads. ``open`` and
 * ``stepIndex`` are deliberately session-scoped: a refresh mid-tour restarts
 * it from the beginning rather than resuming into a half-read state.
 */
interface TourState {
  /** Set once the user finishes or skips the tour. Persisted. */
  hasSeenTour: boolean;
  open: boolean;
  stepIndex: number;

  /** Open the tour from step 0 (used by the header Guide button). */
  start: () => void;
  next: (totalSteps: number) => void;
  back: () => void;
  goTo: (index: number) => void;
  /** Close and remember — the tour will not auto-start again. */
  dismiss: () => void;
  /** Close without remembering (Escape during an auto-started first run). */
  close: () => void;
}

export const useTourStore = create<TourState>()(
  persist(
    (set, get) => ({
      hasSeenTour: false,
      open: false,
      stepIndex: 0,

      start: () => set({ open: true, stepIndex: 0 }),
      next: (totalSteps) => {
        const { stepIndex } = get();
        if (stepIndex + 1 >= totalSteps) {
          set({ open: false, stepIndex: 0, hasSeenTour: true });
        } else {
          set({ stepIndex: stepIndex + 1 });
        }
      },
      back: () => set((s) => ({ stepIndex: Math.max(0, s.stepIndex - 1) })),
      goTo: (index) => set({ stepIndex: Math.max(0, index) }),
      dismiss: () => set({ open: false, stepIndex: 0, hasSeenTour: true }),
      close: () => set({ open: false, stepIndex: 0 }),
    }),
    {
      name: "incois-onboarding",
      // Persist ONLY the "seen" flag. Persisting `open` would re-open the tour
      // after every refresh for a user who closed it with Escape.
      partialize: (s) => ({ hasSeenTour: s.hasSeenTour }),
    }
  )
);
