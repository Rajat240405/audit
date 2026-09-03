import { create } from "zustand";
import type { ToastKind } from "@/components/common/Toast";

export interface ToastItem {
  id: string;
  kind: ToastKind;
  message: string;
  /** Sticky toasts never auto-dismiss — they stay until the user closes them
   *  (used for cross-verification notices the user may need to act on). */
  sticky?: boolean;
}

interface ToastState {
  toasts: ToastItem[];
  push: (kind: ToastKind, message: string) => void;
  /** Persistent variant of push(): no auto-dismiss; the item stays in the
   *  list until the user explicitly dismisses it. */
  pushSticky: (kind: ToastKind, message: string) => void;
  dismiss: (id: string) => void;
}

let counter = 0;

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (kind, message) => {
    const id = `toast-${Date.now()}-${counter++}`;
    set((s) => ({ toasts: [...s.toasts, { id, kind, message }] }));
    // auto-dismiss after 4s (transient toasts only)
    setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    }, 4000);
  },
  pushSticky: (kind, message) => {
    const id = `toast-${Date.now()}-${counter++}`;
    set((s) => ({ toasts: [...s.toasts, { id, kind, message, sticky: true }] }));
    // no timeout — sticky toasts persist until the user dismisses them
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));
