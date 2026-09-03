import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, Info, X, XCircle } from "lucide-react";
import { cn } from "@/utils/cn";

export type ToastKind = "info" | "success" | "error";

export interface ToastItem {
  id: string;
  kind: ToastKind;
  message: string;
  /** Sticky toasts never auto-dismiss — used for cross-verification notices
   *  that must stay visible until the user closes them. */
  sticky?: boolean;
}

export function Toast({ toasts, onDismiss }: { toasts: ToastItem[]; onDismiss?: (id: string) => void }) {
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-80 flex-col gap-2">
      <AnimatePresence>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 40 }}
            className={cn(
              "pointer-events-auto flex items-start gap-2 rounded-md border bg-surface p-3 text-xs shadow-lg",
              t.kind === "success" && "border-success/40",
              t.kind === "error" && "border-danger/40",
              t.kind === "info" && "border-border",
              t.sticky && "border-l-2"
            )}
          >
            {t.kind === "success" && <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />}
            {t.kind === "error" && <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-danger" />}
            {t.kind === "info" && <Info className="mt-0.5 h-4 w-4 shrink-0 text-accent" />}
            <span className="leading-relaxed">{t.message}</span>
            {t.sticky && onDismiss && (
              <button
                type="button"
                aria-label="Dismiss"
                title="Dismiss"
                onClick={() => onDismiss(t.id)}
                className="ml-auto -mr-1 -mt-1 shrink-0 rounded p-0.5 text-muted hover:bg-surface-2 hover:text-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
