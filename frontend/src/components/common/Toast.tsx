import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, Info, XCircle } from "lucide-react";
import { cn } from "@/utils/cn";

export type ToastKind = "info" | "success" | "error";

export interface ToastItem {
  id: string;
  kind: ToastKind;
  message: string;
}

export function Toast({ toasts }: { toasts: ToastItem[] }) {
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
              t.kind === "info" && "border-border"
            )}
          >
            {t.kind === "success" && <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />}
            {t.kind === "error" && <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-danger" />}
            {t.kind === "info" && <Info className="mt-0.5 h-4 w-4 shrink-0 text-accent" />}
            <span className="leading-relaxed">{t.message}</span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
