import { motion } from "framer-motion";

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-10 text-muted">
      <motion.div
        className="h-6 w-6 rounded-full border-2 border-accent border-t-transparent"
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }}
      />
      <span className="text-xs">{label}</span>
    </div>
  );
}
