import { useState } from "react";
import { Download } from "lucide-react";
import { useDraftStore } from "@/store/useDraftStore";
import { exportDraft, type ExportFormat } from "@/services/export";
import { Button } from "@/components/ui/button";

const FORMATS: Array<{ value: ExportFormat; label: string }> = [
  { value: "md", label: "Markdown" },
  { value: "docx", label: "DOCX" },
  { value: "txt", label: "TXT" },
];

/** Export menu — DOCX / PDF / Markdown / TXT (PDF is a future target). */
export function ExportMenu() {
  const content = useDraftStore((s) => s.content);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const doExport = async (format: ExportFormat) => {
    setBusy(true);
    try {
      const title = `incois-audit-draft-${new Date().toISOString().slice(0, 10)}`;
      await exportDraft(format, title, content || "(empty draft)");
      setOpen(false);
    } catch (err) {
      alert(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="relative">
      <Button variant="secondary" size="sm" onClick={() => setOpen((v) => !v)} disabled={!content}>
        <Download className="h-3.5 w-3.5" />
        Export
      </Button>
      {open && (
        <div className="absolute right-0 top-9 z-30 w-44 rounded-md border border-border bg-surface p-1 shadow-lg">
          {FORMATS.map((f) => (
            <button
              key={f.value}
              className="flex w-full items-center justify-between rounded px-2 py-1.5 text-xs hover:bg-surface-2 disabled:opacity-50"
              disabled={busy}
              onClick={() => doExport(f.value)}
            >
              {f.label}
              {busy && <span className="text-[10px] text-muted">…</span>}
            </button>
          ))}
          <div className="mt-1 border-t border-border pt-1 text-[10px] text-muted">
            PDF &amp; gov templates: coming soon
          </div>
        </div>
      )}
    </div>
  );
}
