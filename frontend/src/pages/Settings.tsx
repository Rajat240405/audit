import { useEffect, useRef, useState } from "react";
import { useAppStore } from "@/store/useAppStore";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Modal } from "@/components/common/Modal";
import {
  fetchIngestStatus,
  triggerIngest,
  uploadDocument,
  type IngestStatus,
} from "@/api/model";
import { useToastStore } from "@/store/useToastStore";
import { FileUp, RefreshCw, Upload, X } from "lucide-react";

/**
 * Settings panel — big ingest-focused card. The ONE action is
 * "Upload & Ingest": it saves files to data/inbox AND triggers the backend
 * (convert -> embed -> index) in one click. "Ingest inbox files" is kept for
 * files dropped directly into the folder by hand.
 */
export function Settings() {
  const open = useAppStore((s) => s.settingsOpen);
  const setOpen = useAppStore((s) => s.setSettingsOpen);
  const pushToast = useToastStore((s) => s.push);
  const [ingest, setIngest] = useState<IngestStatus | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const refreshIngest = () => {
    fetchIngestStatus()
      .then(setIngest)
      .catch(() => setIngest(null));
  };

  useEffect(() => {
    if (open) refreshIngest();
  }, [open]);

  useEffect(() => {
    if (!ingest?.running) return;
    const t = setInterval(refreshIngest, 2000);
    return () => clearInterval(t);
  }, [ingest?.running]);

  const pickFiles = (list: FileList | null) => {
    if (!list) return;
    setFiles((prev) => [...prev, ...Array.from(list)]);
  };

  const onUploadAndIngest = async () => {
    if (files.length === 0) {
      pushToast("info", "Choose files to upload first");
      return;
    }
    setUploading(true);
    try {
      for (const f of files) {
        await uploadDocument(f);
      }
      pushToast("success", `${files.length} file(s) saved to inbox`);
      setFiles([]);
      await triggerIngest();
      pushToast("info", "Ingest started — converting, embedding, indexing…");
      refreshIngest();
    } catch (e) {
      pushToast("error", e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const busy = uploading || ingest?.running;

  return (
    <Modal
      open={open}
      onOpenChange={setOpen}
      title="Ingest Documents"
      className="max-w-2xl"
    >
      <div className="space-y-4 text-sm">
        <p className="text-[12px] leading-relaxed text-muted">
          Add new documents to the knowledge base. One click: files are saved,
          converted, embedded and indexed — they become queryable (sources +
          Q&amp;A) immediately.
        </p>

        {/* ONE-LINE action bar: choose -> upload & ingest -> refresh */}
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface-2/50 p-3">
          <input
            ref={fileRef}
            type="file"
            multiple
            accept=".pdf,.txt,.md,.json,.jsonl"
            className="hidden"
            onChange={(e) => pickFiles(e.target.files)}
          />
          <Button variant="secondary" onClick={() => fileRef.current?.click()} disabled={busy}>
            <FileUp className="mr-1.5 h-4 w-4" />
            Choose files
          </Button>

          <Button variant="success" onClick={onUploadAndIngest} disabled={busy || files.length === 0} className="px-4">
            <Upload className="mr-1.5 h-4 w-4" />
            {uploading ? "Uploading…" : ingest?.running ? "Ingesting…" : files.length > 0 ? `Upload & Ingest (${files.length})` : "Upload & Ingest"}
          </Button>

          <Button variant="outline" onClick={() => triggerIngest().then(refreshIngest).catch(() => pushToast("error", "Ingest failed"))} disabled={busy}>
            <RefreshCw className="mr-1.5 h-4 w-4" />
            Ingest inbox files
          </Button>

          <span className="ml-auto flex items-center gap-2">
            <Badge variant={ingest?.pending ? "accent" : "default"} className="px-2 py-1 text-[11px]">
              {ingest?.pending ?? 0} in inbox
            </Badge>
            <Button variant="ghost" size="icon" onClick={refreshIngest} title="Refresh status">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </span>
        </div>

        {/* selected files */}
        {files.length > 0 && (
          <div className="space-y-1">
            {files.map((f, i) => (
              <div key={i} className="flex items-center justify-between rounded-md border border-border bg-surface-2/40 px-3 py-1.5 text-[12px]">
                <span className="truncate font-medium">{f.name}</span>
                <span className="ml-2 shrink-0 text-[11px] text-muted">{(f.size / 1024).toFixed(1)} KB</span>
                <button
                  className="ml-2 shrink-0 text-muted hover:text-danger"
                  onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                  title="Remove"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* status — embeddings confirmation */}
        <div className="rounded-lg border border-border bg-surface-2/30 p-3">
          <p className="mb-1 text-[11px] font-bold uppercase tracking-widest text-muted">
            Status
          </p>
          {ingest?.running ? (
            <p className="text-[12px] text-accent">
              ⏳ Running — converting → embedding (bge-m3) → indexing…
            </p>
          ) : ingest?.last ? (
            <div className="space-y-0.5 text-[12px]">
              <p className="text-foreground/80">{ingest.last.message}</p>
              <p className="text-muted">
                at {new Date(ingest.last.at).toLocaleTimeString()} · {ingest.last.ok} file(s) ok ·{" "}
                {ingest.last.records} record(s) added
              </p>
            </div>
          ) : (
            <p className="text-[12px] text-muted">No ingest run yet.</p>
          )}
          {!ingest?.running && ingest?.last && (
            <p className="mt-1 text-[11px] text-muted">
              Tip: if it says “0 records added”, the document may already be in the corpus (dedup) —
              try a different file, or check it appears in Sources after a query.
            </p>
          )}
        </div>
      </div>
    </Modal>
  );
}
