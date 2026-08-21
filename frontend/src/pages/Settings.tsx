import { useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "@/store/useAppStore";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Modal } from "@/components/common/Modal";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  fetchIngestStatus,
  fetchIngestTargets,
  triggerIngest,
  uploadDocument,
  uploadToTarget,
  type IngestStatus,
  type IngestTargets,
} from "@/api/model";
import { useToastStore } from "@/store/useToastStore";
import { FileUp, FolderTree, RefreshCw, Upload, X } from "lucide-react";

/**
 * Settings panel — big ingest-focused card. Operators either drop files into
 * the INBOX (flat, auto-detect — the legacy flow) or pick a hierarchical
 * target: Source (Ministry/Parent) → Organization → Document Type. The whole
 * hierarchy is server-discovered from GET /api/ingest/targets (the same
 * registry the CLI uses) — nothing about orgs or document types is hardcoded
 * here. Uploads then run through the SAME ingestion pipeline as the CLI:
 * convert → dedup → embed new only → FAISS/BM25 update.
 */
export function Settings() {
  const open = useAppStore((s) => s.settingsOpen);
  const setOpen = useAppStore((s) => s.setSettingsOpen);
  const pushToast = useToastStore((s) => s.push);
  const [ingest, setIngest] = useState<IngestStatus | null>(null);
  const [targets, setTargets] = useState<IngestTargets | null>(null);
  const [targetsError, setTargetsError] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // hierarchy selection ("inbox" = flat legacy flow)
  const [sourceSel, setSourceSel] = useState("inbox");
  const [orgSel, setOrgSel] = useState("");
  const [docTypeSel, setDocTypeSel] = useState("");

  const refreshIngest = () => {
    fetchIngestStatus()
      .then(setIngest)
      .catch(() => setIngest(null));
  };

  useEffect(() => {
    if (!open) return;
    refreshIngest();
    fetchIngestTargets()
      .then((t) => {
        setTargets(t);
        setTargetsError(null);
      })
      .catch((e) => setTargetsError(e instanceof Error ? e.message : "targets unavailable"));
  }, [open]);

  useEffect(() => {
    if (!ingest?.running) return;
    const t = setInterval(refreshIngest, 2000);
    return () => clearInterval(t);
  }, [ingest?.running]);

  const uploadSources = useMemo(
    () => (targets?.sources ?? []).filter((s) => s.upload !== false),
    [targets],
  );
  const source = uploadSources.find((s) => s.name === sourceSel) ?? null;
  const hierarchical = !!source?.hierarchical;
  const org = source?.orgs.find((o) => o.slug === orgSel) ?? null;
  const category = org?.categories.find((c) => c.document_type === docTypeSel) ?? null;

  const pickSource = (name: string) => {
    setSourceSel(name);
    setOrgSel("");
    setDocTypeSel("");
  };

  const pickFiles = (list: FileList | null) => {
    if (!list) return;
    setFiles((prev) => [...prev, ...Array.from(list)]);
  };

  const onUploadAndIngest = async () => {
    if (files.length === 0) {
      pushToast("info", "Choose files to upload first");
      return;
    }
    if (hierarchical && !orgSel) {
      pushToast("error", "Organization is required");
      return;
    }
    if (hierarchical && !docTypeSel) {
      pushToast("error", "Document type is required");
      return;
    }
    setUploading(true);
    try {
      for (const f of files) {
        if (hierarchical) {
          await uploadToTarget(f, { source: sourceSel, org: orgSel, document_type: docTypeSel });
        } else {
          await uploadDocument(f);
        }
      }
      pushToast(
        "success",
        hierarchical
          ? `${files.length} file(s) staged → ${source?.label} / ${org?.label} / ${category?.label}`
          : `${files.length} file(s) saved to inbox`,
      );
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
  const last = ingest?.last;
  const lastVerdicts = last?.files ?? [];

  return (
    <Modal
      open={open}
      onOpenChange={setOpen}
      title="Ingest Documents"
      className="max-w-2xl"
    >
      <div className="space-y-4 text-sm">
        <p className="text-[12px] leading-relaxed text-muted">
          Add new documents to the knowledge base. Pick a place in the
          ministry/organization tree (or the auto-detect inbox), then upload —
          files are converted, embedded and indexed; they become queryable
          (sources + Q&amp;A) immediately. Existing documents are never
          re-embedded: only new ones are.
        </p>

        {/* ── Hierarchy selector (server-discovered; no hardcoded orgs) ── */}
        <div className="space-y-2 rounded-lg border border-border bg-surface-2/40 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] font-bold uppercase tracking-widest text-muted">
              Ingest into
            </span>
            <Select value={sourceSel} onValueChange={pickSource} disabled={busy}>
              <SelectTrigger className="w-44">
                <SelectValue placeholder="Source" />
              </SelectTrigger>
              <SelectContent>
                {uploadSources.map((s) => (
                  <SelectItem key={s.name} value={s.name}>
                    {s.hierarchical ? s.label : "Inbox (auto-detect)"}
                    {s.discovered ? " · discovered" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {hierarchical && (
              <Select value={orgSel} onValueChange={setOrgSel} disabled={busy}>
                <SelectTrigger className="w-44">
                  <SelectValue placeholder="Organization" />
                </SelectTrigger>
                <SelectContent>
                  {(source?.orgs ?? []).map((o) => (
                    <SelectItem key={o.slug} value={o.slug}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}

            {hierarchical && (
              <Select value={docTypeSel} onValueChange={setDocTypeSel} disabled={busy || !orgSel}>
                <SelectTrigger className="w-44">
                  <SelectValue placeholder="Document type" />
                </SelectTrigger>
                <SelectContent>
                  {(org?.categories ?? []).map((c) => (
                    <SelectItem key={c.document_type} value={c.document_type}>
                      {c.label}
                      {c.files > 0 ? ` (${c.files})` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          {targetsError && (
            <p className="text-[11px] text-warning">
              Hierarchy unavailable ({targetsError}) — inbox upload still works.
            </p>
          )}

          {/* selected destination + mini knowledge-tree preview */}
          {hierarchical && source && (
            <div className="rounded-md border border-border/60 bg-surface-2/30 p-2.5">
              <div className="flex items-center gap-1.5 text-[11px] text-muted">
                <FolderTree className="h-3.5 w-3.5" />
                {category ? (
                  <span>
                    Destination: <span className="font-mono text-foreground/80">data/{category.path}/</span>
                    {category.files > 0 ? ` · ${category.files} file(s)` : " · new folder"}
                  </span>
                ) : (
                  <span>Select an organization and document type</span>
                )}
              </div>
              <div className="mt-1.5 max-h-36 space-y-1 overflow-y-auto pl-1 text-[11px] leading-relaxed">
                {source.orgs.map((o) => (
                  <div key={o.slug}>
                    <span className={o.slug === orgSel ? "font-semibold text-accent" : "text-foreground/70"}>
                      {o.label}
                    </span>
                    <span className="ml-2 text-muted">
                      {o.categories
                        .filter((c) => c.files > 0)
                        .map((c) => `${c.label}: ${c.files}`)
                        .join(" · ") || "empty"}
                    </span>
                    {o.slug === orgSel && category && category.file_names.length > 0 && (
                      <div className="ml-4 mt-0.5 space-y-0.5 text-muted">
                        {category.file_names.slice(0, 8).map((n) => (
                          <div key={n} className="truncate font-mono">└ {n}</div>
                        ))}
                        {category.file_names.length > 8 && (
                          <div className="italic">… {category.file_names.length - 8} more</div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* action bar: choose -> upload & ingest -> refresh */}
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

          <Button
            variant="success"
            onClick={onUploadAndIngest}
            disabled={busy || files.length === 0 || (hierarchical && (!orgSel || !docTypeSel))}
            className="px-4"
          >
            <Upload className="mr-1.5 h-4 w-4" />
            {uploading ? "Uploading…" : ingest?.running ? "Ingesting…" : files.length > 0 ? `Upload & Ingest (${files.length})` : "Upload & Ingest"}
          </Button>

          <Button variant="outline" onClick={() => triggerIngest().then(refreshIngest).catch(() => pushToast("error", "Ingest failed"))} disabled={busy}>
            <RefreshCw className="mr-1.5 h-4 w-4" />
            Ingest staged files
          </Button>

          <span className="ml-auto flex items-center gap-2">
            {(ingest?.pending_uploads ?? 0) > 0 && (
              <Badge variant="warning" className="px-2 py-1 text-[11px]">
                {ingest?.pending_uploads} staged
              </Badge>
            )}
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

        {/* status — embeddings + per-file verdicts */}
        <div className="rounded-lg border border-border bg-surface-2/30 p-3">
          <p className="mb-1 text-[11px] font-bold uppercase tracking-widest text-muted">
            Status
          </p>
          {ingest?.running ? (
            <p className="text-[12px] text-accent">
              ⏳ Running — converting → embedding (bge-m3) → indexing…
            </p>
          ) : last ? (
            <div className="space-y-1.5 text-[12px]">
              <p className="text-foreground/80">{last.message}</p>
              <p className="text-muted">
                at {new Date(last.at).toLocaleTimeString()} · {last.ok} file(s) ok ·{" "}
                {last.records} record(s) added
              </p>
              {typeof last.received === "number" && last.received > 0 && (
                <p className="text-muted">
                  Documents received: {last.received} · New: {last.new_documents ?? 0} ·
                  Duplicates skipped: {last.duplicates ?? 0} · Embedded:{" "}
                  {last.records_embedded ?? 0} · Failed: {last.failed_documents ?? 0}
                </p>
              )}
              {lastVerdicts.length > 0 && (
                <div className="space-y-0.5 pt-0.5">
                  {lastVerdicts.map((v, i) => (
                    <div key={i} className="flex items-center gap-2 text-[11px]">
                      <Badge
                        variant={
                          v.verdict === "new"
                            ? "success"
                            : v.verdict === "failed"
                              ? "danger"
                              : "muted"
                        }
                      >
                        {v.verdict === "new"
                          ? "new"
                          : v.verdict === "duplicate" || v.verdict === "skipped_duplicate_pdf"
                            ? "duplicate"
                            : "failed"}
                      </Badge>
                      <span className="truncate font-mono">{v.name}</span>
                      {v.message && <span className="truncate text-muted">— {v.message}</span>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <p className="text-[12px] text-muted">No ingest run yet.</p>
          )}
          {!ingest?.running && last && (
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
