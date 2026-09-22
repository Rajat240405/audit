import { useEffect, useState } from "react";
import { Modal } from "@/components/common/Modal";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  deleteKnowledge,
  myKnowledge,
  updateKnowledge,
} from "@/api/model";
import { useAppStore } from "@/store/useAppStore";
import { useToastStore } from "@/store/useToastStore";
import type { KnowledgeRecord } from "@/types";
import { getKnowledgeOwner, KNOWLEDGE_OWNER_KEY } from "@/components/chat/SaveKnowledgeModal";

/**
 * "My Saved Knowledge" (Settings area): the current user's own contributions,
 * each editable and deletable. Deleting removes ONLY that user's record —
 * another saver's identical contribution is a separate record and survives.
 */
export function MyKnowledgeModal() {
  const open = useAppStore((s) => s.knowledgeOpen);
  const setOpen = useAppStore((s) => s.setKnowledgeOpen);
  const pushToast = useToastStore((s) => s.push);

  const [owner, setOwner] = useState("");
  const [records, setRecords] = useState<KnowledgeRecord[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState<KnowledgeRecord | null>(null);
  const [editQuestion, setEditQuestion] = useState("");
  const [editAnswer, setEditAnswer] = useState("");
  const [deleting, setDeleting] = useState<KnowledgeRecord | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = (name: string) => {
    if (!name) return;
    setLoading(true);
    myKnowledge(name)
      .then((r) => setRecords(r.records))
      .catch((e) => {
        setRecords([]);
        pushToast("error", `Could not load saved knowledge: ${e.message}`);
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (!open) return;
    const name = getKnowledgeOwner();
    setOwner(name);
    setRecords(null);
    setEditing(null);
    setDeleting(null);
    if (name) refresh(name);
  }, [open]);

  const saveOwner = () => {
    const name = owner.trim();
    if (!name) return;
    try {
      localStorage.setItem(KNOWLEDGE_OWNER_KEY, name);
    } catch {
      /* storage unavailable — works for this session only */
    }
    refresh(name);
  };

  const startEdit = (rec: KnowledgeRecord) => {
    setEditing(rec);
    setEditQuestion(rec.question);
    setEditAnswer(rec.answer);
  };

  const saveEdit = async () => {
    if (!editing || busy) return;
    setBusy(true);
    try {
      await updateKnowledge(editing.knowledge_id, {
        question: editQuestion,
        answer: editAnswer,
        savedBy: owner,
      });
      pushToast("success", "Saved knowledge updated");
      setEditing(null);
      refresh(owner);
    } catch (e) {
      pushToast("error", `Update failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const confirmDelete = async () => {
    if (!deleting || busy) return;
    setBusy(true);
    try {
      await deleteKnowledge(deleting.knowledge_id, owner);
      pushToast("success", "Your contribution was removed");
      setDeleting(null);
      refresh(owner);
    } catch (e) {
      pushToast("error", `Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={open}
      onOpenChange={setOpen}
      title="My Saved Knowledge"
      className="max-w-2xl"
    >
      <div className="space-y-4 text-sm" data-testid="my-knowledge">
        {!owner ? (
          <div className="space-y-2">
            <p className="text-[12px] text-muted">
              Enter the name you save knowledge under to see your contributions.
            </p>
            <div className="flex gap-2">
              <input
                data-testid="my-kn-name"
                value={owner}
                onChange={(e) => setOwner(e.target.value)}
                placeholder="e.g. Rajat"
                className="flex-1 rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
              />
              <Button data-testid="my-kn-name-save" onClick={saveOwner} disabled={!owner.trim()}>
                Show my knowledge
              </Button>
            </div>
          </div>
        ) : loading ? (
          <p className="text-muted">Loading…</p>
        ) : !records || records.length === 0 ? (
          <p className="text-muted">
            No saved contributions yet for “{owner}”. Use “Save to Knowledge” under an answer.
          </p>
        ) : (
          <>
            <p className="text-[12px] text-muted">
              {records.length} contribution{records.length === 1 ? "" : "s"} saved as{" "}
              <span className="font-medium text-foreground/80">{owner}</span>. Deleting yours
              never removes someone else&apos;s copy of the same answer.
            </p>
            <div className="max-h-[50vh] space-y-2 overflow-y-auto pr-1">
              {records.map((rec) => (
                <div
                  key={rec.knowledge_id}
                  data-testid="my-kn-record"
                  className="rounded-lg border border-border bg-surface-2/40 p-3"
                >
                  {editing?.knowledge_id === rec.knowledge_id ? (
                    <div className="space-y-2">
                      <input
                        data-testid="my-kn-edit-question"
                        value={editQuestion}
                        onChange={(e) => setEditQuestion(e.target.value)}
                        className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent"
                      />
                      <textarea
                        data-testid="my-kn-edit-answer"
                        value={editAnswer}
                        onChange={(e) => setEditAnswer(e.target.value)}
                        rows={6}
                        className="w-full resize-y rounded border border-border bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent"
                      />
                      <div className="flex justify-end gap-2">
                        <Button variant="ghost" onClick={() => setEditing(null)} disabled={busy}>
                          Cancel
                        </Button>
                        <Button data-testid="my-kn-edit-save" onClick={() => void saveEdit()} disabled={busy}>
                          {busy ? "Saving…" : "Save changes"}
                        </Button>
                      </div>
                    </div>
                  ) : deleting?.knowledge_id === rec.knowledge_id ? (
                    <div className="space-y-2">
                      <p className="text-[13px] text-foreground/90">
                        Delete your saved answer for “{rec.question.slice(0, 80)}”?
                        {rec.version > 1 && " (version " + rec.version + ")"}
                      </p>
                      <p className="text-[11px] text-muted">
                        Only YOUR contribution is removed. If someone else saved the same
                        answer, theirs stays.
                      </p>
                      <div className="flex justify-end gap-2">
                        <Button variant="ghost" onClick={() => setDeleting(null)} disabled={busy}>
                          Keep it
                        </Button>
                        <Button
                          data-testid="my-kn-delete-confirm"
                          variant="destructive"
                          onClick={() => void confirmDelete()}
                          disabled={busy}
                        >
                          {busy ? "Deleting…" : "Delete"}
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <p className="line-clamp-1 text-[13px] font-medium text-foreground/90">
                        {rec.question}
                      </p>
                      <p className="line-clamp-2 whitespace-pre-wrap text-[12px] text-muted">
                        {rec.answer}
                      </p>
                      <div className="mt-1 flex flex-wrap items-center gap-1.5">
                        <Badge variant="muted">v{rec.version}</Badge>
                        <Badge variant="muted">
                          {rec.sources.length} source{rec.sources.length === 1 ? "" : "s"}
                        </Badge>
                        <Badge variant="muted">{rec.updated_at.slice(0, 10)}</Badge>
                        <span className="flex-1" />
                        <Button
                          data-testid="my-kn-edit"
                          variant="ghost"
                          size="sm"
                          onClick={() => startEdit(rec)}
                        >
                          Edit
                        </Button>
                        <Button
                          data-testid="my-kn-delete"
                          variant="ghost"
                          size="sm"
                          onClick={() => setDeleting(rec)}
                        >
                          Delete
                        </Button>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </Modal>
  );
}
