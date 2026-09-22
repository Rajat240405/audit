import { useEffect, useState } from "react";
import { Modal } from "@/components/common/Modal";
import { Button } from "@/components/ui/button";
import { saveKnowledge } from "@/api/model";
import { useToastStore } from "@/store/useToastStore";
import type { SourceItem } from "@/types";

/** localStorage key for the manually entered display name (v1 identity). */
export const KNOWLEDGE_OWNER_KEY = "knowledge.ownerName";

/** Read the remembered display name (empty string when never set). */
export function getKnowledgeOwner(): string {
  try {
    return localStorage.getItem(KNOWLEDGE_OWNER_KEY) ?? "";
  } catch {
    return "";
  }
}

/** Remember the display name for the next save. */
function setKnowledgeOwner(name: string): void {
  try {
    localStorage.setItem(KNOWLEDGE_OWNER_KEY, name);
  } catch {
    /* storage unavailable (private mode) — saving still works, just no memory */
  }
}

interface SaveKnowledgeModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The user question this answer belongs to (never a fallback slice). */
  question: string;
  answer: string;
  sources: SourceItem[];
}

/**
 * Themed save dialog: who is saving, what question, which answer, how many
 * sources. Replaces the old `alert()` — and the old fallback that silently
 * used the ANSWER as the question when no preceding user message existed:
 * callers must pass a real question or keep the button disabled.
 */
export function SaveKnowledgeModal({
  open,
  onOpenChange,
  question,
  answer,
  sources,
}: SaveKnowledgeModalProps) {
  const pushToast = useToastStore((s) => s.push);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (open) setName(getKnowledgeOwner());
  }, [open]);

  const save = async () => {
    const owner = name.trim();
    if (!owner || busy) return;
    setBusy(true);
    try {
      setKnowledgeOwner(owner); // remember for the next save
      await saveKnowledge({
        question,
        answer,
        sources,
        savedBy: owner,
      });
      pushToast("success", "Saved to Knowledge ✓ (shared with everyone)");
      onOpenChange(false);
    } catch (e) {
      pushToast("error", `Save failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      title="Save to Knowledge"
      className="max-w-xl"
    >
      <div className="space-y-4">
        <div>
          <label htmlFor="kn-owner" className="mb-1 block text-xs font-medium text-muted">
            Your name
          </label>
          <input
            id="kn-owner"
            data-testid="kn-owner-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Rajat"
            autoComplete="off"
            className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-foreground outline-none focus:border-accent"
          />
          <p className="mt-1 text-[11px] text-muted">
            Saved answers are shared globally; your name shows as the contributor.
          </p>
        </div>

        <div>
          <span className="mb-1 block text-xs font-medium text-muted">Question</span>
          <div
            data-testid="kn-question-preview"
            className="max-h-24 overflow-y-auto rounded-lg border border-border bg-surface-2 px-3 py-2 text-sm text-foreground/90"
          >
            {question}
          </div>
        </div>

        <div>
          <span className="mb-1 block text-xs font-medium text-muted">Answer</span>
          <div
            data-testid="kn-answer-preview"
            className="max-h-40 overflow-y-auto whitespace-pre-wrap rounded-lg border border-border bg-surface-2 px-3 py-2 text-sm leading-relaxed text-foreground/90"
          >
            {answer.slice(0, 2000)}
            {answer.length > 2000 ? "…" : ""}
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-muted">
            {sources.length} source{sources.length === 1 ? "" : "s"} attached
          </span>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={busy}>
              Cancel
            </Button>
            <Button
              data-testid="kn-save-btn"
              onClick={() => void save()}
              disabled={busy || !name.trim()}
            >
              {busy ? "Saving…" : "Save to Knowledge"}
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  );
}
