import { useCallback, useRef } from "react";
import { streamEdit } from "@/api/chat";
import { useAppStore } from "@/store/useAppStore";
import { useDraftStore } from "@/store/useDraftStore";
import { useToastStore } from "@/store/useToastStore";
import { useEditStore } from "@/store/useEditStore";

/** Stream an AI edit into a temp buffer, then STAGE it for user review.
 *  Editing state lives in the shared useEditStore; on completion the result
 *  is staged as pendingEdit — the user Accepts or Rejects (never auto-replace).
 */
export function useEditDraft() {
  const abortRef = useRef<AbortController | null>(null);
  const draftStyle = useAppStore((s) => s.draftStyle);
  const editing = useEditStore((s) => s.editing);
  const editingLabel = useEditStore((s) => s.editingLabel);
  const pendingEdit = useEditStore((s) => s.pendingEdit);

  const edit = useCallback(
    (instruction: string, label?: string) => {
      const draft = useDraftStore.getState();
      if (!draft.content.trim()) return;
      const original = draft.content;
      const lbl = label ?? "edit";
      useEditStore.getState().start(lbl);
      const abort = new AbortController();
      abortRef.current = abort;
      let acc = "";
      streamEdit({
        document: original,
        instruction,
        draftStyle,
        signal: abort.signal,
        handlers: {
          onTokens: (t) => {
            acc += t;
          },
          onError: (msg) => {
            console.error("edit failed:", msg);
            useToastStore.getState().push("error", `Edit failed: ${msg}`);
            useEditStore.getState().fail();
          },
          onDone: () => {
            if (acc.trim()) {
              // STAGE for accept/reject instead of auto-applying
              useEditStore.getState().stage(original, acc, lbl);
              useToastStore.getState().push(
                "info",
                `Edit "${lbl}" ready — review & accept or reject`
              );
            } else {
              useToastStore.getState().push("error", "Edit produced no output");
              useEditStore.getState().fail();
            }
          },
        },
      });
    },
    [draftStyle]
  );

  const accept = useCallback(() => {
    const p = useEditStore.getState().pendingEdit;
    if (p) {
      useDraftStore.getState().applyEdit(p.revised, `AI edit: ${p.label}`);
      useToastStore.getState().push("success", `✓ Accepted "${p.label}"`);
      useEditStore.getState().accept();
    }
  }, []);

  const reject = useCallback(() => {
    const p = useEditStore.getState().pendingEdit;
    if (p) {
      useToastStore.getState().push("info", `Rejected "${p.label}" — draft unchanged`);
      useEditStore.getState().reject();
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    useEditStore.getState().finish();
  }, []);

  return { edit, editing, editingLabel, pendingEdit, accept, reject, cancel };
}
