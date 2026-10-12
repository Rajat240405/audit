import { useCallback } from "react";
import { streamEdit } from "@/api/chat";
import { useAppStore } from "@/store/useAppStore";
import { useDraftStore } from "@/store/useDraftStore";
import { useSessionStore } from "@/store/useSessionStore";
import { useToastStore } from "@/store/useToastStore";
import {
  beginEditRequest,
  cancelInFlightEdit,
  endEditRequest,
  useEditStore,
  type EditOrigin,
} from "@/store/useEditStore";

/** True only while the canvas is still bound to the answer the edit came from. */
function isBoundTo(origin: EditOrigin): boolean {
  const d = useDraftStore.getState();
  return (
    origin.messageId !== null &&
    d.activeSessionId === origin.sessionId &&
    d.activeMessageId === origin.messageId
  );
}

/** Stream an AI edit into a temp buffer, then STAGE it for user review.
 *  Editing state lives in the shared useEditStore; on completion the result
 *  is staged as pendingEdit — the user Accepts or Rejects (never auto-replace).
 *  The edit is bound to the answer it started from (see useEditStore).
 */
export function useEditDraft() {
  const draftStyle = useAppStore((s) => s.draftStyle);
  const editing = useEditStore((s) => s.editing);
  const editingLabel = useEditStore((s) => s.editingLabel);
  const pendingEdit = useEditStore((s) => s.pendingEdit);

  const edit = useCallback(
    (instruction: string, label?: string) => {
      const draft = useDraftStore.getState();
      if (!draft.content.trim()) return;
      const original = draft.content;
      const origin: EditOrigin = {
        sessionId: draft.activeSessionId,
        messageId: draft.activeMessageId,
      };
      const lbl = label ?? "edit";
      useEditStore.getState().start(lbl);
      const abort = beginEditRequest();
      let acc = "";
      streamEdit({
        document: original,
        instruction,
        draftStyle,
        signal: abort.signal,
        handlers: {
          onTokens: (t) => {
            if (!abort.signal.aborted) acc += t;
          },
          onError: (msg) => {
            if (abort.signal.aborted) return;
            console.error("edit failed:", msg);
            useToastStore.getState().push("error", `Edit failed: ${msg}`);
            useEditStore.getState().fail();
          },
          onDone: () => {
            endEditRequest(abort);
            // Aborted (session switch, new query, cancel): drop the result.
            if (abort.signal.aborted) return;
            if (acc.trim()) {
              if (!isBoundTo(origin)) {
                useEditStore.getState().finish();
                useToastStore.getState().push(
                  "info",
                  "Edit discarded: the canvas now shows a different answer."
                );
                return;
              }
              // STAGE for accept/reject instead of auto-applying
              useEditStore.getState().stage(original, acc, lbl, origin);
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
    if (!p) return;
    if (!isBoundTo(p)) {
      // The user moved to another answer after this edit was staged.
      useToastStore.getState().push(
        "error",
        "This edit belongs to a different answer and was discarded."
      );
      useEditStore.getState().reject();
      return;
    }
    useDraftStore.getState().applyEdit(p.revised);
    // In-place: update the message this edit was made for (same message the
    // canvas is bound to), so the sidebar transcript shows the edited answer.
    if (p.sessionId && p.messageId) {
      useSessionStore.getState().updateMessage(p.sessionId, p.messageId, { content: p.revised });
    }
    useToastStore.getState().push("success", `✓ Accepted "${p.label}"`);
    useEditStore.getState().accept();
  }, []);

  const reject = useCallback(() => {
    const p = useEditStore.getState().pendingEdit;
    if (p) {
      useToastStore.getState().push("info", `Rejected "${p.label}" — draft unchanged`);
      useEditStore.getState().reject();
    }
  }, []);

  const cancel = useCallback(() => {
    cancelInFlightEdit();
  }, []);

  return { edit, editing, editingLabel, pendingEdit, accept, reject, cancel };
}
