import { useCallback, useRef, useState } from "react";
import { streamEdit } from "@/api/chat";
import { useAppStore } from "@/store/useAppStore";
import { useDraftStore } from "@/store/useDraftStore";

/** Stream an AI edit into a temp buffer, then commit via applyEdit on done. */
export function useEditDraft() {
  const [editing, setEditing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const draftStyle = useAppStore((s) => s.draftStyle);

  const edit = useCallback(
    (instruction: string) => {
      const draft = useDraftStore.getState();
      if (!draft.content.trim()) return;
      setEditing(true);
      const abort = new AbortController();
      abortRef.current = abort;
      let acc = "";
      streamEdit({
        document: draft.content,
        instruction,
        draftStyle,
        signal: abort.signal,
        handlers: {
          onTokens: (t) => {
            acc += t;
          },
          onError: (msg) => {
            console.error("edit failed:", msg);
            setEditing(false);
          },
          onDone: () => {
            if (acc.trim()) {
              useDraftStore.getState().applyEdit(acc);
            }
            setEditing(false);
          },
        },
      });
    },
    [draftStyle]
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setEditing(false);
  }, []);

  return { edit, editing, cancel };
}
