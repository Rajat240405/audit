import { useCallback, useEffect } from "react";
import { useAppStore } from "@/store/useAppStore";
import { useSessionStore } from "@/store/useSessionStore";
import { usePipelineStore } from "@/store/usePipelineStore";
import { showsHybridStages } from "@/lib/retrievalMode";
import { hybridStages, graphStages } from "@/utils/formatters";
import { getActiveRun, startChatRun, stopActiveRun, useChatRunStore } from "@/lib/chatRun";

/**
 * Thin React binding over the run registry (lib/chatRun.ts), which owns the
 * streaming flow. Components keep a declarative API:
 *
 *  - `send(question)` starts a run in the active chat (refused while any run is
 *    active).
 *  - `stop()` stops the active run, from any chat.
 *  - `running` is true when the active run belongs to the chat on screen.
 *  - `busyElsewhere` is true when a run is generating in a different chat.
 */
export function useChatStream() {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const run = useChatRunStore((s) => s.active);
  const retrievalMode = useAppStore((s) => s.retrievalMode);

  const running = !!run && run.sessionId === activeSessionId;
  const busyElsewhere = !!run && !running;

  const send = useCallback((question: string) => {
    startChatRun(question);
  }, []);

  const stop = useCallback(() => {
    stopActiveRun();
  }, []);

  // Stage definitions follow the retrieval mode. Skip while a run owns the
  // pipeline view, so a mode change cannot overwrite a live run's stages.
  useEffect(() => {
    if (getActiveRun()) return;
    usePipelineStore
      .getState()
      .setStages(showsHybridStages(retrievalMode) ? hybridStages() : graphStages());
  }, [retrievalMode]);

  return { send, stop, running, busyElsewhere };
}
