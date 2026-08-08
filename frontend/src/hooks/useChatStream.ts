import { useCallback, useEffect, useRef, useState } from "react";
import { streamChat } from "@/api/chat";
import { useAppStore } from "@/store/useAppStore";
import { useDraftStore } from "@/store/useDraftStore";
import { usePipelineStore } from "@/store/usePipelineStore";
import { useSessionStore } from "@/store/useSessionStore";
import { graphStages, hybridStages } from "@/utils/formatters";
import type { ChatMessage, SourceItem } from "@/types";

/**
 * Drives the full streaming flow: pipeline stages -> sources -> trace ->
 * token-by-token generation -> meta. Coordinates all stores; components stay
 * declarative.
 */
export function useChatStream() {
  const retrievalMode = useAppStore((s) => s.retrievalMode);
  const draftStore = useDraftStore;

  const abortRef = useRef<AbortController | null>(null);
  const [running, setRunning] = useState(false);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    draftStore.getState().cancelStream();
    usePipelineStore.getState().finish();
    setRunning(false);
  }, [draftStore]);

  // reset pipeline stages when mode changes
  useEffect(() => {
    usePipelineStore
      .getState()
      .setStages(retrievalMode === "graph" ? graphStages() : hybridStages());
  }, [retrievalMode]);

  const send = useCallback(
    (question: string) => {
      const app = useAppStore.getState();
      const sessions = useSessionStore.getState();
      let sessionId = sessions.activeSessionId;
      if (!sessionId) {
        sessionId = sessions.createSession();
      }
      const userMsg: ChatMessage = {
        id: Math.random().toString(36).slice(2),
        role: "user",
        content: question,
        createdAt: Date.now(),
      };
      sessions.addMessage(sessionId, userMsg);

      // prepare a placeholder assistant message we'll update live
      const assistantId = Math.random().toString(36).slice(2);
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
      };
      sessions.addMessage(sessionId, assistantMsg);

      const pipeline = usePipelineStore.getState();
      pipeline.start();
      pipeline.setStages(retrievalMode === "graph" ? graphStages() : hybridStages());

      const draft = useDraftStore.getState();
      draft.startStream();

      const abort = new AbortController();
      abortRef.current = abort;
      setRunning(true);

      streamChat({
        message: question,
        mode: app.mode,
        retrievalMode: app.retrievalMode,
        draftStyle: app.draftStyle,
        signal: abort.signal,
        handlers: {
          onStatus: (stage, message, done) => {
            usePipelineStore.getState().markStage(stage, done ? "done" : "running");
            usePipelineStore.getState().setMessage(message);
          },
          onSources: (sources) => {
            const typed = sources as SourceItem[];
            useDraftStore.getState().setSources(typed);
            sessions.updateMessage(sessionId, assistantId, { sources: typed });
          },
          onTrace: (trace) => {
            useDraftStore.getState().setTrace(trace as never);
            sessions.updateMessage(sessionId, assistantId, { trace: trace as never });
          },
          onTokens: (text) => {
            useDraftStore.getState().appendToken(text);
          },
          onMeta: (meta) => {
            const draft = useDraftStore.getState();
            draft.commitStream();
            draft.setLastMeta(meta as Record<string, unknown>);
            const finalContent = useDraftStore.getState().content;
            sessions.updateMessage(sessionId, assistantId, {
              content: finalContent,
              meta: meta as never,
            });
          },
          onGrounding: (grounding) => {
            // server-verified grounding overrides the client-side heuristic
            useDraftStore.setState({
              grounding: (grounding as Array<{
                text: string;
                found: boolean;
                source?: string;
              }>).map((g) => ({ text: g.text, found: g.found, source: g.source })),
            });
          },
          onFinal: (text, droppedCount, _dropped, judgeRewritten) => {
            const draft = useDraftStore.getState();
            if (judgeRewritten && text && text !== draft.content) {
              // The judge rewrote the answer to remove unsupported claims —
              // replace the draft with the corrected version.
              draft.applyEdit(text);
            } else if (text && text !== draft.content && draft.isStreaming) {
              // no rewrite — just commit the stream if it hasn't committed yet
              draft.commitStream();
            }
            if (droppedCount > 0 || judgeRewritten) {
              console.info(
                `[grounding] ${droppedCount} flagged, judge-rewrite=${judgeRewritten}`
              );
            }
            const finalContent = useDraftStore.getState().content || text;
            sessions.updateMessage(sessionId, assistantId, { content: finalContent });
          },
          onError: (message) => {
            useDraftStore.getState().cancelStream();
            const content = useDraftStore.getState().content || `⚠️ ${message}`;
            sessions.updateMessage(sessionId, assistantId, { content });
            usePipelineStore.getState().finish();
            setRunning(false);
          },
          onDone: () => {
            const draftState = useDraftStore.getState();
            if (draftState.isStreaming) {
              draftState.commitStream();
              const content = draftState.content;
              const { sources, trace, meta } = {
                sources: draftState.sources,
                trace: draftState.trace,
                meta: undefined,
              };
              void meta;
              sessions.updateMessage(sessionId, assistantId, { content, sources, trace: trace ?? undefined });
            }
            usePipelineStore.getState().finish();
            setRunning(false);
          },
        },
      });
    },
    [retrievalMode]
  );

  return { send, stop, running };
}
