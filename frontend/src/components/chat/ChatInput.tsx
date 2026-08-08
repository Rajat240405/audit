import { useState } from "react";
import { Send, Square } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { useDraftStore } from "@/store/useDraftStore";
import { useSessionStore } from "@/store/useSessionStore";
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";
import type { ExecutionMode } from "@/types";

interface ChatInputProps {
  onSend: (q: string) => void;
  onStop: () => void;
  streaming: boolean;
}

/** Query input docked in the left sidebar (matches the Stitch design). */
export function ChatInput({ onSend, onStop, streaming }: ChatInputProps) {
  const [value, setValue] = useState("");
  const retrievalMode = useAppStore((s) => s.retrievalMode);
  const setRetrievalMode = useAppStore((s) => s.setRetrievalMode);
  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);
  const saveVersion = useDraftStore((s) => s.saveVersion);
  const resetDraft = useDraftStore((s) => s.reset);
  const clearMessages = useSessionStore((s) => s.clearMessages);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const turns = useSessionStore(
    (s) => s.sessions.find((x) => x.id === s.activeSessionId)?.messages.length ?? 0
  );

  const submit = () => {
    const q = value.trim();
    if (!q || streaming) return;
    setValue("");
    onSend(q);
  };

  const clearAll = () => {
    resetDraft();
    if (activeSessionId) clearMessages(activeSessionId);
  };

  return (
    <div className="border-t border-border bg-background p-3">
      <div className="mb-1.5 flex items-center justify-between text-[11px] text-muted">
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-gray-400" />
          <span>Memory: {turns} turns</span>
        </div>
        <button className="text-accent hover:underline" onClick={clearAll}>
          clear
        </button>
      </div>

      <div className="rounded-xl border border-border bg-surface p-3 shadow-sm">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={2}
          placeholder="Enter audit query..."
          className="w-full resize-none border-none bg-transparent p-0 text-sm text-foreground placeholder:text-muted focus:outline-none"
        />

        <div className="mt-2 flex w-fit items-center gap-0.5 rounded-full bg-surface-2 p-0.5">
          {(["hybrid", "graph"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setRetrievalMode(m)}
              className={cn(
                "rounded-full px-2.5 py-0.5 text-[10px] font-medium",
                retrievalMode === m
                  ? "bg-foreground text-background"
                  : "text-muted hover:text-foreground"
              )}
            >
              {m === "hybrid" ? "Hybrid RAG" : "GraphRAG"}
            </button>
          ))}
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-2">
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as ExecutionMode)}
            className="rounded-full border border-border bg-surface-2 px-2.5 py-1 text-[11px] text-foreground focus:outline-none"
          >
            <option value="fast">Standard</option>
            <option value="deep">Smart Auto</option>
          </select>
          <Button variant="secondary" size="sm" onClick={() => saveVersion()}>
            Save Last Reply
          </Button>
          <Button variant="secondary" size="sm" onClick={() => saveVersion()}>
            Save Draft
          </Button>
        </div>

        <div className="mt-2 flex items-center justify-between border-t border-border pt-2">
          <span className="text-[11px] text-muted">{turns} turns</span>
          <div className="flex items-center gap-2">
            <button
              className="text-[11px] font-semibold text-muted hover:text-foreground"
              onClick={clearAll}
            >
              CLEAR
            </button>
            {streaming ? (
              <button
                onClick={onStop}
                className="flex h-7 w-7 items-center justify-center rounded-full bg-danger text-white"
                title="Stop"
              >
                <Square className="h-3.5 w-3.5" />
              </button>
            ) : (
              <button
                onClick={submit}
                disabled={!value.trim()}
                className="flex h-7 w-7 items-center justify-center rounded-full bg-foreground text-background disabled:opacity-40"
                title="Send"
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
