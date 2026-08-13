import { useEffect, useRef } from "react";
import {
  Brain,
  CheckCircle2,
  FileText,
  Loader2,
  Sparkles,
  X,
} from "lucide-react";
import { useActivityStore } from "@/store/useActivityStore";
import { useDocViewerStore } from "@/store/useDocViewerStore";
import { useChatActionsStore } from "@/store/useChatActionsStore";
import { cn } from "@/utils/cn";
import { reasoningWaitMessage } from "@/utils/reasoningLabel";

const STEPS = [
  { key: "retrieving", label: "Retrieving" },
  { key: "thinking", label: "Thinking" },
  { key: "generating", label: "Generating" },
] as const;

/**
 * Model Activity panel — a right-side drawer that shows, live, what the
 * backend is doing: the documents the model received, its chain-of-thought
 * (qwen3 reasoning), and the visible answer streaming in. Once the answer
 * starts, a "Go to canvas" button jumps straight to the draft.
 */
export function ModelActivityPanel() {
  const open = useActivityStore((s) => s.open);
  const phase = useActivityStore((s) => s.phase);
  const sources = useActivityStore((s) => s.sources);
  const reasoning = useActivityStore((s) => s.reasoning);
  const answerChars = useActivityStore((s) => s.answerChars);
  const model = useActivityStore((s) => s.model);
  const question = useActivityStore((s) => s.question);
  const error = useActivityStore((s) => s.error);
  const closePanel = useActivityStore((s) => s.closePanel);
  const openDoc = useDocViewerStore((s) => s.openDoc);
  const setTab = useChatActionsStore((s) => s.setTab);

  const reasonRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    reasonRef.current?.scrollTo({ top: reasonRef.current.scrollHeight });
  }, [reasoning]);

  if (!open) return null;

  const goToCanvas = () => {
    setTab?.("draft");
    closePanel();
  };

  const stepIndex = STEPS.findIndex((s) => s.key === phase);
  const activeStep = phase === "done" || phase === "error" ? 3 : Math.max(0, stepIndex);

  return (
    <div className="fixed inset-y-0 right-0 z-40 flex w-[26rem] max-w-[90vw] flex-col border-l border-border bg-surface shadow-2xl">
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between border-b border-border bg-surface-2/60 px-4 py-3">
        <div className="flex items-center gap-2">
          <Brain className="h-4 w-4 text-accent" />
          <span className="text-sm font-semibold">Model Activity</span>
        </div>
        <button
          onClick={closePanel}
          className="rounded p-1 text-muted hover:bg-surface-2 hover:text-foreground"
          title="Close"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-3">
        {question && (
          <p className="rounded-md border border-border bg-surface-2/50 px-3 py-2 text-[11px] text-muted">
            <span className="font-semibold text-foreground">Query:</span> {question}
          </p>
        )}

        {/* Phase stepper */}
        <div className="flex items-center gap-1">
          {STEPS.map((s, i) => {
            const state =
              phase === "idle"
                ? "pending"
                : phase === "error" && i >= activeStep
                  ? "error"
                  : i < activeStep || phase === "done"
                    ? "done"
                    : i === activeStep
                      ? "current"
                      : "pending";
            return (
              <div key={s.key} className="flex flex-1 items-center gap-1">
                <div
                  className={cn(
                    "flex flex-1 flex-col items-center gap-1 rounded-md border px-1 py-1.5 text-[9px] font-semibold uppercase tracking-wide",
                    state === "done" && "border-success/40 bg-success/10 text-success",
                    state === "current" && "border-accent/40 bg-accent/10 text-accent",
                    state === "error" && "border-danger/40 bg-danger/10 text-danger",
                    state === "pending" && "border-border text-muted"
                  )}
                >
                  {state === "done" ? (
                    <CheckCircle2 className="h-3 w-3" />
                  ) : state === "current" ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <span className="h-3 w-3 rounded-full border border-current opacity-50" />
                  )}
                  {s.label}
                </div>
                {i < STEPS.length - 1 && <div className="h-px w-2 bg-border" />}
              </div>
            );
          })}
        </div>

        {model && (
          <p className="text-[10px] text-muted">
            Model: <span className="font-mono text-foreground/80">{model}</span>
          </p>
        )}

        {phase === "error" && (
          <p className="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-[11px] text-danger">
            {error ?? "Generation failed."}
          </p>
        )}

        {/* What the model received */}
        <section>
          <p className="mb-1.5 text-[10px] font-bold uppercase tracking-widest text-muted">
            What the model received ({sources.length} documents)
          </p>
          {sources.length === 0 ? (
            <p className="text-[11px] text-muted">Retrieving documents…</p>
          ) : (
            <div className="space-y-1">
              {sources.map((s, i) => (
                <button
                  key={s.doc_id + i}
                  onClick={() => openDoc(s)}
                  className="flex w-full items-center gap-2 rounded-md border border-border bg-background/40 px-2 py-1.5 text-left hover:border-accent/40 hover:bg-surface-2"
                  title="Open the full document"
                >
                  <FileText className="h-3 w-3 shrink-0 text-accent" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[11px] font-medium text-foreground/90">
                      {s.doc_id}
                    </span>
                    <span className="block truncate text-[10px] text-muted">
                      {s.subject || "—"} · {s.answer.length.toLocaleString()} chars
                    </span>
                  </span>
                </button>
              ))}
            </div>
          )}
        </section>

        {/* Thinking */}
        <section>
          <p className="mb-1.5 flex items-center gap-1 text-[10px] font-bold uppercase tracking-widest text-muted">
            <Sparkles className="h-3 w-3" />
            Model thinking
          </p>
          <div
            ref={reasonRef}
            className={cn(
              "max-h-56 overflow-y-auto whitespace-pre-wrap rounded-md border border-border bg-background/60 p-2.5 font-mono text-[11px] leading-relaxed text-muted",
              !reasoning && phase === "thinking" && "flex items-center gap-2 text-foreground/70"
            )}
          >
            {reasoning ? (
              reasoning
            ) : phase === "thinking" ? (
              <>
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-accent" />
                Model is reasoning… (qwen3 thinks before answering; this can take a bit)
              </>
            ) : phase === "generating" || phase === "done" ? (
              "No reasoning streamed — model produced the answer directly."
            ) : (
              "Waiting for the model…"
            )}
          </div>
        </section>

        {/* Live answer progress + redirect */}
        {(phase === "generating" || phase === "done") && (
          <section className="flex items-center justify-between gap-3 rounded-md border border-accent/30 bg-accent/5 px-3 py-2">
            <div>
              <p className="text-[11px] font-semibold text-foreground">
                {phase === "generating" ? "Answer is generating…" : "Answer ready ✓"}
              </p>
              <p className="text-[10px] text-muted">
                {answerChars.toLocaleString()} characters streamed
              </p>
            </div>
            <button
              onClick={goToCanvas}
              className="shrink-0 rounded-md bg-accent px-3 py-1.5 text-[11px] font-bold text-white hover:opacity-90"
            >
              Go to canvas →
            </button>
          </section>
        )}

        {phase === "done" && !answerChars && (
          <p className="text-[11px] text-muted">Generation finished.</p>
        )}
      </div>
    </div>
  );
}
