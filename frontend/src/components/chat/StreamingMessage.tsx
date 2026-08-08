import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useDraftStore } from "@/store/useDraftStore";
import { usePipelineStore } from "@/store/usePipelineStore";

/**
 * Live token-by-token render of the in-progress generation. Shown in the
 * drafting workspace while streaming; the final content is committed to the
 * session transcript on done.
 */
export function StreamingMessage() {
  const isStreaming = useDraftStore((s) => s.isStreaming);
  const text = useDraftStore((s) => s.streamingText);
  const stageMessage = usePipelineStore((s) => s.message);
  const running = usePipelineStore((s) => s.running);

  if (!isStreaming && !running) return null;

  return (
    <div className="mx-auto w-full max-w-3xl">
      {!text ? (
        <div className="flex items-center gap-2 py-3 text-xs text-muted">
          <span className="h-3 w-3 animate-pulse rounded-full bg-accent" />
          {stageMessage || "Working…"}
        </div>
      ) : (
        <div className="rounded-lg border border-accent/30 bg-surface px-4 py-3">
          <div className="mb-1 flex items-center gap-2 text-[11px] text-accent">
            <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
            Generating…
          </div>
          <div className="md-body text-sm">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
            <span className="inline-block h-4 w-0.5 animate-pulse bg-accent align-middle" />
          </div>
        </div>
      )}
    </div>
  );
}
