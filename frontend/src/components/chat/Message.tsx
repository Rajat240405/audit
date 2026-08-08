import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "@/types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/utils/cn";

/** Message card in the left assistant column (matches the Stitch design). */
export function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-xl rounded-tr-none bg-accent/10 px-4 py-2.5 text-sm leading-relaxed text-foreground">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <div
        className={cn(
          "rounded-xl border bg-surface p-4 shadow-sm",
          message.meta?.is_fallback ? "border-warning/50" : "border-border"
        )}
      >
        {message.content ? (
          <div className="md-body text-sm text-foreground/90">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        ) : (
          <p className="text-sm text-muted">…</p>
        )}
      </div>

      {message.meta && (
        <div className="flex flex-wrap items-center gap-1.5 px-1">
          <Badge variant="muted">
            {message.meta.provider} · {message.meta.model}
          </Badge>
          <Badge variant="muted">{message.meta.profile}</Badge>
          {message.meta.total_tokens != null && (
            <Badge variant="muted">{message.meta.total_tokens} tok</Badge>
          )}
          {message.meta.is_fallback && <Badge variant="warning">fallback</Badge>}
        </div>
      )}

      <div className="flex gap-2 pl-1">
        <button
          className="rounded-full border border-border bg-surface-2 px-3 py-1 text-[11px] text-foreground/70 hover:bg-surface"
          onClick={() => {
            try {
              void navigator.clipboard.writeText(message.content);
            } catch {
              /* clipboard unavailable */
            }
          }}
        >
          Copy
        </button>
        <button className="rounded-full border border-border bg-surface-2 px-3 py-1 text-[11px] text-foreground/70 hover:bg-surface">
          Save to Knowledge
        </button>
      </div>
    </div>
  );
}
