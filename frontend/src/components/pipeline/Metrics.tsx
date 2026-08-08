import { useDraftStore } from "@/store/useDraftStore";
import { formatMs, formatNumber, traceToRows } from "@/utils/formatters";

/**
 * Performance panel — mode-aware metrics. Collapsible detail rows.
 */
export function Metrics({ isGraph }: { isGraph: boolean }) {
  const trace = useDraftStore((s) => s.trace);
  const sources = useDraftStore((s) => s.sources);
  const meta = useDraftStore((s) => s.lastMeta);

  void isGraph;
  const rows: Array<[string, string]> = [
    ["Model", meta ? String(meta.model ?? "—") : "—"],
    ["Provider", meta ? String(meta.provider ?? "—") : "—"],
    ["Execution Profile", meta ? String(meta.profile ?? "—") : "—"],
    ["Response Time", meta && meta.response_time_ms != null ? formatMs(Number(meta.response_time_ms)) : "—"],
    ["Retrieved Documents", meta && meta.retrieved_documents != null ? formatNumber(Number(meta.retrieved_documents)) : formatNumber(sources.length)],
    ["Retrieved Chunks", meta && meta.retrieved_chunks != null ? formatNumber(Number(meta.retrieved_chunks)) : "—"],
    ["Confidence", meta ? (meta.is_fallback ? "Fallback" : "High") : "—"],
    ["Context Used", meta && meta.total_tokens != null ? `${formatNumber(Number(meta.total_tokens))} tokens` : "—"],
  ];

  return (
    <div className="rounded-lg border border-border bg-surface-2">
      <details open={false}>
        <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold text-muted">
          Performance
        </summary>
        <div className="space-y-1 px-3 pb-2">
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-center justify-between text-[11px]">
              <span className="text-muted">{k}</span>
              <span className="font-mono">{v}</span>
            </div>
          ))}

          {trace && (
            <div className="mt-2 border-t border-border pt-2">
              <p className="mb-1 text-[10px] font-semibold uppercase text-muted">Retrieval stages</p>
              {traceToRows(trace).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between text-[11px]">
                  <span className="text-muted">{k}</span>
                  <span className="font-mono">{formatMs(v)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </details>
    </div>
  );
}
