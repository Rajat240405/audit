import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { cn } from "@/utils/cn";
import type { GraphModelNode } from "@/lib/graphModel";

export interface GraphNodeData extends Record<string, unknown> {
  model: GraphModelNode;
  dimmed: boolean;
  selected: boolean;
}

const KIND_STYLE: Record<GraphModelNode["kind"], string> = {
  anchor: "border-accent/60 bg-accent/10 text-accent",
  entity: "border-warning/50 bg-warning/10 text-warning",
  document: "border-success/50 bg-success/10 text-success",
};

/**
 * One graph node. Anchors (query matches), expanded entities and result
 * documents are visually distinct, and each node states its Neo4j label so
 * node TYPES are readable without consulting the legend.
 */
function GraphNodeCardImpl({ data }: NodeProps) {
  const { model, dimmed, selected } = data as GraphNodeData;
  return (
    <div
      data-testid={`graph-node-${model.id}`}
      data-kind={model.kind}
      className={cn(
        "min-w-[130px] max-w-[190px] rounded-lg border px-2.5 py-1.5 shadow-sm transition-opacity",
        KIND_STYLE[model.kind],
        dimmed && "opacity-25",
        selected && "ring-2 ring-accent ring-offset-1 ring-offset-surface"
      )}
    >
      <Handle type="target" position={Position.Top} className="!h-1.5 !w-1.5 !bg-border" />
      <p className="truncate text-[11px] font-semibold leading-tight" title={model.name}>
        {model.name}
      </p>
      <p className="mt-0.5 truncate text-[9px] uppercase tracking-wide opacity-70">
        {model.label}
        {model.kind === "anchor" && model.via ? ` · ${model.via}` : ""}
      </p>
      <Handle type="source" position={Position.Bottom} className="!h-1.5 !w-1.5 !bg-border" />
    </div>
  );
}

export const GraphNodeCard = memo(GraphNodeCardImpl);
