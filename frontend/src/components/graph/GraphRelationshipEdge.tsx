import { memo } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";

export interface GraphEdgeData extends Record<string, unknown> {
  rel: string;
  /** Vertical nudge so pills in the same band do not stack. */
  labelOffset: number;
  /** Hidden by the density heuristic (still revealed on hover/selection). */
  labelHidden: boolean;
  /** Dimmed because another node is selected. */
  dimmed: boolean;
  /** On the highlighted traversal path (selection neighbourhood). */
  highlighted: boolean;
}

/**
 * The label pill itself — presentational and portal-free so it can be unit
 * tested directly (EdgeLabelRenderer needs a live ReactFlow store, which jsdom
 * cannot provide without measured nodes).
 */
export function RelationshipPill({
  id,
  rel,
  labelOffset,
  labelHidden,
  dimmed,
  highlighted,
  labelX,
  labelY,
}: Pick<GraphEdgeData, "rel" | "labelOffset" | "labelHidden" | "dimmed" | "highlighted"> & {
  id: string;
  labelX: number;
  labelY: number;
}) {
  return (
    <div
      data-testid={`graph-edge-label-${id}`}
      data-hidden={labelHidden ? "true" : "false"}
      className={[
        "graph-edge-pill nodrag nopan",
        labelHidden ? "graph-edge-pill--auto" : "",
        highlighted ? "graph-edge-pill--on" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      style={{
        transform: `translate(-50%, -50%) translate(${labelX}px, ${
          labelY + (labelOffset ?? 0)
        }px)`,
        opacity: dimmed ? 0 : undefined,
      }}
    >
      {rel}
    </div>
  );
}

/**
 * Relationship edge with a compact, rounded, semi-transparent label pill.
 *
 * Replaces React Flow's built-in rectangular label (opaque black box, fixed at
 * the midpoint) which was unreadable over crossing edges.
 *
 * Density handling: when the graph is busy the pill is hidden by default and
 * revealed on hover or when the edge is part of the selected neighbourhood, so
 * readability wins over showing every label at once. CSS `:hover` is used
 * rather than React state so hovering never re-renders the canvas.
 */
function GraphRelationshipEdgeImpl({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  data,
}: EdgeProps) {
  const { rel, labelOffset, labelHidden, dimmed, highlighted } =
    (data ?? {}) as GraphEdgeData;

  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    curvature: 0.28,
  });

  const stroke = highlighted
    ? "var(--color-accent, #6ea8fe)"
    : "var(--color-border, #6b7280)";

  return (
    <g className="graph-edge" data-testid={`graph-edge-${id}`}>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={{
          stroke,
          strokeWidth: highlighted ? 1.8 : 1.35,
          // Inactive relationships must read as REAL edges, not noise, while
          // staying clearly subordinate to the hovered/selected ones. The
          // ordering dimmed < idle < highlighted is what keeps the hierarchy
          // legible; only the absolute values were raised.
          opacity: dimmed ? 0.28 : highlighted ? 0.95 : 0.72,
        }}
      />
      <EdgeLabelRenderer>
        <RelationshipPill
          id={id}
          rel={rel}
          labelOffset={labelOffset}
          labelHidden={labelHidden}
          dimmed={dimmed}
          highlighted={highlighted}
          labelX={labelX}
          labelY={labelY}
        />
      </EdgeLabelRenderer>
    </g>
  );
}

export const GraphRelationshipEdge = memo(GraphRelationshipEdgeImpl);
