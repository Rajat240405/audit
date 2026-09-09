/**
 * Deterministic layered layout for the GraphRAG node-link view.
 *
 * A force simulation was deliberately avoided: it is non-deterministic, keeps
 * animating after a new query (stale motion), and makes assertions impossible.
 * This produces stable, reproducible coordinates for the same model, which
 * matters because the graph must visibly RESET when a new query arrives.
 *
 * Layers mirror the retrieval story the placeholder promised:
 *
 *     anchors (matched entities)   -> top
 *     other entities (expansion)   -> middle
 *     documents (results)          -> bottom
 */
import type { GraphModel, GraphModelNode } from "./graphModel";

export interface LayoutPosition {
  x: number;
  y: number;
}

export const LAYER_GAP_Y = 180;
export const NODE_GAP_X = 220;

function layerOf(n: GraphModelNode): number {
  if (n.kind === "anchor") return 0;
  if (n.kind === "document") return 2;
  return 1;
}

/**
 * Assign positions. Nodes are ordered inside a layer by (docIds desc, id) so
 * the busiest nodes sit centrally and the result is stable across renders.
 */
export function layoutGraph(model: GraphModel): Map<string, LayoutPosition> {
  const layers = new Map<number, GraphModelNode[]>();
  for (const n of model.nodes) {
    const l = layerOf(n);
    const bucket = layers.get(l);
    if (bucket) bucket.push(n);
    else layers.set(l, [n]);
  }

  const positions = new Map<string, LayoutPosition>();
  for (const [layer, rawNodes] of [...layers.entries()].sort((a, b) => a[0] - b[0])) {
    const ordered = [...rawNodes].sort((a, b) => {
      if (b.docIds.length !== a.docIds.length) return b.docIds.length - a.docIds.length;
      return a.id.localeCompare(b.id);
    });
    const width = (ordered.length - 1) * NODE_GAP_X;
    ordered.forEach((n, i) => {
      positions.set(n.id, {
        x: i * NODE_GAP_X - width / 2,
        y: layer * LAYER_GAP_Y,
      });
    });
  }
  return positions;
}
