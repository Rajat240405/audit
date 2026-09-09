/**
 * Deterministic layered layout with crossing reduction.
 *
 * A force simulation is deliberately avoided: it is non-deterministic, keeps
 * animating after a new query (stale motion), and makes assertions impossible.
 * This is a small Sugiyama-style pipeline — stable and reproducible for the
 * same model, which matters because the graph must visibly RESET per query.
 *
 *   1. layer      anchors -> entities -> documents (semantic grouping kept)
 *   2. order      barycenter sweeps to cut edge crossings
 *   3. place      pull each node toward its neighbours' average x, then
 *                 enforce a minimum gap so cards never overlap
 *   4. stagger    alternate a small y offset inside a layer so rows read as
 *                 bands rather than rigid lines, and label corridors open up
 *
 * Every tie is broken by node id, so the output is identical across renders.
 */
import type { GraphModel, GraphModelNode } from "./graphModel";

export interface LayoutPosition {
  x: number;
  y: number;
}

/** Vertical distance between semantic layers. */
export const LAYER_GAP_Y = 190;
/** Minimum horizontal distance between two node cards (card is ~170px wide). */
export const MIN_GAP_X = 194;
/** Alternating vertical offset inside a layer (breaks the rigid row). */
export const STAGGER_Y = 26;
/** Barycenter sweeps. Fixed count keeps the result deterministic. */
const ORDER_SWEEPS = 4;
/** Horizontal relaxation passes. */
const PLACEMENT_PASSES = 6;

function layerOf(n: GraphModelNode): number {
  if (n.kind === "anchor") return 0;
  if (n.kind === "document") return 2;
  return 1;
}

interface Adjacency {
  /** neighbours in other layers (drive ordering) */
  cross: Map<string, string[]>;
  /** every neighbour (drives horizontal relaxation) */
  all: Map<string, string[]>;
}

function buildAdjacency(model: GraphModel, layer: Map<string, number>): Adjacency {
  const cross = new Map<string, string[]>();
  const all = new Map<string, string[]>();
  const push = (m: Map<string, string[]>, k: string, v: string) => {
    const cur = m.get(k);
    if (cur) cur.push(v);
    else m.set(k, [v]);
  };
  for (const e of model.edges) {
    if (!layer.has(e.source) || !layer.has(e.target)) continue;
    push(all, e.source, e.target);
    push(all, e.target, e.source);
    if (layer.get(e.source) !== layer.get(e.target)) {
      push(cross, e.source, e.target);
      push(cross, e.target, e.source);
    }
  }
  return { cross, all };
}

/**
 * Barycenter ordering: repeatedly place each node at the average index of its
 * neighbours in the adjacent layers. Classic, cheap crossing reduction.
 */
function orderLayers(
  layers: Map<number, GraphModelNode[]>,
  adj: Adjacency
): Map<string, number> {
  const index = new Map<string, number>();
  const ordered = new Map<number, GraphModelNode[]>();

  // Seed: busiest nodes first so hubs start centrally; id breaks ties.
  for (const [l, nodes] of layers) {
    const seeded = [...nodes].sort((a, b) => {
      const da = adj.all.get(a.id)?.length ?? 0;
      const db = adj.all.get(b.id)?.length ?? 0;
      if (db !== da) return db - da;
      return a.id.localeCompare(b.id);
    });
    ordered.set(l, seeded);
    seeded.forEach((n, i) => index.set(n.id, i));
  }

  const layerIds = [...layers.keys()].sort((a, b) => a - b);
  for (let sweep = 0; sweep < ORDER_SWEEPS; sweep += 1) {
    // alternate direction so information flows both ways
    const order = sweep % 2 === 0 ? layerIds : [...layerIds].reverse();
    for (const l of order) {
      const nodes = ordered.get(l);
      if (!nodes || nodes.length < 2) continue;
      const bary = new Map<string, number>();
      nodes.forEach((n, i) => {
        const nbrs = adj.cross.get(n.id) ?? [];
        if (nbrs.length === 0) {
          bary.set(n.id, i); // keep isolated nodes where they are
          return;
        }
        let sum = 0;
        for (const nb of nbrs) sum += index.get(nb) ?? 0;
        bary.set(n.id, sum / nbrs.length);
      });
      nodes.sort((a, b) => {
        const d = (bary.get(a.id) ?? 0) - (bary.get(b.id) ?? 0);
        if (Math.abs(d) > 1e-9) return d;
        const pi = (index.get(a.id) ?? 0) - (index.get(b.id) ?? 0);
        if (pi !== 0) return pi;
        return a.id.localeCompare(b.id);
      });
      nodes.forEach((n, i) => index.set(n.id, i));
    }
  }
  return index;
}

/** Push nodes apart in-place so no two cards overlap, preserving order. */
function enforceGap(xs: number[]): number[] {
  const out = [...xs];
  for (let i = 1; i < out.length; i += 1) {
    if (out[i] - out[i - 1] < MIN_GAP_X) out[i] = out[i - 1] + MIN_GAP_X;
  }
  // recentre the layer on 0 so layers stay visually aligned
  const mid = (out[0] + out[out.length - 1]) / 2;
  return out.map((x) => x - mid);
}

/**
 * Assign positions for every node in the model.
 *
 * Related nodes end up near each other (shorter edges, fewer long diagonals)
 * while cards keep a guaranteed minimum separation.
 */
export function layoutGraph(model: GraphModel): Map<string, LayoutPosition> {
  const positions = new Map<string, LayoutPosition>();
  if (model.nodes.length === 0) return positions;

  const layerOfNode = new Map<string, number>();
  const layers = new Map<number, GraphModelNode[]>();
  for (const n of model.nodes) {
    const l = layerOf(n);
    layerOfNode.set(n.id, l);
    const bucket = layers.get(l);
    if (bucket) bucket.push(n);
    else layers.set(l, [n]);
  }

  const adj = buildAdjacency(model, layerOfNode);
  const index = orderLayers(layers, adj);

  const layerIds = [...layers.keys()].sort((a, b) => a - b);
  // initial x from the reduced ordering
  const x = new Map<string, number>();
  for (const l of layerIds) {
    const nodes = [...(layers.get(l) ?? [])].sort(
      (a, b) => (index.get(a.id) ?? 0) - (index.get(b.id) ?? 0)
    );
    const raw = nodes.map((_, i) => i * MIN_GAP_X);
    const spaced = enforceGap(raw);
    nodes.forEach((n, i) => x.set(n.id, spaced[i]));
  }

  // Relaxation: pull each node toward the mean x of ALL its neighbours, then
  // re-separate. Shortens edges without breaking the crossing-reduced order.
  for (let pass = 0; pass < PLACEMENT_PASSES; pass += 1) {
    const sweep = pass % 2 === 0 ? layerIds : [...layerIds].reverse();
    for (const l of sweep) {
      const nodes = [...(layers.get(l) ?? [])].sort(
        (a, b) => (index.get(a.id) ?? 0) - (index.get(b.id) ?? 0)
      );
      if (nodes.length === 0) continue;
      const desired = nodes.map((n) => {
        const nbrs = adj.all.get(n.id) ?? [];
        if (nbrs.length === 0) return x.get(n.id) ?? 0;
        let sum = 0;
        let count = 0;
        for (const nb of nbrs) {
          const nx = x.get(nb);
          if (nx === undefined) continue;
          sum += nx;
          count += 1;
        }
        if (count === 0) return x.get(n.id) ?? 0;
        const target = sum / count;
        const cur = x.get(n.id) ?? 0;
        return cur + (target - cur) * 0.6; // damped, keeps it stable
      });
      const spaced = enforceGap(desired);
      nodes.forEach((n, i) => x.set(n.id, spaced[i]));
    }
  }

  for (const l of layerIds) {
    const nodes = [...(layers.get(l) ?? [])].sort(
      (a, b) => (index.get(a.id) ?? 0) - (index.get(b.id) ?? 0)
    );
    nodes.forEach((n, i) => {
      // Stagger every other card so a layer reads as a band, not a rigid row.
      // This also opens vertical corridors for the relationship pills.
      const offset = nodes.length > 1 ? (i % 2 === 0 ? -STAGGER_Y : STAGGER_Y) : 0;
      positions.set(n.id, { x: x.get(n.id) ?? 0, y: l * LAYER_GAP_Y + offset });
    });
  }
  return positions;
}

/**
 * Count edge crossings for a set of positions — used by tests to prove the
 * ordering step actually reduces crossings rather than just reordering.
 */
export function countCrossings(
  model: GraphModel,
  positions: Map<string, LayoutPosition>
): number {
  const segs = model.edges
    .map((e) => {
      const a = positions.get(e.source);
      const b = positions.get(e.target);
      return a && b ? { a, b } : null;
    })
    .filter((s): s is { a: LayoutPosition; b: LayoutPosition } => s !== null);

  const orient = (p: LayoutPosition, q: LayoutPosition, r: LayoutPosition) => {
    const v = (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
    if (Math.abs(v) < 1e-9) return 0;
    return v > 0 ? 1 : -1;
  };
  const shares = (s: { a: LayoutPosition; b: LayoutPosition },
                  t: { a: LayoutPosition; b: LayoutPosition }) => {
    const same = (p: LayoutPosition, q: LayoutPosition) =>
      Math.abs(p.x - q.x) < 1e-9 && Math.abs(p.y - q.y) < 1e-9;
    return same(s.a, t.a) || same(s.a, t.b) || same(s.b, t.a) || same(s.b, t.b);
  };

  let n = 0;
  for (let i = 0; i < segs.length; i += 1) {
    for (let j = i + 1; j < segs.length; j += 1) {
      const s = segs[i];
      const t = segs[j];
      if (shares(s, t)) continue;
      const d1 = orient(s.a, s.b, t.a);
      const d2 = orient(s.a, s.b, t.b);
      const d3 = orient(t.a, t.b, s.a);
      const d4 = orient(t.a, t.b, s.b);
      if (d1 !== d2 && d3 !== d4) n += 1;
    }
  }
  return n;
}

/**
 * Vertical offsets that keep relationship pills from stacking on top of each
 * other. Edges whose midpoints land in the same band get fanned out
 * deterministically (0, -18, +18, -36, +36 …).
 */
export function labelOffsets(
  model: GraphModel,
  positions: Map<string, LayoutPosition>
): Map<string, number> {
  const BAND = 60;
  const STEP = 18;
  const buckets = new Map<string, string[]>();
  const sorted = [...model.edges].sort((a, b) => a.id.localeCompare(b.id));
  for (const e of sorted) {
    const a = positions.get(e.source);
    const b = positions.get(e.target);
    if (!a || !b) continue;
    const my = (a.y + b.y) / 2;
    const mx = (a.x + b.x) / 2;
    const key = `${Math.round(my / BAND)}:${Math.round(mx / (BAND * 3))}`;
    const cur = buckets.get(key);
    if (cur) cur.push(e.id);
    else buckets.set(key, [e.id]);
  }
  const offsets = new Map<string, number>();
  for (const ids of buckets.values()) {
    ids.forEach((id, i) => {
      const rank = Math.ceil(i / 2) * STEP;
      offsets.set(id, i === 0 ? 0 : i % 2 === 1 ? -rank : rank);
    });
  }
  return offsets;
}
