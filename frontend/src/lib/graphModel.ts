/**
 * GraphRAG result -> node-link model.
 *
 * Pure functions only (no React, no layout library) so the whole data path is
 * unit-testable and the visualization layer stays swappable.
 *
 * Input is the `graph` provenance the backend attaches to graph-mode
 * SourceItems (anchors matched, traversal `via`, and the `fact_keys`
 * traversed). A fact key is emitted by GraphFact as:
 *
 *     "<REL>:<src_key>-><dst_key>"        e.g. "OPERATED_BY:u:twc->c:incois"
 *
 * Entity keys themselves contain ":" ("c:incois", "u:twc"), so the relation is
 * split on the FIRST colon and the endpoints on the FIRST "->" only.
 */
import type { GraphAnchor, GraphProvenance, SourceItem } from "@/types";

export type GraphNodeKind = "document" | "anchor" | "entity";

export interface GraphModelNode {
  id: string;
  /** Display label. */
  name: string;
  /** Neo4j label: Organization | Facility | Programme | Place | Document … */
  label: string;
  kind: GraphNodeKind;
  /** True when this node lies on a traversal path that produced a result. */
  onPath: boolean;
  /** How the anchor was matched (anchors only). */
  via?: string;
  /** Documents whose traversal touched this node. */
  docIds: string[];
  /** Source record, for document nodes. */
  source?: SourceItem;
}

export interface GraphModelEdge {
  id: string;
  source: string;
  target: string;
  /** Relationship type, e.g. MENTIONS / OPERATED_BY. */
  rel: string;
  factKey: string;
  onPath: boolean;
  docIds: string[];
}

export interface GraphModel {
  nodes: GraphModelNode[];
  edges: GraphModelEdge[];
  /** Result documents that carried graph provenance. */
  documentCount: number;
  /** Distinct entities (non-document nodes). */
  entityCount: number;
  /** Fact keys that could not be parsed (diagnostics; never silently hidden). */
  malformedFactKeys: string[];
  /** True when the model was truncated by `maxNodes`. */
  truncated: boolean;
}

export interface ParsedFactKey {
  rel: string;
  src: string;
  dst: string;
}

/** Cap the rendered graph so a hub entity cannot lock up the browser. */
export const DEFAULT_MAX_NODES = 300;

/**
 * Parse "<REL>:<src>-><dst>". Returns null when the shape does not match.
 * Split on the FIRST ":" (relation) and the FIRST "->" (endpoints) because
 * entity keys legitimately contain ":".
 */
export function parseFactKey(factKey: string): ParsedFactKey | null {
  if (typeof factKey !== "string") return null;
  const colon = factKey.indexOf(":");
  if (colon <= 0) return null;
  const rel = factKey.slice(0, colon);
  const rest = factKey.slice(colon + 1);
  const arrow = rest.indexOf("->");
  if (arrow <= 0) return null;
  const src = rest.slice(0, arrow);
  const dst = rest.slice(arrow + 2);
  if (!rel || !src || !dst) return null;
  return { rel, src, dst };
}

/** Human label for an entity key when no anchor/document supplies a name. */
export function labelForKey(key: string): string {
  const colon = key.indexOf(":");
  const bare = colon >= 0 ? key.slice(colon + 1) : key;
  return bare.replace(/[-_]+/g, " ").trim() || key;
}

/**
 * Infer a node label ("Organization", "Facility", …) for an entity that only
 * appears as a fact endpoint. Anchors and documents carry a real label; other
 * endpoints are reported as "Entity" rather than guessed.
 */
function inferLabel(key: string, docIds: Set<string>): string {
  return docIds.has(key) ? "Document" : "Entity";
}

/** True when this source carries usable graph provenance. */
export function hasGraphProvenance(s: SourceItem | undefined | null): boolean {
  const g = s?.graph;
  if (!g) return false;
  return (
    (Array.isArray(g.anchors) && g.anchors.length > 0) ||
    (Array.isArray(g.fact_keys) && g.fact_keys.length > 0)
  );
}

/** True when ANY source in the list is a graph result. */
export function hasGraphResults(sources: SourceItem[] | undefined | null): boolean {
  return Array.isArray(sources) && sources.some(hasGraphProvenance);
}

/**
 * Build the node-link model from graph-mode sources.
 *
 * Hybrid results (no `graph` field) contribute nothing, so a Hybrid RAG
 * response can never be rendered as a GraphRAG graph.
 */
export function buildGraphModel(
  sources: SourceItem[] | undefined | null,
  opts: { maxNodes?: number } = {}
): GraphModel {
  const maxNodes = opts.maxNodes ?? DEFAULT_MAX_NODES;
  const nodes = new Map<string, GraphModelNode>();
  const edges = new Map<string, GraphModelEdge>();
  const malformed: string[] = [];
  let documentCount = 0;

  const list = Array.isArray(sources) ? sources : [];
  const docIds = new Set(
    list.filter(hasGraphProvenance).map((s) => s.doc_id)
  );

  const addNode = (
    id: string,
    patch: Partial<GraphModelNode> & { name: string; label: string; kind: GraphNodeKind }
  ): GraphModelNode | null => {
    const existing = nodes.get(id);
    if (existing) {
      // Anchors outrank inferred entities; onPath is sticky.
      if (patch.kind === "anchor") {
        existing.kind = "anchor";
        existing.label = patch.label;
        existing.name = patch.name;
        if (patch.via) existing.via = patch.via;
      }
      if (patch.onPath) existing.onPath = true;
      for (const d of patch.docIds ?? []) {
        if (!existing.docIds.includes(d)) existing.docIds.push(d);
      }
      return existing;
    }
    if (nodes.size >= maxNodes) return null;
    const node: GraphModelNode = {
      id,
      name: patch.name,
      label: patch.label,
      kind: patch.kind,
      onPath: Boolean(patch.onPath),
      via: patch.via,
      docIds: [...(patch.docIds ?? [])],
      source: patch.source,
    };
    nodes.set(id, node);
    return node;
  };

  for (const s of list) {
    if (!hasGraphProvenance(s)) continue;
    const g = s.graph as GraphProvenance;
    documentCount += 1;

    // 1. the result document itself
    addNode(s.doc_id, {
      name: s.doc_id,
      label: "Document",
      kind: "document",
      onPath: true,
      docIds: [s.doc_id],
      source: s,
    });

    // 2. anchors — entities matched directly from the query text
    const anchors: GraphAnchor[] = Array.isArray(g.anchors) ? g.anchors : [];
    for (const a of anchors) {
      if (!a?.key) continue;
      addNode(a.key, {
        name: a.name || labelForKey(a.key),
        label: a.label || "Entity",
        kind: "anchor",
        onPath: true,
        via: a.via,
        docIds: [s.doc_id],
      });
    }

    // 3. traversed relationships
    const factKeys: string[] = Array.isArray(g.fact_keys) ? g.fact_keys : [];
    for (const fk of factKeys) {
      const parsed = parseFactKey(fk);
      if (!parsed) {
        if (!malformed.includes(fk)) malformed.push(String(fk));
        continue;
      }
      const { rel, src, dst } = parsed;
      const srcNode = addNode(src, {
        name: labelForKey(src),
        label: inferLabel(src, docIds),
        kind: docIds.has(src) ? "document" : "entity",
        onPath: true,
        docIds: [s.doc_id],
      });
      const dstNode = addNode(dst, {
        name: labelForKey(dst),
        label: inferLabel(dst, docIds),
        kind: docIds.has(dst) ? "document" : "entity",
        onPath: true,
        docIds: [s.doc_id],
      });
      // Skip the edge if either endpoint was dropped by the node cap.
      if (!srcNode || !dstNode) continue;

      const existing = edges.get(fk);
      if (existing) {
        if (!existing.docIds.includes(s.doc_id)) existing.docIds.push(s.doc_id);
        continue;
      }
      edges.set(fk, {
        id: fk,
        source: src,
        target: dst,
        rel,
        factKey: fk,
        onPath: true,
        docIds: [s.doc_id],
      });
    }
  }

  const nodeList = [...nodes.values()];
  return {
    nodes: nodeList,
    edges: [...edges.values()],
    documentCount,
    entityCount: nodeList.filter((n) => n.kind !== "document").length,
    malformedFactKeys: malformed,
    truncated: nodes.size >= maxNodes,
  };
}

/** Distinct relationship types present, sorted — drives the legend. */
export function relationshipTypes(model: GraphModel): string[] {
  return [...new Set(model.edges.map((e) => e.rel))].sort();
}

/** Distinct node labels present, sorted — drives the legend. */
export function nodeLabels(model: GraphModel): string[] {
  return [...new Set(model.nodes.map((n) => n.label))].sort();
}
