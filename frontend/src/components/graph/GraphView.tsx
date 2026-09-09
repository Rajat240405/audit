import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  ReactFlow,
  ReactFlowProvider,
  applyNodeChanges,
  useReactFlow,
  type Edge,
  type Node,
  type NodeChange,
  type NodeMouseHandler,
  type OnNodeDrag,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Crosshair, RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { GraphNodeCard, type GraphNodeData } from "./GraphNodeCard";
import {
  GraphRelationshipEdge,
  type GraphEdgeData,
} from "./GraphRelationshipEdge";
import { labelOffsets, layoutGraph } from "@/lib/graphLayout";
import {
  DEFAULT_MAX_NODES,
  buildGraphModel,
  nodeLabels,
  relationshipTypes,
  type GraphModel,
} from "@/lib/graphModel";
import type { SourceItem } from "@/types";

const nodeTypes = { graphNode: GraphNodeCard };
const edgeTypes = { graphEdge: GraphRelationshipEdge };

/**
 * Above this many edges the graph is treated as dense: relationship pills are
 * hidden by default and revealed on hover / selection, so crossing edges stay
 * readable instead of drowning in overlapping labels.
 */
export const DENSE_EDGE_THRESHOLD = 14;

/** Neighbourhood of a selected node (itself + directly connected nodes). */
function neighbourhood(model: GraphModel, id: string | null): Set<string> | null {
  if (!id) return null;
  const keep = new Set<string>([id]);
  for (const e of model.edges) {
    if (e.source === id) keep.add(e.target);
    if (e.target === id) keep.add(e.source);
  }
  return keep;
}

function GraphCanvas({ model }: { model: GraphModel }) {
  const [selected, setSelected] = useState<string | null>(null);
  const { fitView } = useReactFlow();

  // A new query produces a new model: clear selection so no stale highlight
  // survives, and refit the viewport to the new graph.
  useEffect(() => {
    setSelected(null);
  }, [model]);

  useEffect(() => {
    const t = window.setTimeout(() => {
      void fitView({ padding: 0.2, duration: 200 });
    }, 0);
    return () => window.clearTimeout(t);
  }, [model, fitView]);

  const focus = useMemo(() => neighbourhood(model, selected), [model, selected]);

  const positions = useMemo(() => layoutGraph(model), [model]);
  const offsets = useMemo(() => labelOffsets(model, positions), [model, positions]);
  const dense = model.edges.length > DENSE_EDGE_THRESHOLD;

  // Node POSITIONS are owned by React Flow state, not recomputed from the
  // layout on every render. Previously `nodes` was a useMemo over `positions`
  // with no onNodesChange handler, so a drag was immediately overwritten by
  // the next render and the node snapped back. The deterministic layout is now
  // the INITIAL placement only; manual positions persist until Reset.
  const [nodeState, setNodeState] = useState<Node<GraphNodeData>[]>([]);

  // Re-seed only when the graph itself changes (new query), never on
  // selection/hover — so dragging one node cannot rearrange the others.
  useEffect(() => {
    setNodeState(
      model.nodes.map((n) => ({
        id: n.id,
        type: "graphNode",
        position: positions.get(n.id) ?? { x: 0, y: 0 },
        data: { model: n, dimmed: false, selected: false },
        draggable: true,
      }))
    );
  }, [model, positions]);

  const onNodesChange = useCallback((changes: NodeChange<Node<GraphNodeData>>[]) => {
    setNodeState((cur) => applyNodeChanges(changes, cur));
  }, []);

  /** Restore the deterministic initial layout (Reset). */
  const resetLayout = useCallback(() => {
    setNodeState((cur) =>
      cur.map((n) => ({ ...n, position: positions.get(n.id) ?? n.position }))
    );
    setSelected(null);
    window.setTimeout(() => void fitView({ padding: 0.2, duration: 200 }), 0);
  }, [positions, fitView, setSelected]);

  // Visual-only props are merged in at render time so highlighting never
  // touches the stored positions.
  const nodes: Node<GraphNodeData>[] = useMemo(
    () =>
      nodeState.map((n) => ({
        ...n,
        data: {
          ...n.data,
          dimmed: Boolean(focus) && !focus?.has(n.id),
          selected: selected === n.id,
        },
      })),
    [nodeState, focus, selected]
  );

  const edges: Edge<GraphEdgeData>[] = useMemo(
    () =>
      model.edges.map((e) => {
        const inFocus = Boolean(focus) && focus?.has(e.source) && focus?.has(e.target);
        const dimmed = Boolean(focus) && !inFocus;
        return {
          id: e.id,
          source: e.source,
          target: e.target,
          type: "graphEdge",
          data: {
            rel: e.rel,
            labelOffset: offsets.get(e.id) ?? 0,
            // In a dense graph, only the selected neighbourhood keeps its
            // labels on; everything else reveals on hover.
            labelHidden: dense && !inFocus,
            dimmed,
            highlighted: Boolean(inFocus),
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            width: 12,
            height: 12,
            color: inFocus
              ? "var(--color-accent, #6ea8fe)"
              : "var(--color-border, #6b7280)",
          },
        } satisfies Edge<GraphEdgeData>;
      }),
    [model, focus, offsets, dense]
  );

  // A drag ends with a click event on the node. Track whether the pointer
  // actually moved so dragging never toggles selection/expansion.
  const dragOrigin = useRef<{ x: number; y: number } | null>(null);
  const draggedRef = useRef(false);

  const onNodeDragStart: OnNodeDrag<Node<GraphNodeData>> = useCallback((_, node) => {
    dragOrigin.current = { x: node.position.x, y: node.position.y };
    draggedRef.current = false;
  }, []);

  const onNodeDragStop: OnNodeDrag<Node<GraphNodeData>> = useCallback((_, node) => {
    const start = dragOrigin.current;
    if (start) {
      const moved =
        Math.abs(node.position.x - start.x) > 2 ||
        Math.abs(node.position.y - start.y) > 2;
      draggedRef.current = moved;
    }
    dragOrigin.current = null;
  }, []);

  const onNodeClick: NodeMouseHandler = useCallback((_, node) => {
    if (draggedRef.current) {
      draggedRef.current = false; // consume the click that ends a drag
      return;
    }
    setSelected((cur) => (cur === node.id ? null : node.id));
  }, []);

  const selectedNode = selected ? model.nodes.find((n) => n.id === selected) : undefined;

  return (
    <div className="relative h-full w-full" data-testid="graph-canvas">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onNodeDragStart={onNodeDragStart}
        onNodeDragStop={onNodeDragStop}
        onNodeClick={onNodeClick}
        onPaneClick={() => setSelected(null)}
        fitView
        minZoom={0.15}
        maxZoom={2.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={18} size={1} />
        <Controls showInteractive={false} className="!shadow-none" />
      </ReactFlow>

      <div className="pointer-events-none absolute right-2 top-2 flex flex-col items-end gap-1.5">
        <Button
          size="sm"
          variant="secondary"
          className="pointer-events-auto gap-1"
          onClick={() => void fitView({ padding: 0.2, duration: 200 })}
          title="Fit graph to view"
        >
          <Crosshair className="h-3 w-3" /> Fit
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="pointer-events-auto gap-1"
          onClick={resetLayout}
          title="Restore the initial layout and clear selection"
        >
          <RotateCcw className="h-3 w-3" /> Reset
        </Button>
      </div>

      {selectedNode && (
        <div
          data-testid="graph-inspector"
          className="absolute bottom-2 left-2 max-w-[280px] rounded-md border border-border bg-surface-2/95 p-2.5 text-xs backdrop-blur"
        >
          <p className="truncate font-semibold" title={selectedNode.name}>
            {selectedNode.name}
          </p>
          <div className="mt-1 flex flex-wrap gap-1">
            <Badge variant="muted">{selectedNode.label}</Badge>
            {selectedNode.via && <Badge variant="accent">{selectedNode.via}</Badge>}
          </div>
          {selectedNode.source && (
            <p className="mt-1.5 line-clamp-3 text-[11px] text-muted">
              {selectedNode.source.question || selectedNode.source.answer}
            </p>
          )}
          {selectedNode.docIds.length > 0 && (
            <p className="mt-1.5 text-[10px] text-muted">
              Supporting documents: {selectedNode.docIds.length}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Interactive GraphRAG node-link visualization.
 *
 * Renders ONLY from graph provenance the backend attached to the current
 * answer's sources, so a Hybrid RAG response can never appear here.
 */
export function GraphView({ sources }: { sources: SourceItem[] }) {
  const model = useMemo(
    () => buildGraphModel(sources, { maxNodes: DEFAULT_MAX_NODES }),
    [sources]
  );
  const rels = useMemo(() => relationshipTypes(model), [model]);
  const labels = useMemo(() => nodeLabels(model), [model]);
  const dense = model.edges.length > DENSE_EDGE_THRESHOLD;

  return (
    <div className="flex h-full flex-col" data-testid="graph-view">
      <div className="flex flex-wrap items-center gap-1.5 border-b border-border px-3 py-2">
        <Badge variant="accent">{model.entityCount} entities</Badge>
        <Badge variant="success">{model.documentCount} documents</Badge>
        <Badge variant="muted">{model.edges.length} relationships</Badge>
        {dense && (
          <Badge variant="muted" title="Labels reveal on hover or selection">
            labels on hover
          </Badge>
        )}
        {model.truncated && (
          <Badge variant="warning" title={`Showing the first ${DEFAULT_MAX_NODES} nodes`}>
            truncated
          </Badge>
        )}
        {model.malformedFactKeys.length > 0 && (
          <Badge
            variant="danger"
            title={model.malformedFactKeys.slice(0, 5).join(", ")}
          >
            {model.malformedFactKeys.length} unparsed
          </Badge>
        )}
      </div>

      <div className="min-h-0 flex-1">
        <GraphCanvas model={model} />
      </div>

      <div className="flex flex-wrap gap-1.5 border-t border-border px-3 py-2">
        {labels.map((l) => (
          <Badge key={l} variant="default" className="text-[9px]">
            {l}
          </Badge>
        ))}
        {rels.map((r) => (
          <Badge key={r} variant="muted" className="font-mono text-[9px]">
            {r}
          </Badge>
        ))}
      </div>
    </div>
  );
}

/** Provider wrapper — ReactFlow hooks require it. */
export function GraphViewPanel({ sources }: { sources: SourceItem[] }) {
  return (
    <ReactFlowProvider>
      <GraphView sources={sources} />
    </ReactFlowProvider>
  );
}
