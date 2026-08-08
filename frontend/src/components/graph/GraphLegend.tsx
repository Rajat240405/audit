import { Badge } from "@/components/ui/badge";

/** Legend for the future graph visualization. */
export function GraphLegend() {
  return (
    <div className="flex flex-wrap gap-1.5 px-3 pb-2">
      <Badge variant="accent">Entity</Badge>
      <Badge variant="muted">Relationship</Badge>
      <Badge variant="success">Traversal path</Badge>
      <Badge variant="warning">Neighbour</Badge>
    </div>
  );
}
