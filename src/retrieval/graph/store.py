"""
Graph store for GraphRAG Phase 5.
Uses NetworkX in memory to build, serialize, and analyze metadata-driven graphs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx


class GraphStore:
    """
    Manages the in-memory NetworkX directed metadata graph.
    Supports node/edge creation, serialization to JSON, and graph analytics.
    """

    def __init__(self, storage_dir: str = "storage/graphrag") -> None:
        self.storage_dir = Path(storage_dir)
        self.graph_file = self.storage_dir / "graph.json"
        self.graph = nx.DiGraph()

    def build_graph(self, doc_map: Dict[str, Any]) -> None:
        """
        Build the metadata graph entirely in-memory from the loaded doc map.
        Nodes: Document, Ministry, Subject, MP, Parliament Session, Question Type, Date
        Edges: HAS_MINISTRY, HAS_SUBJECT, ASKED_BY, BELONGS_TO_SESSION, QUESTION_TYPE, ANSWERED_ON
        """
        self.graph.clear()

        for doc_id, record in doc_map.items():
            # Extract metadata safely
            metadata = record.get("metadata", {})
            ministry = metadata.get("ministry")
            subject = metadata.get("subject")
            member = metadata.get("member") or record.get("member")  # handle both schemas
            session = metadata.get("session")
            qtype = metadata.get("question_type") or metadata.get("type")
            date = metadata.get("date")

            # 1. Create Document Node (Holds core Q&A properties)
            doc_node = f"Document:{doc_id}"
            self.graph.add_node(
                doc_node,
                type="Document",
                doc_id=doc_id,
                question=record.get("question_text", record.get("question", "")),
                answer=record.get("answer_text", record.get("answer", "")),
            )

            # 2. Add Ministry Node & Edge
            if ministry:
                min_node = f"Ministry:{ministry}"
                self.graph.add_node(min_node, type="Ministry", name=ministry)
                self.graph.add_edge(doc_node, min_node, relation="HAS_MINISTRY")

            # 3. Add Subject Node & Edge
            if subject:
                sub_node = f"Subject:{subject}"
                self.graph.add_node(sub_node, type="Subject", name=subject)
                self.graph.add_edge(doc_node, sub_node, relation="HAS_SUBJECT")

            # 4. Add MP Node & Edge
            if member:
                # Member might be a string list representing multiple MPs (clubbed questions)
                members_list = []
                if isinstance(member, str):
                    if member.startswith("[") and member.endswith("]"):
                        try:
                            # Safely load list from string representation
                            members_list = json.loads(member.replace("'", '"'))
                        except Exception:
                            members_list = [member]
                    else:
                        members_list = [member]
                elif isinstance(member, list):
                    members_list = member
                else:
                    members_list = [str(member)]

                for m in members_list:
                    m = str(m).strip()
                    if m:
                        mp_node = f"MP:{m}"
                        self.graph.add_node(mp_node, type="MP", name=m)
                        self.graph.add_edge(doc_node, mp_node, relation="ASKED_BY")

            # 5. Add Session Node & Edge
            if session:
                sess_node = f"Session:{session}"
                self.graph.add_node(sess_node, type="Session", number=session)
                self.graph.add_edge(doc_node, sess_node, relation="BELONGS_TO_SESSION")

            # 6. Add Question Type Node & Edge
            if qtype:
                qtype_node = f"QuestionType:{qtype}"
                self.graph.add_node(qtype_node, type="QuestionType", value=qtype)
                self.graph.add_edge(doc_node, qtype_node, relation="QUESTION_TYPE")

            # 7. Add Date Node & Edge
            if date:
                date_node = f"Date:{date}"
                self.graph.add_node(date_node, type="Date", date=date)
                self.graph.add_edge(doc_node, date_node, relation="ANSWERED_ON")

    def save(self) -> None:
        """Serialize and save the graph to disk in standard portable node-link JSON format."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(self.graph_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Deserialize and load the graph from disk."""
        if not self.graph_file.exists():
            raise FileNotFoundError(f"Graph file not found at {self.graph_file}")
        with open(self.graph_file, encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)

    def get_stats(self) -> Dict[str, Any]:
        """Compute comprehensive graph analytics and statistics."""
        nodes = self.graph.nodes(data=True)
        edges = self.graph.edges(data=True)

        total_nodes = len(nodes)
        total_edges = len(edges)

        # Node counts by type
        node_types: Dict[str, int] = {}
        for _, attrs in nodes:
            ntype = attrs.get("type", "Unknown")
            node_types[ntype] = node_types.get(ntype, 0) + 1

        # Average Degree
        avg_degree = sum(dict(self.graph.degree()).values()) / total_nodes if total_nodes > 0 else 0.0

        # Connected Components (undirected baseline)
        undirected_g = self.graph.to_undirected()
        num_components = nx.number_connected_components(undirected_g)

        # Top Ministries (by degree connections)
        ministry_connections = []
        for n, attrs in nodes:
            if attrs.get("type") == "Ministry":
                degree = self.graph.degree(n)
                ministry_connections.append((attrs.get("name"), degree))
        top_ministries = sorted(ministry_connections, key=lambda x: x[1], reverse=True)[:5]

        # Top MPs (by questions asked)
        mp_connections = []
        for n, attrs in nodes:
            if attrs.get("type") == "MP":
                degree = self.graph.degree(n)
                mp_connections.append((attrs.get("name"), degree))
        top_mps = sorted(mp_connections, key=lambda x: x[1], reverse=True)[:5]

        # Most Connected Subjects
        subject_connections = []
        for n, attrs in nodes:
            if attrs.get("type") == "Subject":
                degree = self.graph.degree(n)
                subject_connections.append((attrs.get("name"), degree))
        top_subjects = sorted(subject_connections, key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_types": node_types,
            "average_degree": avg_degree,
            "num_components": num_components,
            "top_ministries": top_ministries,
            "top_mps": top_mps,
            "top_subjects": top_subjects
        }
