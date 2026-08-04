"""
GraphRAG Retriever for Phase 5.
Implements metadata-driven graph traversal queries returning standard RetrievedResult objects.
"""

from __future__ import annotations

import difflib
from typing import Any, List, Optional

from src.retrieval.graph.store import GraphStore
from src.retrieval.result import RetrievedResult


class GraphRetriever:
    """
    Traverses the NetworkX metadata graph to query and retrieve matching Q&A records.
    Returns standard RetrievedResult objects for downstream pipeline compatibility.
    """

    def __init__(self, store: GraphStore | None = None) -> None:
        self.store = store or GraphStore()
        try:
            self.store.load()
        except FileNotFoundError:
            # Let it remain empty if not built yet (build will populate it)
            pass

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedResult]:
        """
        Unified retrieve command for GraphRAG.
        Detects if the query mentions any known MP, Ministry, or Subject,
        traverses those nodes, aggregates the document neighbors, and ranks them.
        """
        if not self.store.graph or len(self.store.graph) == 0:
            return []

        matched_docs: dict[str, float] = {}

        # 1. Scan nodes for occurrences inside the query text (case-insensitive)
        query_lower = query.lower()
        
        for node, attrs in self.store.graph.nodes(data=True):
            ntype = attrs.get("type", "")
            if ntype in ("Ministry", "Subject", "MP"):
                node_name = attrs.get("name", "").lower()
                # Check for substring match or high similarity fuzzy match
                if node_name in query_lower or len(node_name) > 3 and difflib.SequenceMatcher(None, node_name, query_lower).ratio() > 0.7:
                    # Traversal: predecessors of this node are the Document nodes
                    for doc_node in self.store.graph.predecessors(node):
                        doc_id = self.store.graph.nodes[doc_node].get("doc_id")
                        if doc_id:
                            # Accumulate matching score (higher degree matches rank higher)
                            matched_docs[doc_id] = matched_docs.get(doc_id, 0.0) + 1.0

        # 2. Fallback: If no direct metadata matches, search document node attributes directly
        if not matched_docs:
            for node, attrs in self.store.graph.nodes(data=True):
                if attrs.get("type") == "Document":
                    q_text = attrs.get("question", "").lower()
                    a_text = attrs.get("answer", "").lower()
                    if query_lower in q_text or query_lower in a_text:
                        doc_id = attrs.get("doc_id")
                        matched_docs[doc_id] = matched_docs.get(doc_id, 0.0) + 0.5

        # 3. Sort matches by score descending and build RetrievedResult objects
        sorted_docs = sorted(matched_docs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            doc_node = f"Document:{doc_id}"
            attrs = self.store.graph.nodes[doc_node]
            
            # Find metadata associations by scanning successors
            ministry = None
            subject = None
            date = None
            
            for succ in self.store.graph.successors(doc_node):
                s_attrs = self.store.graph.nodes[succ]
                s_type = s_attrs.get("type")
                if s_type == "Ministry":
                    ministry = s_attrs.get("name")
                elif s_type == "Subject":
                    subject = s_attrs.get("name")
                elif s_type == "Date":
                    date = s_attrs.get("date")

            results.append(RetrievedResult(
                doc_id=doc_id,
                question=attrs.get("question", ""),
                answer=attrs.get("answer", ""),
                score=score,
                retrieval_method="graph_traversal",
                metadata={
                    "ministry": ministry,
                    "subject": subject,
                    "date": date
                }
            ))

        return results

    def get_docs_by_ministry(self, ministry_name: str) -> List[RetrievedResult]:
        """Traverse graph to retrieve all Documents linked to the specified Ministry."""
        return self._get_docs_by_metadata_node("Ministry", ministry_name, "HAS_MINISTRY")

    def get_docs_by_subject(self, subject_name: str) -> List[RetrievedResult]:
        """Traverse graph to retrieve all Documents linked to the specified Subject."""
        return self._get_docs_by_metadata_node("Subject", subject_name, "HAS_SUBJECT")

    def get_docs_by_mp(self, mp_name: str) -> List[RetrievedResult]:
        """Traverse graph to retrieve all Documents asked by the specified MP."""
        return self._get_docs_by_metadata_node("MP", mp_name, "ASKED_BY")

    def get_docs_by_session(self, session_num: int) -> List[RetrievedResult]:
        """Traverse graph to retrieve all Documents belonging to the specified Parliament Session."""
        target = f"Session:{session_num}"
        if not self.store.graph.has_node(target):
            return []
        doc_nodes = list(self.store.graph.predecessors(target))
        return self._build_results(doc_nodes, "graph_traversal:BELONGS_TO_SESSION")

    # ── Private Traversal Helpers ───────────────────────────────────────────

    def _get_docs_by_metadata_node(self, node_type: str, name: str, relation_label: str) -> List[RetrievedResult]:
        target = f"{node_type}:{name}"
        if not self.store.graph.has_node(target):
            # Try fuzzy search
            target = self._find_fuzzy_node(node_type, name)
            if not target:
                return []

        doc_nodes = list(self.store.graph.predecessors(target))
        return self._build_results(doc_nodes, f"graph_traversal:{relation_label}")

    def _find_fuzzy_node(self, node_type: str, name: str) -> str | None:
        """Find nodes using a case-insensitive fuzzy match threshold of 0.6."""
        nodes_of_type = [n for n, attrs in self.store.graph.nodes(data=True) if attrs.get("type") == node_type]
        name_lower = name.lower()
        
        best_match = None
        best_ratio = 0.0
        
        for n in nodes_of_type:
            n_clean = n.split(":", 1)[1].lower()
            ratio = difflib.SequenceMatcher(None, n_clean, name_lower).ratio()
            if ratio > best_ratio and ratio > 0.6:
                best_ratio = ratio
                best_match = n
                
        return best_match

    def _build_results(self, doc_nodes: List[str], method: str) -> List[RetrievedResult]:
        """Construct standard RetrievedResult objects from a list of Document node IDs."""
        results = []
        for doc_node in doc_nodes:
            attrs = self.store.graph.nodes[doc_node]
            results.append(RetrievedResult(
                doc_id=attrs.get("doc_id", ""),
                question=attrs.get("question", ""),
                answer=attrs.get("answer", ""),
                score=1.0,  # Unweighted categorical traversal match
                retrieval_method=method,
            ))
        return results
