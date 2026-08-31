"""
GraphRAG Retriever: metadata-driven graph traversal returning standard RetrievedResult objects.
Has zero neural dependencies (torch, transformers, sentence-transformers) to remain extremely lightweight.
"""

from __future__ import annotations

import difflib
import re
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
        Performs high-quality query resolution with strict whole-word priority,
        abbreviation mappings, and fuzzy matching.
        """
        if not self.store.graph or len(self.store.graph) == 0:
            return []

        # 1. Normalize Query (lowercase, remove punctuation)
        query_clean = re.sub(r"[^\w\s]", " ", query.lower()).strip()
        query_tokens = [w for w in query_clean.split() if w]

        if not query_tokens:
            return []

        # Abbreviation and Alias mapping expansion
        abbreviation_map = {
            "gst": ["gst collection", "overview of gst collection"],
            "finance": ["finance"],
            "health": ["health and family welfare", "national health mission"],
            "agriculture": ["agriculture and farmers welfare"],
            "transport": ["road transport and highways", "railways"],
            "railways": ["railways"]
        }

        # Expand search terms if there's any abbreviation match
        expanded_search_terms = [query_clean]
        for token in query_tokens:
            if token in abbreviation_map:
                expanded_search_terms.extend(abbreviation_map[token])

        matched_docs: dict[str, float] = {}
        matched_nodes = set()

        # Step 2: Iterate nodes and find matching nodes using the strict priority rules
        for node, attrs in self.store.graph.nodes(data=True):
            ntype = attrs.get("type", "")
            if ntype not in ("Ministry", "Subject", "MP"):
                continue

            node_name = str(attrs.get("name", "")).strip()
            node_name_lower = node_name.lower()
            node_name_clean = re.sub(r"[^\w\s]", " ", node_name_lower).strip()
            node_tokens = [w for w in node_name_clean.split() if w]

            is_match = False
            match_score = 0.0

            # Priority 1: Exact Metadata Match
            if any(node_name_lower == term for term in expanded_search_terms):
                is_match = True
                match_score = 2.0
            # Priority 2: Case-Insensitive Normalized Match
            elif any(node_name_clean == term for term in expanded_search_terms):
                is_match = True
                match_score = 1.8
            # Priority 3: Whole-Token Match (Ensure full word match, NOT substring!)
            else:
                for term in expanded_search_terms:
                    term_words = term.split()
                    # Check if the node's normalized name matches any full word of the query term
                    if node_name_clean in term_words:
                        is_match = True
                        match_score = 1.5
                        break
                    # Also check if the node tokens are completely contained in the term words
                    if all(t in term_words for t in node_tokens):
                        is_match = True
                        match_score = 1.2
                        break

            # Priority 4: Fuzzy Match (Only if not matched by whole-word/exact)
            if not is_match:
                for term in expanded_search_terms:
                    # Avoid fuzzy matching short abbreviations like GST
                    if len(node_name_clean) > 3 and len(term) > 3:
                        ratio = difflib.SequenceMatcher(None, node_name_clean, term).ratio()
                        if ratio > 0.8:
                            is_match = True
                            match_score = ratio
                            break

            if is_match:
                matched_nodes.add(node)
                # Traverse predecessors (which are the Document nodes)
                for doc_node in self.store.graph.predecessors(node):
                    doc_id = self.store.graph.nodes[doc_node].get("doc_id")
                    if doc_id:
                        # Accumulate matching score weighted by traversal match score
                        matched_docs[doc_id] = matched_docs.get(doc_id, 0.0) + match_score

        # Priority 5: Fallback Content Search (if no metadata nodes matched)
        if not matched_docs:
            for node, attrs in self.store.graph.nodes(data=True):
                if attrs.get("type") == "Document":
                    q_text = re.sub(r"[^\w\s]", " ", attrs.get("question", "").lower()).strip()
                    a_text = re.sub(r"[^\w\s]", " ", attrs.get("answer", "").lower()).strip()
                    
                    # Whole-word search inside Document question/answer text
                    if any(term in q_text.split() or term in a_text.split() for term in expanded_search_terms):
                        doc_id = attrs.get("doc_id")
                        matched_docs[doc_id] = matched_docs.get(doc_id, 0.0) + 0.5

        # Sort matches by score descending and build standard RetrievedResult list
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
