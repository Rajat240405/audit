"""
Unit and integration tests for Phase 5 GraphRAG.
"""

from __future__ import annotations

import tempfile
import json
from pathlib import Path
import pytest

from src.retrieval.graph.store import GraphStore
from src.retrieval.graph.retriever import GraphRetriever


def test_graph_build_and_save(tmp_path):
    """Verify building, saving, and loading NetworkX graph store works."""
    dummy_doc_map = {
        "18-101": {
            "question_text": "What are the details of schemes to improve healthcare infrastructure?",
            "answer_text": "The Ministry has allocated substantial funds for rural clinics.",
            "metadata": {
                "ministry": "Health and Family Welfare",
                "subject": "Healthcare Infrastructure",
                "member": "Dr. Shashi Tharoor",
                "session": 18,
                "type": "unstarred",
                "date": "2024-07-15"
            }
        }
    }

    # Build Graph
    store = GraphStore(storage_dir=str(tmp_path))
    store.build_graph(dummy_doc_map)
    
    assert store.graph.has_node("Document:18-101")
    assert store.graph.has_node("Ministry:Health and Family Welfare")
    assert store.graph.has_node("MP:Dr. Shashi Tharoor")
    assert store.graph.has_edge("Document:18-101", "Ministry:Health and Family Welfare")

    # Stats Verification
    stats = store.get_stats()
    assert stats["total_nodes"] == 7  # Doc, Min, Sub, MP, Session, QType, Date
    assert stats["total_edges"] == 6

    # Save & Load Round-trip
    store.save()
    assert (tmp_path / "graph.json").exists()

    new_store = GraphStore(storage_dir=str(tmp_path))
    new_store.load()
    assert len(new_store.graph) == 7
    assert new_store.graph.has_node("Document:18-101")


def test_graph_retrieval(tmp_path):
    """Verify traversing the graph matches documents correctly."""
    dummy_doc_map = {
        "18-101": {
            "question": "What are the details of schemes to improve healthcare infrastructure?",
            "answer": "The Ministry has allocated substantial funds for rural clinics.",
            "metadata": {
                "ministry": "Health and Family Welfare",
                "subject": "Healthcare Infrastructure",
                "member": "Dr. Shashi Tharoor",
                "session": 18,
                "type": "unstarred",
                "date": "2024-07-15"
            }
        }
    }

    store = GraphStore(storage_dir=str(tmp_path))
    store.build_graph(dummy_doc_map)
    store.save()

    retriever = GraphRetriever(store=store)
    
    # Retrieval by Ministry
    results = retriever.get_docs_by_ministry("Health and Family Welfare")
    assert len(results) == 1
    assert results[0].doc_id == "18-101"

    # Retrieval by MP
    results_mp = retriever.get_docs_by_mp("Dr. Shashi Tharoor")
    assert len(results_mp) == 1
    assert results_mp[0].doc_id == "18-101"

    # Retrieval by general query (fuzzy matching the MP name or subject)
    results_q = retriever.retrieve("Shashi Tharoor", top_k=5)
    assert len(results_q) == 1
    assert results_q[0].doc_id == "18-101"
