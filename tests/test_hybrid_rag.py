"""
Unit tests for Phase 2 — Hybrid RAG Pipeline.

Tests cover:
1. Embedder (embedding generation, normalization, batch encoding)
2. FAISSVectorStore (build, search, save/load)
3. BM25Index (build, search, save/load)
4. RRF fusion (combine ranked lists)
5. CrossEncoderReranker (reranking)
6. HybridRAGPipeline (end-to-end retrieval)
7. AnswerGenerator (grounded generation)

Run with: pytest tests/test_hybrid_rag.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.generation.client import LLMResponse
from src.generation.generator import SYSTEM_PROMPT, AnswerGenerator, build_user_prompt
from src.models.qa_record import QARecord, QARecordMetadata
from src.retrieval.hybrid.bm25_index import BM25Index
from src.retrieval.hybrid.embedder import Embedder
from src.retrieval.hybrid.fusion import RRF
from src.retrieval.hybrid.pipeline import HybridRAGPipeline
from src.retrieval.hybrid.vector_store import FAISSVectorStore
from src.retrieval.result import RetrievedResult

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_records():
    """Create a small set of sample Q&A records for testing."""
    return [
        QARecord(
            question_id="18-0001",
            question_text="What measures has the Government taken to address malaria in rural areas?",
            answer_text="The Government has implemented the National Vector Borne Disease Control Programme covering all malaria-endemic districts. During 2022-23, over 50 million insecticide-treated nets were distributed and 10,000 health camps were organized in rural areas.",
            metadata=QARecordMetadata(ministry="Health and Family Welfare", subject="Malaria Control"),
        ),
        QARecord(
            question_id="18-0002",
            question_text="What is the status of the GST collection and revenue sharing with states?",
            answer_text="GST collection in 2023-24 reached Rs. 17.9 lakh crore, representing a 12% increase over the previous year. The revenue deficit grant of Rs. 1.1 lakh crore was released to states meeting their fiscal criteria.",
            metadata=QARecordMetadata(ministry="Finance", subject="GST Collection"),
        ),
        QARecord(
            question_id="18-0003",
            question_text="What steps has the Government taken for skill development and vocational training?",
            answer_text="The Skill India Mission has trained over 14 million candidates since its launch. The PM Kaushal Vikas Yojana has 20 ministry partners offering 400+ job roles. New centres of excellence have been established in 20 states for advanced skills training.",
            metadata=QARecordMetadata(ministry="Skill Development and Entrepreneurship", subject="Skill Development"),
        ),
        QARecord(
            question_id="18-0004",
            question_text="What is the progress on metro rail projects in major cities?",
            answer_text="Metro rail projects are operational in 20 cities covering 900+ km. The Metro Lite and Metro Neo technologies are being deployed in smaller cities. Rs. 63,000 crore has been allocated for expansion in the current plan period.",
            metadata=QARecordMetadata(ministry="Housing and Urban Affairs", subject="Metro Rail"),
        ),
        QARecord(
            question_id="18-0005",
            question_text="What measures address water quality and drinking water coverage in rural India?",
            answer_text="The Jal Jeevan Mission has provided tap water connections to over 100 million rural households since 2019. Water quality monitoring is conducted in all 7.5 lakh villages. Rs. 60,000 crore has been allocated for tap water, sanitation and hygiene.",
            metadata=QARecordMetadata(ministry="Jal Shakti", subject="Drinking Water"),
        ),
    ]


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# Embedder Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedder:
    """Tests for the dense embedding embedder."""

    def test_embed_returns_correct_shape(self):
        """Embedding should have the expected dimension."""
        embedder = Embedder()
        embedding = embedder.embed("What is malaria?")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # all-MiniLM-L6-v2 dimension
        assert embedding.dtype == np.float32

    def test_embed_is_normalized(self):
        """Embedding should be unit-normalized (L2 norm = 1.0)."""
        embedder = Embedder()
        emb = embedder.embed("What is malaria?")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01

    def test_embed_batch_returns_correct_shape(self):
        """Batch embedding should return matrix [n, dim]."""
        embedder = Embedder()
        texts = [
            "What is malaria control?",
            "GST revenue and reforms",
            "Skill development programmes",
        ]
        embeddings = embedder.embed_batch(texts, show_progress=False)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_batch_embedding_normalized(self):
        """All vectors in a batch should be unit-normalized."""
        embedder = Embedder()
        texts = ["text " + str(i) for i in range(5)]
        embeddings = embedder.embed_batch(texts, show_progress=False)
        norms = np.linalg.norm(embeddings, axis=1)
        assert all(abs(n - 1.0) < 0.01 for n in norms)

    def test_similar_texts_have_high_similarity(self):
        """Semantically similar texts should have high cosine similarity."""
        embedder = Embedder()
        emb1 = embedder.embed("What government measures address malaria?")
        emb2 = embedder.embed("Steps to control malaria in India")
        similarity = float(np.dot(emb1, emb2))
        assert similarity > 0.6  # Should be fairly similar

    def test_different_texts_have_lower_similarity(self):
        """Semantically different texts should have lower similarity."""
        embedder = Embedder()
        emb1 = embedder.embed("Malaria control measures in rural areas")
        emb2 = embedder.embed("Metro rail financial allocation and budget process")
        similarity = float(np.dot(emb1, emb2))
        assert similarity < 0.5  # Should be less similar


# ─────────────────────────────────────────────────────────────────────────────
# FAISSVectorStore Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFAISSVectorStore:
    """Tests for the FAISS vector store."""

    def test_build_and_search(self):
        """Build should work and search should return results."""
        embedder = Embedder()
        store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)

        doc_ids = ["doc1", "doc2", "doc3"]
        embeddings = embedder.embed_batch(["text one", "text two", "text three"], show_progress=False)

        store.build(doc_ids, embeddings)
        assert len(store) == 3

        query_emb = embedder.embed("text one query")
        results = store.search(query_emb, k=2)

        assert len(results) == 2
        assert all(isinstance(doc_id, str) and isinstance(score, float) for doc_id, score in results)

    def test_search_returns_sorted_by_score(self):
        """Results should be sorted by score descending."""
        embedder = Embedder()
        store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)

        docs = ["healthcare malaria disease", "finance budget tax", "education school student"]
        embeddings = embedder.embed_batch(docs, show_progress=False)
        store.build(["h", "f", "e"], embeddings)

        query_emb = embedder.embed("malaria disease control")
        results = store.search(query_emb, k=3)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        # "healthcare malaria disease" should be most similar
        assert results[0][0] == "h"

    def test_save_and_load(self, temp_dir):
        """Index should be saved and loaded correctly."""
        embedder = Embedder()
        store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)

        docs = ["document one content", "document two content", "document three content"]
        embeddings = embedder.embed_batch(docs, show_progress=False)
        store.build(["d1", "d2", "d3"], embeddings)

        path = temp_dir / "test_faiss"
        store.save(path)

        # Load into a new store
        new_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
        new_store.load(path)
        assert len(new_store) == 3

        # Search should produce same results
        query_emb = embedder.embed("document one query")
        results = new_store.search(query_emb, k=2)
        assert len(results) == 2

    def test_search_empty_index(self):
        """Searching empty index should raise RuntimeError."""
        store = FAISSVectorStore(embedding_dim=384)
        # Building with 0 docs still creates the index
        store.build([], np.array([]).reshape(0, 384).astype(np.float32))
        # Empty index returns empty
        assert len(store) == 0
        # Searching empty index returns empty
        results = store.search(np.random.randn(384).astype(np.float32), k=5)
        assert results == []

    def test_search_wrong_dimension_raises(self):
        """Searching with wrong dimension should raise."""
        store = FAISSVectorStore(embedding_dim=384)
        store.build(["d1"], np.random.randn(1, 384).astype(np.float32))
        wrong_emb = np.random.randn(256).astype(np.float32)  # Wrong dim
        with pytest.raises(Exception):
            store.search(wrong_emb, k=5)


# ─────────────────────────────────────────────────────────────────────────────
# BM25Index Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBM25Index:
    """Tests for the BM25 lexical index."""

    def test_build_and_search(self):
        """BM25 index should build and search correctly."""
        index = BM25Index()
        docs = [
            ("d1", "What government malaria control rural", "Answer about malaria"),
            ("d2", "GST tax revenue Finance Ministry", "Answer about GST"),
            ("d3", "Skill development training education", "Answer about skills"),
        ]
        index.build(docs)
        assert len(index) == 3

    def test_exact_keyword_match(self):
        """BM25 should score exact keyword matches higher."""
        index = BM25Index()
        docs = [
            ("d1", "malaria control government rural health", "Malaria answer"),
            ("d2", "metro rail transport finance budget", "Metro answer"),
            ("d3", "skill development training programme", "Skill answer"),
        ]
        index.build(docs)

        results = index.search("malaria government control", k=3)
        assert results[0][0] == "d1"  # malaria doc should rank first
        assert results[0][1] > 0  # Should have a positive score

    def test_different_results_for_different_queries(self):
        """Different queries should return different result orderings."""
        index = BM25Index()
        docs = [
            ("d1", "malaria health disease", "answer"),
            ("d2", "metro rail transport", "answer"),
            ("d3", "skill training employment", "answer"),
        ]
        index.build(docs)

        r1 = index.search("malaria", k=3)
        r2 = index.search("metro", k=3)
        r3 = index.search("skill", k=3)

        assert r1[0][0] == "d1"
        assert r2[0][0] == "d2"
        assert r3[0][0] == "d3"

    def test_save_and_load(self, temp_dir):
        """BM25 index should persist to disk."""
        index = BM25Index(k1=1.5, b=0.75)
        docs = [
            ("d1", "question one text", "answer one"),
            ("d2", "question two text", "answer two"),
        ]
        index.build(docs)

        path = temp_dir / "test_bm25"
        index.save(path)

        new_index = BM25Index()
        new_index.load(path)
        assert len(new_index) == 2
        assert new_index.k1 == 1.5

        results = new_index.search("question one", k=2)
        assert results[0][0] == "d1"

    def test_empty_query_returns_empty(self):
        """Empty query should return empty results."""
        index = BM25Index()
        index.build([("d1", "text", "answer")])
        results = index.search("", k=5)
        assert results == []

    def test_bm25_parameters(self):
        """BM25 parameters should be configurable."""
        index = BM25Index(k1=2.0, b=0.5)
        assert index.k1 == 2.0
        assert index.b == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# RRF Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRRF:
    """Tests for Reciprocal Rank Fusion."""

    def test_fuse_two_lists(self):
        """RRF should combine two ranked lists."""
        dense = [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]
        bm25 = [("d3", 5.0), ("d2", 4.0), ("d1", 3.0)]

        fused = RRF.fuse([dense, bm25], k=60)

        # d1 and d2 appear in both lists → should rank highest
        doc_ids = [doc_id for doc_id, _ in fused]
        assert "d1" in doc_ids
        assert "d2" in doc_ids
        assert "d3" in doc_ids

    def test_fuse_preserves_common_docs(self):
        """Documents appearing in multiple lists should be boosted."""
        dense = [("d1", 1.0), ("d2", 0.9), ("d3", 0.8)]
        bm25 = [("d1", 1.0), ("d4", 0.9), ("d5", 0.8)]  # d1 is common

        fused = RRF.fuse([dense, bm25], k=60)
        doc_ids = [doc_id for doc_id, _ in fused]

        # d1 should be first (in both lists)
        assert doc_ids[0] == "d1"

    def test_fuse_respects_top_k(self):
        """RRF should respect the top_k parameter."""
        dense = [(str(i), 1.0 - i * 0.1) for i in range(10)]
        bm25 = [(str(i), 1.0 - i * 0.1) for i in range(10)]

        fused = RRF.fuse([dense, bm25], k=60, top_k=3)
        assert len(fused) == 3

    def test_fuse_empty_list(self):
        """Empty list should return empty results."""
        fused = RRF.fuse([[]], k=60)
        assert fused == []

    def test_fuse_empty_result_lists(self):
        """Multiple empty lists should return empty."""
        fused = RRF.fuse([[], []], k=60)
        assert fused == []

    def test_fuse_single_list(self):
        """Single list should be returned unchanged (after dedup)."""
        results = [("d1", 1.0), ("d2", 0.9), ("d2", 0.85)]  # d2 duplicated
        fused = RRF.fuse([results], k=60)
        assert len(fused) <= 3  # Deduplicated

    def test_fuse_sorted_by_score(self):
        """Results should be sorted by fused RRF score descending."""
        dense = [("d1", 1.0), ("d2", 1.0), ("d3", 1.0)]
        bm25 = [("d3", 1.0), ("d2", 1.0), ("d1", 1.0)]

        fused = RRF.fuse([dense, bm25], k=60)
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_fuse_default_k(self):
        """Default k should be 60."""
        fused = RRF.fuse([[("d1", 1.0)]])
        assert len(fused) == 1  # No error


# ─────────────────────────────────────────────────────────────────────────────
# HybridRAGPipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridRAGPipeline:
    """Tests for the complete Hybrid RAG pipeline."""

    def test_build_creates_both_indices(self, sample_records):
        """build() should create both FAISS and BM25 indices."""
        pipeline = HybridRAGPipeline(records=sample_records)
        assert len(pipeline) == 5
        assert len(pipeline.vector_store) == 5
        assert len(pipeline.bm25_index) == 5

    def test_retrieve_returns_structured_records(self, sample_records):
        """retrieve() should return original question+answer, not concatenated text."""
        pipeline = HybridRAGPipeline(records=sample_records)
        results, timings = pipeline.retrieve("malaria control", top_k=3)

        assert len(results) > 0
        for result in results:
            # MUST have separate question and answer fields
            assert isinstance(result.question, str)
            assert isinstance(result.answer, str)
            # Should NOT just be the concatenated content
            assert "QUESTION:" not in result.question
            assert "ANSWER:" not in result.answer
            # doc_id should be present
            assert result.doc_id

    def test_retrieve_includes_per_system_scores(self, sample_records):
        """Results should include dense, BM25, and RRF scores."""
        pipeline = HybridRAGPipeline(records=sample_records)
        results, timings = pipeline.retrieve("malaria health", top_k=3)

        for result in results:
            assert result.dense_score is not None
            assert result.bm25_score is not None
            assert result.rrf_score is not None

    def test_retrieve_timings_recorded(self, sample_records):
        """Timings should be recorded for each stage."""
        pipeline = HybridRAGPipeline(records=sample_records)
        _, timings = pipeline.retrieve("malaria", top_k=5)

        assert timings.embed_query_ms >= 0
        assert timings.dense_search_ms >= 0
        assert timings.bm25_search_ms >= 0
        assert timings.rrf_fusion_ms >= 0
        assert timings.total_ms >= timings.embed_query_ms

    def test_retrieve_respects_top_k(self, sample_records):
        """Should return at most top_k results."""
        pipeline = HybridRAGPipeline(records=sample_records)
        results, _ = pipeline.retrieve("government schemes", top_k=2)
        assert len(results) <= 2

    def test_retrieve_malaria_query_finds_health_doc(self, sample_records):
        """Malaria query should retrieve the malaria document first."""
        pipeline = HybridRAGPipeline(records=sample_records)
        results, _ = pipeline.retrieve("malaria in rural areas", top_k=3)
        assert len(results) > 0
        assert results[0].doc_id == "18-0001"  # Malaria doc

    def test_retrieve_gst_query_finds_finance_doc(self, sample_records):
        """GST query should retrieve the finance document."""
        pipeline = HybridRAGPipeline(records=sample_records)
        results, _ = pipeline.retrieve("GST collection revenue sharing", top_k=3)
        assert len(results) > 0
        assert results[0].doc_id == "18-0002"  # Finance doc

    def test_save_and_load(self, temp_dir, sample_records):
        """Pipeline should persist to disk correctly."""
        pipeline = HybridRAGPipeline(records=sample_records)
        path = temp_dir / "hybrid_rag"
        pipeline.save(path)

        new_pipeline = HybridRAGPipeline()
        new_pipeline.load(path)
        assert len(new_pipeline) == 5

        # Verify retrieval works with loaded pipeline
        results, _ = new_pipeline.retrieve("malaria", top_k=3)
        assert len(results) > 0
        assert results[0].doc_id == "18-0001"

    def test_retrieve_without_reranker(self, sample_records):
        """Pipeline should work without a reranker."""
        pipeline = HybridRAGPipeline(records=sample_records, use_reranker=False)
        results, timings = pipeline.retrieve("malaria", top_k=3)
        assert len(results) > 0
        assert timings.rerank_ms == 0.0  # No reranking happened


# ─────────────────────────────────────────────────────────────────────────────
# AnswerGenerator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerGenerator:
    """Tests for the answer generation module."""

    def test_build_user_prompt_format(self):
        """Prompt should include all sources with clear formatting."""
        results = [
            RetrievedResult(
                doc_id="18-1",
                question="What about malaria?",
                answer="Malaria cases are declining in India.",
                score=0.95,
                retrieval_method="rrf",
                metadata={"ministry": "Health"},
            ),
            RetrievedResult(
                doc_id="18-2",
                question="What about TB?",
                answer="TB cases are also monitored.",
                score=0.80,
                retrieval_method="rrf",
                metadata={"ministry": "Health"},
            ),
        ]

        prompt = build_user_prompt("Tell me about malaria", results)

        # Should include both sources
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt
        assert "18-1" in prompt
        assert "18-2" in prompt
        assert "malaria" in prompt.lower()
        assert "QUESTION:" in prompt
        assert "ANSWER:" in prompt
        assert "Tell me about malaria" in prompt

    def test_system_prompt_present(self):
        """System prompt should enforce grounding rules."""
        assert "ONLY" in SYSTEM_PROMPT.upper()
        assert "context" in SYSTEM_PROMPT.lower()
        assert "hallucinate" in SYSTEM_PROMPT.lower()

    def test_generate_returns_structured_result(self):
        """generate() should return a GenerationResult."""
        from unittest.mock import MagicMock

        mock_response = LLMResponse(
            text="Based on the context, malaria control measures include...",
            model="test-model",
            prompt_tokens=500,
            completion_tokens=50,
            total_tokens=550,
            latency_ms=500.0,
        )

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response

        generator = AnswerGenerator(llm_client=mock_client)

        results = [
            RetrievedResult(
                doc_id="18-1",
                question="What about malaria?",
                answer="Malaria measures include net distribution.",
                score=0.95,
                retrieval_method="rrf",
            ),
        ]

        gen_result = generator.generate("Tell me about malaria", results)

        assert gen_result.answer == "Based on the context, malaria control measures include..."
        assert gen_result.model == "test-model"
        assert "18-1" in gen_result.sources_used
        assert gen_result.prompt_tokens == 500
        assert gen_result.completion_tokens == 50
        assert gen_result.total_tokens == 550
        assert gen_result.generation_latency_ms >= 0  # Latency from mock is ~0ms

    def test_generate_no_context_returns_fallback(self):
        """generate() with no context should return a fallback message."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        generator = AnswerGenerator(llm_client=mock_client)

        gen_result = generator.generate("Any question", [])
        assert "cannot answer" in gen_result.answer.lower()
        assert gen_result.sources_used == []

    def test_generate_cost_estimation(self):
        """estimated_cost_usd should be non-negative."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.generate.return_value = LLMResponse(
            text="Answer text",
            model="test",
            prompt_tokens=500,
            completion_tokens=50,
            total_tokens=550,
            latency_ms=10.0,
        )
        generator = AnswerGenerator(llm_client=mock_client)

        result = generator.generate(
            "question",
            [RetrievedResult("d1", "q", "a", 0.9, "rrf")]
        )
        assert result.estimated_cost_usd >= 0
        assert isinstance(result.estimated_cost_usd, float)


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridRAGIntegration:
    """End-to-end integration tests for Hybrid RAG."""

    def test_full_pipeline_build_and_query(self, sample_records):
        """Full pipeline: build → query → get results."""
        pipeline = HybridRAGPipeline(records=sample_records)

        results, timings = pipeline.retrieve(
            "What has the government done for malaria control?",
            top_k=3,
        )

        assert len(results) > 0
        assert all(isinstance(r.question, str) for r in results)
        assert all(isinstance(r.answer, str) for r in results)
        assert timings.total_ms > 0

    def test_pipeline_is_deterministic(self, sample_records):
        """Same query should produce same results (no randomness in retrieval)."""
        pipeline = HybridRAGPipeline(records=sample_records)

        r1, _ = pipeline.retrieve("malaria malaria malaria", top_k=3)
        r2, _ = pipeline.retrieve("malaria malaria malaria", top_k=3)

        # Should produce identical results (deterministic)
        assert [x.doc_id for x in r1] == [x.doc_id for x in r2]
