"""Application-side pre-Docker fixes: paths, long chunks, fusion_top_k, saved_by, build_meta."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.models.qa_record import QARecord, QARecordMetadata
from src.retrieval.hybrid.pipeline import HybridRAGPipeline


class FakeEmbedder:
    model_name = "fake-test-embedder"
    embedding_dim = 8

    def embed(self, text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2**31)
        rng = np.random.RandomState(seed)
        v = rng.randn(8).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v

    def embed_batch(self, texts, batch_size=1, show_progress=False):
        return np.stack([self.embed(t) for t in texts])


class FakeReranker:
    last_truncated_docs: set[str] = set()

    def rerank(self, query, candidates, k=5, doc_texts=None):
        return list(candidates)[:k]


def _pipe(records) -> HybridRAGPipeline:
    return HybridRAGPipeline(
        records=records,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        use_reranker=True,
    )


def test_default_fusion_top_k_is_50():
    p = HybridRAGPipeline(embedder=FakeEmbedder(), reranker=FakeReranker())
    assert p.fusion_top_k == 50


def test_prompt_debug_path_is_under_data_dir(monkeypatch, tmp_path):
    from src.utils import app_paths as ap

    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "data"))
    assert ap.prompt_debug_path() == tmp_path / "data" / "generation_prompt_debug.txt"
    assert ap.prompt_debug_path().parent == ap.data_dir()


def test_start_server_is_single_worker():
    import inspect
    import src.retrieval.frontend.server as srv

    src = inspect.getsource(srv.start_server)
    assert "workers=1" in src
    assert "workers=2" not in src


def test_app_paths_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    monkeypatch.delenv("APP_INDEX_DIR", raising=False)
    monkeypatch.delenv("APP_MODEL_DIR", raising=False)
    from src.utils import app_paths as ap

    assert ap.data_dir() == ap.project_root() / "data"
    assert ap.index_dir() == ap.project_root() / "storage" / "hybrid_rag"
    assert ap.model_dir() == ap.project_root() / "models"

    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APP_INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("APP_MODEL_DIR", str(tmp_path / "models"))
    assert ap.data_dir() == tmp_path / "data"
    assert ap.index_dir() == tmp_path / "idx"
    assert ap.model_dir() == tmp_path / "models"
    assert ap.corpus_path() == tmp_path / "data" / "corpus_reports.jsonl"


def test_long_chunks_and_build_meta_survive_save_load(tmp_path):
    long_answer = "\n".join(
        [f"Paragraph {i} UNIQUE_TOKEN_{i} " + ("x" * 80) for i in range(80)]
    )
    assert len(long_answer) > 4000
    records = [
        QARecord(
            question_id="long-1",
            question_text="What is in the long annexure?",
            answer_text=long_answer,
            metadata=QARecordMetadata(
                ministry="EARTH SCIENCES",
                subject="INCOIS annual report",
                document_type="annual_report",
            ),
        ),
        QARecord(
            question_id="short-1",
            question_text="Short parliamentary Q?",
            answer_text="Short answer about GST.",
            metadata=QARecordMetadata(
                ministry="Finance",
                subject="GST",
                document_type="parliamentary_qa",
            ),
        ),
    ]
    p = _pipe(records)
    assert len(p._long_chunk_map) > 0
    chunk_ids = set(p._long_chunk_map)

    out = tmp_path / "idx"
    p.save(out)
    assert (out / "long_chunk_map.json").exists()
    assert (out / "build_meta.json").exists()
    meta = json.loads((out / "build_meta.json").read_text(encoding="utf-8"))
    assert meta["embed_model"] == "fake-test-embedder"
    assert meta["embed_dim"] == 8
    assert meta["row_count"] == 2
    assert meta["long_chunk_count"] == len(chunk_ids)
    assert meta["fusion_top_k"] == 50
    assert meta["rows_sha256"]
    assert meta["built_at"]

    loaded = HybridRAGPipeline(embedder=FakeEmbedder(), reranker=FakeReranker())
    loaded.load(out)
    assert set(loaded._long_chunk_map) == chunk_ids
    assert loaded._long_chunk_texts
    # retrieve a unique buried token after reload
    hits, _ = loaded.retrieve("UNIQUE_TOKEN_55", top_k=3)
    assert any(h.doc_id == "long-1" for h in hits)


def test_source_filters_still_work():
    records = [
        QARecord(
            question_id="a1",
            question_text="INCOIS ocean observation network",
            answer_text="INCOIS operates ocean observation systems.",
            metadata=QARecordMetadata(
                ministry="EARTH SCIENCES",
                subject="Document: INCOIS Annual Report",
                document_type="annual_report",
            ),
        ),
        QARecord(
            question_id="p1",
            question_text="Lok Sabha question on GST",
            answer_text="GST collection increased.",
            metadata=QARecordMetadata(
                ministry="Finance",
                subject="GST",
                document_type="parliamentary_qa",
            ),
        ),
    ]
    p = _pipe(records)
    by_type, _ = p.retrieve("ocean GST", top_k=5, doc_types=["annual_report"])
    assert by_type
    assert all((h.metadata.get("document_type") or "") == "annual_report" for h in by_type)

    by_cat, _ = p.retrieve("ocean GST", top_k=5, doc_categories=["annual"])
    assert by_cat
    assert all(h.doc_id == "a1" for h in by_cat)


def test_save_knowledge_writes_saved_by(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("APP_USER", "win-dev")
    import src.retrieval.frontend.server as srv

    monkeypatch.setattr(srv, "USER_KNOWLEDGE_DIR", tmp_path)
    client = TestClient(srv.app)
    res = client.post(
        "/api/save-knowledge",
        json={
            "question": "What is INCOIS?",
            "answer": "Indian National Centre for Ocean Information Services.",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["saved_by"] == "win-dev"
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    stored = json.loads(files[0].read_text(encoding="utf-8"))
    assert stored["saved_by"] == "win-dev"
    assert stored["answer"].startswith("Indian National")

    # restart-style reload from disk
    look = client.get("/api/knowledge-lookup", params={"q": "What is INCOIS?"})
    assert look.status_code == 200
    data = look.json()
    assert data["found"] is True
    assert data.get("saved_by") == "win-dev"

    # header override
    res2 = client.post(
        "/api/save-knowledge",
        json={"question": "Header identity?", "answer": "From X-User header."},
        headers={"X-User": "scientist-a"},
    )
    assert res2.json()["saved_by"] == "scientist-a"


def test_sources_catalogue_from_index_not_corpus(tmp_path, monkeypatch):
    from src.retrieval.frontend.org_tree import build_sources_catalogue
    from src.models.qa_record import QARecord, QARecordMetadata

    indexed = [
        QARecord(
            question_id="idx-1",
            question_text="INCOIS report",
            answer_text="Ocean observation.",
            metadata=QARecordMetadata(
                ministry="EARTH SCIENCES",
                subject="Document: INCOIS Annual Report",
                document_type="annual_report",
            ),
        ),
    ]
    cat = build_sources_catalogue(indexed)
    assert cat["total"] == 1
    assert cat["source"] == "index"
    types = {t["type"]: t["count"] for t in cat["types"]}
    assert types.get("annual_report") == 1
    # corpus-only records must not appear
    assert all(t["type"] != "only_in_corpus" for t in cat["types"])


def test_knowledge_fuzzy_threshold_default_and_override(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_FUZZY_THRESHOLD", raising=False)
    import src.retrieval.frontend.server as srv

    assert srv.knowledge_fuzzy_threshold() == 0.85
    monkeypatch.setenv("KNOWLEDGE_FUZZY_THRESHOLD", "0.9")
    assert srv.knowledge_fuzzy_threshold() == 0.9
    monkeypatch.setenv("KNOWLEDGE_FUZZY_THRESHOLD", "nope")
    assert srv.knowledge_fuzzy_threshold() == 0.85


def test_fuzzy_lookup_uses_085(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import json

    monkeypatch.delenv("KNOWLEDGE_FUZZY_THRESHOLD", raising=False)
    import src.retrieval.frontend.server as srv

    monkeypatch.setattr(srv, "USER_KNOWLEDGE_DIR", tmp_path)
    (tmp_path / "k.json").write_text(
        json.dumps({"question": "How many Doppler Weather Radars?", "answer": "48"}),
        encoding="utf-8",
    )
    client = TestClient(srv.app)
    # similar enough (~0.91)
    hit = client.get(
        "/api/knowledge-lookup",
        params={"q": "How many Doppler Radars?"},
    ).json()
    assert hit["found"] is True
    assert hit["matched"] == "fuzzy"
    # too loose for 0.85
    miss = client.get(
        "/api/knowledge-lookup",
        params={"q": "What is the ocean temperature?"},
    ).json()
    assert miss["found"] is False


def test_append_jsonl_atomic_leaves_original_on_success(tmp_path):
    from src.utils.atomic_io import append_jsonl_atomic

    corpus = tmp_path / "corpus_reports.jsonl"
    corpus.write_text('{"question_id":"a"}\n', encoding="utf-8")
    n = append_jsonl_atomic(corpus, ['{"question_id":"b"}'])
    assert n == 1
    lines = [ln for ln in corpus.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_write_text_atomic_no_tmp_left(tmp_path):
    from src.utils.atomic_io import write_text_atomic

    dest = tmp_path / "doc_map.json"
    dest.write_text("OLD", encoding="utf-8")
    write_text_atomic(dest, "NEW")
    assert dest.read_text(encoding="utf-8") == "NEW"
    assert not list(tmp_path.glob("*.tmp"))


def test_health_live():
    from fastapi.testclient import TestClient
    import src.retrieval.frontend.server as srv

    client = TestClient(srv.app)
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"
