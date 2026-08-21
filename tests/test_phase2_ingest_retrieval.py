"""Phase 2 validation — incremental ingestion + retrieval/org filters.

Evidence-driven: this suite builds REAL FAISS + BM25 index artifacts on disk
(faiss-cpu + rank_bm25, no bge-m3 download) using a DETERMINISTIC stub
embedder, then proves at the vector level that:

  A. new-file ingestion appends exactly the new record
  B. duplicate/no-op ingestion touches nothing (0 appends, 0 re-embeds)
  C. only new records reach the embedder
  D. FAISS/BM25 update incrementally (old vectors byte-stable, ntotal +N only)
  E. org filtering works on stamped orgs, heuristic legacy orgs, and is safe
     for unknown/absent orgs
  F. doc_types / doc_categories filtering works on stamped metadata
  G-J. CLI/operator workflow (flat compat, hierarchical, discovery, all)

All filesystem writes go to tmp dirs; the production corpus/index is never
touched (this sandbox checkout has no production corpus — verified: data/
contains only finetune + a prompt debug file).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import src.scripts.ingest as ingest_cli
import src.scripts.ingest_folder as engine
from src.models.qa_record import QARecord
from src.retrieval.hybrid.pipeline import HybridRAGPipeline
from src.utils.app_paths import corpus_path, data_dir
from tests.test_ingest_cli import make_record, read_corpus_lines, env_data, no_embed  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic embedder — REAL vectors for REAL FAISS/BM25, zero ML downloads
# ─────────────────────────────────────────────────────────────────────────────

class DeterministicEmbedder:
    """Seeded-by-content vectors: same text -> same normalized float32 vector.
    Dim 32 keeps a 1432-doc FAISS trivial for CI while remaining a REAL index
    (IndexFlatIP = cosine on normalized vectors, reconstructable)."""

    def __init__(self, dim: int = 32):
        self.embedding_dim = dim
        self.model_name = "deterministic-stub"

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.random(self.embedding_dim).astype(np.float32)
        return v / (np.linalg.norm(v) + np.float32(1e-12))

    def embed(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_batch(self, texts, batch_size: int = 32, show_progress: bool = False):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)


class _NoopReranker:
    def rerank(self, *a, **k):  # pragma: no cover — must never be called
        raise AssertionError("reranker must not run in offline tests")


def _alpha_token(i: int) -> str:
    """Pure-alpha unique token (BM25 tokenizer keeps isalpha() tokens only —
    digits in a token get it dropped at BOTH index and query time)."""
    chars = []
    i += 26 * 26
    while i:
        i, r = divmod(i, 26)
        chars.append(chr(97 + r))
    return "tok" + "".join(reversed(chars))


def build_real_index(records: list[QARecord], idx_dir: Path) -> HybridRAGPipeline:
    p = HybridRAGPipeline(
        records=records,
        embedder=DeterministicEmbedder(),
        reranker=_NoopReranker(),
        use_reranker=False,
    )
    idx_dir.mkdir(parents=True, exist_ok=True)
    p.save(str(idx_dir))
    return p


def load_real_index(idx_dir: Path) -> HybridRAGPipeline:
    p = HybridRAGPipeline(
        embedder=DeterministicEmbedder(),
        reranker=_NoopReranker(),
        use_reranker=False,
    )
    p.load(str(idx_dir))
    return p


def stamped_record(qid: str, *, org=None, source=None, doc_type="document",
                   ministry="EARTH SCIENCES", q=None, a=None, subject=None):
    rec = make_record(qid, q=q or f"What is reported about {qid} operations?",
                      a=a or f"Details and figures for {qid} are documented here.")
    m = rec.metadata
    m.org, m.source, m.document_type, m.ministry = org, source, doc_type, ministry
    if subject is not None:
        m.subject = subject
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# D (centerpiece): the "1432 existing + 1 new" acceptance scenario on a REAL
# saved index — old vectors byte-stable, ntotal grows by exactly the new ids.
# ─────────────────────────────────────────────────────────────────────────────

class TestIndexLevelIncremental:
    N_EXISTING = 1432

    def _make_corpus_records(self) -> list[QARecord]:
        recs = []
        for i in range(self.N_EXISTING):
            org = ["incois", "imd", "moes_hq", "isro"][i % 4]
            recs.append(stamped_record(
                f"incdoc-{i:06d}", org=org, source=["moes", "moes", "moes", "isro"][i % 4],
                doc_type=["annual_report", "audit_report", "research_publication", "document"][i % 4],
                q=f"What does report {i} say about ocean observation programme {i}?",
                a=f"Report {i} documents programme {i} outcomes; {_alpha_token(i)} present.",
            ))
        return recs

    def test_1432_plus_1_index_integrity(self, tmp_path):
        idx = tmp_path / "idx"
        originals = self._make_corpus_records()
        emb = DeterministicEmbedder()
        build_real_index(originals, idx)

        # index artifact sanity: the SAME marker files CLI _index_exists checks
        for marker in ("vector_store.index", "bm25_index.pkl",
                       "doc_map.json", "pipeline_metadata.json"):
            assert (idx / marker).exists()

        p = load_real_index(idx)
        assert len(p._doc_map) == self.N_EXISTING
        ntotal_before = p.vector_store._index.ntotal
        assert ntotal_before == self.N_EXISTING

        # snapshot a sample of OLD vectors (position -> id -> vector)
        sample_positions = [0, 1, 500, 999, self.N_EXISTING - 1]
        before_vectors = {
            pos: p.vector_store._index.reconstruct(pos).copy()
            for pos in sample_positions
        }
        before_ids = list(p.vector_store._doc_ids)

        # tomorrow: ONE genuinely new document arrives
        new_rec = stamped_record(
            "incdoc-999999", org="incois", source="moes", doc_type="annual_report",
            q="What is the ZSPLUNGE unique buoy token report about?",
            a="The ZSPLUNGE token marks the 2026 deep-sea buoy audit annexure.",
        )
        added = p.add_records([*originals, new_rec])   # full corpus, as the engine does

        # ONLY the new id was embedded/added
        assert added == 1
        assert p.vector_store._index.ntotal == ntotal_before + 1
        assert len(p.vector_store._doc_ids) == self.N_EXISTING + 1
        assert len(p._doc_map) == self.N_EXISTING + 1
        assert len(set(p.vector_store._doc_ids)) == self.N_EXISTING + 1  # no duplicate ids

        # old vectors untouched: same positions, same bytes
        for pos, vec in before_vectors.items():
            after = p.vector_store._index.reconstruct(pos)
            assert np.allclose(after, vec, atol=1e-6), f"vector at pos {pos} changed!"
        assert p.vector_store._doc_ids[: self.N_EXISTING] == before_ids  # order preserved

        # the new vector is EXACTLY the deterministic embedding of the new record
        new_pos = p.vector_store._doc_ids.index(new_rec.question_id)
        assert np.allclose(
            p.vector_store._index.reconstruct(new_pos),
            emb.embed(new_rec.document_content), atol=1e-6)

        # save + reload: counts persist; new + old both searchable
        p.save(str(idx))
        p2 = load_real_index(idx)
        assert p2.vector_store._index.ntotal == self.N_EXISTING + 1
        assert len(p2._doc_map) == self.N_EXISTING + 1

        hits = p2.bm25_index.search("ZSPLUNGE buoy token", k=3)
        assert hits and hits[0][0] == new_rec.question_id      # NEW searchable
        hits_old = p2.bm25_index.search(_alpha_token(500), k=3)
        assert hits_old and hits_old[0][0] == "incdoc-000500"  # OLD searchable

        # second identical add: total no-op at the VECTOR level
        ntotal_now = p2.vector_store._index.ntotal
        assert p2.add_records([*originals, new_rec]) == 0
        assert p2.vector_store._index.ntotal == ntotal_now


# ─────────────────────────────────────────────────────────────────────────────
# A/B/C: corpus-level acceptance through the source CLI (engine + registry)
# ─────────────────────────────────────────────────────────────────────────────

class TestCorpusLevelIncremental:
    def test_existing_corpus_1432_plus_one_new_file(self, env_data, no_embed):
        """Corpus-side acceptance: 1432 -> 1433 lines, first 1432 untouched."""
        existing = [
            stamped_record(f"incdoc-{i:06d}", org="incois", source="moes",
                           doc_type="annual_report",
                           q=f"Question about programme {i}?",
                           a=f"Answer documenting programme {i}.")
            for i in range(1432)
        ]
        corpus = corpus_path()
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text("".join(r.model_dump_json() + "\n" for r in existing),
                          encoding="utf-8")
        before = corpus.read_bytes()

        leaf = data_dir() / "moes" / "incois" / "annual_reports"
        leaf.mkdir(parents=True)
        (leaf / "report_2026.txt").write_text(
            "The 2026 INCOIS annual activities and audited outcomes report.",
            encoding="utf-8")

        specs, _, cmap = ingest_cli.resolve_sources("moes/incois/annual_reports")
        res = ingest_cli.run_sources(specs, category_map=cmap)

        after = corpus.read_bytes()
        assert len(read_corpus_lines()) == 1433
        assert res["added"] == 1
        assert after.startswith(before)  # the existing 1432 lines are untouched
        new_line = json.loads(read_corpus_lines()[-1])
        assert new_line["metadata"]["org"] == "incois"
        assert res["embed"] == "incremental"   # NOT a rebuild
        assert no_embed["incremental"] == 1 and no_embed["rebuild"] == 0

    def test_identical_rerun_is_full_noop(self, env_data, no_embed):
        leaf = data_dir() / "moes" / "incois" / "annual_reports"
        leaf.mkdir(parents=True)
        (leaf / "ar.txt").write_text("Annual activities and buoy programme review.",
                                     encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("moes")
        r1 = ingest_cli.run_sources(specs, category_map=cmap)
        no_embed["incremental"] = 0
        r2 = ingest_cli.run_sources(specs, category_map=cmap)
        assert (r1["added"], r2["added"]) == (1, 0)
        assert r2["embed"] == "skip"
        assert no_embed["incremental"] == 0 and no_embed["rebuild"] == 0

    def test_document_moved_between_tree_paths_stays_one_record(self, env_data, no_embed):
        """Identity is content-based: moving x.txt incois -> imd is a NO-OP."""
        incois_leaf = data_dir() / "moes" / "incois" / "annual_reports"
        imd_leaf = data_dir() / "moes" / "imd" / "annual_reports"
        incois_leaf.mkdir(parents=True)
        imd_leaf.mkdir(parents=True)
        src = incois_leaf / "shared_report.txt"
        src.write_text("One report physically relocated between organizations.",
                       encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("moes")
        r1 = ingest_cli.run_sources(specs, category_map=cmap)
        src.replace(imd_leaf / "shared_report.txt")  # move, don't copy
        r2 = ingest_cli.run_sources(specs, category_map=cmap)
        assert (r1["added"], r2["added"]) == (1, 0)
        assert len(read_corpus_lines()) == 1  # moved file was skipped by dedup

    def test_existing_files_are_skipped_not_reprocessed(self, env_data, no_embed):
        """Mixed leaf: pre-ingested file + one new file -> ONLY new appended."""
        leaf = data_dir() / "moes" / "incois" / "audit_reports"
        leaf.mkdir(parents=True)
        (leaf / "old_audit.txt").write_text("Earlier audit paragraphs on stores.",
                                            encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("moes")
        ingest_cli.run_sources(specs, category_map=cmap)
        (leaf / "new_audit.txt").write_text("Fresh audit paragraphs on procurement.",
                                            encoding="utf-8")
        res = ingest_cli.run_sources(specs, category_map=cmap)
        assert res["added"] == 1
        assert len(read_corpus_lines()) == 2


# ─────────────────────────────────────────────────────────────────────────────
# E/F: retrieval org/doc_type/doc_category filtering on a REAL pipeline
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def filter_pipeline(tmp_path):
    """4-doc real index mixing stamped + legacy (heuristic) org identity."""
    recs = [
        stamped_record("d-inc", org="incois", source="moes", doc_type="annual_report",
                       q="ALPHAtoken buoy question", a="ALPHAtoken INCOIS annual buoy answer"),
        stamped_record("d-imd", org="imd", source="moes", doc_type="audit_report",
                       q="BETAtoken met question", a="BETAtoken IMD audit answer"),
        stamped_record("d-legacy", org=None, source=None, doc_type="annual_report",
                       subject="INCOIS Annual Report 2023",
                       q="GAMMAtoken legacy question", a="GAMMAtoken scanned legacy INCOIS answer"),
        stamped_record("d-isro", org="isro", source="isro", doc_type="research_publication",
                       q="DELTAtoken launch question", a="DELTAtoken ISRO research answer"),
    ]
    return build_real_index(recs, tmp_path / "idx")


class TestRetrievalOrgFiltering:
    def _ids(self, results):
        return {r.doc_id for r in results}

    def test_stamped_org_filter(self, filter_pipeline):
        """Q: 'information from INCOIS annual reports' -> orgs=['incois'].
        Only incois-org candidates survive — both stamped AND legacy-derived."""
        results, _ = filter_pipeline.retrieve("buoy annual report", top_k=10,
                                              orgs=["incois"])
        got = self._ids(results)
        assert got <= {"d-inc", "d-legacy"}   # nothing else leaks through
        assert "d-inc" in got                 # stamped org honored

    def test_other_org_filter(self, filter_pipeline):
        """Q: 'information from IMD reports' -> orgs=['imd']."""
        results, _ = filter_pipeline.retrieve("audit question", top_k=10, orgs=["imd"])
        got = self._ids(results)
        assert got <= {"d-imd"}

    def test_unknown_new_org_slug_filter(self, filter_pipeline):
        """org slugs NOT in ORG_TREE (isro) still filter — explicit stamp wins."""
        results, _ = filter_pipeline.retrieve("launch programmes", top_k=10,
                                              orgs=["isro"])
        assert self._ids(results) <= {"d-isro"}

    def test_no_org_filter_is_unchanged_normal_retrieval(self, filter_pipeline):
        """Org filtering is OPT-IN: a query without orgs must see everything."""
        results, _ = filter_pipeline.retrieve("question about tokens", top_k=10)
        assert self._ids(results)  # non-empty; no crash; no restriction

    def test_empty_orgs_list_means_no_filter(self, filter_pipeline):
        unfiltered, _ = filter_pipeline.retrieve("question", top_k=10)
        filtered, _ = filter_pipeline.retrieve("question", top_k=10, orgs=[])
        assert self._ids(unfiltered) == self._ids(filtered)  # [] = omitted

    def test_doc_type_filter(self, filter_pipeline):
        results, _ = filter_pipeline.retrieve("question", top_k=10,
                                              doc_types=["annual_report"])
        assert self._ids(results) <= {"d-inc", "d-legacy"}

    def test_doc_category_filter_uses_new_audit_category(self, filter_pipeline):
        results, _ = filter_pipeline.retrieve("question", top_k=10,
                                              doc_categories=["audit"])
        assert self._ids(results) <= {"d-imd"}

    def test_org_and_type_compose(self, filter_pipeline):
        results, _ = filter_pipeline.retrieve("question", top_k=10,
                                              orgs=["incois"], doc_types=["annual_report"])
        assert self._ids(results) <= {"d-inc", "d-legacy"}


# ─────────────────────────────────────────────────────────────────────────────
# G-J: CLI/operator workflow through main() (exact operator commands)
# ─────────────────────────────────────────────────────────────────────────────

class TestCliOperatorWorkflow:
    def _run(self, argv):
        old = sys.argv
        try:
            sys.argv = ["prog", *argv]
            ingest_cli.main()
        finally:
            sys.argv = old

    def _moes_tree(self):
        f = data_dir() / "moes" / "incois" / "annual_reports" / "ar.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("INCOIS annual report activities and figures.", encoding="utf-8")

    def test_cmd_moes_ingest(self, env_data, no_embed):
        self._moes_tree()
        self._run(["moes", "--ingest", "--no-rebuild"])
        assert len(read_corpus_lines()) == 1

    def test_cmd_moes_incois_subpath(self, env_data, no_embed):
        self._moes_tree()
        other = data_dir() / "moes" / "imd" / "audit_reports" / "a.txt"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_text("IMD audit paragraphs for cyclone warnings.", encoding="utf-8")
        self._run(["moes/incois", "--ingest", "--no-rebuild"])
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        assert saved["metadata"]["org"] == "incois"  # ONLY the branch ingested

    def test_cmd_category_leaf_subpath(self, env_data, no_embed):
        self._moes_tree()
        self._run(["moes/incois/annual_reports", "--ingest", "--no-rebuild"])
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        assert saved["metadata"]["document_type"] == "annual_report"

    def test_cmd_isro_discovery_no_code(self, env_data, no_embed):
        """data/isro/ -> `ingest isro --ingest` works with zero registration."""
        f = data_dir() / "isro" / "annual_reports" / "isro_ar.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("Annual review of the space agency missions.", encoding="utf-8")
        self._run(["isro", "--ingest", "--no-rebuild"])
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        assert saved["metadata"]["org"] == "isro"
        assert saved["metadata"]["ministry"] is None

    def test_cmd_all_ingest_mixed_tree(self, env_data, no_embed):
        self._moes_tree()
        f = data_dir() / "isro" / "audit_reports" / "a.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("Audit observations of the space programme.", encoding="utf-8")
        self._run(["all", "--ingest", "--no-rebuild"])
        orgs = {json.loads(l)["metadata"]["org"] for l in read_corpus_lines()}
        assert {"incois", "isro"} <= orgs

    def test_cmd_all_ingest_rerun_noop(self, env_data, no_embed, capsys):
        self._moes_tree()
        self._run(["all", "--ingest", "--no-rebuild"])
        self._run(["all", "--ingest"])
        out = capsys.readouterr().out
        assert "0 new record(s)" in out


# ─────────────────────────────────────────────────────────────────────────────
# Operator runbook (G): the exact procedure for a new source — pinned by test
# ─────────────────────────────────────────────────────────────────────────────

class TestNewSourceProcedure:
    def test_folder_only_is_sufficient(self, env_data, no_embed):
        """The operator procedure for a NEW ministry/source is exactly:
        mkdir data/<name>/<category>/; copy files; ingest <name> --ingest.
        No config edit, no code edit — proven here."""
        leaf = data_dir() / "ministry_xyz" / "research_papers"
        leaf.mkdir(parents=True)
        (leaf / "paper.txt").write_text("Research summary for the new ministry.",
                                        encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("ministry_xyz")
        res = ingest_cli.run_sources(specs, category_map=cmap)
        assert res["added"] == 1
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        assert saved["metadata"]["document_type"] == "research_publication"

    def test_config_formalization_optional_ministry_stamp(self, env_data, tmp_path, no_embed):
        """Formalizing in sources.yaml is OPTIONAL and only adds labels
        (ministry stamp, org_map names) — ingestion works without it."""
        leaf = data_dir() / "ministry_xyz" / "audit_reports"
        leaf.mkdir(parents=True)
        (leaf / "audit.txt").write_text("Audit paragraphs for the new ministry.",
                                        encoding="utf-8")
        cfg = tmp_path / "sources.yaml"
        cfg.write_text(
            "sources:\n"
            "  ministry_xyz:\n"
            "    kind: folders\n"
            "    folders: [ministry_xyz]\n"
            "    hierarchical: true\n"
            "    ministry: MINISTRY XYZ\n"
            "    default_org: ministry_xyz_hq\n",
            encoding="utf-8",
        )
        specs, _, cmap = ingest_cli.resolve_sources("ministry_xyz", str(cfg))
        res = ingest_cli.run_sources(specs, category_map=cmap)
        assert res["added"] == 1
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        meta = saved["metadata"]
        assert meta["document_type"] == "audit_report"
        assert meta["ministry"] == "MINISTRY XYZ"          # config stamp applies
        assert meta["org"] == "ministry_xyz_hq"            # default_org for HQ files
