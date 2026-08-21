"""Focused tests for the source-aware ingestion CLI (src/scripts/ingest.py).

Scope: all tests are OFFLINE and self-contained — every filesystem path is
redirected to tmp dirs via APP_DATA_DIR / APP_INDEX_DIR, so neither the real
`data/` tree nor `storage/hybrid_rag` is ever touched.

The heavy embed steps (bge-m3) are monkeypatched at the seam where the CLI
delegates into the ingest_folder engine — EXCEPT one test that exercises
HybridRAGPipeline.add_records directly with stub collaborators to prove the
core incremental invariant: existing vectors are never re-embedded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.scripts.ingest as ingest_cli
import src.scripts.ingest_folder as engine
from src.models.qa_record import QARecord
from src.utils.app_paths import config_path, corpus_path, data_dir


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_record(qid: str, q: str = "What is the status of the coastal buoy network?",
                a: str = "The buoy network is operational with all stations reporting.") -> QARecord:
    return QARecord(question_id=qid, question_text=q, answer_text=a)


def write_corpus(records: list[QARecord]) -> Path:
    corpus = corpus_path()
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    return corpus


def read_corpus_lines() -> list[str]:
    corpus = corpus_path()
    if not corpus.exists():
        return []
    return [l for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture()
def env_data(tmp_path, monkeypatch):
    """Redirect all app paths into a tmp tree; neutralize the embed phase."""
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APP_INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("APP_MODEL_DIR", str(tmp_path / "models"))
    return tmp_path


@pytest.fixture()
def no_embed(monkeypatch):
    """Patch the engine's index functions; return call log."""
    calls = {"incremental": 0, "rebuild": 0, "exists": True}
    monkeypatch.setattr(engine, "incremental_update",
                        lambda: calls.__setitem__("incremental", calls["incremental"] + 1))
    monkeypatch.setattr(engine, "rebuild_index",
                        lambda: calls.__setitem__("rebuild", calls["rebuild"] + 1))
    monkeypatch.setattr(engine, "_index_exists", lambda: calls["exists"])
    return calls


# ─────────────────────────────────────────────────────────────────────────────
# 1-3. Source resolution (registry)
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceResolution:
    def test_parliament_resolves_records_kind(self, env_data):
        specs, _, _cm = ingest_cli.resolve_sources("parliament")
        spec = specs["parliament"]
        assert spec.kind == "records"
        # Phase-1 output dirs, enriched first (richer copy wins id conflicts)
        assert spec.record_dirs[:2] == ["enriched", "processed"]
        assert not spec.folders

    def test_incois_resolves_nested_section_folders(self, env_data):
        specs, _, _cm = ingest_cli.resolve_sources("incois")
        spec = specs["incois"]
        assert spec.kind == "folders"
        # the nested structure must be preserved, not flattened
        for nested in ("incois_reports/AnnualReports", "incois_reports/Others",
                       "incois_reports/TechnicalReports", "incois_reports/ResearchPublications"):
            assert nested in spec.folders

    def test_moes_resolves_knowledge_folder(self, env_data):
        # Registry restructured in Phase 1: 'moes' is now the hierarchical
        # ministry root (data/moes/); the legacy flat CCPS knowledge folder
        # became its own source 'moes_reports'. Both behaviors pinned.
        specs, _, _cm2 = ingest_cli.resolve_sources("moes")
        spec = specs["moes"]
        assert spec.kind == "folders"
        assert spec.hierarchical is True and spec.folders == ["moes"]
        specs2, _, _cm3 = ingest_cli.resolve_sources("moes_reports")
        assert specs2["moes_reports"].folders == ["moes_reports/knowledge"]

    def test_builtin_fallback_matches_shipped_yaml(self, env_data):
        """If config/sources.yaml is unreadable, the built-in mirror must
        produce an IDENTICAL registry (no behavioral drift by fallback)."""
        from_file, exc_file, cm_file = ingest_cli.load_sources(config_path("sources.yaml"))
        from_builtin, exc_builtin, cm_builtin = ingest_cli.load_sources(Path("/nonexistent/sources.yaml"))
        assert set(from_file) == set(from_builtin) == {"inbox", "parliament", "incois", "moes", "moes_reports"}
        for name in from_file:
            a, b = from_file[name], from_builtin[name]
            assert (a.kind, a.folders, a.record_dirs, a.move_processed,
                    a.hierarchical, a.org, a.default_org, a.org_map, a.ministry) == \
                   (b.kind, b.folders, b.record_dirs, b.move_processed,
                    b.hierarchical, b.org, b.default_org, b.org_map, b.ministry)
        assert exc_file == exc_builtin
        assert cm_file == cm_builtin == ingest_cli._BUILTIN_CATEGORY_MAP

    def test_future_source_via_config_without_code_change(self, env_data, tmp_path):
        """A new ministry can be registered by EDITING CONFIG ONLY."""
        cfg = tmp_path / "sources.yaml"
        cfg.write_text(
            "sources:\n"
            "  ministry_xyz:\n"
            "    kind: folders\n"
            "    folders: [ministry_xyz]\n",
            encoding="utf-8",
        )
        sources, _, _cm = ingest_cli.load_sources(cfg)
        assert "ministry_xyz" in sources
        assert sources["ministry_xyz"].folders == ["ministry_xyz"]


# ─────────────────────────────────────────────────────────────────────────────
# 4-5. Discovery + `all`
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscovery:
    def test_discovery_finds_future_source_dir(self, env_data):
        future = data_dir() / "ministry_new"
        future.mkdir(parents=True)
        (future / "note.txt").write_text("A document from the new ministry.", encoding="utf-8")
        registered, excludes, _cmap = ingest_cli.load_sources(Path("/nonexistent/none.yaml"))
        found = ingest_cli.discover_sources(registered, excludes)
        assert "ministry_new" in found
        assert found["ministry_new"].discovered is True
        # resolvable by name — the "<future-source> --ingest" contract
        specs, _, _cm = ingest_cli.resolve_sources("ministry_new")
        assert specs["ministry_new"].folders == ["ministry_new"]

    def test_discovery_excludes_reserved_and_registered(self, env_data):
        registered, excludes, _cmap = ingest_cli.load_sources(Path("/nonexistent/none.yaml"))
        for name in ("processed", "enriched", "raw", "finetune", "user-knowledge", "inbox"):
            d = data_dir() / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "x.txt").write_text("some content here for testing", encoding="utf-8")
        found = ingest_cli.discover_sources(registered, excludes)
        assert found == {}  # nothing reserved or registered leaks into discovery

    def test_discovery_requires_ingestible_files(self, env_data):
        empty = data_dir() / "empty_dir"
        empty.mkdir(parents=True)
        (empty / "readme.docx").write_text("not ingestible by the engine", encoding="utf-8")
        registered, excludes, _cmap = ingest_cli.load_sources(Path("/nonexistent/none.yaml"))
        assert "empty_dir" not in ingest_cli.discover_sources(registered, excludes)

    def test_all_resolves_registered_plus_discovered(self, env_data):
        future = data_dir() / "ministry_new"
        future.mkdir(parents=True)
        (future / "note.txt").write_text("content from future ministry", encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("all")
        assert {"inbox", "parliament", "incois", "moes", "moes_reports", "ministry_new"} <= set(specs)

    def test_unknown_source_error_lists_known(self, env_data):
        with pytest.raises(KeyError) as ei:
            ingest_cli.resolve_sources("does_not_exist")
        msg = str(ei.value)
        assert "parliament" in msg and "incois" in msg


# ─────────────────────────────────────────────────────────────────────────────
# 6-9. Incremental semantics (the critical requirement)
# ─────────────────────────────────────────────────────────────────────────────

class TestIncrementalSemantics:
    def test_run_twice_adds_nothing_second_time(self, env_data, no_embed):
        future = data_dir() / "ministry_new"
        future.mkdir(parents=True)
        (future / "note.txt").write_text(
            "Installation report for the new deep-sea observatory.", encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("ministry_new")
        r1 = ingest_cli.run_sources(specs)
        lines1 = read_corpus_lines()
        r2 = ingest_cli.run_sources(specs)
        lines2 = read_corpus_lines()
        assert r1["added"] == 1 and r2["added"] == 0
        assert lines1 == lines2 and len(lines2) == 1  # deterministic dedup
        assert r2["embed"] == "skip"  # no additions -> no index work at all

    def test_records_merge_preserves_parliament_ids_and_dedups(self, env_data, no_embed):
        processed = data_dir() / "processed"
        processed.mkdir(parents=True)
        rec = make_record("18-4-3035")
        (processed / "processed_20240101.jsonl").write_text(
            rec.model_dump_json() + "\n", encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("parliament")
        r1 = ingest_cli.run_sources(specs)
        r2 = ingest_cli.run_sources(specs)
        assert (r1["added"], r2["added"]) == (1, 0)
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        assert saved["question_id"] == "18-4-3035"  # NOT re-hashed to incdoc-*

    def test_enriched_takes_precedence_over_processed_on_same_id(self, env_data):
        enriched = data_dir() / "enriched"
        processed = data_dir() / "processed"
        enriched.mkdir(parents=True)
        processed.mkdir(parents=True)
        rich = make_record("18-1-100")
        rich.metadata.subject = "enriched subject"
        (enriched / "enriched_a.jsonl").write_text(rich.model_dump_json() + "\n", encoding="utf-8")
        plain = make_record("18-1-100")
        plain.metadata.subject = "processed subject"
        (processed / "processed_b.jsonl").write_text(plain.model_dump_json() + "\n", encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("parliament")
        ingest_cli._sync_engine_paths()
        res = ingest_cli.ingest_source(specs["parliament"])
        assert res["added"] == 1
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        assert saved["metadata"]["subject"] == "enriched subject"

    def test_append_is_prefix_stable_for_existing_records(self, env_data, no_embed):
        """Existing corpus lines are byte-identical after appending one new doc."""
        existing = [make_record("18-4-1"), make_record("18-4-2")]
        corpus = write_corpus(existing)
        before = corpus.read_bytes()
        future = data_dir() / "ministry_new"
        future.mkdir(parents=True)
        (future / "report.txt").write_text("MoES approved the new marine station.", encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("ministry_new")
        res = ingest_cli.run_sources(specs)
        after = corpus.read_bytes()
        assert res["added"] == 1
        assert after.startswith(before)  # append-only: untouched prefix
        assert len(read_corpus_lines()) == 3

    def test_incremental_selected_when_index_usable(self, env_data, no_embed):
        future = data_dir() / "ministry_new"
        future.mkdir(parents=True)
        (future / "note.txt").write_text("A new report to embed incrementally.", encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("ministry_new")
        res = ingest_cli.run_sources(specs)
        assert res["embed"] == "incremental"
        assert no_embed["incremental"] == 1 and no_embed["rebuild"] == 0

    def test_rebuild_ONLY_when_index_missing_or_explicit(self, env_data, no_embed):
        future = data_dir() / "ministry_new"
        future.mkdir(parents=True)
        (future / "note.txt").write_text("A report that will need a first build.", encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("ministry_new")
        no_embed["exists"] = False  # no usable index -> first build (the allowed case)
        res = ingest_cli.run_sources(specs)
        assert res["embed"] == "rebuild" and no_embed["rebuild"] == 1

    def test_full_rebuild_forces_rebuild_even_with_usable_index(self, env_data, no_embed):
        future = data_dir() / "ministry_new"
        future.mkdir(parents=True)
        (future / "note.txt").write_text("Report to be reindexed entirely.", encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("ministry_new")
        no_embed["exists"] = True  # usable index present
        res = ingest_cli.run_sources(specs, full_rebuild=True)
        assert res["embed"] == "rebuild"
        assert no_embed["rebuild"] == 1 and no_embed["incremental"] == 0

    def test_no_rebuild_defers_index(self, env_data, no_embed):
        future = data_dir() / "ministry_new"
        future.mkdir(parents=True)
        (future / "note.txt").write_text("Report appended but not yet indexed.", encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("ministry_new")
        res = ingest_cli.run_sources(specs, no_rebuild=True)
        assert res["embed"] == "defer"
        assert no_embed["incremental"] == 0 and no_embed["rebuild"] == 0
        assert len(read_corpus_lines()) == 1  # corpus still updated

    def test_choose_embed_action_contract(self):
        assert ingest_cli.choose_embed_action(0, no_rebuild=False, full_rebuild=True) == "skip"
        assert ingest_cli.choose_embed_action(3, no_rebuild=True, full_rebuild=False) == "defer"


# ─────────────────────────────────────────────────────────────────────────────
# 8. add_records: existing vectors are never re-embedded (offline, stubbed)
# ─────────────────────────────────────────────────────────────────────────────

class TestAddRecordsIncrementalInvariant:
    def test_existing_ids_never_reach_the_embedder(self):
        from src.retrieval.hybrid.pipeline import HybridRAGPipeline

        old1 = make_record("18-4-1")
        old2 = make_record("18-4-2")
        new = make_record("18-4-3", q="What new funding was announced for ocean observations?",
                          a="Funding of Rs. 42 crore was announced in the latest session.")

        # Bypass __init__ (it loads bge-m3) — attach stub collaborators only.
        p = HybridRAGPipeline.__new__(HybridRAGPipeline)
        p._doc_map = {old1.question_id: old1, old2.question_id: old2}
        p._doc_texts = {}
        p._long_chunk_map = {}
        p._long_chunk_texts = {}
        p.long_doc_chars = 10**9  # no long-doc chunking in this test

        embedded_texts: list[list[str]] = []

        class StubEmbedder:
            def embed_batch(self, texts, batch_size=1, show_progress=False):
                embedded_texts.append(list(texts))
                return [[0.1, 0.2]] * len(texts)

        added_ids: list[list[str]] = []

        class StubVectorStore:
            def add(self, ids, embeddings):
                added_ids.append(list(ids))

        bm25_payloads: list[list] = []

        class StubBM25:
            def build(self, docs):
                bm25_payloads.append(list(docs))

        p.embedder = StubEmbedder()
        p.vector_store = StubVectorStore()
        p.bm25_index = StubBM25()

        n = p.add_records([old1, old2, new])

        assert n == 1
        # the ONLY text sent to the embedding model is the new record's
        assert embedded_texts == [[new.document_content]]
        # the ONLY vectors appended belong to the new id
        assert added_ids == [[new.question_id]]
        # pre-existing doc_map entries are the identical objects (untouched)
        assert p._doc_map["18-4-1"] is old1 and p._doc_map["18-4-2"] is old2
        assert p._doc_map["18-4-3"] is new
        # BM25 rebuild is text-only by DESIGN (no embeddings involved) and
        # covers all docs — that is the documented add_records contract
        (payload,) = bm25_payloads
        assert sorted(doc_id for doc_id, *_ in payload) == ["18-4-1", "18-4-2", "18-4-3"]


# ─────────────────────────────────────────────────────────────────────────────
# 10. Frontend/inbox compatibility + reuse-not-duplication pins
# ─────────────────────────────────────────────────────────────────────────────

class TestFrontendInboxCompat:
    def test_reuse_not_duplication_pins(self):
        """The CLI must reuse the engine's index implementations verbatim —
        the same engine server._run_ingest_job drives (via ingest_service)."""
        assert ingest_cli.embed_incremental is engine.incremental_update
        assert ingest_cli.embed_full_rebuild is engine.rebuild_index
        assert ingest_cli.index_is_usable is engine._index_exists

    def test_inbox_source_defaults_move_processed(self, env_data):
        """UI contract: files that ingest successfully move to inbox/processed
        (mirrors ingest_folder.main's `"inbox" in folder` default and what
        server /api/ingest does with move_processed=True)."""
        registered, _, _cm = ingest_cli.load_sources(Path("/nonexistent/none.yaml"))
        assert registered["inbox"].move_processed is True
        for name in ("incois", "moes", "parliament"):
            assert registered[name].move_processed is False

    def test_inbox_ingest_moves_file_to_processed(self, env_data, no_embed):
        inbox = data_dir() / "inbox"
        inbox.mkdir(parents=True)
        src = inbox / "circular.txt"
        src.write_text("Office circular on revised audit timelines for FY 2025-26.",
                       encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("inbox")
        res = ingest_cli.run_sources(specs)
        assert res["added"] == 1
        assert not src.exists()
        assert (inbox / "processed" / "circular.txt").exists()

    def test_qa_jsonl_via_folder_ingest_keeps_audit_qa_type(self, env_data, no_embed):
        future = data_dir() / "ministry_new"
        future.mkdir(parents=True)
        (future / "qa.jsonl").write_text(
            json.dumps({"Question": "What is the INCOIS tsunami warning latency?",
                        "Answer": "Under ten minutes for Indian Ocean events."}) + "\n",
            encoding="utf-8")
        specs, _, _cm = ingest_cli.resolve_sources("ministry_new")
        ingest_cli.run_sources(specs)
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        assert saved["metadata"]["document_type"] == "audit_qa"  # detection preserved
        assert saved["question_id"].startswith("incdoc-")  # deterministic hash id


# ─────────────────────────────────────────────────────────────────────────────
# CLI surface
# ─────────────────────────────────────────────────────────────────────────────

class TestCliSurface:
    def test_main_lists_sources_when_no_args(self, env_data, capsys):
        monkeypatch_argv = ["prog"]
        import sys
        old = sys.argv
        try:
            sys.argv = monkeypatch_argv
            ingest_cli.main()
        finally:
            sys.argv = old
        out = capsys.readouterr().out
        assert "parliament" in out and "incois" in out and "Pass a source name" in out

    def test_main_unknown_source_exits_2(self, env_data, capsys):
        import sys
        old = sys.argv
        try:
            sys.argv = ["prog", "nope", "--ingest"]
            with pytest.raises(SystemExit) as se:
                ingest_cli.main()
        finally:
            sys.argv = old
        assert se.value.code == 2
        assert "unknown source" in capsys.readouterr().err
