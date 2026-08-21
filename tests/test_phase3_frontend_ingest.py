"""Phase 3 validation — frontend hierarchical upload + ingestion safety.

Covers (per the Phase-3 brief, groups A–K), all OFFLINE and tmp-redirected:

  A. frontend source discovery (registered ∪ discovered, config-driven)
  B. organization discovery (org_map ∪ on-disk org dirs)
  C. document-type discovery (category_map-driven)
  D. upload metadata propagation (source/org/document_type stamped; the
     physical path and the metadata can never drift — walker parity)
  E. frontend → SAME ingestion pipeline as the CLI (reuse-not-duplication)
  F. frontend incremental ingestion on a REAL FAISS index (only new embedded)
  G. duplicate upload (0 new, explicit verdict, index untouched)
  H. multiple-document upload (mixed new/dup batch counts)
  I. invalid hierarchy → clear operator-facing 4xx messages
  J. `retrieve build` canonical-corpus selection (safety fix #1)
  K. `sync_sources` append safety — never shrinks the canonical corpus (fix #2)

The production corpus/index is never touched: every path is redirected via
APP_DATA_DIR / APP_INDEX_DIR to pytest tmp dirs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

import src.scripts.ingest as ingest_cli
import src.scripts.ingest_folder as engine
import src.scripts.ingest_service as svc
import src.scripts.sync_sources as sync_mod
from src.models.qa_record import QARecord
from src.retrieval.hybrid.pipeline import HybridRAGPipeline
from src.utils.app_paths import corpus_path, data_dir, index_dir

from tests.test_ingest_cli import (  # noqa: F401  (shared fixtures/helpers)
    env_data, make_record, no_embed, read_corpus_lines, write_corpus,
)
from tests.test_phase2_ingest_retrieval import (  # noqa: F401
    DeterministicEmbedder, _NoopReranker, build_real_index, load_real_index,
    stamped_record,
)


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_neutral_doc(path: Path, token: str) -> Path:
    """A .txt whose content/filename fires NO legacy detection rule, so the
    category hint is what decides the document_type (tests hint propagation,
    not content overrides). Pure-alpha token keeps it BM25-visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Operations note concerning the {token} programme deployment status.\n"
        f"Monitoring outcomes and inventory figures for {token} are recorded here.\n",
        encoding="utf-8",
    )
    return path


def det_factory(records=None):
    """pipeline_factory seam for svc.update_index_in_process: deterministic
    embedder, real FAISS/BM25, zero ML downloads (Phase-2 pattern)."""
    if records is not None:
        return HybridRAGPipeline(records=records, embedder=DeterministicEmbedder(),
                                 reranker=_NoopReranker(), use_reranker=False)
    return HybridRAGPipeline(embedder=DeterministicEmbedder(),
                             reranker=_NoopReranker(), use_reranker=False)


@pytest.fixture()
def client(env_data):
    """FastAPI TestClient with sandboxed paths + clean per-test ingest state."""
    import src.retrieval.frontend.server as server
    from fastapi.testclient import TestClient

    server._INGEST_STATE.update(
        {"running": False, "last": None, "pending": 0, "pending_uploads": []}
    )
    return TestClient(server.app)


def _source(tree: dict, name: str) -> dict | None:
    for s in tree["sources"]:
        if s["name"] == name:
            return s
    return None


def _org(entry: dict, slug: str) -> dict | None:
    for o in entry.get("orgs", []):
        if o["slug"] == slug:
            return o
    return None


# ─────────────────────────────────────────────────────────────────────────────
# A. Frontend source discovery — one authoritative registry, no hardcoding
# ─────────────────────────────────────────────────────────────────────────────

class TestFrontendSourceDiscovery:
    def test_targets_tree_registered_and_discovered(self, client):
        # a registered hierarchical source with content
        write_neutral_doc(data_dir() / "moes" / "incois" / "annual_reports" / "axone.txt",
                          "axone")
        # a brand-new ZERO-CONFIG source (data/isro/ — design requirement)
        write_neutral_doc(data_dir() / "isro" / "research_papers" / "bemer.txt", "bemer")

        r = client.get("/api/ingest/targets")
        assert r.status_code == 200
        tree = r.json()

        moes = _source(tree, "moes")
        assert moes and moes["hierarchical"] is True and moes["upload"] is True
        assert moes["ministry"] == "EARTH SCIENCES"

        isro = _source(tree, "isro")  # discovered, never registered anywhere
        assert isro and isro["hierarchical"] is True and isro["discovered"] is True
        assert isro["ministry"] is None  # temporary scope never becomes a label

        inbox = _source(tree, "inbox")
        assert inbox and inbox["hierarchical"] is False and inbox["upload"] is True

        # crawler-owned flat sources are visible but NOT upload targets
        for legacy in ("incois", "moes_reports"):
            e = _source(tree, legacy)
            assert e is None or e.get("upload") is False

        # records sources (parliament) are never upload targets
        assert _source(tree, "parliament") is None

        # the active category map is echoed for the UI
        assert set(tree["document_types"]) == {
            "annual_report", "audit_report", "research_publication", "document"}
        assert tree["category_map"]["annual_reports"] == "annual_report"

    def test_tree_is_config_driven_not_python(self, env_data, tmp_path):
        """A config edit alone (new org in org_map) changes the tree — NO
        Python/frontend change (single authoritative configuration)."""
        cfg = yaml.safe_load((Path("config/sources.yaml")).read_text(encoding="utf-8"))
        cfg["sources"]["moes"]["org_map"]["essao"] = "essao"
        cfg_file = tmp_path / "sources_custom.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        tree = svc.discover_ingest_tree(cfg_file)
        moes = _source(tree, "moes")
        assert _org(moes, "essao") is not None  # appeared from CONFIG only


# ─────────────────────────────────────────────────────────────────────────────
# B. Organization discovery
# ─────────────────────────────────────────────────────────────────────────────

class TestOrganizationDiscovery:
    def test_orgs_from_org_map_dirs_and_default(self, client):
        write_neutral_doc(data_dir() / "moes" / "vssc" / "other" / "cipex.txt", "cipex")
        tree = client.get("/api/ingest/targets").json()
        moes = _source(tree, "moes")
        slugs = {o["slug"] for o in moes["orgs"]}
        # org_map values are offered even before any directory exists
        assert {"moes_hq", "incois", "imd", "iitm", "niot"} <= slugs
        # an unmapped on-disk org dir appears verbatim (config: "unmapped
        # segment becomes the org slug")
        assert "vssc" in slugs

    def test_discovered_source_orgs(self, client):
        write_neutral_doc(data_dir() / "isro" / "vssc" / "research_papers" / "doky.txt",
                          "doky")
        isro = _source(client.get("/api/ingest/targets").json(), "isro")
        slugs = {o["slug"] for o in isro["orgs"]}
        assert "isro" in slugs   # default_org always selectable
        assert "vssc" in slugs   # on-disk org dir discovered


# ─────────────────────────────────────────────────────────────────────────────
# C. Document-type discovery
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentTypeDiscovery:
    def test_categories_counts_and_file_previews(self, client):
        write_neutral_doc(data_dir() / "moes" / "incois" / "audit_reports" / "elqon.txt",
                          "elqon")
        write_neutral_doc(data_dir() / "moes" / "incois" / "audit_reports" / "fymek.txt",
                          "fymek")
        tree = client.get("/api/ingest/targets").json()
        incois = _org(_source(tree, "moes"), "incois")
        by_type = {c["document_type"]: c for c in incois["categories"]}

        audit = by_type["audit_report"]
        assert audit["category_dir"] == "audit_reports"
        assert audit["path"] == "moes/incois/audit_reports"
        assert audit["exists"] is True and audit["files"] == 2
        assert audit["file_names"] == ["elqon.txt", "fymek.txt"]

        empty = by_type["annual_report"]  # offered even before the dir exists
        assert empty["exists"] is False and empty["files"] == 0

    def test_custom_category_map_extends_types_via_config(self, env_data, tmp_path):
        cfg = yaml.safe_load(Path("config/sources.yaml").read_text(encoding="utf-8"))
        cfg["hierarchy"]["category_map"]["technical_reports"] = "technical_report"
        cfg_file = tmp_path / "sources_cat.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        tree = svc.discover_ingest_tree(cfg_file)
        assert "technical_report" in tree["document_types"]
        target = svc.resolve_upload_target(
            "moes", "incois", "technical_report", config_file=cfg_file)
        assert target.folder.name == "technical_reports"
        assert target.document_type == "technical_report"


# ─────────────────────────────────────────────────────────────────────────────
# D. Upload metadata propagation — hierarchy is authoritative, path==metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadMetadataPropagation:
    def test_hierarchical_upload_stamps_metadata(self, env_data, no_embed):
        target = svc.resolve_upload_target("moes", "incois", "annual_report")
        dest = write_neutral_doc(target.folder / "gyxon.txt", "gyxon")

        stats = svc.ingest_uploaded_files(target, [dest.name])
        assert stats["records_added"] == 1

        lines = read_corpus_lines()
        assert len(lines) == 1
        rec = QARecord.model_validate_json(lines[0])
        m = rec.metadata
        assert m.source == "moes"               # provenance: top source
        assert m.org == "incois"                # hierarchy is authoritative…
        assert m.document_type == "annual_report"
        assert m.ministry == "EARTH SCIENCES"   # configured ministry stamps

        assert stats["files"][0]["verdict"] == "new"
        assert stats["files"][0]["message"] == "Document uploaded successfully"

    def test_path_and_metadata_can_never_drift(self, env_data):
        """CONVERGENCE INVARIANT: the folder resolve_upload_target picks for
        (source, org, doc_type) is exactly the folder whose WALKER-resolved
        identity equals those values — CLI rescans restate the upload stamp."""
        spec, _, cm = ingest_cli.resolve_sources("moes")
        spec = spec["moes"]
        # org may be given as an org_map VALUE ("moes_hq") or KEY ("ministry");
        # both resolve to the same leaf identity the walker would derive
        for org, dt in (("incois", "annual_report"),
                        ("moes_hq", "audit_report"),
                        ("ministry", "audit_report"),
                        ("imd", "research_publication")):
            target = svc.resolve_upload_target("moes", org, dt)
            rel = target.folder.relative_to(data_dir() / "moes").parts
            w_org, w_hint = ingest_cli.resolve_path_context(rel, spec, cm)
            assert (w_org, w_hint) == (target.org, target.document_type)

        # discovered default-org placement is root-level (walker parity too)
        write_neutral_doc(data_dir() / "isro" / "other" / "hyvok.txt", "hyvok")
        isro_spec = ingest_cli.resolve_sources("isro")[0]["isro"]
        target = svc.resolve_upload_target("isro", "isro", "audit_report")
        rel = target.folder.relative_to(data_dir() / "isro").parts
        assert rel == ("audit_reports",)  # no bogus org segment for default org
        w_org, w_hint = ingest_cli.resolve_path_context(rel, isro_spec, cm)
        assert (w_org, w_hint) == (target.org, target.document_type)

    def test_root_level_default_org_upload(self, env_data, no_embed):
        # a new source appears via DISCOVERY once its directory exists on disk
        # (unknown names 404 instead of silently creating data/<typo>/)
        write_neutral_doc(data_dir() / "isro" / "other" / "seedoc.txt", "seedoc")
        target = svc.resolve_upload_target("isro", "isro", "research_publication")
        assert target.meta_context["default_ministry"] is None  # no legacy label
        dest = write_neutral_doc(target.folder / "julip.txt", "julip")
        svc.ingest_uploaded_files(target, [dest.name])
        rec = QARecord.model_validate_json(read_corpus_lines()[0])
        assert rec.metadata.org == "isro"
        assert rec.metadata.ministry is None    # Earth Sciences not stamped
        assert rec.metadata.document_type == "research_publication"


# ─────────────────────────────────────────────────────────────────────────────
# E. Frontend → the SAME ingestion pipeline as the CLI (no pipeline B)
# ─────────────────────────────────────────────────────────────────────────────

class TestSharedPipelineConvergence:
    def test_service_uses_the_engine_not_a_copy(self):
        # module identity: the service drives the very engine module the CLI uses
        assert svc._engine is ingest_cli._engine is engine
        assert svc._registry is ingest_cli
        # no second conversion implementation lives in the service
        assert [n for n in vars(svc) if n.startswith("convert_")] == []
        # the per-file probe IS the engine's converter entry point
        assert svc._engine.convert_one_detected is engine.convert_one_detected

    def test_ingest_uploaded_files_delegates_with_context(self, env_data, monkeypatch):
        seen_kwargs = {}
        real = engine.ingest_folder

        def spy(folder, move_processed=False, meta_context=None, only_files=None):
            seen_kwargs.update(folder=folder, move_processed=move_processed,
                               meta_context=meta_context, only_files=only_files)
            return real(folder, move_processed=move_processed,
                        meta_context=meta_context, only_files=only_files)

        monkeypatch.setattr(engine, "ingest_folder", spy)
        target = svc.resolve_upload_target("moes", "imd", "audit_report")
        dest = write_neutral_doc(target.folder / "kewal.txt", "kewal")
        stats = svc.ingest_uploaded_files(target, [dest.name])

        assert stats["records_added"] == 1
        assert seen_kwargs["only_files"] == {dest.name}          # scoped, not whole-leaf
        assert seen_kwargs["move_processed"] is False            # tree files stay put
        ctx = seen_kwargs["meta_context"]
        assert ctx["org"] == "imd" and ctx["source"] == "moes"
        assert ctx["doc_type_hint"] == "audit_report"

    def test_server_job_flows_through_service(self, client, monkeypatch):
        """The background job calls svc.ingest_uploaded_files + the shared
        in-process index updater (spies) — never a server-side re-implementation."""
        import src.retrieval.frontend.server as server

        calls = {"ingest_uploaded_files": 0, "update_index": 0}
        real_ingest = svc.ingest_uploaded_files

        def spy_ingest(target, filenames):
            calls["ingest_uploaded_files"] += 1
            return real_ingest(target, filenames)

        def fake_index():
            calls["update_index"] += 1
            return 0, None

        monkeypatch.setattr(svc, "ingest_uploaded_files", spy_ingest)
        monkeypatch.setattr(svc, "update_index_in_process", fake_index)

        r = client.post(
            "/api/ingest/upload?filename=loquen.txt&source=moes&org=incois"
            "&document_type=annual_report",
            content="Operations note concerning the loquen programme deployment status. "
                    "Monitoring outcomes for loquen are recorded here.".encode(),
        )
        assert r.status_code == 200, r.text
        server._run_ingest_job()
        assert calls == {"ingest_uploaded_files": 1, "update_index": 1}

        last = server._INGEST_STATE["last"]
        assert last["received"] == 1 and last["new_documents"] == 1
        assert last["duplicates"] == 0 and last["failed_documents"] == 0
        assert last["files"][0]["verdict"] == "new"
        # legacy keys still present and shaped (old UI contract intact)
        assert {"at", "ok", "failed", "records", "message"} <= set(last)


# ─────────────────────────────────────────────────────────────────────────────
# F. Frontend incremental ingestion — REAL index, only the new doc embedded
# ─────────────────────────────────────────────────────────────────────────────

class TestFrontendIncremental:
    def _seed_corpus_and_index(self, n: int):
        idx = index_dir()
        recs = [
            stamped_record(
                f"incdoc-{i:06d}", org="incois", source="moes",
                doc_type="annual_report",
                q=f"What does the note say about programme item {i}?",
                a=f"Programme item {i} outcomes documented; seafan{i:04d} catalogued.",
            )
            for i in range(n)
        ]
        write_corpus(recs)
        build_real_index(recs, idx)
        return recs, idx

    def test_upload_one_doc_embeds_only_that_doc(self, env_data):
        n0 = 60  # the "1432 + 1" scenario, scaled for CI (Phase 2 pins 1432)
        recs, idx = self._seed_corpus_and_index(n0)
        ntotal_before = load_real_index(idx).vector_store._index.ntotal
        assert ntotal_before == n0

        # ── ONE new document uploaded through the FRONTEND path ──
        target = svc.resolve_upload_target("moes", "incois", "audit_report")
        dest = write_neutral_doc(target.folder / "muvek.txt", "muvek")
        stats = svc.ingest_uploaded_files(target, [dest.name])
        assert stats["received"] == 1 and stats["new"] == 1
        assert stats["records_added"] == 1            # only the new doc appended
        assert len(read_corpus_lines()) == n0 + 1     # 1432 + 1 shape

        # index update through the shared in-process path the server job calls
        n_embedded, pipe = svc.update_index_in_process(pipeline_factory=det_factory)
        assert n_embedded == 1                        # ONLY the new doc embedded
        assert pipe.vector_store._index.ntotal == n0 + 1

        # reload from disk: old doc ids intact, new doc searchable
        p2 = load_real_index(idx)
        assert len(p2._doc_map) == n0 + 1
        assert all(r.question_id in p2._doc_map for r in recs[:5])
        hits = p2.bm25_index.search("muvek programme deployment", k=3)
        assert hits and hits[0][0] not in {r.question_id for r in recs}
        new_rec = p2._doc_map[hits[0][0]]  # loaded doc_map holds QARecord objects
        assert new_rec.metadata.org == "incois"
        assert new_rec.metadata.source == "moes"
        assert new_rec.metadata.document_type == "audit_report"

    def test_second_identical_upload_is_a_full_noop(self, env_data):
        n0 = 25
        self._seed_corpus_and_index(n0)
        target = svc.resolve_upload_target("moes", "incois", "annual_report")
        dest = write_neutral_doc(target.folder / "nytol.txt", "nytol")
        svc.ingest_uploaded_files(target, [dest.name])
        n1, _ = svc.update_index_in_process(pipeline_factory=det_factory)
        assert n1 == 1
        idx_blob = (index_dir() / "vector_store.index").read_bytes()
        corpus_blob = corpus_path().read_bytes()

        # exact same document again: 0 records, 0 embeddings, 0 index writes
        stats2 = svc.ingest_uploaded_files(target, [dest.name])
        assert stats2["received"] == 1 and stats2["new"] == 0
        assert stats2["duplicates"] == 1 and stats2["records_added"] == 0
        assert stats2["files"][0]["verdict"] == "duplicate"

        n2, pipe2 = svc.update_index_in_process(pipeline_factory=det_factory)
        assert n2 == 0
        assert pipe2.vector_store._index.ntotal == n0 + 1
        assert (index_dir() / "vector_store.index").read_bytes() == idx_blob
        assert corpus_path().read_bytes() == corpus_blob


# ─────────────────────────────────────────────────────────────────────────────
# G/H. Duplicate + multi-document uploads
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadBatches:
    def test_duplicate_upload_message(self, env_data, no_embed):
        target = svc.resolve_upload_target("moes", "incois", "annual_report")
        dest = write_neutral_doc(target.folder / "oqarel.txt", "oqarel")
        first = svc.ingest_uploaded_files(target, [dest.name])
        assert first["files"][0]["message"] == "Document uploaded successfully"

        # "the exact same document again" — same name, same content (the file
        # is overwritten in place, then conversion dedups on the content-hash
        # id which embeds the stem for document records)
        second = svc.ingest_uploaded_files(target, [dest.name])
        assert second["files"][0]["verdict"] == "duplicate"
        assert second["files"][0]["message"] == "Document already exists — skipped"
        assert second["records_added"] == 0
        assert len(read_corpus_lines()) == 1

        # honest note on EXISTING engine identity semantics (unchanged by
        # Phase 3): a document record's id hashes ("Document: <stem>" + text),
        # so the same bytes under a DIFFERENT filename are a distinct record —
        # exactly what a CLI rerun in this folder would also produce.
        dest2 = target.folder / "oqarel_copy.txt"
        dest2.write_bytes(dest.read_bytes())
        third = svc.ingest_uploaded_files(target, [dest2.name])
        assert third["files"][0]["verdict"] == "new"
        assert third["records_added"] == 1
        assert len(read_corpus_lines()) == 2

    def test_mixed_batch_counts(self, env_data, no_embed):
        target = svc.resolve_upload_target("moes", "imd", "research_publication")
        pre = write_neutral_doc(target.folder / "pafen.txt", "pafen")
        svc.ingest_uploaded_files(target, [pre.name])  # one doc already in corpus

        batch = [
            write_neutral_doc(target.folder / "qorin.txt", "qorin").name,   # new
            write_neutral_doc(target.folder / "ruget.txt", "ruget").name,   # new
            pre.name,                                                       # dup
        ]
        stats = svc.ingest_uploaded_files(target, batch)
        assert stats["received"] == 3
        assert stats["new"] == 2
        assert stats["duplicates"] == 1
        assert stats["failed"] == 0
        assert stats["records_added"] == 2
        assert len(read_corpus_lines()) == 3  # 1 pre-existing + 2 new

        verdicts = {f["name"]: f["verdict"] for f in stats["files"]}
        assert verdicts == {"qorin.txt": "new", "ruget.txt": "new", "pafen.txt": "duplicate"}

    def test_invalid_document_upload_is_a_failed_verdict(self, env_data, no_embed):
        """A readable-but-unextractable file → failed verdict, no crash, 0 added."""
        target = svc.resolve_upload_target("moes", "incois", "document")
        bad = target.folder / "sogar.json"
        bad.write_text('{"unexpected": "shape without any known keys"}', encoding="utf-8")
        stats = svc.ingest_uploaded_files(target, [bad.name])
        assert stats["failed"] == 1 and stats["records_added"] == 0
        assert stats["files"][0]["verdict"] == "failed"


# ─────────────────────────────────────────────────────────────────────────────
# I. Invalid hierarchy / upload validation — operator-facing messages
# ─────────────────────────────────────────────────────────────────────────────

class TestInvalidHierarchy:
    def _post(self, client, qs: str, body: bytes = b"0123456789abcdef"):
        return client.post(f"/api/ingest/upload?{qs}", content=body)

    def test_unknown_source_404(self, client):
        r = self._post(client, "filename=x.txt&source=nosuchorg")
        assert r.status_code == 404
        assert "Unknown source 'nosuchorg'" in r.json()["detail"]
        assert "Known upload targets" in r.json()["detail"]

    def test_missing_organization_400(self, client):
        r = self._post(client, "filename=x.txt&source=moes&document_type=annual_report")
        assert r.status_code == 400
        assert r.json()["detail"] == "Organization is required for source 'moes'"

    def test_unknown_organization_400_lists_known(self, client):
        r = self._post(client, "filename=x.txt&source=moes&org=nimat&document_type=annual_report")
        assert r.status_code == 400
        d = r.json()["detail"]
        assert "Unknown organization 'nimat'" in d and "incois" in d

    def test_missing_and_invalid_document_type_400(self, client):
        r = self._post(client, "filename=x.txt&source=moes&org=incois")
        assert r.status_code == 400
        assert r.json()["detail"] == "Document type is required for source 'moes'"

        r = self._post(client, "filename=x.txt&source=moes&org=incois&document_type=menu")
        assert r.status_code == 400
        d = r.json()["detail"]
        assert "Invalid document type 'menu'" in d and "annual_report" in d

    def test_crawler_flat_source_rejected_with_guidance(self, client):
        r = self._post(client, "filename=x.txt&source=incois&org=incois&document_type=document")
        assert r.status_code == 400
        assert "crawler-managed flat source" in r.json()["detail"]

    def test_records_source_rejected(self, client):
        r = self._post(client, "filename=x.txt&source=parliament")
        assert r.status_code == 400
        assert "records source" in r.json()["detail"]

    def test_bad_extension_and_empty_and_oversize(self, client, monkeypatch):
        r = self._post(client, "filename=x.exe&source=moes&org=incois&document_type=document")
        assert r.status_code == 400 and "Unsupported file type .exe" in r.json()["detail"]

        r = client.post("/api/ingest/upload?filename=x.txt&source=moes&org=incois"
                        "&document_type=document", content=b"tiny")
        assert r.status_code == 400 and r.json()["detail"] == "File is empty"

        # the pre-read header guard (2GB upload must never enter RAM)
        r = client.post(
            "/api/ingest/upload?filename=x.txt&source=moes&org=incois&document_type=document",
            content=b"0123456789abcdef",
            headers={"content-length": str(201 * 1024 * 1024)},
        )
        assert r.status_code == 413 and "File too large" in r.json()["detail"]

        # the post-read body guard (same policy, second line of defence)
        import src.retrieval.frontend.server as server
        monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 32)
        r = client.post("/api/ingest/upload?filename=x.txt&source=moes&org=incois"
                        "&document_type=document",
                        content=b"0" * 64)
        assert r.status_code == 413 and "File too large" in r.json()["detail"]

    def test_missing_filename_400(self, client):
        r = client.post("/api/ingest/upload?source=moes", content=b"0123456789abcdef")
        assert r.status_code == 400 and "filename query param required" in r.json()["detail"]

    def test_validation_failure_stages_nothing(self, client):
        self._post(client, "filename=x.txt&source=moes&org=bogus&document_type=document")
        import src.retrieval.frontend.server as server
        assert server._INGEST_STATE["pending_uploads"] == []


# ─────────────────────────────────────────────────────────────────────────────
# J. retrieve build — canonical corpus selection (safety fix #1)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCanonicalCorpusSelection:
    def _capture_build(self, monkeypatch, tmp_path):
        """Run get_pipeline's data-selection with the heavy build neutralized;
        capture the data file it chose."""
        import src.retrieval.cli as cli

        chosen = {}

        def fake_load_jsonl(path, *a, **k):
            chosen["data_path"] = Path(path)
            return []

        class _DummyPipeline:
            def __init__(self, records=None, **kw):
                self.records = records or []

            def save(self, path, *a, **k):
                Path(path).mkdir(parents=True, exist_ok=True)

            def __len__(self):
                return len(self.records)

        monkeypatch.setattr(cli.DataLoader, "load_jsonl", staticmethod(fake_load_jsonl))
        monkeypatch.setattr(cli, "HybridRAGPipeline", _DummyPipeline)
        return cli, chosen

    def test_canonical_corpus_preferred_over_enriched(self, env_data, monkeypatch, tmp_path):
        cli, chosen = self._capture_build(monkeypatch, tmp_path)
        # canonical corpus present…
        write_corpus([make_record("incdoc-aaaaaa")])
        # …plus a NEWER parliament-subset file — the historic wrong pick
        enr = data_dir() / "enriched"
        enr.mkdir(parents=True, exist_ok=True)
        subset = enr / "zz_newest.jsonl"
        subset.write_text(make_record("18-4-9999").model_dump_json() + "\n", encoding="utf-8")
        os.utime(subset, (2_000_000_000, 2_000_000_000))

        cli.get_pipeline(force_rebuild=True, all_ministries=True,
                         index_dir=str(tmp_path / "idx-empty"))
        assert chosen["data_path"] == corpus_path()  # canonical, not the subset

    def test_legacy_fallback_preserved_without_corpus(self, env_data, monkeypatch, tmp_path):
        cli, chosen = self._capture_build(monkeypatch, tmp_path)
        enr = data_dir() / "enriched"
        enr.mkdir(parents=True, exist_ok=True)
        legacy = enr / "qa.jsonl"
        legacy.write_text(make_record("18-1-1").model_dump_json() + "\n", encoding="utf-8")

        cli.get_pipeline(force_rebuild=True, all_ministries=True,
                         index_dir=str(tmp_path / "idx-empty"))
        assert chosen["data_path"] == legacy  # old behavior when no canonical corpus

    def test_explicit_data_always_wins(self, env_data, monkeypatch, tmp_path):
        cli, chosen = self._capture_build(monkeypatch, tmp_path)
        write_corpus([make_record("incdoc-bbbbbb")])
        custom = tmp_path / "custom.jsonl"
        custom.write_text(make_record("z").model_dump_json() + "\n", encoding="utf-8")

        cli.get_pipeline(data_file=str(custom), force_rebuild=True,
                         all_ministries=True, index_dir=str(tmp_path / "idx-empty"))
        assert chosen["data_path"] == custom


# ─────────────────────────────────────────────────────────────────────────────
# K. sync_sources append safety — the canonical corpus can never shrink (F.1)
# ─────────────────────────────────────────────────────────────────────────────

class TestSyncSourcesAppendSafety:
    def test_merge_preserves_foreign_records_and_refreshes_own(self, tmp_path):
        corpus = tmp_path / "corpus_reports.jsonl"
        own_old = stamped_record("incdoc-own001", q="Old sync question phrasing?",
                                 a="Old sync answer text.")
        foreign = [stamped_record(f"18-4-{3000+i}", q=f"Parliament question {i}?")
                   for i in range(5)]
        raw_line = '{"broken": true'  # historical malformed line — never dropped
        corpus.write_text(
            "".join(r.model_dump_json() + "\n" for r in [own_old, *foreign]) + raw_line + "\n",
            encoding="utf-8",
        )

        own_new = stamped_record("incdoc-own001", q="Old sync question phrasing?",
                                 a="REFRESHED sync answer text.")
        brand_new = stamped_record("incdoc-own002", q="A genuinely new sync document?")
        scratch = tmp_path / ".sync_ingest_scratch.jsonl"
        scratch.write_text(
            own_new.model_dump_json() + "\n" + brand_new.model_dump_json() + "\n",
            encoding="utf-8",
        )

        stats = sync_mod.merge_scratch_into_corpus(scratch, corpus)

        final = [l for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_id = {}
        for l in final:
            try:
                by_id[json.loads(l)["question_id"]] = l
            except Exception:  # noqa: BLE001
                pass
        # union, never shrink: 5 foreign + refreshed own + brand new + raw line
        assert stats["existing"] == 7  # 6 records + 1 raw line counted
        assert stats["updated"] == 1 and stats["added"] == 1
        assert stats["preserved_raw"] == 1 and raw_line in final
        assert all(f"18-4-{3000+i}" in by_id for i in range(5))  # foreign intact
        assert "REFRESHED" in by_id["incdoc-own001"]               # sync refresh flows
        assert "incdoc-own002" in by_id                            # new appended
        assert len(final) == 8

    def test_merge_with_smaller_sync_subset_cannot_shrink(self, tmp_path):
        """The regression scenario: canonical corpus has 100 records, the sync
        output covers only 3 — the result stays 100."""
        corpus = tmp_path / "corpus_reports.jsonl"
        recs = [stamped_record(f"incdoc-{i:06d}", q=f"What is the status of item {i}?")
                for i in range(100)]
        corpus.write_text("".join(r.model_dump_json() + "\n" for r in recs), encoding="utf-8")

        scratch = tmp_path / "scratch.jsonl"
        sub = [stamped_record("incdoc-000001", q="What is the status of item 1?",
                              a="Refreshed answer text one."),
               stamped_record("incdoc-000050", q="What is the status of item 50?",
                              a="Refreshed answer text fifty."),
               stamped_record("incdoc-new-99", q="Newly synced document?")]
        scratch.write_text("".join(r.model_dump_json() + "\n" for r in sub), encoding="utf-8")

        stats = sync_mod.merge_scratch_into_corpus(scratch, corpus)
        lines = [l for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 101            # 100 preserved + 1 new — never 3
        assert stats["added"] == 1 and stats["updated"] == 2

    def test_merge_missing_scratch_is_a_noop(self, tmp_path):
        corpus = tmp_path / "corpus_reports.jsonl"
        corpus.write_text(make_record("x-1").model_dump_json() + "\n", encoding="utf-8")
        before = corpus.read_bytes()
        stats = sync_mod.merge_scratch_into_corpus(tmp_path / "absent.jsonl", corpus)
        assert stats["skipped"] is True
        assert corpus.read_bytes() == before

    def test_main_wires_ingest_through_scratch_then_merge(self, env_data, monkeypatch):
        """Offline end-to-end of --manual: downloads are faked, ingest_all is
        simulated by writing its scratch output; the canonical corpus (seeded
        with foreign records) must absorb the merge, never be replaced."""
        corpus = corpus_path()
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(stamped_record("18-4-3035").model_dump_json() + "\n", encoding="utf-8")

        # sync module keeps CWD-relative paths (pre-existing F.13) — redirect here
        monkeypatch.setattr(sync_mod, "MANIFEST", data_dir() / "sync_manifest.json")
        monkeypatch.setattr(sync_mod, "LOG", data_dir() / "sync.log")
        monkeypatch.setattr(sync_mod, "DOWNLOAD_DIR", data_dir() / "incois_reports")
        monkeypatch.setattr(sync_mod, "MOES_DIR", data_dir() / "moes_reports")
        monkeypatch.setattr(sync_mod, "OCR_DIR", data_dir() / "scanned_ocr")
        monkeypatch.setattr(sync_mod, "CORPUS", corpus)
        monkeypatch.setattr(sync_mod, "discover_incois", lambda: {"https://x/AR25.pdf": "annual"})
        monkeypatch.setattr(sync_mod, "discover_moes", lambda: {})

        def fake_download(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"%PDF-1.4 fake bytes for the sync test")
            return True

        new_sync_rec = stamped_record("incdoc-syncnew", q="Freshly synced annual note?")
        ran_cmds = []

        def fake_run(*cmd):
            ran_cmds.append(list(cmd))
            if "src.scripts.ingest_all" in cmd:  # simulate its --out scratch file
                out = Path(cmd[cmd.index("--out") + 1])
                out.write_text(new_sync_rec.model_dump_json() + "\n", encoding="utf-8")

        monkeypatch.setattr(sync_mod, "download", fake_download)
        monkeypatch.setattr(sync_mod, "run", fake_run)
        monkeypatch.setattr(sync_mod, "_server_running", lambda: True)
        monkeypatch.setattr("sys.argv", ["sync_sources", "--manual"])

        sync_mod.main()

        ingest_cmds = [c for c in ran_cmds if "src.scripts.ingest_all" in c]
        assert len(ingest_cmds) == 1
        out_arg = Path(ingest_cmds[0][ingest_cmds[0].index("--out") + 1])
        assert out_arg.name == ".sync_ingest_scratch.jsonl"  # NOT the canonical corpus
        assert out_arg.parent == corpus.parent

        lines = [l for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
        ids = {json.loads(l)["question_id"] for l in lines}
        assert ids == {"18-4-3035", "incdoc-syncnew"}  # foreign preserved + new merged
        assert not out_arg.exists()                    # scratch cleaned up
