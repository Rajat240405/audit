"""Phase 1 tests — hierarchical data structure + config-driven discovery.

Covers the task's focused areas:
  * hierarchical path discovery (moes/<org>/<category>/ walking)
  * source/org/category extraction from paths (path grammar)
  * metadata propagation raw -> corpus (org/source/document_type/ministry)
  * existing flat source compatibility (byte-identical legacy behavior)
  * new unknown source discovery (data/isro with zero config)
  * category mapping (category_map -> document_type, content still wins)
  * no regression to existing ingestion behavior (flat engine contract)

All tests offline; APP_* paths redirected to tmp; embed phase monkeypatched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.scripts.ingest as ingest_cli
import src.scripts.ingest_folder as engine
from src.models.qa_record import QARecord
from src.utils.app_paths import corpus_path, data_dir
from tests.test_ingest_cli import make_record, read_corpus_lines, env_data, no_embed  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Tree builder
# ─────────────────────────────────────────────────────────────────────────────

def build_moes_tree(root: Path) -> None:
    """data/moes/{ministry,incois,imd,iitm}/{categories}/<files>"""
    files = {
        "ministry/annual_reports/annual_2025.txt":
            "Ministry overview and budget outlay for ocean services.",
        "incois/annual_reports/AR_2024_25.txt":
            "INCOIS yearly activities summary document text.",
        "incois/audit_reports/audit_2024.txt":
            "Observations on procurement procedures at the centre.",
        "incois/research_papers/paper_buoys.txt":
            "A study on moored buoy networks in the Indian Ocean.",
        "incois/other/misc_note.txt":
            "General information note about operations.",
        "imd/audit_reports/imd_audit.txt":
            "Audit paragraphs for the meteorological department.",
    }
    for rel, text in files.items():
        f = root / "moes" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")


def moes_spec() -> ingest_cli.SourceSpec:
    specs, _, cmap = ingest_cli.resolve_sources("moes")
    return specs["moes"]


def _rel(job: ingest_cli.LeafJob) -> str:
    """Leaf path relative to data/moes, OS-independent. ``str(Path)`` renders
    ``\\`` on Windows; ``.as_posix()`` normalizes the separator at the
    comparison boundary so the path-grammar assertions hold on any platform
    (production paths stay real ``Path`` objects — only the test renders
    them)."""
    return job.folder.relative_to(data_dir() / "moes").as_posix()


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical path discovery + source/org/category extraction (path grammar)
# ─────────────────────────────────────────────────────────────────────────────

class TestPathGrammar:
    def test_walker_finds_all_leaf_dirs(self, env_data, no_embed):
        build_moes_tree(data_dir())
        jobs = ingest_cli.expand_source(moes_spec(), ingest_cli._BUILTIN_CATEGORY_MAP)
        leaves = {_rel(j) for j in jobs}
        assert leaves == {
            "ministry/annual_reports",
            "incois/annual_reports",
            "incois/audit_reports",
            "incois/research_papers",
            "incois/other",
            "imd/audit_reports",
        }

    def test_org_and_category_extraction_table(self, env_data, no_embed):
        build_moes_tree(data_dir())
        jobs = ingest_cli.expand_source(moes_spec(), ingest_cli._BUILTIN_CATEGORY_MAP)
        by_rel = {_rel(j): (j.org, j.doc_type_hint) for j in jobs}
        assert by_rel["ministry/annual_reports"] == ("moes_hq", "annual_report")
        assert by_rel["incois/annual_reports"] == ("incois", "annual_report")
        assert by_rel["incois/audit_reports"] == ("incois", "audit_report")
        assert by_rel["incois/research_papers"] == ("incois", "research_publication")
        assert by_rel["incois/other"] == ("incois", "document")
        assert by_rel["imd/audit_reports"] == ("imd", "audit_report")

    def test_root_level_files_use_default_org_and_legacY_detection(self, env_data):
        (data_dir() / "moes").mkdir(parents=True)
        (data_dir() / "moes" / "policy.txt").write_text(
            "Standing order on data sharing between centres.", encoding="utf-8")
        jobs = ingest_cli.expand_source(moes_spec(), ingest_cli._BUILTIN_CATEGORY_MAP)
        (job,) = jobs
        assert job.folder == data_dir() / "moes"
        assert job.org == "moes_hq" and job.doc_type_hint is None

    def test_unknown_org_segment_becomes_org_slug_verbatim(self, env_data):
        # no org_map entry, no config needed: cola/ becomes its own slug
        f = data_dir() / "moes" / "cola" / "audit_reports" / "x.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("Centre-specific audit narrative for the year.", encoding="utf-8")
        jobs = ingest_cli.expand_source(moes_spec(), ingest_cli._BUILTIN_CATEGORY_MAP)
        (job,) = jobs
        assert job.org == "cola" and job.doc_type_hint == "audit_report"

    def test_walker_skips_processed_and_hidden_dirs(self, env_data):
        for rel in ("incois/processed/skip.txt", "incois/.hidden/skip.txt",
                    "incois/__pycache__/skip.txt"):
            f = data_dir() / "moes" / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("must never be ingested by the walker", encoding="utf-8")
        good = data_dir() / "moes" / "incois" / "audit_reports" / "keep.txt"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_text("Real audit content to keep and ingest.", encoding="utf-8")
        jobs = ingest_cli.expand_source(moes_spec(), ingest_cli._BUILTIN_CATEGORY_MAP)
        assert [_rel(j) for j in jobs] == ["incois/audit_reports"]


# ─────────────────────────────────────────────────────────────────────────────
# Subpath selectors: moes/incois, moes/incois/annual_reports
# ─────────────────────────────────────────────────────────────────────────────

class TestSubpathSelectors:
    def test_org_branch_selector(self, env_data, no_embed):
        build_moes_tree(data_dir())
        specs, _, cmap = ingest_cli.resolve_sources("moes/incois")
        spec = specs["moes"]
        assert spec.subpath == "incois"
        jobs = ingest_cli.expand_source(spec, cmap)
        rels = sorted(_rel(j) for j in jobs)
        assert rels == ["incois/annual_reports", "incois/audit_reports",
                        "incois/other", "incois/research_papers"]

    def test_category_leaf_selector(self, env_data, no_embed):
        build_moes_tree(data_dir())
        specs, _, cmap = ingest_cli.resolve_sources("moes/incois/annual_reports")
        jobs = ingest_cli.expand_source(specs["moes"], cmap)
        (job,) = jobs
        assert job.org == "incois" and job.doc_type_hint == "annual_report"

    def test_invalid_subpath_errors(self, env_data):
        build_moes_tree(data_dir())
        with pytest.raises(KeyError):
            ingest_cli.resolve_sources("moes/no_such_org")

    def test_subpath_rejected_on_flat_source(self, env_data):
        with pytest.raises(KeyError, match="not a single-root hierarchical"):
            ingest_cli.resolve_sources("inbox/anything")


# ─────────────────────────────────────────────────────────────────────────────
# Metadata propagation end-to-end (path -> engine -> corpus)
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataPropagation:
    def test_moes_incois_annual_reports_stamp(self, env_data, no_embed):
        """The canonical example: moes/incois/annual_reports/x.txt must carry
        source=moes, org=incois, document_type=annual_report, ministry stamp."""
        build_moes_tree(data_dir())
        specs, _, cmap = ingest_cli.resolve_sources("moes/incois/annual_reports")
        res = ingest_cli.run_sources(specs, category_map=cmap)
        assert res["added"] == 1
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        meta = saved["metadata"]
        assert meta["source"] == "moes"
        assert meta["org"] == "incois"
        assert meta["document_type"] == "annual_report"
        assert meta["ministry"] == "EARTH SCIENCES"

    def test_full_tree_stamps_each_leaf_correctly(self, env_data, no_embed):
        build_moes_tree(data_dir())
        specs, _, cmap = ingest_cli.resolve_sources("moes")
        res = ingest_cli.run_sources(specs, category_map=cmap)
        assert res["added"] == 6
        got = {}
        for line in read_corpus_lines():
            rec = json.loads(line)
            got[Path(rec["metadata"]["source_url"]).name] = rec["metadata"]
        assert got["AR_2024_25.txt"]["document_type"] == "annual_report"
        assert got["AR_2024_25.txt"]["org"] == "incois"
        assert got["audit_2024.txt"]["document_type"] == "audit_report"
        assert got["paper_buoys.txt"]["document_type"] == "research_publication"
        assert got["misc_note.txt"]["document_type"] == "document"
        assert got["annual_2025.txt"]["org"] == "moes_hq"      # ministry/ -> HQ
        assert got["imd_audit.txt"]["org"] == "imd"
        for m in got.values():
            assert m["source"] == "moes" and m["ministry"] == "EARTH SCIENCES"

    def test_audit_qa_under_tree_keeps_type_but_stamps_org_source(self, env_data, no_embed):
        f = data_dir() / "moes" / "incois" / "audit_reports" / "qa.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"Question": "What did the audit find on buoys?",
                                 "Answer": "Ten paragraphs on procurement."}) + "\n",
                     encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("moes")
        ingest_cli.run_sources(specs, category_map=cmap)
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        meta = saved["metadata"]
        assert meta["document_type"] == "audit_qa"   # QA type never overridden
        assert meta["org"] == "incois" and meta["source"] == "moes"

    def test_content_still_beats_category_hint(self, env_data, no_embed):
        """A wrongly-filed doc self-declares: content header outranks the
        folder's category hint (preserves the detection philosophy)."""
        f = data_dir() / "moes" / "incois" / "annual_reports" / "tr_misfiled.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("TECHNICAL REPORT on drifter deployments 2024 — misfiled copy.",
                     encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("moes")
        ingest_cli.run_sources(specs, category_map=cmap)
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        assert saved["metadata"]["document_type"] == "technical_report"  # content wins
        assert saved["metadata"]["org"] == "incois"  # org still from path

    def test_record_id_is_metadata_independent(self, env_data, no_embed):
        """Same content in two places -> same incdoc id -> single corpus record.
        (Ids hash Q|A only; stamping org/source can never duplicate.)"""
        text = "Identical report text placed in two different organizations."
        f1 = data_dir() / "moes" / "incois" / "other" / "same.txt"
        f2 = data_dir() / "moes" / "imd" / "other" / "same.txt"
        for f in (f1, f2):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text, encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("moes")
        res = ingest_cli.run_sources(specs, category_map=cmap)
        assert res["added"] == 1
        lines = [json.loads(l) for l in read_corpus_lines()]
        assert len(lines) == 1
        assert lines[0]["question_id"].startswith("incdoc-")


# ─────────────────────────────────────────────────────────────────────────────
# New unknown source discovery (data/isro with zero config)
# ─────────────────────────────────────────────────────────────────────────────

class TestFutureSourceDiscovery:
    def test_isro_discovered_recursively(self, env_data):
        f = data_dir() / "isro" / "annual_reports" / "isro_ar_2024.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("Launch manifest and annual achievements of the agency.",
                     encoding="utf-8")
        registered, excludes, _ = ingest_cli.load_sources(Path("/nonexistent/none.yaml"))
        found = ingest_cli.discover_sources(registered, excludes)
        assert "isro" in found
        assert found["isro"].hierarchical is True
        assert found["isro"].default_org == "isro"
        assert found["isro"].ministry is None  # unknown — never guessed

    def test_isro_top_level_category_collapses_org_to_default(self, env_data, no_embed):
        """data/isro/annual_reports/x (no org segment) -> org=isro (default_org),
        document_type=annual_report, ministry=None (not EARTH SCIENCES)."""
        f = data_dir() / "isro" / "annual_reports" / "isro_ar_2024.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("Annual review of missions and satellite programmes.",
                     encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("isro")
        res = ingest_cli.run_sources(specs, category_map=cmap)
        assert res["added"] == 1
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        meta = saved["metadata"]
        assert meta["org"] == "isro"
        assert meta["source"] == "isro"
        assert meta["document_type"] == "annual_report"
        assert meta["ministry"] is None  # Earth Sciences NOT stamped on new sources

    def test_isro_nested_org_without_config(self, env_data, no_embed):
        """data/isro/vssc/research_papers -> org=vssc verbatim, no config."""
        f = data_dir() / "isro" / "vssc" / "research_papers" / "prop.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("Study on semi-cryogenic propulsion test results.",
                     encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("isro")
        ingest_cli.run_sources(specs, category_map=cmap)
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        meta = saved["metadata"]
        assert meta["org"] == "vssc"
        assert meta["document_type"] == "research_publication"

    def test_all_includes_discovered_hierarchical(self, env_data, no_embed):
        build_moes_tree(data_dir())
        f = data_dir() / "isro" / "audit_reports" / "a.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("Audit observations for the space programme office.",
                     encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("all")
        assert "isro" in specs and specs["isro"].hierarchical is True
        res = ingest_cli.run_sources(specs, category_map=cmap)
        saved = [json.loads(l)["metadata"]["org"] for l in read_corpus_lines()]
        assert "isro" in saved  # the discovered tree was ingested too


# ─────────────────────────────────────────────────────────────────────────────
# Flat (legacy) compatibility — no behavior / record changes for flat sources
# ─────────────────────────────────────────────────────────────────────────────

class TestFlatCompatibility:
    def test_inbox_records_keep_legacy_shape(self, env_data, no_embed):
        """Inbox has NO meta_context: records stay byte-shaped like the legacy
        default (ministry stamped, no org/source)."""
        inbox = data_dir() / "inbox"
        inbox.mkdir(parents=True)
        (inbox / "memo.txt").write_text("Audit cell circular on evidence retention.",
                                        encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("inbox")
        ingest_cli.run_sources(specs, category_map=cmap)
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        meta = saved["metadata"]
        assert meta["ministry"] == "EARTH SCIENCES"
        assert meta["org"] is None and meta["source"] is None

    def test_flat_incois_stamps_org_but_keeps_ministry(self, env_data, no_embed):
        """Legacy flat source WITH explicit org: new records gain org=incois;
        ministry default unchanged (explicit config value)."""
        folder = data_dir() / "incois_reports" / "TechnicalReports"
        folder.mkdir(parents=True)
        (folder / "tr_buoys.txt").write_text("TECHNICAL REPORT on buoy maintenance.",
                                             encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("incois")
        res = ingest_cli.run_sources(specs, category_map=cmap)
        assert res["added"] == 1
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        meta = saved["metadata"]
        assert meta["org"] == "incois"
        assert meta["source"] == "incois"
        assert meta["ministry"] == "EARTH SCIENCES"
        assert meta["document_type"] == "technical_report"  # legacy folder rule intact

    def test_parliament_merge_stamps_source_provenance(self, env_data, no_embed):
        processed = data_dir() / "processed"
        processed.mkdir(parents=True)
        (processed / "processed_x.jsonl").write_text(
            make_record("18-4-3035").model_dump_json() + "\n", encoding="utf-8")
        specs, _, cmap = ingest_cli.resolve_sources("parliament")
        ingest_cli.run_sources(specs, category_map=cmap)
        (saved,) = [json.loads(l) for l in read_corpus_lines()]
        assert saved["metadata"]["source"] == "parliament"
        assert saved["question_id"] == "18-4-3035"  # ids still preserved

    def test_old_corpus_lines_without_org_parse_fine(self, env_data, no_embed):
        """Pre-Phase-1 corpus lines (no org/source keys) load with defaults."""
        legacy_line = make_record("18-1-55").model_dump_json()
        parsed = QARecord.model_validate_json(legacy_line)
        assert parsed.metadata.org is None and parsed.metadata.source is None


# ─────────────────────────────────────────────────────────────────────────────
# Incremental guarantee inside the tree
# ─────────────────────────────────────────────────────────────────────────────

class TestTreeIncremental:
    def test_new_file_in_tree_appends_and_embeds_only_new(self, env_data, no_embed):
        build_moes_tree(data_dir())
        specs, _, cmap = ingest_cli.resolve_sources("moes")
        r1 = ingest_cli.run_sources(specs, category_map=cmap)
        assert r1["added"] == 6
        n_lines_1 = len(read_corpus_lines())
        no_embed["incremental"] = 0  # isolate the SECOND run's embed behavior
        # tomorrow: one new report lands in the tree
        new = data_dir() / "moes" / "incois" / "annual_reports" / "report_2026.txt"
        new.write_text("New year INCOIS annual activities and achievements.",
                       encoding="utf-8")
        r2 = ingest_cli.run_sources(specs, category_map=cmap)
        assert r2["added"] == 1               # ONLY the new one appended
        assert len(read_corpus_lines()) == n_lines_1 + 1
        assert r2["embed"] == "incremental"   # existing vectors untouched
        assert no_embed["incremental"] == 1 and no_embed["rebuild"] == 0
        new_meta = json.loads(read_corpus_lines()[-1])["metadata"]
        assert new_meta["org"] == "incois" and new_meta["document_type"] == "annual_report"

    def test_rescan_whole_tree_is_noop(self, env_data, no_embed):
        build_moes_tree(data_dir())
        specs, _, cmap = ingest_cli.resolve_sources("moes")
        first = ingest_cli.run_sources(specs, category_map=cmap)
        no_embed["exists"] = True
        second = ingest_cli.run_sources(specs, category_map=cmap)
        assert first["added"] == 6 and second["added"] == 0
        assert second["embed"] == "skip"  # zero index work on a clean rescan


# ─────────────────────────────────────────────────────────────────────────────
# Query-side integration pins (org filter can SEE the stamp)
# ─────────────────────────────────────────────────────────────────────────────

class TestQuerySideIntegration:
    def test_pipeline_org_of_feeds_stamped_org(self):
        """pipeline._org_of must pass metadata.org to derive_org — else stamped
        orgs show in /api/sources but the chat org-filter can't see them.
        Pinned at the exact seam (source-level bind to the filter input)."""
        import inspect
        from src.retrieval.hybrid import pipeline as _pipeline
        src = inspect.getsource(_pipeline.HybridRAGPipeline.retrieve)
        assert '"org": getattr(rec.metadata, "org", None)' in src

    def test_derive_org_explicit_stamp_wins(self):
        from src.retrieval.frontend.org_tree import derive_org
        # exactly the dict shape pipeline._org_of feeds after the fix
        assert derive_org({"org": "incois", "document_type": "annual_report",
                           "subject": "anything", "source_url": "/x"}) == "incois"
        # unknown/new org slug (ISRO) passes through identically
        assert derive_org({"org": "isro", "document_type": "audit_report"}) == "isro"
        # legacy record (no org) still falls back to heuristics
        assert derive_org({"org": None, "document_type": "annual_report",
                           "subject": "INCOIS Annual Report 2024"}) == "incois"

    def test_audit_report_maps_to_audit_category(self):
        from src.retrieval.frontend.org_tree import derive_category
        assert derive_category({"document_type": "audit_report"}) == "audit"

    def test_category_map_comes_from_config(self):
        """category_map is configuration — a yaml override changes detection
        without touching code."""
        _, _, cmap = ingest_cli.load_sources()
        assert cmap["annual_reports"] == "annual_report"
        assert cmap["audit_reports"] == "audit_report"
        assert cmap["research_papers"] == "research_publication"
        assert cmap["other"] == "document"
