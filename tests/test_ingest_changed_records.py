"""Step 5 validation: changed-record detection in ingest.py (kind=records).

Tests:
  _qa_content_hash()
    - stable across corpus roundtrip (same hash before and after model_dump_json → validate)
    - changes when question_text changes
    - changes when answer_text changes
    - changes when metadata changes
    - stable across scraped_at differences (volatile key excluded)
    - stable across question_id changes — wait, question_id IS in the hash

  _seed_seen_with_hash()
    - empty dict when corpus absent
    - correct {id: hash} for each corpus record
    - malformed lines silently skipped
    - hash is consistent with _qa_content_hash(record)

  merge_record_dirs() with seen_hashes
    - new id → added=1, changed=0
    - existing id + identical content → added=0, changed=0
    - existing id + changed content → added=0, changed=1
    - seen_hashes=None → old behaviour (no change detection, returns (added,0))
    - multiple changed records counted correctly
    - mixed: some new, some unchanged, some changed

  ingest_source() for kind=records
    - no corpus → all records are new, changed=0
    - all records unchanged → added=0, changed=0, no corpus write
    - one record changed → added=0, changed=1
    - one record new + one changed → added=1, changed=1
    - returns {"added": N, "changed": M, "folders": 0}
    - folder-kind path is completely unaffected (changed=0 in return)

  choose_embed_action() with total_changed
    - added=0, changed=0 → "skip"
    - added=5, changed=0, index exists → "incremental"
    - added=0, changed=1, index exists → "rebuild"   (changed forces rebuild)
    - added=3, changed=2, index exists → "rebuild"
    - added=0, changed=1, no_rebuild=True → "defer"  (--no-rebuild takes precedence)
    - added=0, changed=0, full_rebuild=True → "skip"  (nothing to rebuild)
    - added=1, changed=0, no index → "rebuild"        (first build)
    - existing behaviour preserved for total_changed=0 (default)

  run_sources()
    - changed flows through to embed action
    - result dict contains "changed" key

No network, no real crawlers, no ML. All I/O to tmp_path fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import src.scripts.ingest as ingest_cli
from src.models.qa_record import QARecord
from src.scripts.ingest import (
    SourceSpec,
    _qa_content_hash,
    _seed_seen_with_hash,
    choose_embed_action,
    ingest_source,
    merge_record_dirs,
    run_sources,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_qa(
    qid: str = "rs-1-0001",
    question: str = "What is the ocean depth policy?",
    answer: str = "The government monitors ocean depth regularly.",
    ministry: str = "earth-sciences",
    session: int = 271,
) -> QARecord:
    return QARecord(
        question_id=qid,
        question_text=question,
        answer_text=answer,
        metadata=ingest_cli.QARecord.__fields__["metadata"].default_factory() if False
        else __import__("src.models.qa_record", fromlist=["QARecordMetadata"])
               .QARecordMetadata(ministry=ministry, session=session),
    )


def _write_staging_jsonl(path: Path, records: list[QARecord]) -> None:
    """Write records to a staging qa.jsonl (raw dict form, as crawlers produce)."""
    lines = []
    for rec in records:
        # Staging files are raw dicts with the crawler's format
        d = {
            "question_id": rec.question_id,
            "question_text": rec.question_text,
            "answer_text": rec.answer_text,
            "metadata": json.loads(rec.metadata.model_dump_json()),
            "scraped_at": "2026-08-24T06:00:00Z",
        }
        lines.append(json.dumps(d))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_corpus(path: Path, records: list[QARecord]) -> None:
    """Write records to a corpus file (QARecord.model_dump_json() form)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(r.model_dump_json() for r in records) + "\n",
        encoding="utf-8",
    )


# ── _qa_content_hash ──────────────────────────────────────────────────────────

class TestQaContentHash:

    def test_stable_across_corpus_roundtrip(self):
        """Hash must be identical before and after model_dump_json → validate."""
        rec = _make_qa()
        h1 = _qa_content_hash(rec)
        roundtrip = QARecord.model_validate_json(rec.model_dump_json())
        h2 = _qa_content_hash(roundtrip)
        assert h1 == h2, "Hash must be stable across corpus serialization roundtrip"

    def test_changes_on_question_text_change(self):
        r1 = _make_qa(question="Original question text?")
        r2 = _make_qa(question="Updated question text?")
        assert _qa_content_hash(r1) != _qa_content_hash(r2)

    def test_changes_on_answer_text_change(self):
        r1 = _make_qa(answer="Original answer.")
        r2 = _make_qa(answer="Updated answer with new information.")
        assert _qa_content_hash(r1) != _qa_content_hash(r2)

    def test_changes_on_metadata_change(self):
        r1 = _make_qa(ministry="earth-sciences")
        r2 = _make_qa(ministry="ocean-development")
        assert _qa_content_hash(r1) != _qa_content_hash(r2)

    def test_stable_across_different_scraped_at(self):
        """scraped_at is volatile — two records differing only in scraped_at must hash identically."""
        from src.models.qa_record import QARecordMetadata
        from datetime import datetime, timezone

        r1 = QARecord(
            question_id="rs-1-0001",
            question_text="What is the policy?",
            answer_text="The policy is good.",
            scraped_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        r2 = QARecord(
            question_id="rs-1-0001",
            question_text="What is the policy?",
            answer_text="The policy is good.",
            scraped_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
        )
        assert _qa_content_hash(r1) == _qa_content_hash(r2)

    def test_identical_records_have_identical_hash(self):
        r1 = _make_qa()
        r2 = _make_qa()
        assert _qa_content_hash(r1) == _qa_content_hash(r2)

    def test_returns_hex_string(self):
        h = _qa_content_hash(_make_qa())
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex = 64 chars
        int(h, 16)  # raises if not hex


# ── _seed_seen_with_hash ──────────────────────────────────────────────────────

class TestSeedSeenWithHash:

    def test_empty_when_corpus_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        result = _seed_seen_with_hash()
        assert result == {}

    def test_returns_id_to_hash_mapping(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        rec = _make_qa(qid="rs-1-0001")
        _write_corpus(tmp_path / "corpus_reports.jsonl", [rec])
        result = _seed_seen_with_hash()
        assert "rs-1-0001" in result
        assert result["rs-1-0001"] == _qa_content_hash(rec)

    def test_hash_consistent_with_qa_content_hash(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        rec = _make_qa(qid="rs-1-0002")
        _write_corpus(tmp_path / "corpus_reports.jsonl", [rec])
        result = _seed_seen_with_hash()
        # The stored hash must equal what _qa_content_hash computes directly
        expected = _qa_content_hash(rec)
        assert result["rs-1-0002"] == expected

    def test_multiple_records(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        recs = [_make_qa(qid=f"rs-1-{i:04d}") for i in range(5)]
        _write_corpus(tmp_path / "corpus_reports.jsonl", recs)
        result = _seed_seen_with_hash()
        assert len(result) == 5
        for rec in recs:
            assert rec.question_id in result
            assert result[rec.question_id] == _qa_content_hash(rec)

    def test_malformed_lines_silently_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        corpus = tmp_path / "corpus_reports.jsonl"
        good = _make_qa(qid="rs-1-0001")
        corpus.write_text(
            "not valid json\n" + good.model_dump_json() + "\n",
            encoding="utf-8",
        )
        result = _seed_seen_with_hash()
        assert "rs-1-0001" in result
        assert len(result) == 1  # bad line skipped, good line present


# ── merge_record_dirs with seen_hashes ───────────────────────────────────────

class TestMergeRecordDirsWithHashes:

    def _staging_dir(self, tmp_path: Path, records: list[QARecord]) -> Path:
        d = tmp_path / "staging"
        d.mkdir()
        _write_staging_jsonl(d / "qa.jsonl", records)
        return d

    def _run_merge(self, staging_dir: Path, seen: set, seen_hashes: dict | None,
                   data_root: Path) -> tuple[int, int]:
        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: data_root / rel):
            out: list[QARecord] = []
            added, changed = merge_record_dirs(
                ["staging"], out, seen,
                seen_hashes=seen_hashes,
            )
        return added, changed

    def test_new_id_counts_as_added(self, tmp_path):
        rec = _make_qa(qid="rs-1-0001")
        self._staging_dir(tmp_path, [rec])
        seen: set = set()
        hashes: dict = {}
        added, changed = self._run_merge(tmp_path / "staging", seen, hashes, tmp_path)
        assert added == 1
        assert changed == 0

    def test_unchanged_id_skipped(self, tmp_path):
        rec = _make_qa(qid="rs-1-0001")
        self._staging_dir(tmp_path, [rec])
        # Seed seen with the same record — it's already in corpus
        corpus_hash = _qa_content_hash(rec)
        seen = {"rs-1-0001"}
        hashes = {"rs-1-0001": corpus_hash}
        added, changed = self._run_merge(tmp_path / "staging", seen, hashes, tmp_path)
        assert added == 0
        assert changed == 0

    def test_changed_id_counts_as_changed(self, tmp_path):
        old_rec = _make_qa(qid="rs-1-0001", answer="Original answer.")
        new_rec = _make_qa(qid="rs-1-0001", answer="Updated answer — content changed.")
        self._staging_dir(tmp_path, [new_rec])  # staging has NEW content
        # Corpus has OLD content
        old_hash = _qa_content_hash(old_rec)
        seen = {"rs-1-0001"}
        hashes = {"rs-1-0001": old_hash}
        added, changed = self._run_merge(tmp_path / "staging", seen, hashes, tmp_path)
        assert added == 0
        assert changed == 1

    def test_seen_hashes_none_returns_zero_changed(self, tmp_path):
        """seen_hashes=None → old id-only behaviour → always (added, 0)."""
        rec = _make_qa(qid="rs-1-0001")
        self._staging_dir(tmp_path, [rec])
        seen = {"rs-1-0001"}  # already seen
        added, changed = self._run_merge(tmp_path / "staging", seen, None, tmp_path)
        assert added == 0
        assert changed == 0  # no change detection without seen_hashes

    def test_multiple_changed_records(self, tmp_path):
        old_recs = [_make_qa(qid=f"rs-1-{i:04d}", answer="Old answer.") for i in range(3)]
        new_recs = [_make_qa(qid=f"rs-1-{i:04d}", answer="New answer.") for i in range(3)]
        self._staging_dir(tmp_path, new_recs)
        seen = {r.question_id for r in old_recs}
        hashes = {r.question_id: _qa_content_hash(r) for r in old_recs}
        added, changed = self._run_merge(tmp_path / "staging", seen, hashes, tmp_path)
        assert added == 0
        assert changed == 3

    def test_mixed_new_unchanged_changed(self, tmp_path):
        old_unchanged = _make_qa(qid="rs-1-0001", answer="Same answer.")
        old_changed = _make_qa(qid="rs-1-0002", answer="Old answer.")
        new_incoming = _make_qa(qid="rs-1-0003", answer="Brand new record.")
        new_changed_version = _make_qa(qid="rs-1-0002", answer="UPDATED answer.")

        self._staging_dir(tmp_path, [old_unchanged, new_changed_version, new_incoming])
        seen = {"rs-1-0001", "rs-1-0002"}
        hashes = {
            "rs-1-0001": _qa_content_hash(old_unchanged),
            "rs-1-0002": _qa_content_hash(old_changed),
        }
        added, changed = self._run_merge(tmp_path / "staging", seen, hashes, tmp_path)
        assert added == 1    # rs-1-0003 is new
        assert changed == 1  # rs-1-0002 content changed


# ── ingest_source for kind=records ────────────────────────────────────────────

class TestIngestSourceRecordsKind:

    def _make_spec(self, staging_rel: str) -> SourceSpec:
        return SourceSpec(
            name="test_source",
            kind="records",
            record_dirs=[staging_rel],
            recursive=False,
        )

    def test_no_corpus_all_new(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        staging = tmp_path / "test_staging"
        staging.mkdir()
        recs = [_make_qa(qid=f"rs-1-{i:04d}") for i in range(3)]
        _write_staging_jsonl(staging / "qa.jsonl", recs)

        spec = self._make_spec("test_staging")
        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: tmp_path / rel):
            result = ingest_source(spec)

        assert result["added"] == 3
        assert result["changed"] == 0
        assert result["folders"] == 0

    def test_all_unchanged_nothing_written(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        staging = tmp_path / "test_staging"
        staging.mkdir()
        recs = [_make_qa(qid=f"rs-1-{i:04d}") for i in range(2)]
        _write_staging_jsonl(staging / "qa.jsonl", recs)
        # Pre-populate corpus with the same records
        _write_corpus(tmp_path / "corpus_reports.jsonl", recs)

        spec = self._make_spec("test_staging")
        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: tmp_path / rel):
            result = ingest_source(spec)

        assert result["added"] == 0
        assert result["changed"] == 0

    def test_one_record_changed_detected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        staging = tmp_path / "test_staging"
        staging.mkdir()

        old_rec = _make_qa(qid="rs-1-0001", answer="Original answer.")
        new_rec = _make_qa(qid="rs-1-0001", answer="Corrected upstream answer.")

        # Corpus has old version
        _write_corpus(tmp_path / "corpus_reports.jsonl", [old_rec])
        # Staging has new version
        _write_staging_jsonl(staging / "qa.jsonl", [new_rec])

        spec = self._make_spec("test_staging")
        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: tmp_path / rel):
            result = ingest_source(spec)

        assert result["added"] == 0
        assert result["changed"] == 1

    def test_one_new_one_changed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        staging = tmp_path / "test_staging"
        staging.mkdir()

        old_rec = _make_qa(qid="rs-1-0001", answer="Original answer text here.")
        new_content = _make_qa(qid="rs-1-0001", answer="Updated answer with new information.")
        brand_new = _make_qa(qid="rs-1-0002", answer="Never seen before in corpus.")

        _write_corpus(tmp_path / "corpus_reports.jsonl", [old_rec])
        _write_staging_jsonl(staging / "qa.jsonl", [new_content, brand_new])

        spec = self._make_spec("test_staging")
        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: tmp_path / rel):
            result = ingest_source(spec)

        assert result["added"] == 1
        assert result["changed"] == 1

    def test_result_dict_has_changed_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        staging = tmp_path / "test_staging"
        staging.mkdir()
        _write_staging_jsonl(staging / "qa.jsonl", [])

        spec = self._make_spec("test_staging")
        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: tmp_path / rel):
            result = ingest_source(spec)

        assert "changed" in result, "'changed' key must always be present in records-kind result"

    def test_folder_kind_unchanged(self, tmp_path, monkeypatch):
        """folder-kind ingest_source returns 'changed': 0 when no files changed."""
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        spec = SourceSpec(
            name="test_folders",
            kind="folders",
            folders=["nonexistent_folder"],
        )
        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: tmp_path / rel), \
             patch("src.scripts.ingest.expand_source", return_value=[]), \
             patch("src.scripts.ingest._moes_website_dedup_excludes", return_value=set()), \
             patch("src.scripts.ingest._seed_seen_hashes_by_url", return_value={}):
            result = ingest_source(spec)

        # Step 7: folder-kind now tracks changed records; zero when nothing changed
        assert result.get("changed", 0) == 0


# ── choose_embed_action with total_changed ────────────────────────────────────

class TestChooseEmbedActionWithChanged:

    def _action(self, added=0, changed=0, no_rebuild=False,
                full_rebuild=False, index_exists=True):
        with patch.object(ingest_cli._engine, "_index_exists",
                          return_value=index_exists):
            return choose_embed_action(
                total_added=added,
                no_rebuild=no_rebuild,
                full_rebuild=full_rebuild,
                total_changed=changed,
            )

    def test_skip_when_nothing_added_and_nothing_changed(self):
        assert self._action(added=0, changed=0) == "skip"

    def test_incremental_when_added_no_changed_index_exists(self):
        assert self._action(added=5, changed=0, index_exists=True) == "incremental"

    def test_rebuild_when_changed_even_if_nothing_added(self):
        """changed > 0 must force full rebuild even with added=0."""
        assert self._action(added=0, changed=1, index_exists=True) == "rebuild"

    def test_rebuild_when_added_and_changed(self):
        assert self._action(added=3, changed=2, index_exists=True) == "rebuild"

    def test_defer_when_no_rebuild_flag_overrides_changed(self):
        """--no-rebuild takes precedence: operator explicitly defers index work."""
        assert self._action(added=0, changed=1, no_rebuild=True) == "defer"

    def test_defer_when_no_rebuild_flag_overrides_added(self):
        assert self._action(added=5, changed=0, no_rebuild=True) == "defer"

    def test_rebuild_when_full_rebuild_requested_even_if_nothing_added(self):
        """--full-rebuild with nothing added/changed → still 'rebuild' (explicit operator intent)."""
        assert self._action(added=0, changed=0, full_rebuild=True) == "rebuild"

    def test_rebuild_on_first_build_no_index(self):
        assert self._action(added=1, changed=0, index_exists=False) == "rebuild"

    def test_rebuild_when_full_rebuild_explicit(self):
        assert self._action(added=1, changed=0, full_rebuild=True, index_exists=True) == "rebuild"

    # ── backward compatibility: total_changed defaults to 0 ─────────────────

    def test_default_changed_zero_unchanged_behaviour_skip(self):
        """Old callers omitting total_changed must still get 'skip' when nothing added."""
        with patch.object(ingest_cli._engine, "_index_exists", return_value=True):
            assert choose_embed_action(0, no_rebuild=False, full_rebuild=False) == "skip"

    def test_default_changed_zero_unchanged_behaviour_incremental(self):
        with patch.object(ingest_cli._engine, "_index_exists", return_value=True):
            assert choose_embed_action(5, no_rebuild=False, full_rebuild=False) == "incremental"

    def test_default_changed_zero_unchanged_behaviour_defer(self):
        assert choose_embed_action(5, no_rebuild=True, full_rebuild=False) == "defer"

    # ── existing test_ingest_cli.py contract preserved ───────────────────────

    def test_full_rebuild_zero_records_returns_rebuild(self):
        """Regression (bug fix): choose_embed_action(0, False, True, 0) must return 'rebuild'.
        --full-rebuild is explicit operator intent and must not be short-circuited
        by the 'nothing added' skip guard."""
        with patch.object(ingest_cli._engine, "_index_exists", return_value=True):
            assert choose_embed_action(0, no_rebuild=False, full_rebuild=True, total_changed=0) == "rebuild"

    def test_existing_contract_defer(self):
        assert choose_embed_action(3, no_rebuild=True, full_rebuild=False) == "defer"


# ── run_sources changed accumulation ─────────────────────────────────────────

class TestRunSourcesChangedAccumulation:

    def test_changed_flows_into_result_dict(self, tmp_path, monkeypatch):
        """run_sources must include 'changed' in its return dict."""
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))

        staging = tmp_path / "test_staging"
        staging.mkdir()
        old_rec = _make_qa(qid="rs-1-0001", answer="Original answer text here.")
        new_rec = _make_qa(qid="rs-1-0001", answer="Completely rewritten answer text.")
        _write_corpus(tmp_path / "corpus_reports.jsonl", [old_rec])
        _write_staging_jsonl(staging / "qa.jsonl", [new_rec])

        spec = SourceSpec(
            name="test_source", kind="records",
            record_dirs=["test_staging"], recursive=False,
        )

        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: tmp_path / rel), \
             patch.object(ingest_cli._engine, "_index_exists", return_value=True), \
             patch.object(ingest_cli._engine, "rebuild_index"), \
             patch.object(ingest_cli._engine, "incremental_update"):
            result = run_sources({"test_source": spec})

        assert "changed" in result
        assert result["changed"] == 1

    def test_changed_triggers_rebuild_not_incremental(self, tmp_path, monkeypatch):
        """When changed > 0, run_sources must call rebuild_index, not incremental_update."""
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))

        staging = tmp_path / "test_staging"
        staging.mkdir()
        old_rec = _make_qa(qid="rs-1-0001", answer="Old answer.")
        new_rec = _make_qa(qid="rs-1-0001", answer="Completely different answer.")
        _write_corpus(tmp_path / "corpus_reports.jsonl", [old_rec])
        _write_staging_jsonl(staging / "qa.jsonl", [new_rec])

        spec = SourceSpec(
            name="test_source", kind="records",
            record_dirs=["test_staging"], recursive=False,
        )

        rebuild_called = []
        incremental_called = []

        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: tmp_path / rel), \
             patch.object(ingest_cli._engine, "_index_exists", return_value=True), \
             patch.object(ingest_cli._engine, "rebuild_index",
                          side_effect=lambda: rebuild_called.append(1)), \
             patch.object(ingest_cli._engine, "incremental_update",
                          side_effect=lambda: incremental_called.append(1)):
            run_sources({"test_source": spec})

        assert len(rebuild_called) == 1, "rebuild_index must be called when records changed"
        assert len(incremental_called) == 0, "incremental_update must NOT be called"

    def test_no_changed_uses_incremental(self, tmp_path, monkeypatch):
        """All new records (no changed) → incremental_update, not rebuild."""
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))

        staging = tmp_path / "test_staging"
        staging.mkdir()
        recs = [_make_qa(qid=f"rs-1-{i:04d}") for i in range(2)]
        _write_staging_jsonl(staging / "qa.jsonl", recs)
        # No pre-existing corpus → all records are new

        spec = SourceSpec(
            name="test_source", kind="records",
            record_dirs=["test_staging"], recursive=False,
        )

        rebuild_called = []
        incremental_called = []

        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: tmp_path / rel), \
             patch.object(ingest_cli._engine, "_index_exists", return_value=True), \
             patch.object(ingest_cli._engine, "rebuild_index",
                          side_effect=lambda: rebuild_called.append(1)), \
             patch.object(ingest_cli._engine, "incremental_update",
                          side_effect=lambda: incremental_called.append(1)):
            result = run_sources({"test_source": spec})

        assert result["added"] == 2
        assert result["changed"] == 0
        assert len(incremental_called) == 1
        assert len(rebuild_called) == 0

    def test_no_rebuild_flag_defers_even_on_changed(self, tmp_path, monkeypatch):
        """--no-rebuild must defer index work even when changed > 0."""
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))

        staging = tmp_path / "test_staging"
        staging.mkdir()
        old_rec = _make_qa(qid="rs-1-0001", answer="Original answer text here.")
        new_rec = _make_qa(qid="rs-1-0001", answer="Rewritten answer with new facts.")
        _write_corpus(tmp_path / "corpus_reports.jsonl", [old_rec])
        _write_staging_jsonl(staging / "qa.jsonl", [new_rec])

        spec = SourceSpec(
            name="test_source", kind="records",
            record_dirs=["test_staging"], recursive=False,
        )

        with patch("src.scripts.ingest._data_path",
                   side_effect=lambda rel: tmp_path / rel), \
             patch.object(ingest_cli._engine, "_index_exists", return_value=True), \
             patch.object(ingest_cli._engine, "rebuild_index") as mock_rebuild, \
             patch.object(ingest_cli._engine, "incremental_update") as mock_incr:
            result = run_sources({"test_source": spec}, no_rebuild=True)

        assert result["embed"] == "defer"
        mock_rebuild.assert_not_called()
        mock_incr.assert_not_called()


# ── Step 7: changed-record detection for kind=folders sources ─────────────────
# These tests verify the complete pipeline:
#   source file changed → ingest_folder detects changed → ingest_source
#   propagates changed count → choose_embed_action triggers full rebuild.

class TestFolderSourceChangedDetection:
    """ingest_folder + ingest_source: changed-record detection for folder sources.

    The key difference from kind=records: question_id is content-derived
    (_hash_id(q+"|"+a)), so a changed file produces a new question_id.
    Detection must therefore key on source_url (= str(path)), not question_id.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_corpus_record(self, source_url: str, answer: str = "Answer text long enough.") -> QARecord:
        from src.scripts.convert_sirs_knowledge import _make_record
        rec = _make_record(
            f"Document: {Path(source_url).stem}",
            answer,
            subject=Path(source_url).stem,
            source_url=source_url,
            document_type="document",
        )
        assert rec is not None
        return rec

    def _write_corpus(self, path: Path, records: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")

    def _write_pdf(self, path: Path, content: str) -> None:
        """Write a minimal PDF-like bytes file for mocking the converter."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 stub")

    # ── _seed_seen_hashes_by_url ──────────────────────────────────────────────

    def test_seed_by_url_empty_when_no_corpus(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        from src.scripts.ingest import _seed_seen_hashes_by_url
        with patch("src.scripts.ingest.corpus_path",
                   return_value=tmp_path / "corpus_reports.jsonl"):
            result = _seed_seen_hashes_by_url()
        assert result == {}

    def test_seed_by_url_indexes_source_url(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        corpus = tmp_path / "corpus_reports.jsonl"
        rec = self._make_corpus_record("/data/annual_reports/AR_2024.pdf")
        self._write_corpus(corpus, [rec])
        from src.scripts.ingest import _seed_seen_hashes_by_url
        with patch("src.scripts.ingest.corpus_path", return_value=corpus):
            result = _seed_seen_hashes_by_url()
        assert "/data/annual_reports/AR_2024.pdf" in result
        assert isinstance(result["/data/annual_reports/AR_2024.pdf"], str)

    def test_seed_by_url_skips_records_without_source_url(self, tmp_path, monkeypatch):
        """RS/LS QA records have no source_url — must not appear in the map."""
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        corpus = tmp_path / "corpus_reports.jsonl"
        # RS-style record: no source_url on metadata
        rec = _make_qa(qid="rs-1-0001", answer="Parliamentary answer text here.")
        self._write_corpus(corpus, [rec])
        from src.scripts.ingest import _seed_seen_hashes_by_url
        with patch("src.scripts.ingest.corpus_path", return_value=corpus):
            result = _seed_seen_hashes_by_url()
        # source_url is None on RS records → must not appear
        assert result == {}

    def test_seed_by_url_multiple_records(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
        corpus = tmp_path / "corpus_reports.jsonl"
        recs = [
            self._make_corpus_record("/data/doc1.pdf", "Answer one is here."),
            self._make_corpus_record("/data/doc2.pdf", "Answer two is here."),
        ]
        self._write_corpus(corpus, recs)
        from src.scripts.ingest import _seed_seen_hashes_by_url
        with patch("src.scripts.ingest.corpus_path", return_value=corpus):
            result = _seed_seen_hashes_by_url()
        assert len(result) == 2
        assert "/data/doc1.pdf" in result
        assert "/data/doc2.pdf" in result

    # ── ingest_folder: seen_hashes detection ────────────────────────────────

    def _run_ingest_folder(self, tmp_path: Path, folder: Path,
                           corpus_records: list,
                           converted_records: list,
                           seen_hashes: dict | None = None) -> dict:
        """Helper: run ingest_folder with mocked converter and corpus."""
        import json as _json
        corpus_path = tmp_path / "corpus_reports.jsonl"
        self._write_corpus(corpus_path, corpus_records)

        convert_calls = iter(converted_records)

        def fake_convert(path, out, seen, move_after, meta_context=None):
            try:
                recs = next(convert_calls)
            except StopIteration:
                return 0
            n = 0
            for r in recs:
                if r.question_id not in seen:
                    seen.add(r.question_id)
                    out.append(r)
                    n += 1
            return n

        with patch("src.scripts.ingest_folder.CORPUS", corpus_path), \
             patch("src.scripts.ingest_folder.convert_one_detected",
                   side_effect=fake_convert):
            from src.scripts.ingest_folder import ingest_folder
            return ingest_folder(str(folder), seen_hashes=seen_hashes)

    def test_ingest_folder_unchanged_file_zero_changed(self, tmp_path):
        """File in corpus, same content → added=0, changed=0."""
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "doc.pdf").write_bytes(b"%PDF stub")

        rec = self._make_corpus_record(str(folder / "doc.pdf"), "Same answer text here.")
        from src.scripts.ingest import _qa_content_hash
        seen_hashes = {str(folder / "doc.pdf"): _qa_content_hash(rec)}

        result = self._run_ingest_folder(
            tmp_path, folder,
            corpus_records=[rec],
            converted_records=[[rec]],  # converter produces same record
            seen_hashes=seen_hashes,
        )
        assert result["added"] == 0, "unchanged file must not be added"
        assert result["changed"] == 0, "unchanged file must not count as changed"

    def test_ingest_folder_new_file_added_not_changed(self, tmp_path):
        """New file (source_url not in seen_hashes) → added=1, changed=0."""
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "new.pdf").write_bytes(b"%PDF stub")

        new_rec = self._make_corpus_record(str(folder / "new.pdf"), "Brand new answer text.")
        seen_hashes: dict[str, str] = {}  # nothing known yet

        result = self._run_ingest_folder(
            tmp_path, folder,
            corpus_records=[],
            converted_records=[[new_rec]],
            seen_hashes=seen_hashes,
        )
        assert result["added"] == 1, "new file must be counted as added"
        assert result["changed"] == 0, "new file must not be counted as changed"

    def test_ingest_folder_changed_file_detected(self, tmp_path):
        """File already in corpus, content changed → added=1 (new id), changed=1."""
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "doc.pdf").write_bytes(b"%PDF stub")

        old_rec = self._make_corpus_record(str(folder / "doc.pdf"), "Original answer text here.")
        new_rec = self._make_corpus_record(str(folder / "doc.pdf"), "Completely rewritten answer.")

        from src.scripts.ingest import _qa_content_hash
        seen_hashes = {str(folder / "doc.pdf"): _qa_content_hash(old_rec)}

        result = self._run_ingest_folder(
            tmp_path, folder,
            corpus_records=[old_rec],
            converted_records=[[new_rec]],
            seen_hashes=seen_hashes,
        )
        # new_rec has a different question_id (content-derived) → added to out
        assert result["added"] == 1
        assert result["changed"] == 1, "changed file must be counted as changed"

    def test_ingest_folder_no_seen_hashes_zero_changed(self, tmp_path):
        """Legacy callers that omit seen_hashes get changed=0 always."""
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "doc.pdf").write_bytes(b"%PDF stub")

        old_rec = self._make_corpus_record(str(folder / "doc.pdf"), "Original answer text here.")
        new_rec = self._make_corpus_record(str(folder / "doc.pdf"), "Completely rewritten answer.")

        result = self._run_ingest_folder(
            tmp_path, folder,
            corpus_records=[old_rec],
            converted_records=[[new_rec]],
            seen_hashes=None,  # legacy caller
        )
        assert result["changed"] == 0, "legacy callers must always get changed=0"

    def test_ingest_folder_changed_key_always_present(self, tmp_path):
        """Return dict must always have 'changed' key regardless of seen_hashes."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_ingest_folder(
            tmp_path, folder,
            corpus_records=[], converted_records=[], seen_hashes=None,
        )
        assert "changed" in result

    # ── ingest_source kind=folders end-to-end ───────────────────────────────

    def _run_ingest_source_folders(
        self, tmp_path: Path, folder: Path,
        ingest_folder_result: dict,
        seen_hashes_by_url: dict | None = None,
    ) -> dict:
        """Run ingest_source for a folders spec with mocked engine."""
        from src.scripts.ingest import SourceSpec, ingest_source, LeafJob

        spec = SourceSpec(name="incois", kind="folders", folders=["annual_reports"])
        job = LeafJob(folder=folder, org="incois", doc_type_hint=None,
                      meta_context={}, exclude_files=())

        with patch("src.scripts.ingest.expand_source", return_value=[job]), \
             patch("src.scripts.ingest._moes_website_dedup_excludes", return_value=set()), \
             patch("src.scripts.ingest._seed_seen_hashes_by_url",
                   return_value=seen_hashes_by_url or {}), \
             patch("src.scripts.ingest._engine.ingest_folder",
                   return_value=ingest_folder_result), \
             patch("src.scripts.ingest._engine.log"):
            return ingest_source(spec)

    def test_ingest_source_unchanged_result(self, tmp_path):
        """All files unchanged → added=0, changed=0."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_ingest_source_folders(
            tmp_path, folder,
            ingest_folder_result={"added": 0, "files": 1, "failed": 0, "changed": 0},
        )
        assert result["added"] == 0
        assert result["changed"] == 0

    def test_ingest_source_new_file_result(self, tmp_path):
        """New file ingested → added=1, changed=0."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_ingest_source_folders(
            tmp_path, folder,
            ingest_folder_result={"added": 1, "files": 1, "failed": 0, "changed": 0},
        )
        assert result["added"] == 1
        assert result["changed"] == 0

    def test_ingest_source_changed_file_propagated(self, tmp_path):
        """Changed file → changed=1 propagated from engine to ingest_source result."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_ingest_source_folders(
            tmp_path, folder,
            ingest_folder_result={"added": 1, "files": 1, "failed": 0, "changed": 1},
        )
        assert result["changed"] == 1

    def test_ingest_source_changed_plus_new(self, tmp_path):
        """1 changed + 1 new → changed=1, added=2."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_ingest_source_folders(
            tmp_path, folder,
            ingest_folder_result={"added": 2, "files": 2, "failed": 0, "changed": 1},
        )
        assert result["added"] == 2
        assert result["changed"] == 1

    def test_ingest_source_result_has_changed_key(self, tmp_path):
        """folders ingest_source result dict must always have 'changed' key."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_ingest_source_folders(
            tmp_path, folder,
            ingest_folder_result={"added": 0, "files": 0, "failed": 0, "changed": 0},
        )
        assert "changed" in result

    # ── full pipeline: run_sources → choose_embed_action ────────────────────

    def _run_full_pipeline(self, tmp_path: Path, folder: Path,
                           ingest_folder_result: dict,
                           no_rebuild: bool = False,
                           full_rebuild: bool = False) -> dict:
        from src.scripts.ingest import SourceSpec, run_sources, LeafJob

        spec = SourceSpec(name="incois", kind="folders", folders=["annual_reports"])
        job = LeafJob(folder=folder, org="incois", doc_type_hint=None,
                      meta_context={}, exclude_files=())

        with patch("src.scripts.ingest.expand_source", return_value=[job]), \
             patch("src.scripts.ingest._moes_website_dedup_excludes", return_value=set()), \
             patch("src.scripts.ingest._seed_seen_hashes_by_url", return_value={}), \
             patch("src.scripts.ingest._engine.ingest_folder",
                   return_value=ingest_folder_result), \
             patch("src.scripts.ingest._engine.log"), \
             patch("src.scripts.ingest._engine._index_exists", return_value=True), \
             patch("src.scripts.ingest._engine.rebuild_index") as mock_rebuild, \
             patch("src.scripts.ingest._engine.incremental_update") as mock_incr:
            result = run_sources(
                {"incois": spec},
                no_rebuild=no_rebuild,
                full_rebuild=full_rebuild,
            )
        result["_mock_rebuild"] = mock_rebuild
        result["_mock_incr"] = mock_incr
        return result

    def test_pipeline_unchanged_skips_index(self, tmp_path):
        """No changes → embed action is 'skip'."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_full_pipeline(
            tmp_path, folder,
            ingest_folder_result={"added": 0, "files": 1, "failed": 0, "changed": 0},
        )
        assert result["embed"] == "skip"
        result["_mock_rebuild"].assert_not_called()
        result["_mock_incr"].assert_not_called()

    def test_pipeline_new_file_incremental(self, tmp_path):
        """New file only → embed action is 'incremental'."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_full_pipeline(
            tmp_path, folder,
            ingest_folder_result={"added": 1, "files": 1, "failed": 0, "changed": 0},
        )
        assert result["embed"] == "incremental"
        result["_mock_rebuild"].assert_not_called()
        result["_mock_incr"].assert_called_once()

    def test_pipeline_changed_file_triggers_full_rebuild(self, tmp_path):
        """Changed file → embed action is 'rebuild' (full re-embed required)."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_full_pipeline(
            tmp_path, folder,
            ingest_folder_result={"added": 1, "files": 1, "failed": 0, "changed": 1},
        )
        assert result["embed"] == "rebuild"
        result["_mock_rebuild"].assert_called_once()
        result["_mock_incr"].assert_not_called()

    def test_pipeline_changed_plus_new_triggers_rebuild(self, tmp_path):
        """Changed + new → embed action is 'rebuild'."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_full_pipeline(
            tmp_path, folder,
            ingest_folder_result={"added": 2, "files": 2, "failed": 0, "changed": 1},
        )
        assert result["embed"] == "rebuild"
        result["_mock_rebuild"].assert_called_once()

    def test_pipeline_changed_no_rebuild_flag_defers(self, tmp_path):
        """Changed file + --no-rebuild → embed action is 'defer'."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_full_pipeline(
            tmp_path, folder,
            ingest_folder_result={"added": 1, "files": 1, "failed": 0, "changed": 1},
            no_rebuild=True,
        )
        assert result["embed"] == "defer"
        result["_mock_rebuild"].assert_not_called()
        result["_mock_incr"].assert_not_called()

    def test_pipeline_explicit_full_rebuild_zero_changes(self, tmp_path):
        """--full-rebuild with zero changes/adds → embed action is 'rebuild'."""
        folder = tmp_path / "docs"
        folder.mkdir()
        result = self._run_full_pipeline(
            tmp_path, folder,
            ingest_folder_result={"added": 0, "files": 1, "failed": 0, "changed": 0},
            full_rebuild=True,
        )
        assert result["embed"] == "rebuild"
        result["_mock_rebuild"].assert_called_once()
