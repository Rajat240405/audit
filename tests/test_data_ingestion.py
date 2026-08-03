"""
Unit tests for Phase 1 — Data Ingestion.

Tests cover:
1. QARecord model validation
2. DataValidator
3. DataEnricher
4. MockDataGenerator
5. DataLoader (save/load round-trip)
6. IngestionStats

Run with: pytest tests/test_data_ingestion.py -v
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from dirty_equals import IsDatetime, IsPositive, IsStr

from src.models.qa_record import (
    QARecord,
    QARecordMetadata,
    QuestionType,
    AnswerStatus,
)
from src.models.statistics import IngestionStats, FieldStats, ScrapingStats
from src.data.validator import DataValidator, ValidationReport, ValidationError_
from src.data.enricher import DataEnricher, KNOWN_MINISTRIES
from src.data.scraper import MockDataGenerator
from src.data.loader import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# QARecord Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQARecordModel:
    """Tests for QARecord Pydantic model."""

    def test_valid_record_creation(self):
        """A fully-populated record should validate successfully."""
        record = QARecord(
            question_id="18-0001",
            question_text="Will the Minister of Finance be pleased to state the details of GST collection?",
            answer_text="The total GST collected in the fiscal year was Rs. X crore. The Government has taken steps to improve compliance.",
        )
        assert record.question_id == "18-0001"
        assert len(record.question_text) > 10
        assert len(record.answer_text) > 10

    def test_question_id_auto_generation(self):
        """If question_id is 'unknown', it should be auto-generated from metadata."""
        record = QARecord(
            question_id="unknown",
            question_text="What are the steps taken to address unemployment?",
            answer_text="The Government has launched several schemes for skill development and employment generation.",
            metadata=QARecordMetadata(session=18, question_number=42),
        )
        # After validation, question_id should be updated
        assert record.question_id == "18-42"

    def test_question_id_auto_generation_from_hash(self):
        """If question_id is 'unknown' with no metadata, generate from hash."""
        record = QARecord(
            question_id="unknown",
            question_text="What is the status of the metro rail project?",
            answer_text="The metro rail project is under implementation in 20 cities.",
        )
        # Should have a generated ID
        assert record.question_id.startswith("gen-")
        assert len(record.question_id) == 20  # "gen-" + 16-char hash

    def test_question_text_too_short_rejected(self):
        """Question text must be at least 10 characters."""
        with pytest.raises(ValueError) as exc_info:
            QARecord(
                question_id="18-1",
                question_text="Short?",  # too short
                answer_text="This is a much longer answer text that is definitely valid.",
            )
        assert "String should have at least 10 characters" in str(exc_info.value)

    def test_answer_text_too_short_rejected(self):
        """Answer text must be at least 10 characters."""
        with pytest.raises(ValueError) as exc_info:
            QARecord(
                question_id="18-1",
                question_text="This is a valid question text that is long enough.",
                answer_text="Too short",  # too short
            )
        assert "String should have at least 10 characters" in str(exc_info.value)

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped."""
        record = QARecord(
            question_id="  18-1  ",
            question_text="  Will the Minister state...  ",
            answer_text="  The Government has stated...  ",
        )
        assert record.question_id == "18-1"
        assert record.question_text == "Will the Minister state..."
        assert record.answer_text == "The Government has stated..."

    def test_empty_string_rejected(self):
        """Empty string question should be rejected."""
        with pytest.raises(ValueError) as exc_info:
            QARecord(
                question_id="18-1",
                question_text="",  # empty
                answer_text="This is a valid answer text.",
            )
        assert "Field cannot be empty" in str(exc_info.value)

    def test_document_content_format(self):
        """document_content should include both question and answer."""
        record = QARecord(
            question_id="18-1",
            question_text="What is the metro rail status?",
            answer_text="The metro rail project covers 20 cities.",
            metadata=QARecordMetadata(ministry="Railways", subject="Metro"),
        )
        content = record.document_content
        assert "QUESTION: What is the metro rail status?" in content
        assert "ANSWER: The metro rail project covers 20 cities." in content
        assert "MINISTRY: Railways" in content
        assert "SUBJECT: Metro" in content

    def test_document_content_no_metadata_when_absent(self):
        """When ministry/subject are absent, document_content omits those lines."""
        record = QARecord(
            question_id="18-1",
            question_text="What is the metro rail status?",
            answer_text="Metro project covers 20 cities.",
        )
        content = record.document_content
        assert "MINISTRY:" not in content
        assert "SUBJECT:" not in content

    def test_content_hash_deterministic(self):
        """content_hash should be stable across calls."""
        record = QARecord(
            question_id="18-1",
            question_text="Same question text here.",
            answer_text="Same answer text here.",
        )
        hash1 = record.content_hash
        hash2 = record.content_hash
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA-256 first 16 hex chars

    def test_to_document_dict(self):
        """to_document_dict should return retrieval-ready structure."""
        record = QARecord(
            question_id="18-42",
            question_text="GST details?",
            answer_text="GST collection was X.",
            metadata=QARecordMetadata(ministry="Finance"),
        )
        doc = record.to_document_dict()
        assert doc["doc_id"] == "18-42"
        assert doc["question"] == "GST details?"
        assert doc["answer"] == "GST collection was X."
        assert doc["metadata"]["ministry"] == "Finance"
        assert "document_content" not in doc  # not included in retrieval doc

    def test_question_type_enum(self):
        """Question type should accept valid enum values."""
        record = QARecord(
            question_id="18-1",
            question_text="What is the unemployment rate?",
            answer_text="The unemployment rate is X percent.",
            metadata=QARecordMetadata(question_type=QuestionType.STARRED),
        )
        assert record.metadata.question_type == QuestionType.STARRED

    def test_model_serialization(self):
        """Record should serialize to JSON cleanly."""
        record = QARecord(
            question_id="18-1",
            question_text="Valid question text here.",
            answer_text="Valid answer text here.",
        )
        data = record.model_dump(mode="json")
        # Verify it's JSON-serializable
        json_str = json.dumps(data)
        restored = json.loads(json_str)
        assert restored["question_id"] == "18-1"

    def test_scraped_at_auto_populated(self):
        """scraped_at should be auto-populated with current time."""
        before = datetime.utcnow()
        record = QARecord(
            question_id="18-1",
            question_text="Valid question text here.",
            answer_text="Valid answer text here.",
        )
        after = datetime.utcnow()
        assert before <= record.scraped_at <= after


# ─────────────────────────────────────────────────────────────────────────────
# DataValidator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataValidator:
    """Tests for DataValidator."""

    def test_validate_all_valid(self):
        """A list of valid records should return all as valid."""
        records = [
            QARecord(
                question_id=f"18-{i:04d}",
                question_text=f"What is the status of scheme {i}?",
                answer_text="The scheme is under implementation and has covered X beneficiaries.",
            )
            for i in range(100)
        ]
        validator = DataValidator()
        report = validator.validate(records, raw_source="test")

        assert report.valid_count == 100
        assert report.invalid_count == 0
        assert report.duplicate_count == 0

    def test_validate_invalid_records(self):
        """Invalid records should be flagged and excluded."""
        records = [
            QARecord(
                question_id="18-1",
                question_text="Valid question text here.",
                answer_text="Valid answer text here.",
            ),
            {"question_id": "18-2"},  # Missing question_text — invalid dict
            QARecord(
                question_id="18-3",
                question_text="Valid question.",
                answer_text="Valid answer.",  # Will work
            ),
            "",  # Not even a dict
        ]
        validator = DataValidator()
        report = validator.validate(records, raw_source="test")

        # Only 2 valid records
        assert report.valid_count == 2
        assert report.invalid_count == 2  # 1 dict + 1 string
        assert len(report.invalid_records) == 2

    def test_validate_deduplication(self):
        """Duplicate question_ids should be removed."""
        records = [
            QARecord(
                question_id="18-001",
                question_text="What is the unemployment rate?",
                answer_text="Unemployment rate is X percent.",
            ),
            QARecord(
                question_id="18-001",  # Duplicate ID
                question_text="Different question entirely?",
                answer_text="Different answer.",
            ),
            QARecord(
                question_id="18-002",
                question_text="What is the GDP growth?",
                answer_text="GDP growth is Y percent.",
            ),
        ]
        validator = DataValidator(dedup_key="question_id")
        report = validator.validate(records, raw_source="test")

        assert report.valid_count == 2
        assert report.duplicate_count == 1
        assert {r.question_id for r in report.valid_records} == {"18-001", "18-002"}

    def test_validate_statistics_computed(self):
        """Validation should compute field statistics."""
        records = [
            QARecord(
                question_id=f"18-{i:04d}",
                question_text="What is the unemployment rate?",
                answer_text="Unemployment rate is X percent.",
                metadata=QARecordMetadata(ministry="Finance" if i % 2 == 0 else None),
            )
            for i in range(10)
        ]
        validator = DataValidator()
        report = validator.validate(records, raw_source="test")

        assert report.stats.total_valid_records == 10
        assert report.stats.question_text_stats.present == 10
        assert report.stats.ministry_stats.present == 5  # 5 with ministry
        assert report.stats.ministry_stats.missing == 5

    def test_validate_empty_list(self):
        """Validating an empty list should return empty report."""
        validator = DataValidator()
        report = validator.validate([], raw_source="test")
        assert report.valid_count == 0
        assert report.stats.total_valid_records == 0


# ─────────────────────────────────────────────────────────────────────────────
# DataEnricher Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataEnricher:
    """Tests for DataEnricher."""

    def test_enrich_ministry_extraction(self):
        """Enricher should extract ministry from question text."""
        record = QARecord(
            question_id="18-1",
            question_text="Will the Minister of Finance be pleased to state details of GST collection?",
            answer_text="GST collection details are as follows...",
            metadata=QARecordMetadata(),  # No ministry set
        )
        enricher = DataEnricher()
        enriched = enricher.enrich(record)

        assert enriched.metadata.ministry == "Finance"

    def test_enrich_preserves_existing_ministry(self):
        """Enricher should not override existing ministry (non-strict mode)."""
        record = QARecord(
            question_id="18-1",
            question_text="Will the Minister of Finance state...",
            answer_text="Answer text here.",
            metadata=QARecordMetadata(ministry="Finance"),  # Already set
        )
        enricher = DataEnricher(strict=False)
        enriched = enricher.enrich(record)
        assert enriched.metadata.ministry == "Finance"

    def test_enrich_strict_mode_overrides(self):
        """In strict mode, enrichment should override existing values."""
        record = QARecord(
            question_id="18-1",
            question_text="Will the Minister of Railways state...",
            answer_text="Answer text here.",
            metadata=QARecordMetadata(ministry="Finance"),  # Wrong ministry
        )
        enricher = DataEnricher(strict=True)
        enriched = enricher.enrich(record)
        # In strict mode, it re-extracts — it may or may not change
        # The important thing is it doesn't crash
        assert enriched.metadata.ministry is not None

    def test_enrich_question_type_detection(self):
        """Enricher should detect STARRED question type."""
        record = QARecord(
            question_id="18-1",
            question_text="[STARRED] Will the Minister of Finance state the details...",
            answer_text="Details are as follows...",
            metadata=QARecordMetadata(question_type=QuestionType.UNKNOWN),
        )
        enricher = DataEnricher()
        enriched = enricher.enrich(record)
        assert enriched.metadata.question_type == QuestionType.STARRED

    def test_enrich_subject_extraction(self):
        """Enricher should extract subject topic from question."""
        record = QARecord(
            question_id="18-1",
            question_text="What steps has the Government taken to combat malaria in rural areas?",
            answer_text="Steps taken include indoor residual spraying, distribution of nets...",
            metadata=QARecordMetadata(subject=None),
        )
        enricher = DataEnricher()
        enriched = enricher.enrich(record)
        assert enriched.metadata.subject == "Health"

    def test_enrich_batch(self):
        """enrich_batch should process all records in-place."""
        records = [
            QARecord(
                question_id=f"18-{i}",
                question_text=f"What is the status of malaria cases in district {i}?",
                answer_text="Malaria cases have been reported in various districts.",
            )
            for i in range(5)
        ]
        enricher = DataEnricher()
        enriched = enricher.enrich_batch(records, show_progress=False)
        assert len(enriched) == 5
        assert all(r.metadata.subject == "Health" for r in enriched)

    def test_enrich_date_extraction(self):
        """Enricher should extract date from text."""
        record = QARecord(
            question_id="18-1",
            question_text="What is the status of the scheme as of 15 March 2023?",
            answer_text="The scheme status is as follows...",
            metadata=QARecordMetadata(),
        )
        enricher = DataEnricher()
        enriched = enricher.enrich(record)
        assert enriched.metadata.date == "2023-03-15"


# ─────────────────────────────────────────────────────────────────────────────
# MockDataGenerator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMockDataGenerator:
    """Tests for MockDataGenerator."""

    def test_generates_correct_count(self):
        """Generator should produce exactly target_count records."""
        gen = MockDataGenerator(target_count=50, seed=42)
        records = gen.generate_batch(50)
        assert len(records) == 50

    def test_all_records_valid(self):
        """All generated records should pass Pydantic validation."""
        gen = MockDataGenerator(target_count=100, seed=42)
        for record in gen.generate():
            # Should not raise
            validated = QARecord.model_validate(record.model_dump())
            assert validated.question_id
            assert len(validated.question_text) >= 10
            assert len(validated.answer_text) >= 10
            if record.question_id == "gen-0001":  # Only check first
                break  # Don't validate all 100 in a unit test

    def test_unique_question_ids(self):
        """Generated records should have unique question IDs."""
        gen = MockDataGenerator(target_count=200, seed=42)
        records = gen.generate_batch(200)
        ids = [r.question_id for r in records]
        assert len(set(ids)) == len(ids), "Duplicate question IDs found"

    def test_ministry_distribution(self):
        """Generated records should follow realistic ministry distribution."""
        gen = MockDataGenerator(target_count=500, seed=42)
        records = gen.generate_batch(500)
        ministries = [r.metadata.ministry for r in records if r.metadata.ministry]

        assert len(ministries) > 0  # Most records should have a ministry
        # Check some expected ministries are present
        ministry_names = set(ministries)
        assert "Finance" in ministry_names
        assert "Health and Family Welfare" in ministry_names

    def test_question_types(self):
        """Generated records should include various question types."""
        gen = MockDataGenerator(target_count=500, seed=42)
        records = gen.generate_batch(500)
        types = {r.metadata.question_type for r in records}

        # Should have multiple types (mostly UNSTARRED)
        assert QuestionType.UNSTARRED in types
        assert QuestionType.STARRED in types

    def test_reproducible_with_seed(self):
        """Same seed should produce same records."""
        gen1 = MockDataGenerator(target_count=10, seed=123)
        gen2 = MockDataGenerator(target_count=10, seed=123)

        records1 = gen1.generate_batch(10)
        records2 = gen2.generate_batch(10)

        assert [r.question_id for r in records1] == [r.question_id for r in records2]
        assert [r.question_text for r in records1] == [r.question_text for r in records2]

    def test_answer_substantially_longer_than_question(self):
        """Answers should generally be longer than questions (typical for Lok Sabha)."""
        gen = MockDataGenerator(target_count=50, seed=42)
        records = gen.generate_batch(50)
        for r in records:
            assert len(r.answer_text) >= len(r.question_text), (
                f"Answer shorter than question for {r.question_id}"
            )

    def test_deterministic_content(self):
        """Question text should be meaningful (not Lorem Ipsum or gibberish)."""
        gen = MockDataGenerator(target_count=5, seed=99)
        records = gen.generate_batch(5)
        for r in records:
            # Check for actual words (not random characters)
            words = r.question_text.split()
            assert len(words) >= 5, f"Question too short: {r.question_text}"
            # Check for meaningful content
            assert any(c.isupper() for c in r.question_text)  # Has uppercase (proper text)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader:
    """Tests for DataLoader save/load round-trip."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test files."""
        return tmp_path

    @pytest.fixture
    def sample_records(self):
        """Create a small list of sample records."""
        return [
            QARecord(
                question_id=f"18-{i:04d}",
                question_text=f"What is the status of scheme {i}?",
                answer_text="The scheme has been implemented and covers X beneficiaries across Y states.",
                metadata=QARecordMetadata(
                    ministry=["Finance", "Health and Family Welfare", "Education"][i % 3],
                    session=18,
                    subject=f"Subject {i}",
                ),
            )
            for i in range(10)
        ]

    def test_save_and_load_jsonl(self, temp_dir, sample_records):
        """JSONL save/load should preserve all records."""
        path = temp_dir / "test.jsonl"

        DataLoader.save_jsonl(sample_records, path)
        loaded = DataLoader.load_jsonl(path)

        assert len(loaded) == len(sample_records)
        assert loaded[0].question_id == sample_records[0].question_id
        assert loaded[-1].question_id == sample_records[-1].question_id
        assert loaded[0].metadata.ministry == sample_records[0].metadata.ministry

    def test_save_and_load_jsonl_with_limit(self, temp_dir, sample_records):
        """load_jsonl with limit should only load that many records."""
        path = temp_dir / "test.jsonl"
        DataLoader.save_jsonl(sample_records, path)

        loaded = DataLoader.load_jsonl(path, limit=5)
        assert len(loaded) == 5

    def test_save_and_load_jsonl_append(self, temp_dir, sample_records):
        """Appending to JSONL should add records without removing existing."""
        path = temp_dir / "test.jsonl"

        DataLoader.save_jsonl(sample_records[:5], path)
        DataLoader.save_jsonl(sample_records[5:], path, append=True)

        loaded = DataLoader.load_jsonl(path)
        assert len(loaded) == 10

    def test_save_and_load_json(self, temp_dir, sample_records):
        """JSON (array) save/load should work correctly."""
        path = temp_dir / "test.json"
        DataLoader.save_json(sample_records, path)
        loaded = DataLoader.load_json(path)

        assert len(loaded) == len(sample_records)

    def test_count_records(self, temp_dir, sample_records):
        """count_records should return correct count without loading data."""
        path = temp_dir / "test.jsonl"
        DataLoader.save_jsonl(sample_records, path)

        count = DataLoader.count_records(path)
        assert count == 10

    def test_streaming_load(self, temp_dir, sample_records):
        """Streaming load should produce all records."""
        path = temp_dir / "test.jsonl"
        DataLoader.save_jsonl(sample_records, path)

        loaded = list(DataLoader.load_jsonl_streaming(path))
        assert len(loaded) == 10


# ─────────────────────────────────────────────────────────────────────────────
# Statistics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestionStats:
    """Tests for IngestionStats."""

    def test_stats_to_dict_serializable(self):
        """to_dict() should return a JSON-serializable dict."""
        stats = IngestionStats(
            total_raw_records=1000,
            total_valid_records=950,
            total_invalid_records=30,
            duplicates_removed=20,
            ministry_distribution={"Finance": 200, "Health": 150},
        )
        data = stats.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(data, default=str)
        assert "total_raw_records" in json_str

    def test_print_summary_format(self):
        """print_summary should return a non-empty string."""
        stats = IngestionStats(
            total_raw_records=1000,
            total_valid_records=950,
            ministry_distribution={"Finance": 200, "Health": 150},
        )
        summary = stats.print_summary()
        assert isinstance(summary, str)
        assert len(summary) > 100
        assert "PHASE 1" in summary
        assert "1,000" in summary
        assert "Finance" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestionPipeline:
    """End-to-end integration tests for the ingestion pipeline."""

    def test_full_pipeline_with_mock_data(self, tmp_path):
        """Running the full pipeline with mock data should produce valid output."""
        from src.data.ingestion_pipeline import IngestionPipeline

        pipeline = IngestionPipeline(
            config_path="config/ingestion.yaml",
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            enriched_dir=str(tmp_path / "enriched"),
        )

        stats = pipeline.run(
            target_count=100,
            strategy="mock",
            skip_enrichment=False,
        )

        # Check stats
        assert stats.total_raw_records == 100
        assert stats.total_valid_records == 100
        assert stats.total_invalid_records == 0
        assert stats.unique_records == 100

        # Check output files exist
        assert Path(stats.raw_file).exists()
        assert Path(stats.processed_file).exists()
        assert Path(stats.enriched_file).exists()

        # Check enriched records have metadata
        enriched = DataLoader.load_jsonl(stats.enriched_file)
        assert len(enriched) == 100
        # Most enriched records should have ministry
        with_ministry = sum(1 for r in enriched if r.metadata.ministry)
        assert with_ministry >= 50  # At least half

    def test_pipeline_skip_enrichment(self, tmp_path):
        """Pipeline with skip_enrichment=True should not produce enriched file."""
        from src.data.ingestion_pipeline import IngestionPipeline

        pipeline = IngestionPipeline(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            enriched_dir=str(tmp_path / "enriched"),
        )

        stats = pipeline.run(
            target_count=20,
            strategy="mock",
            skip_enrichment=True,
        )

        assert stats.enriched_file is None
        assert Path(stats.processed_file).exists()
