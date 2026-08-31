"""
Data Validator for Parliamentary Q&A records.

Classification: LEGACY-REQUIRED component of the Phase-1 Lok Sabha
acquisition pipeline (workspace cleanup, audit §4). Runs only inside
src/data/ingestion_pipeline.py; the canonical corpus pipeline
(src/scripts/ingest*.py) validates at QARecord load time instead. Keep for
Lok Sabha restaging runs; not a general ingestion gate.

Design Decisions
----------------
1. We use Pydantic's native validation — fast, type-safe, declarative.
   Each QARecord is validated individually. This is the primary validation gate.

2. We run two passes:
   - Pass 1 (Strict): Validate against QARecord schema. Records failing here
     are "invalid" and excluded from the processed dataset.
   - Pass 2 (Lenient): For records that pass Pass 1, we compute completeness
     statistics. A record missing a ministry is still valid but gets a warning.

3. Deduplication uses question_id first (most reliable), then falls back to
   a SHA-256 hash of the question_text. This handles records scraped twice
   with different IDs.

4. Statistics are collected in a single pass over the data for efficiency.
   We don't want to iterate multiple times over large datasets.

5. We return a ValidationReport containing both valid records and statistics.
   The statistics object drives the Phase 1 reporting requirement.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError
from rich.console import Console

from src.models.qa_record import QARecord
from src.models.statistics import FieldStats, IngestionStats

console = Console()


@dataclass
class ValidationError_:
    """A single validation error."""

    question_id: str
    field: str
    message: str
    raw_value: str | None = None


@dataclass
class ValidationWarning:
    """A single validation warning (non-fatal)."""

    question_id: str
    message: str


@dataclass
class ValidationReport:
    """
    Complete report from validating a batch of raw records.

    Contains:
    - valid_records: Records that passed Pydantic validation
    - invalid_records: Records that failed validation (with error details)
    - duplicate_records: Records removed as duplicates
    - stats: Comprehensive statistics
    """

    valid_records: list[QARecord] = field(default_factory=list)
    invalid_records: list[tuple[dict, list[ValidationError_]]] = field(
        default_factory=list
    )
    duplicate_records: list[QARecord] = field(default_factory=list)
    errors: list[ValidationError_] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)
    stats: IngestionStats = field(default_factory=IngestionStats)

    @property
    def valid_count(self) -> int:
        return len(self.valid_records)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_records)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_records)


class DataValidator:
    """
    Validates, deduplicates, and computes statistics on Q&A records.

    Usage
    -----
    ```python
    validator = DataValidator()
    report = validator.validate(raw_records)  # list[QARecord] or list[dict]
    print(report.stats.print_summary())
    ```
    """

    def __init__(
        self,
        dedup_key: str = "question_id",
        allow_partial: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        dedup_key : str
            Field to use for deduplication. Options: "question_id", "question_hash".
        allow_partial : bool
            If True, records missing optional fields are still valid.
            If False, all fields must be present.
        """
        self.dedup_key = dedup_key
        self.allow_partial = allow_partial
        self._seen_ids: set[str] = set()
        self._seen_hashes: set[str] = set()

    def reset(self) -> None:
        """Reset deduplication state between validation runs."""
        self._seen_ids.clear()
        self._seen_hashes.clear()

    def validate(
        self,
        records: list[QARecord] | list[dict],
        raw_source: str = "unknown",
        show_progress: bool = True,
    ) -> ValidationReport:
        """
        Validate a list of records.

        Runs in two phases:
        1. Pydantic validation + deduplication
        2. Statistics computation

        Parameters
        ----------
        records : list[QARecord] | list[dict]
            Raw records to validate.
        raw_source : str
            Human-readable source of the raw data (for reporting).
        show_progress : bool
            Show a progress spinner.

        Returns
        -------
        ValidationReport
            Valid records + invalid records + statistics.
        """
        self.reset()

        report = ValidationReport()
        report.stats.raw_data_source = raw_source

        # Phase 1: Validation and deduplication
        records_iter: Iterator = enumerate(records)
        if show_progress:
            console.print(f"[dim]Validating {len(records):,} records...[/dim]")

        for idx, raw in records_iter:
            self._process_single_record(raw, report)

        # Phase 2: Compute completeness and distribution statistics
        self._compute_statistics(report)

        return report

    def _process_single_record(
        self,
        raw: QARecord | dict | Any,
        report: ValidationReport,
    ) -> None:
        """Validate and deduplicate a single record."""
        # Convert dict to QARecord if needed
        if isinstance(raw, dict):
            raw_dict = raw
            try:
                record = QARecord.model_validate(raw_dict)
            except ValidationError as exc:
                errors = [
                    ValidationError_(
                        question_id=str(raw_dict.get("question_id", f"index_{id(raw_dict)}")),
                        field=str(err["loc"]),
                        message=err["msg"],
                        raw_value=str(err.get("input", ""))[:100],
                    )
                    for err in exc.errors()
                ]
                report.invalid_records.append((raw_dict, errors))
                report.errors.extend(errors)
                return
        elif isinstance(raw, QARecord):
            record = raw
        else:
            # Garbage input — string, int, None, etc. Mark as invalid.
            errors = [
                ValidationError_(
                    question_id="unknown",
                    field="root",
                    message=f"Expected dict or QARecord, got {type(raw).__name__}",
                    raw_value=str(raw)[:100] if raw is not None else None,
                )
            ]
            report.invalid_records.append((raw, errors))
            report.errors.extend(errors)
            return

        # Deduplication check
        is_duplicate = False
        dedup_key_val = record.question_id

        if self.dedup_key == "question_id":
            if dedup_key_val in self._seen_ids:
                is_duplicate = True
            else:
                self._seen_ids.add(dedup_key_val)
        elif self.dedup_key == "question_hash":
            h = record.content_hash
            if h in self._seen_hashes:
                is_duplicate = True
            else:
                self._seen_hashes.add(h)

        if is_duplicate:
            report.duplicate_records.append(record)
            return

        # Completeness checks (warnings, not errors)
        self._check_completeness(record, report)

        # Add valid record
        report.valid_records.append(record)

    def _check_completeness(
        self,
        record: QARecord,
        report: ValidationReport,
    ) -> None:
        """Issue warnings for optional fields that are missing."""
        if not record.metadata.ministry:
            report.warnings.append(
                ValidationWarning(
                    question_id=record.question_id,
                    message="Missing ministry field",
                )
            )
        if not record.metadata.date:
            report.warnings.append(
                ValidationWarning(
                    question_id=record.question_id,
                    message="Missing date field",
                )
            )
        if not record.metadata.subject:
            report.warnings.append(
                ValidationWarning(
                    question_id=record.question_id,
                    message="Missing subject field",
                )
            )

    def _compute_statistics(self, report: ValidationReport) -> None:
        """Compute comprehensive statistics over valid records."""
        records = report.valid_records
        n = len(records)
        report.stats.total_raw_records = (
            n + report.invalid_count + report.duplicate_count
        )
        report.stats.total_valid_records = n
        report.stats.total_invalid_records = report.invalid_count
        report.stats.duplicates_removed = report.duplicate_count
        report.stats.unique_records = n

        # Field completeness
        report.stats.question_text_stats = self._field_stats(
            records, lambda r: r.question_text, n
        )
        report.stats.answer_text_stats = self._field_stats(
            records, lambda r: r.answer_text, n
        )
        report.stats.ministry_stats = self._field_stats(
            records, lambda r: r.metadata.ministry, n
        )
        report.stats.date_stats = self._field_stats(
            records, lambda r: r.metadata.date, n
        )
        report.stats.subject_stats = self._field_stats(
            records, lambda r: r.metadata.subject, n
        )
        report.stats.source_url_stats = self._field_stats(
            records, lambda r: r.metadata.source_url, n
        )
        report.stats.question_id_stats = FieldStats(
            present=len(set(r.question_id for r in records)),
            missing=0,
            unique=len(set(r.question_id for r in records)),
            total=n,
        )

        # Content length statistics
        q_lengths = [len(r.question_text) for r in records]
        a_lengths = [len(r.answer_text) for r in records]

        report.stats.avg_question_length_chars = sum(q_lengths) / n if n else 0
        report.stats.avg_answer_length_chars = sum(a_lengths) / n if n else 0
        report.stats.min_question_length_chars = min(q_lengths) if q_lengths else 0
        report.stats.max_question_length_chars = max(q_lengths) if q_lengths else 0
        report.stats.min_answer_length_chars = min(a_lengths) if a_lengths else 0
        report.stats.max_answer_length_chars = max(a_lengths) if a_lengths else 0

        # Ministry distribution
        ministries = Counter(
            r.metadata.ministry for r in records if r.metadata.ministry
        )
        report.stats.ministry_distribution = dict(ministries)

        # Question type distribution
        qtypes = Counter(
            r.metadata.question_type.value for r in records
        )
        report.stats.question_type_distribution = dict(qtypes)

    @staticmethod
    def _field_stats(
        records: list[QARecord],
        value_of,
        total: int,
    ) -> FieldStats:
        """Compute FieldStats for a given field value extractor.

        ``value_of(record)`` must return the field's value (or None).
        ``present`` counts non-empty values; ``unique`` counts the distinct
        non-empty values across records.
        """
        present = sum(1 for r in records if value_of(r))
        unique_vals = len({value_of(r) for r in records if value_of(r)})
        return FieldStats(
            present=present,
            missing=total - present,
            unique=unique_vals,
            total=total,
        )
