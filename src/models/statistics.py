"""
Ingestion statistics and reporting.

Collects metrics at every stage of the data pipeline so we can
report a comprehensive statistics report at the end of Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field


@dataclass
class FieldStats:
    """Statistics for a single field across all records."""

    present: int = 0       # Number of records where field is non-null/non-empty
    missing: int = 0        # Number of records where field is null/empty
    unique: int = 0         # Number of unique values
    total: int = 0          # Total records checked

    @property
    def presence_rate(self) -> float:
        """Fraction of records that have this field."""
        return self.present / self.total if self.total > 0 else 0.0

    @property
    def missing_rate(self) -> float:
        """Fraction of records missing this field."""
        return self.missing / self.total if self.total > 0 else 0.0


@dataclass
class ValidationResult:
    """Result of validating a single record."""

    question_id: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ScrapingStats:
    """Statistics from the scraping phase."""

    pages_scraped: int = 0
    question_links_found: int = 0
    individual_pages_attempted: int = 0
    individual_pages_success: int = 0
    individual_pages_failed: int = 0
    http_errors: int = 0
    parse_errors: int = 0
    rate_limit_hits: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # ── Archive ingestion accounting (production path) ─────────────────────
    rows_read: int = 0             # rows processed by the archive scraper
    pdf_parsed: int = 0            # records successfully parsed from PDF
    docx_parsed: int = 0           # records successfully parsed from DOCX
    synthetic_generated: int = 0   # always 0 for archive ingestion (never synthesizes)
    skipped_broken: int = 0        # download/HTTP/resolution failure
    skipped_scanned: int = 0       # image-only PDF, no text layer
    skipped_unsupported: int = 0   # HTML/ZIP/other non-document content
    skipped_parser_failure: int = 0  # document downloaded but extraction failed
    skipped_other: int = 0         # no source URL / pdf mode disabled / unknown

    @property
    def skipped_total(self) -> int:
        return (
            self.skipped_broken
            + self.skipped_scanned
            + self.skipped_unsupported
            + self.skipped_parser_failure
            + self.skipped_other
        )

    @property
    def success_rate(self) -> float:
        if self.individual_pages_attempted == 0:
            return 0.0
        return self.individual_pages_success / self.individual_pages_attempted

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


class IngestionStats(BaseModel):
    """
    Complete statistics report for Phase 1.
    Generated after the full ingestion pipeline completes.
    """

    # Pipeline info
    pipeline_version: str = "1.0.0"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    config_file: str | None = None
    raw_data_source: str = "unknown"

    # Raw scraping stats (if scraped)
    scraping: ScrapingStats = Field(default_factory=ScrapingStats)

    # Record counts
    total_raw_records: int = 0
    total_valid_records: int = 0
    total_invalid_records: int = 0

    # Deduplication
    duplicates_removed: int = 0
    unique_records: int = 0

    # Field completeness
    question_id_stats: FieldStats = Field(default_factory=FieldStats)
    question_text_stats: FieldStats = Field(default_factory=FieldStats)
    answer_text_stats: FieldStats = Field(default_factory=FieldStats)
    ministry_stats: FieldStats = Field(default_factory=FieldStats)
    date_stats: FieldStats = Field(default_factory=FieldStats)
    subject_stats: FieldStats = Field(default_factory=FieldStats)
    source_url_stats: FieldStats = Field(default_factory=FieldStats)
    question_type_stats: dict[str, int] = Field(default_factory=dict)

    # Content statistics
    avg_question_length_chars: float = 0.0
    avg_answer_length_chars: float = 0.0
    min_question_length_chars: int = 0
    max_question_length_chars: int = 0
    min_answer_length_chars: int = 0
    max_answer_length_chars: int = 0

    # Ministry distribution
    ministry_distribution: dict[str, int] = Field(default_factory=dict)
    question_type_distribution: dict[str, int] = Field(default_factory=dict)

    # Files produced
    raw_file: str | None = None
    processed_file: str | None = None
    enriched_file: str | None = None

    # Errors and warnings
    validation_errors: list[str] = Field(default_factory=list)
    processing_warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a dict with a consistent, uniform schema.

        Every ``*_stats`` entry carries ``present``/``missing``/``unique``/
        ``total`` plus the computed ``presence_rate``; the ``scraping`` block
        includes the computed ``success_rate`` and ``duration_seconds``.
        """
        data = self.model_dump(mode="json")

        # Uniform, enriched field stats (adds computed presence_rate)
        for key in (
            "question_id_stats",
            "question_text_stats",
            "answer_text_stats",
            "ministry_stats",
            "date_stats",
            "subject_stats",
            "source_url_stats",
        ):
            fs: FieldStats = getattr(self, key)
            data[key] = {
                "present": fs.present,
                "missing": fs.missing,
                "unique": fs.unique,
                "total": fs.total,
                "presence_rate": round(fs.presence_rate, 4),
            }

        # Enriched scraping stats (adds computed success_rate / duration)
        data["scraping"] = {
            "pages_scraped": self.scraping.pages_scraped,
            "question_links_found": self.scraping.question_links_found,
            "individual_pages_attempted": self.scraping.individual_pages_attempted,
            "individual_pages_success": self.scraping.individual_pages_success,
            "individual_pages_failed": self.scraping.individual_pages_failed,
            "http_errors": self.scraping.http_errors,
            "parse_errors": self.scraping.parse_errors,
            "rate_limit_hits": self.scraping.rate_limit_hits,
            "started_at": self.scraping.started_at.isoformat() if self.scraping.started_at else None,
            "completed_at": self.scraping.completed_at.isoformat() if self.scraping.completed_at else None,
            "success_rate": round(self.scraping.success_rate, 4),
            "duration_seconds": round(self.scraping.duration_seconds, 2),
            # Archive ingestion accounting (production)
            "rows_read": self.scraping.rows_read,
            "pdf_parsed": self.scraping.pdf_parsed,
            "docx_parsed": self.scraping.docx_parsed,
            "synthetic_generated": self.scraping.synthetic_generated,
            "skipped_total": self.scraping.skipped_total,
            "skipped_broken": self.scraping.skipped_broken,
            "skipped_scanned": self.scraping.skipped_scanned,
            "skipped_unsupported": self.scraping.skipped_unsupported,
            "skipped_parser_failure": self.scraping.skipped_parser_failure,
            "skipped_other": self.scraping.skipped_other,
        }

        return data

    def print_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "  PHASE 1 — DATA INGESTION: SUMMARY REPORT",
            "=" * 60,
            "",
            f"  Generated at: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Data source : {self.raw_data_source}",
            "",
            "  ── Record Counts ──────────────────────────────────────",
            f"  Raw records       : {self.total_raw_records:>10,}",
            f"  Valid records     : {self.total_valid_records:>10,}",
            f"  Invalid records   : {self.total_invalid_records:>10,}",
            f"  Duplicates removed: {self.duplicates_removed:>10,}",
            f"  Unique records    : {self.unique_records:>10,}",
            "",
            "  ── Field Completeness ────────────────────────────────",
            f"  question_text : {self.question_text_stats.presence_rate:.1%} present",
            f"  answer_text   : {self.answer_text_stats.presence_rate:.1%} present",
            f"  ministry      : {self.ministry_stats.presence_rate:.1%} present",
            f"  date          : {self.date_stats.presence_rate:.1%} present",
            f"  subject       : {self.subject_stats.presence_rate:.1%} present",
            f"  source_url    : {self.source_url_stats.presence_rate:.1%} present",
            "",
            "  ── Content Lengths ───────────────────────────────────",
            f"  Question length: {self.min_question_length_chars:>5} – {self.max_question_length_chars:>5} chars"
            f"  (avg: {self.avg_question_length_chars:>6.0f})",
            f"  Answer length  : {self.min_answer_length_chars:>5} – {self.max_answer_length_chars:>6} chars"
            f"  (avg: {self.avg_answer_length_chars:>7.0f})",
            "",
            "  ── Top Ministries ────────────────────────────────────",
        ]

        sorted_ministries = sorted(
            self.ministry_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        for ministry, count in sorted_ministries:
            pct = count / max(self.total_valid_records, 1) * 100
            lines.append(f"  {ministry:<40} {count:>5} ({pct:>5.1f}%)")

        if self.scraping.individual_pages_attempted > 0:
            lines.extend([
                "",
                "  ── Scraping ─────────────────────────────────────────",
                f"  Pages attempted : {self.scraping.individual_pages_attempted:>10,}",
                f"  Pages succeeded  : {self.scraping.individual_pages_success:>10,}",
                f"  Pages failed     : {self.scraping.individual_pages_failed:>10,}",
                f"  Success rate     : {self.scraping.success_rate:.1%}",
                f"  Duration          : {self.scraping.duration_seconds:.0f}s",
            ])

        lines.extend([
            "",
            "  ── Output Files ────────────────────────────────────────",
            f"  Raw       : {self.raw_file or 'N/A':<50}",
            f"  Processed : {self.processed_file or 'N/A':<50}",
            f"  Enriched  : {self.enriched_file or 'N/A':<50}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)
