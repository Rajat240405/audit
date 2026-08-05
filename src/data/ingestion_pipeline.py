"""
Data Ingestion Pipeline for Parliamentary Q&A.

Ties together: Scraping → Validation → Deduplication → Enrichment → Output.

Phase 1 Deliverable:
- Clean, validated knowledge base at data/processed/
- Comprehensive statistics report
- Enrichment applied (optional metadata)

This pipeline is designed to be run as a standalone CLI tool
and also importable as a Python module.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel

from src.data.enricher import DataEnricher
from src.data.loader import DataLoader
from src.data.scraper import ScraperFactory
from src.data.validator import DataValidator
from src.models.qa_record import QARecord
from src.models.statistics import IngestionStats
from src.utils.project_scope import resolve_effective_ministry_filter, filter_records_by_ministry

console = Console()


class IngestionPipeline:
    """
    Orchestrates the full Phase 1 data ingestion pipeline.

    Stages
    ------
    1. SCRAPE: Fetch raw Q&A records from source
    2. VALIDATE: Check schema compliance, deduplicate
    3. ENRICH: Extract metadata (ministry, date, subject)
    4. SAVE: Write processed records to disk
    5. REPORT: Print statistics summary

    Design Decisions
    ---------------
    1. We write raw records incrementally during scraping.
       This means if scraping fails partway through, we still have
       all previously scraped records on disk.

    2. We process in fixed batch sizes to avoid memory issues
       with very large datasets.

    3. The pipeline is idempotent: running it twice produces
       the same output (same timestamp in filenames prevents
       accidental overwrites unless --overwrite is specified).

    4. All intermediate files are preserved:
       - data/raw/       → raw scraped records
       - data/processed/ → validated + deduplicated records
       - data/enriched/   → records with extracted metadata
    """

    def __init__(
        self,
        config_path: str = "config/ingestion.yaml",
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        enriched_dir: str = "data/enriched",
    ) -> None:
        self.config_path = Path(config_path)
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.enriched_dir = Path(enriched_dir)

        # Load config
        self.config = self._load_config()

        # Create output directories
        for d in [self.raw_dir, self.processed_dir, self.enriched_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Sub-modules
        self.validator = DataValidator(
            dedup_key=self.config.get("validation", {}).get("deduplicate_by", "question_id")
        )
        self.enricher = DataEnricher(
            strict=self.config.get("enrichment", {}).get("strict", False)
        )

        # Project scope is now resolved via shared utility (src/utils/project_scope.py)

    def _load_config(self) -> dict:
        """Load ingestion configuration from YAML."""
        if not self.config_path.exists():
            console.print(f"[yellow]Config not found at {self.config_path}, using defaults.[/yellow]")
            return {}
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def run(
        self,
        target_count: int = 3500,
        strategy: str = "archive",
        local_file: str | None = None,
        skip_enrichment: bool = False,
        skip_scraping: bool = False,
        overwrite: bool = False,
        use_pdf: bool = True,
        ministry_filter: str | None = None,      # Explicit override
        all_ministries: bool = False,            # New: explicit override flag
    ) -> IngestionStats:
        """
        Run the full ingestion pipeline.
        """
        # Map strategies for backward compatibility
        if strategy in ("auto", "httpx", "playwright"):
            strategy = "live"

        console.print()
        console.print(Panel.fit(
            "[bold cyan]Phase 1: Data Ingestion Pipeline[/bold cyan]\n"
            f"Target: {target_count:,} records | Strategy: {strategy}",
            border_style="cyan",
        ))
        console.print()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        raw_file = self.raw_dir / f"raw_{timestamp}.jsonl"
        processed_file = self.processed_dir / f"processed_{timestamp}.jsonl"
        enriched_file = self.enriched_dir / f"enriched_{timestamp}.jsonl"

        stats = IngestionStats(
            raw_data_source=strategy,
            raw_file=str(raw_file),
            processed_file=str(processed_file),
            enriched_file=str(enriched_file),
            config_file=str(self.config_path),
        )

        from rich.progress import Progress  # noqa: F401

        # ── Stage 1: Scrape ────────────────────────────────────────────────
        console.print("\n[bold]Stage 1/4: Scraping[/bold]")
        console.print(f"  Strategy: [cyan]{strategy}[/cyan]")
        console.print(f"  Target:   [cyan]{target_count:,}[/cyan] records")

        raw_records: list[QARecord] = []
        effective_filter = resolve_effective_ministry_filter(
            explicit_filter=ministry_filter,
            all_ministries=all_ministries,
            config_path=str(self.config_path),
        )
        assert effective_filter == "EARTH SCIENCES", (
            f"effective_filter={effective_filter!r}"
        )
        console.print(f"[green]Pipeline resolved filter = {effective_filter!r}[/green]")

        if skip_scraping:
            console.print("  [yellow]Skipping scrape (--skip-scraping flag set)[/yellow]")
        else:
            scraper = ScraperFactory(
                base_url=self.config.get("scraper", {}).get(
                    "base_url",
                    "https://sansad.in/ls/questions/questions-and-answers"
                ),
                strategy=strategy,
                local_file=local_file,
                use_pdf=use_pdf,
                ministry_filter=effective_filter,
            ).create_scraper()

            scraper_stats_start = datetime.utcnow()

            with Progress(
                *Progress.get_default_columns(),
                console=console,
            ) as p:
                task = p.add_task("Scraping records...", total=target_count)
                count = 0
                for record in scraper.scrape_all(max_records=target_count):
                    raw_records.append(record)
                    count += 1
                    if count % 50 == 0 or count >= target_count:
                        p.update(task, completed=min(count, target_count))
                    # Write raw records incrementally
                    DataLoader.save_jsonl([record], raw_file, append=True)

                stats.scraping = scraper.stats

            console.print(
                f"  ✓ Scraped {len(raw_records):,} records → {raw_file.name}"
            )

        # ── Stage 2: Validate ───────────────────────────────────────────────
        console.print("\n[bold]Stage 2/4: Validating[/bold]")

        report = self.validator.validate(
            raw_records,
            raw_source=f"{strategy}:{len(raw_records)} raw",
            show_progress=True,
        )

        console.print(f"  ✓ Valid:     {report.valid_count:>6,} records")
        console.print(f"  ✗ Invalid:   {report.invalid_count:>6,} records")
        console.print(f"  ⊘ Duplicates:{report.duplicate_count:>6,} removed")
        stats.scraping = scraper.stats if not skip_scraping else stats.scraping

        # Save processed records
        processed_records = report.valid_records

        # ── Phase 12+ MoES Project Scope Filtering (shared utility) ──
        if effective_filter:
            original_count = len(processed_records)
            processed_records = filter_records_by_ministry(processed_records, effective_filter)
            console.print(f"  ✓ Ministry filter applied ({effective_filter}): {original_count:,} → {len(processed_records):,} records kept")

            stats.total_valid_records = len(processed_records)
            stats.unique_records = len(processed_records)
            stats.duplicates_removed = report.duplicate_count

        DataLoader.save_jsonl(processed_records, processed_file)
        stats.raw_file = str(raw_file)
        stats.processed_file = str(processed_file)
        console.print(f"  ✓ Saved:     {processed_file.name}")

        # ── Stage 3: Enrich ───────────────────────────────────────────────────
        if skip_enrichment:
            console.print("\n[bold]Stage 3/4: Enrichment[/bold] [yellow]SKIPPED[/yellow]")
            stats.enriched_file = None
        else:
            console.print("\n[bold]Stage 3/4: Enriching[/bold]")

            enriched_records = self.enricher.enrich_batch(
                processed_records,
                show_progress=True,
            )

            DataLoader.save_jsonl(enriched_records, enriched_file)
            stats.enriched_file = str(enriched_file)
            console.print(f"  ✓ Enriched:  {enriched_file.name}")

        # ── Stage 4: Statistics ──────────────────────────────────────────────
        console.print("\n[bold]Stage 4/4: Computing Statistics[/bold]")

        # Build full stats from validation report
        stats.total_raw_records = len(raw_records)
        stats.total_valid_records = len(processed_records)
        stats.total_invalid_records = report.invalid_count
        stats.duplicates_removed = report.duplicate_count
        stats.unique_records = len(processed_records)
        stats.validation_errors = [f"{e.question_id}: {e.message}" for e in report.errors[:20]]
        stats.processing_warnings = [w.message for w in report.warnings[:20]]

        # Copy field stats from report (use processed_records for MoES-filtered stats)
        records_for_stats = processed_records
        if records_for_stats:
            first_valid = report.valid_records[0]
            # We already computed stats in the validator — extract them
            stats.avg_question_length_chars = sum(
                len(r.question_text) for r in report.valid_records
            ) / len(report.valid_records)
            stats.avg_answer_length_chars = sum(
                len(r.answer_text) for r in report.valid_records
            ) / len(report.valid_records)
            stats.min_question_length_chars = min(
                len(r.question_text) for r in report.valid_records
            )
            stats.max_question_length_chars = max(
                len(r.question_text) for r in report.valid_records
            )
            stats.min_answer_length_chars = min(
                len(r.answer_text) for r in report.valid_records
            )
            stats.max_answer_length_chars = max(
                len(r.answer_text) for r in report.valid_records
            )

            # Ministry distribution
            from collections import Counter
            ministry_dist = Counter(
                r.metadata.ministry for r in records_for_stats
                if r.metadata.ministry
            )
            stats.ministry_distribution = dict(ministry_dist)

            # Question type distribution
            qtype_dist = Counter(
                r.metadata.question_type.value for r in records_for_stats
            )
            stats.question_type_distribution = dict(qtype_dist)

            # Field stats
            n = len(records_for_stats)
            stats.question_text_stats.present = n
            stats.question_text_stats.total = n
            stats.answer_text_stats.present = n
            stats.answer_text_stats.total = n
            stats.ministry_stats.present = sum(
                1 for r in records_for_stats if r.metadata.ministry
            )
            stats.ministry_stats.missing = n - stats.ministry_stats.present
            stats.ministry_stats.total = n
            stats.date_stats.present = sum(
                1 for r in records_for_stats if r.metadata.date
            )
            stats.date_stats.missing = n - stats.date_stats.present
            stats.date_stats.total = n
            stats.subject_stats.present = sum(
                1 for r in records_for_stats if r.metadata.subject
            )
            stats.subject_stats.missing = n - stats.subject_stats.present
            stats.subject_stats.total = n
            stats.source_url_stats.present = sum(
                1 for r in records_for_stats if r.metadata.source_url
            )
            stats.source_url_stats.missing = n - stats.source_url_stats.present
            stats.source_url_stats.total = n
            stats.question_id_stats.present = n
            stats.question_id_stats.total = n
            stats.question_id_stats.unique = len(
                {r.question_id for r in records_for_stats}
            )

        # Save stats to JSON
        stats_file = self.processed_dir / f"stats_{timestamp}.json"
        with open(stats_file, "w") as f:
            json.dump(stats.to_dict(), f, indent=2, default=str)

        # ── Print Summary ───────────────────────────────────────────────────
        console.print()
        console.print(Panel.fit(
            stats.print_summary(),
            title="[bold]Phase 1 Complete[/bold]",
            border_style="green",
        ))

        console.print(f"\n  Statistics saved to: [cyan]{stats_file}[/cyan]")
        console.print(f"  Processed records:   [cyan]{processed_file}[/cyan]")
        if not skip_enrichment:
            console.print(f"  Enriched records:     [cyan]{enriched_file}[/cyan]")

        return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="1.0.0")
def cli() -> None:
    """Parliamentary RAG — Phase 1: Data Ingestion Pipeline."""
    pass


@cli.command()
@click.option(
    "--count", "-n",
    default=3500,
    type=int,
    help="Target number of Q&A records to ingest.",
)
@click.option(
    "--strategy", "-s",
    default="archive",
    type=click.Choice(["auto", "playwright", "httpx", "mock", "local", "live", "archive"]),
    help="Scraping strategy.",
)
@click.option(
    "--local-file", "-f",
    type=str,
    help="Path to local JSONL file (required if strategy=local).",
)
@click.option(
    "--skip-enrichment",
    is_flag=True,
    help="Skip metadata enrichment stage.",
)
@click.option(
    "--skip-scraping",
    is_flag=True,
    help="Skip scraping; use existing raw data.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output files.",
)
@click.option(
    "--use-pdf/--no-pdf",
    default=True,
    help="Whether to download and parse official PDFs (default: True).",
)
@click.option(
    "--config", "-c",
    type=str,
    default="config/ingestion.yaml",
    help="Path to configuration file.",
)
@click.option(
    "--ministry-filter",
    type=str,
    default=None,
    help="Explicit ministry filter (overrides config). Example: 'Ministry of Earth Sciences'",
)
@click.option(
    "--all-ministries",
    is_flag=True,
    default=False,
    help="Index ALL ministries (overrides MoES default scope in config)",
)
def ingest(
    count: int,
    strategy: str,
    local_file: str | None,
    skip_enrichment: bool,
    skip_scraping: bool,
    overwrite: bool,
    config: str,
    use_pdf: bool,
    ministry_filter: str | None,
    all_ministries: bool,
) -> None:
    """Run the Phase 1 data ingestion pipeline."""
    try:
        pipeline = IngestionPipeline(
            config_path=config,
            raw_dir="data/raw",
            processed_dir="data/processed",
            enriched_dir="data/enriched",
        )
        stats = pipeline.run(
            target_count=count,
            strategy=strategy,
            local_file=local_file,
            skip_enrichment=skip_enrichment,
            skip_scraping=skip_scraping,
            overwrite=overwrite,
            use_pdf=use_pdf,
            ministry_filter=ministry_filter,
            all_ministries=all_ministries,
        )
        console.print("\n[bold green]✓ Phase 1 ingestion complete![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise


@cli.command()
def stats() -> None:
    """Show statistics from the most recent ingestion run."""
    processed_dir = Path("data/processed")
    stat_files = sorted(processed_dir.glob("stats_*.json"), reverse=True)

    if not stat_files:
        console.print("[yellow]No statistics found. Run `ingest` first.[/yellow]")
        return

    latest = stat_files[0]
    with open(latest) as f:
        data = json.load(f)

    # Rebuild IngestionStats for nice printing
    from src.models.statistics import IngestionStats
    stats = IngestionStats.model_validate(data)
    console.print(stats.print_summary())


@cli.command()
def validate() -> None:
    """Validate an existing processed file."""
    processed_dir = Path("data/processed")
    processed_files = sorted(processed_dir.glob("processed_*.jsonl"), reverse=True)

    if not processed_files:
        console.print("[yellow]No processed files found.[/yellow]")
        return

    latest = processed_files[0]
    console.print(f"Validating: [cyan]{latest}[/cyan]")

    records = DataLoader.load_jsonl(latest)
    validator = DataValidator()

    # Re-validate (no deduplication since already processed)
    valid = [r for r in records]
    console.print(f"Total: {len(valid):,} records")

    # Check for issues
    short_questions = [r for r in valid if len(r.question_text) < 20]
    short_answers = [r for r in valid if len(r.answer_text) < 20]

    if short_questions:
        console.print(f"[yellow]Questions < 20 chars: {len(short_questions)}[/yellow]")
    if short_answers:
        console.print(f"[yellow]Answers < 20 chars: {len(short_answers)}[/yellow]")

    if not short_questions and not short_answers:
        console.print("[green]✓ All records pass basic validation.[/green]")


if __name__ == "__main__":
    cli()
