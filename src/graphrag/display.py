"""
Live status rendering for long-running GraphRAG builds.

A single multi-line panel (driven by ``rich.live.Live``) showing:

    Document: 512 / 1352 (37.9%)
    ██████████░░░░░░░░░░░░░░░░░  37.9%
    Current ID : 17-12-2138
    Provider   : groq
    Model      : qwen3.6-27b
    API Key    : ...91CD
    Elapsed    : 2h 18m 05s
    ETA        : 3h 47m 12s
    Nodes      : 2,458
    Relationships: 8,194
    Failures/Retries: 0 / 3

No Rich ``Progress`` is used here so the panel renders exactly once per
``Live`` refresh (no nested-refresh conflicts during a multi-hour run).
"""

from __future__ import annotations

import time
from typing import Optional

from rich.panel import Panel
from rich.table import Table

BAR_WIDTH = 34


def format_duration(seconds: float) -> str:
    """Format seconds as e.g. 2h 18m 05s (or 3m 05s / 45s for shorter runs)."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _bar(fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * BAR_WIDTH))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


class BuildStatusPanel:
    """Mutable state rendered as a live build-status panel."""

    def __init__(self, total: int, started: float) -> None:
        self.total = total
        self.started = started
        self.completed = 0
        self.current_id: Optional[str] = None
        self.provider: Optional[str] = None
        self.model: Optional[str] = None
        self.key: Optional[str] = None
        self.nodes = 0
        self.rels = 0
        self.failures = 0
        self.retries = 0
        self.last_event: Optional[str] = None

    def update(self, **kw) -> None:
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @property
    def pct(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.completed / self.total * 100.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def eta(self) -> Optional[float]:
        if self.completed <= 0:
            return None
        per_doc = self.elapsed / self.completed
        return per_doc * (self.total - self.completed)

    def render(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan", justify="right")
        table.add_column()

        pct = self.pct
        table.add_row(
            "Document",
            f"{self.completed:,} / {self.total:,} ({pct:.1f}%)",
        )
        table.add_row("", _bar(pct / 100.0))
        table.add_row("Current ID", self.current_id or "-")
        table.add_row("Provider", self.provider or "-")
        table.add_row("Model", self.model or "-")
        table.add_row("API Key", self.key or "-")
        table.add_row("Elapsed", format_duration(self.elapsed))
        eta = self.eta
        table.add_row("ETA", format_duration(eta) if eta is not None else "-")
        table.add_row("Nodes", f"{self.nodes:,}")
        table.add_row("Relationships", f"{self.rels:,}")
        table.add_row("Failures/Retries", f"{self.failures} / {self.retries}")
        if self.last_event:
            table.add_row("Last event", self.last_event[:90])

        return Panel(table, title="[bold]GraphRAG Build[/bold]", border_style="cyan")
