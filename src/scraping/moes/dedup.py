"""MoES ↔ Parliamentary Q&A cross-source dedup (ingestion integration stage).

Reuses ``check_moes_pq_overlap``'s matching/classification (EXACT_SHA via
SHA-256; TEXTUALLY_NEAR_IDENTICAL via calibrated directional containment). At
ingestion only **confirmed** duplicates are excluded:

  * ``EXACT_SHA`` — byte-identical file exists in the Parliamentary corpus;
  * ``TEXTUALLY_NEAR_IDENTICAL`` — containment >= ``near_identical_threshold``.

Everything uncertain is PRESERVED: ``POTENTIALLY_CORRESPONDING``,
``POTENTIALLY_UNIQUE`` and ``UNCOMPARABLE`` are never dropped. Dedup is
conservative by design — a false exclusion loses a document, so the default on
any computation failure is to exclude nothing (preserve all).

An auditable exclusion report (markdown) is produced whenever the stage runs.

The crawler is not touched: this is a downstream ingestion-only integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.scraping.moes.config import MoesConfigError
from src.scripts import check_moes_pq_overlap as ov
from src.utils.app_paths import config_path, data_dir

DEFAULT_CONFIG = config_path("moes_pq_dedup.yaml")
DEFAULT_REPORT = "moes_pq_dedup_report.md"

#: classes whose documents are CONFIRMED duplicates => excluded at ingestion
EXCLUDED_CLASSES = frozenset({ov.CLASS_EXACT, ov.CLASS_NEAR})


@dataclass(frozen=True)
class DedupThresholds:
    near_identical: float = 0.90
    related: float = 0.50


def load_thresholds(path: str | Path | None = None) -> DedupThresholds:
    """Read dedup thresholds from YAML (config/moes_pq_dedup.yaml by default)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        # Absent config = conservative defaults (near=0.90); never a hard error.
        return DedupThresholds()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    near = float(data.get("near_identical_threshold", 0.90))
    related = float(data.get("related_threshold", 0.50))
    if not (0.0 < related <= near <= 1.0):
        raise MoesConfigError(
            f"{cfg_path}: require 0 < related_threshold <= near_identical_threshold <= 1"
        )
    return DedupThresholds(near_identical=near, related=related)


@dataclass
class ExcludedDoc:
    """One MoES PQ document excluded as a confirmed duplicate."""

    key: str
    record_id: str
    rel_path: str
    filename: str
    klass: str
    score: float | None
    jaccard: float | None
    candidate: str
    reason: str

    @classmethod
    def from_verdict(cls, v: ov.Verdict) -> ExcludedDoc:
        return cls(
            key=v.doc.key,
            record_id=v.doc.record_id,
            rel_path=v.doc.rel_path,
            filename=Path(v.doc.rel_path).name,
            klass=v.klass,
            score=v.score,
            jaccard=v.jaccard,
            candidate=v.candidate or "",
            reason=v.reason,
        )


@dataclass
class DedupResult:
    """Outcome of one dedup computation over the MoES PQ press releases."""

    moes_root: Path
    parl_root: Path
    thresholds: DedupThresholds
    verdicts: list[ov.Verdict] = field(default_factory=list)
    excluded: list[ExcludedDoc] = field(default_factory=list)
    report_text: str = ""

    @property
    def excluded_filenames(self) -> set[str]:
        return {e.filename for e in self.excluded}

    @property
    def counts(self) -> dict[str, int]:
        c = dict.fromkeys(ov.CLASS_ORDER, 0)
        for v in self.verdicts:
            c[v.klass] += 1
        return c


def compute_exclusions(
    moes_root: Path,
    parl_root: Path,
    thresholds: DedupThresholds | None = None,
    *,
    top_k: int = 8,
    boilerplate_fraction: float = 0.5,
    shingle_size: int = 5,
) -> DedupResult:
    """Classify MoES PQ press releases against the Parliamentary corpus and keep
    the confirmed-duplicate exclusions. Never raises for a missing/empty corpus —
    it returns an empty result so ingestion degrades to preserving everything."""
    th = thresholds or load_thresholds()
    # normalize the parliamentary root to the house dir that holds session-*
    # (mirrors ov.resolve_parliamentary_root's handling of the rajya-sabha nest)
    parl_root = _house_dir(parl_root)
    moes_root = Path(moes_root)
    result = DedupResult(
        moes_root=moes_root, parl_root=parl_root, thresholds=th
    )
    if not moes_root.is_dir() or not parl_root.is_dir():
        return result
    try:
        moes_records = ov.load_moes_pq(moes_root)
        parl_records, sha_index = ov.load_parliamentary(parl_root)
    except Exception:  # noqa: BLE001 — safe default: preserve everything
        return result
    if not moes_records or not parl_records:
        return result
    try:
        verdicts, _stats = ov.classify(
            moes_records,
            parl_records,
            sha_index,
            near_threshold=th.near_identical,
            related_threshold=th.related,
            top_k=top_k,
            boilerplate_fraction=boilerplate_fraction,
            shingle_size=shingle_size,
        )
    except Exception:  # noqa: BLE001 — safe default: preserve everything
        return result
    result.verdicts = verdicts
    result.excluded = [
        ExcludedDoc.from_verdict(v) for v in verdicts if v.klass in EXCLUDED_CLASSES
    ]
    return result


def render_exclusion_report(
    result: DedupResult,
    *,
    moes_root: Path | None = None,
    parl_root: Path | None = None,
) -> str:
    """Auditable markdown report of what was excluded and what was preserved."""
    counts = result.counts
    moes = moes_root or result.moes_root
    parl = parl_root or result.parl_root
    lines: list[str] = [
        "# MoES ↔ Parliamentary Q&A dedup exclusion report",
        "",
        "Generated at ingestion. Only **confirmed** duplicates are excluded: "
        "EXACT_SHA and TEXTUALLY_NEAR_IDENTICAL. All other classes are preserved.",
        "",
        f"- MoES corpus root      : {moes}",
        f"- Parliamentary root   : {parl}",
        f"- near_identical_threshold : {result.thresholds.near_identical}",
        f"- related_threshold        : {result.thresholds.related}",
        f"- MoES PQ documents compared : {len(result.verdicts)}",
        f"- EXCLUDED (confirmed dup)  : {len(result.excluded)}",
        "  - " + ", ".join(f"{k}: {counts[k]}" for k in ov.CLASS_ORDER),
        "",
        "## Excluded documents",
        "",
    ]
    if not result.excluded:
        lines.append("(none — nothing was a confirmed duplicate)")
    for e in sorted(result.excluded, key=lambda x: x.rel_path):
        score = f"{e.score:.3f}" if e.score is not None else "-"
        lines.append(f"- `{e.rel_path}` [{e.klass}] containment={score}")
        lines.append(f"    record {e.record_id}; parliamentary: {e.candidate or '-'}")
        lines.append(f"    reason: {e.reason}")
    preserved = [
        v for v in result.verdicts if v.klass not in EXCLUDED_CLASSES
    ]
    lines += ["", "## Preserved documents", ""]
    if not preserved:
        lines.append("(none)")
    for v in sorted(preserved, key=lambda x: x.doc.rel_path):
        score = f"{v.score:.3f}" if v.score is not None else "-"
        lines.append(f"- `{v.doc.rel_path}` [{v.klass}] containment={score}")
    lines += ["", ov.DONE_LINE]
    return "\n".join(lines) + "\n"


def _resolve_moes_root(default: str | Path | None) -> Path:
    """Default MoES staging root WITHOUT raising SystemExit (missing -> path)."""
    if default is not None:
        return Path(default).expanduser()
    base = data_dir()
    for name in (".moes-website", "moes-website"):
        cand = base / name
        if cand.is_dir():
            return cand.resolve()
    return (base / ".moes-website").resolve()


def _house_dir(path: str | Path) -> Path:
    """Normalize a parliamentary root to the house dir that holds session-*
    (mirrors ov.resolve_parliamentary_root: a root pointing at parliamentary-qa
    or at the house dir both resolve to the house dir). Never raises for a
    missing root — returns the path as-is."""
    p = Path(path).expanduser()
    if (p / "rajya-sabha").is_dir():
        return (p / "rajya-sabha").resolve()
    if list(p.glob("session-*")):
        return p.resolve()
    return p.resolve()


def _resolve_parl_root(default: str | Path | None) -> Path:
    """Default parliamentary-qa root WITHOUT raising SystemExit (missing -> path)."""
    if default is not None:
        return _house_dir(default)
    return _house_dir(data_dir() / "parliamentary-qa")


def moes_website_dedup(
    *,
    moes_root: str | Path | None = None,
    parl_root: str | Path | None = None,
    report_path: str | Path | None = None,
    thresholds_path: str | Path | None = None,
) -> DedupResult:
    """High-level stage used by ingestion: resolve default roots, compute
    exclusions, render + persist the auditable report, and return the result.

    Safe by default: any missing corpus/root or computation failure yields an
    empty result (preserve everything) and skips writing a report. Never raises
    (even SystemExit) for absent corpora — ingestion must not break on dedup."""
    mroot = _resolve_moes_root(moes_root)
    proot = _resolve_parl_root(parl_root)
    th = load_thresholds(thresholds_path)
    result = compute_exclusions(mroot, proot, th)
    if not result.verdicts:
        return result
    result.report_text = render_exclusion_report(result, moes_root=mroot, parl_root=proot)
    dest = (Path(report_path).expanduser() if report_path else None) or (
        data_dir() / DEFAULT_REPORT
    )
    # never write inside a corpus root
    for root in (mroot, proot):
        try:
            dest.relative_to(root)
            dest = data_dir() / DEFAULT_REPORT
            break
        except ValueError:
            continue
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result.report_text, encoding="utf-8")
    except Exception:  # noqa: BLE001 — report is auxiliary; never fail ingestion
        pass
    return result
