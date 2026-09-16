"""Stage 10 — retrieval validation battery.

Runs labelled queries through the **production** retrieval stack and reports
per-case outcomes. There is no second retrieval engine here: the battery calls
``HybridRAGPipeline.retrieve`` on whatever pipeline it is given, so BGE-M3 +
FAISS + BM25 + RRF + cross-encoder reranking are exercised exactly as they are
in production.

The point of this module is *epistemic*, not mechanical. Spec §10's headline
numbers came from an int8 ONNX embedder at 160 tokens; production is fp32 BGE-M3
at 512. Those are not the same measurement, and the battery refuses to let them
be confused:

    ``stub``        a stand-in embedder or no reranker. Proves wiring only.
                    Says nothing about retrieval quality.
    ``prototype``   int8 / truncated / ONNX variants. Comparable to the spec's
                    published tables, NOT to production.
    ``production``  real BGE-M3 (1024-d, fp32) AND a loaded cross-encoder.
                    The only tier whose numbers may be quoted as production.

``--require-production`` makes that a hard gate: on a sandbox with a stub
embedder the battery exits non-zero instead of printing numbers that could be
mistaken for a real result.

Nothing here writes an index. It loads one read-only, or is handed a pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

TIER_STUB = "stub"
TIER_PROTOTYPE = "prototype"
TIER_PRODUCTION = "production"

#: BGE-M3's real output dimensionality. A different dim means a different model.
BGE_M3_DIM = 1024
#: Substrings that identify the production embedder.
PRODUCTION_EMBED_HINTS = ("bge-m3", "baai/bge-m3")
#: Substrings that identify an int8 / ONNX / quantised variant. These ARE
#: comparable to the spec's published tables, so they get their own tier.
PROTOTYPE_HINTS = ("int8", "onnx", "quant", "q8", "truncat")
#: Substrings that identify a test stand-in. Checked FIRST: a fake embedder is
#: not a quantised model and must never be reported as comparable to the
#: prototype measurements, let alone to production.
STUB_HINTS = ("fake", "stub", "dummy", "mock", "test", "random")


@dataclass
class BatteryCase:
    """One labelled retrieval expectation.

    ``expect_in_top_k`` is a substring that must appear in the returned text of
    at least one of the top ``k`` results — deliberately coarse, because a
    battery that asserts exact ranking is a battery that breaks on any
    legitimate model change.
    """

    case_id: str
    query: str
    expect_in_top_k: str
    k: int = 5
    note: str = ""


@dataclass
class BatteryResult:
    case_id: str
    query: str
    expected: str
    k: int
    passed: bool
    hit_rank: int | None
    top_ids: list[str] = field(default_factory=list)


@dataclass
class BatteryReport:
    tier: str
    tier_reason: str
    embed_model: str | None
    embed_dim: int | None
    reranker_loaded: bool
    total: int
    passed: int
    results: list[BatteryResult] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rate"] = round(self.rate, 4)
        data["quotable_as_production"] = self.tier == TIER_PRODUCTION
        return data


def detect_tier(pipeline: Any) -> tuple[str, str]:
    """Classify what a pipeline can actually prove. Returns ``(tier, reason)``.

    Conservative by construction: anything ambiguous resolves to a weaker tier.
    """
    embedder = getattr(pipeline, "embedder", None)
    model = str(getattr(embedder, "model_name", "") or "").lower()
    try:
        dim = int(getattr(pipeline, "_embedding_dim", 0) or 0)
    except (TypeError, ValueError):
        dim = 0

    reranker = getattr(pipeline, "reranker", None)
    use_reranker = bool(getattr(pipeline, "use_reranker", False))
    reranker_loaded = use_reranker and reranker is not None and getattr(
        reranker, "model", None
    ) is not None

    if not model:
        return TIER_STUB, "no embedder model name (stand-in or unloaded)"
    if any(hint in model for hint in STUB_HINTS):
        return TIER_STUB, f"embedder {model!r} is a test stand-in, not a real model"
    if any(hint in model for hint in PROTOTYPE_HINTS):
        return TIER_PROTOTYPE, f"embedder {model!r} is a quantised/prototype variant"
    if dim and dim != BGE_M3_DIM:
        return TIER_PROTOTYPE, f"embedding dim {dim} != {BGE_M3_DIM} (truncated or other model)"
    if not any(hint in model for hint in PRODUCTION_EMBED_HINTS):
        return TIER_STUB, f"embedder {model!r} is not the production BGE-M3"
    if not reranker_loaded:
        return TIER_STUB, "cross-encoder reranker not loaded; production ranking unproven"
    return TIER_PRODUCTION, f"BGE-M3 ({model}, dim={dim}) with cross-encoder reranker"


def _result_text(result: Any) -> str:
    """Best-effort text from a ``RetrievedResult``."""
    for attr in ("answer_text", "text", "content", "document_content"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value:
            return value
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("answer_text", "text", "chunk_text"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _result_id(result: Any) -> str:
    for attr in ("document_id", "doc_id", "id", "question_id"):
        value = getattr(result, attr, None)
        if value:
            return str(value)
    return "?"


def run_case(pipeline: Any, case: BatteryCase) -> BatteryResult:
    """Run one case through the real ``retrieve`` path."""
    try:
        results = pipeline.retrieve(case.query, top_k=case.k)
    except Exception as exc:  # noqa: BLE001 - one bad case must not kill a battery
        return BatteryResult(
            case_id=case.case_id, query=case.query, expected=case.expect_in_top_k,
            k=case.k, passed=False, hit_rank=None, top_ids=[f"ERROR: {exc}"],
        )

    top_ids = [_result_id(r) for r in results]
    needle = case.expect_in_top_k.lower()
    hit_rank = None
    for rank, result in enumerate(results, start=1):
        if needle in _result_text(result).lower():
            hit_rank = rank
            break

    return BatteryResult(
        case_id=case.case_id, query=case.query, expected=case.expect_in_top_k,
        k=case.k, passed=hit_rank is not None, hit_rank=hit_rank, top_ids=top_ids,
    )


def run_battery(
    pipeline: Any, cases: list[BatteryCase] | None = None
) -> BatteryReport:
    """Run every case and label the tier honestly."""
    tier, reason = detect_tier(pipeline)
    embedder = getattr(pipeline, "embedder", None)
    cases = cases if cases is not None else default_cases()

    results = [run_case(pipeline, case) for case in cases]
    return BatteryReport(
        tier=tier,
        tier_reason=reason,
        embed_model=str(getattr(embedder, "model_name", "") or "") or None,
        embed_dim=getattr(pipeline, "_embedding_dim", None),
        reranker_loaded=bool(
            getattr(pipeline, "use_reranker", False)
            and getattr(getattr(pipeline, "reranker", None), "model", None) is not None
        ),
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        results=results,
    )


def default_cases() -> list[BatteryCase]:
    """The battery defined by the spec's retrieval expectations.

    These target INCOIS content that V2 is meant to make retrievable: a table
    caption, a rotated-table schedule, a figure caption, and a page-level
    fact that only OCR recovery can surface. They are intentionally phrased
    the way a user would ask, not the way a document is written.
    """
    return [
        BatteryCase(
            "T1-table-caption",
            "How many research vessels does INCOIS operate?",
            "vessel",
            note="table content that flattening previously buried",
        ),
        BatteryCase(
            "T2-rotated-schedule",
            "What are the schedule receipts for the fiscal year?",
            "schedule",
            note="rot90 content, only reachable via the rotated lattice path",
        ),
        BatteryCase(
            "T3-figure-caption",
            "Show the sea surface temperature anomaly figure.",
            "figure",
            note="figure card caption",
        ),
        BatteryCase(
            "T4-scan-ocr",
            "What observations were recorded by the buoy network?",
            "observation",
            note="image-only page recovered by OCR",
        ),
        BatteryCase(
            "T5-negative",
            "What is the capital of France?",
            "paris",
            note="out-of-domain control: expected to MISS on an INCOIS-only index",
        ),
    ]


def load_pipeline(index_path: str, enable_v2: bool = False) -> Any:
    """Load an index read-only. Never builds, never saves."""
    from src.retrieval.hybrid.artifacts import index_is_complete
    from src.retrieval.hybrid.pipeline import HybridRAGPipeline

    if not index_is_complete(index_path):
        raise SystemExit(
            f"no complete index at {index_path!s}. This tool never builds an index; "
            f"run src.data.v2.reindex (V2) or the normal ingest (production) first."
        )
    pipeline = HybridRAGPipeline(use_reranker=True, enable_v2_children=enable_v2)
    pipeline.load(index_path)
    return pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the INCOIS V2 retrieval battery on the production stack."
    )
    parser.add_argument("--index", default="storage/hybrid_rag_v2", help="index to load")
    parser.add_argument("--v2", action="store_true", help="enable V2 child units")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--require-production",
        action="store_true",
        help="exit non-zero unless real BGE-M3 + cross-encoder are loaded",
    )
    args = parser.parse_args(argv)

    pipeline = load_pipeline(args.index, enable_v2=args.v2)
    report = run_battery(pipeline)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        data = report.to_dict()
        for key in ("tier", "tier_reason", "embed_model", "embed_dim",
                    "reranker_loaded", "total", "passed", "rate",
                    "quotable_as_production"):
            print(f"{key:24} {data[key]}")
        print()
        for result in report.results:
            mark = "PASS" if result.passed else "MISS"
            rank = f"@{result.hit_rank}" if result.hit_rank else ""
            print(f"  {mark:4} {result.case_id:20} {rank:4} {result.query}")
        if report.tier != TIER_PRODUCTION:
            print(
                f"\nWARNING: tier={report.tier} — these numbers are NOT "
                f"production-equivalent ({report.tier_reason})."
            )

    if args.require_production and report.tier != TIER_PRODUCTION:
        print(
            f"REFUSED: --require-production but tier={report.tier} "
            f"({report.tier_reason})",
            file=sys.stderr,
        )
        return 3
    return 0


__all__ = [
    "BGE_M3_DIM",
    "TIER_PRODUCTION",
    "TIER_PROTOTYPE",
    "TIER_STUB",
    "BatteryCase",
    "BatteryReport",
    "BatteryResult",
    "default_cases",
    "detect_tier",
    "run_battery",
    "run_case",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
