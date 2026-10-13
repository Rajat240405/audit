#!/usr/bin/env python3
"""Benchmark CrossEncoder prediction batch sizes (RERANK_BATCH_SIZE) on CPU.

Two subcommands:

``capture``  Run ONE real retrieval through the real pipeline and save the
             query plus the candidate texts that the reranker received, as a
             fixed fixture. Read-only: it loads the index and runs retrieval.
             The fixture CONTAINS DOCUMENT TEXT. It is written with 0600
             permissions. Keep it out of git.

``bench``    Time ``CrossEncoderReranker.rerank`` on a fixed fixture for each
             batch size. Uses the existing timing fields (``candidate_count``,
             ``model_access_ms``, ``model_predict_ms``). Each batch size gets
             one untimed warm-up call. The repeats then interleave the batch
             sizes so drift affects them equally. Reports median prediction
             latency per batch size and checks that all pairs are scored and
             that the top-k ranking is identical across batch sizes.
             Without ``--fixture`` it uses a SYNTHETIC fixture, which is
             labelled as such. Synthetic results do not predict production
             speed.

Nothing is written to the index, the database or the application state. Only
the optional ``--json-out`` file and the optional ``capture`` fixture are
written. Neither the query nor any document text is printed.

Run on the HPC CPU node with the same thread settings as the server, for
example::

    TORCH_INTRA_OP_THREADS=8 TORCH_INTER_OP_THREADS=1 \\
    python tools/bench_rerank_batch.py bench --fixture /path/fixture.json \\
        --batch-sizes 8,16,32,50 --repeats 7 --json-out /path/out.json

The model defaults to the one the server resolves (``GRAPHRAG_RERANK_MODEL``,
or the local ``bge-reranker-v2-m3`` folder). Pass ``--model`` to override it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SYNTHETIC_NOTE = (
    "SYNTHETIC fixture: random vocabulary text, NOT real retrieved candidates. "
    "Use it only to check the harness. Do not use it to predict production speed."
)
DEFAULT_BATCH_SIZES = (8, 16, 32, 50)
DEFAULT_REPEATS = 5
TOP_K = 5


def _load_reranker_module():
    """Load reranker.py by path. Importing the package would pull in faiss."""
    path = REPO / "src/retrieval/hybrid/reranker.py"
    spec = importlib.util.spec_from_file_location("_reranker_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def load_fixture(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    query = data.get("query")
    doc_ids = data.get("doc_ids")
    docs = data.get("docs")
    if not isinstance(query, str) or not query.strip():
        raise SystemExit("fixture: 'query' must be a non-empty string")
    if not isinstance(doc_ids, list) or not isinstance(docs, list) or len(doc_ids) != len(docs):
        raise SystemExit("fixture: 'doc_ids' and 'docs' must be lists of equal length")
    if not doc_ids:
        raise SystemExit("fixture: no candidates")
    if len(set(doc_ids)) != len(doc_ids):
        raise SystemExit("fixture: doc_ids must be unique")
    return {
        "source": "captured",
        "path": str(path),
        "query": query,
        "doc_ids": [str(d) for d in doc_ids],
        "docs": [str(d) for d in docs],
    }


def synthetic_fixture(n: int = 50, seed: int = 1) -> dict:
    """Deterministic fixed-length-ish text. Labelled SYNTHETIC in all output."""
    rng = random.Random(seed)
    vocab = (
        "audit finding control procedure report budget ministry fund compliance "
        "expenditure revenue department certificate review sample transaction "
        "approval register ledger variance recommendation response period annual "
        "disbursement grant project financial statement observation risk"
    ).split()
    docs = []
    for _ in range(n):
        length = rng.randint(60, 380)  # roughly 60–380 words, similar to a chunk
        docs.append(" ".join(rng.choice(vocab) for _ in range(length)))
    return {
        "source": "synthetic",
        "path": None,
        "query": "synthetic benchmark question about audit findings",
        "doc_ids": [f"synthetic-{i:03d}" for i in range(n)],
        "docs": docs,
    }


def capture_fixture(out: Path, question: str) -> None:
    """Record the real candidate texts from one real retrieval. Read-only."""
    from src.utils.app_paths import index_dir
    from src.utils.torch_threads import apply_torch_thread_limits, format_thread_report

    print(format_thread_report(apply_torch_thread_limits()))
    from src.retrieval.hybrid.pipeline import HybridRAGPipeline

    pipe = HybridRAGPipeline()
    idx = index_dir()
    if not idx.exists():
        raise SystemExit(f"index directory not found: {idx}")
    pipe.load(idx)

    captured: dict = {}
    original_rerank = pipe.reranker.rerank

    def recording_rerank(*args, **kwargs):
        if not captured:  # first rerank call only: the real query's candidates
            cands = kwargs["candidates"]
            texts = kwargs.get("doc_texts") or {}
            ids = [c[0] for c in cands]
            captured.update(
                query=kwargs["query"],
                doc_ids=ids,
                docs=[texts.get(i, "") for i in ids],
            )
        return original_rerank(*args, **kwargs)

    pipe.reranker.rerank = recording_rerank
    pipe.retrieve(question)
    if not captured:
        raise SystemExit("retrieval did not reach the reranker (no fixture written)")

    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(captured, fh, ensure_ascii=False)
    os.chmod(out, 0o600)
    print(
        f"wrote fixture: candidates={len(captured['doc_ids'])} path={out} "
        "(contains document text: keep out of git)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────


def _parse_batch_sizes(raw: str) -> list[int]:
    mod = _load_reranker_module()
    sizes = []
    for part in raw.split(","):
        value = mod.parse_rerank_batch_size(part)
        if value is None:
            raise SystemExit(f"invalid batch size in --batch-sizes: {part!r}")
        sizes.append(value)
    if not sizes:
        raise SystemExit("--batch-sizes is empty")
    return sizes


def run_benchmark(fixture: dict, model_name: str, batch_sizes: list[int], repeats: int) -> dict:
    from src.utils.torch_threads import apply_torch_thread_limits, format_thread_report

    thread_report = apply_torch_thread_limits()
    mod = _load_reranker_module()
    reranker = mod.CrossEncoderReranker(
        model_name=model_name, device="cpu", batch_size=batch_sizes[0]
    )

    # Model load is timed on its own. It is not part of the per-batch numbers.
    t0 = time.perf_counter()
    model = reranker.model
    load_ms = (time.perf_counter() - t0) * 1000

    # Count every pair that reaches predict(), and the batch_size it receives.
    seen: list[tuple[int, int | None]] = []
    original_predict = model.predict

    def counting_predict(pairs, *args, **kwargs):
        seen.append((len(pairs), kwargs.get("batch_size")))
        return original_predict(pairs, *args, **kwargs)

    model.predict = counting_predict

    query = fixture["query"]
    doc_ids = fixture["doc_ids"]
    doc_texts = dict(zip(doc_ids, fixture["docs"]))
    candidates = [(d, 1.0 - i * 0.001) for i, d in enumerate(doc_ids)]
    n = len(candidates)

    per_size: dict[int, dict] = {
        b: {"predict_ms": [], "total_ms": [], "access_ms": [], "top": None,
            "pairs_scored": set(), "batch_sizes_seen": set()}
        for b in batch_sizes
    }

    def one_call(b: int) -> tuple[dict, list, float]:
        reranker.batch_size = b
        seen.clear()
        info: dict = {}
        t = time.perf_counter()
        top = reranker.rerank(query, candidates, k=TOP_K, doc_texts=doc_texts, timing=info)
        total_ms = (time.perf_counter() - t) * 1000
        if info.get("candidate_count") != n:
            raise SystemExit(f"candidate_count mismatch for batch {b}: {info.get('candidate_count')}")
        if len(seen) != 1:
            raise SystemExit(f"expected one predict() call for batch {b}, got {len(seen)}")
        pairs_n, passed_b = seen[0]
        if pairs_n != n:
            raise SystemExit(f"batch {b}: predict scored {pairs_n} pairs, expected {n}")
        if passed_b != b:
            raise SystemExit(f"batch {b}: predict received batch_size={passed_b}")
        st = per_size[b]
        st["pairs_scored"].add(pairs_n)
        st["batch_sizes_seen"].add(passed_b)
        return info, top, total_ms

    # Untimed warm-up for each batch size.
    for b in batch_sizes:
        one_call(b)

    for _rep in range(repeats):
        for b in batch_sizes:  # interleaved, so drift affects every size equally
            info, top, total_ms = one_call(b)
            st = per_size[b]
            st["predict_ms"].append(info["model_predict_ms"])
            st["access_ms"].append(info["model_access_ms"])
            st["total_ms"].append(total_ms)
            st["top"] = top

    # Ranking check: top-k IDs must match across batch sizes. Scores must agree
    # to float tolerance (batching can change padding and rounding slightly).
    ref_b = batch_sizes[0]
    ref_top = per_size[ref_b]["top"]
    ids_identical = all(
        [d for d, _ in per_size[b]["top"]] == [d for d, _ in ref_top] for b in batch_sizes
    )
    max_score_diff = max(
        (abs(s1 - s2) for b in batch_sizes for (_, s1), (_, s2) in zip(per_size[b]["top"], ref_top)),
        default=0.0,
    )

    rows = []
    for b in batch_sizes:
        st = per_size[b]
        rows.append({
            "batch_size": b,
            "predict_ms_median": statistics.median(st["predict_ms"]),
            "predict_ms_min": min(st["predict_ms"]),
            "predict_ms_max": max(st["predict_ms"]),
            "total_ms_median": statistics.median(st["total_ms"]),
            "model_access_ms_median": statistics.median(st["access_ms"]),
            "predict_batches_per_call": math.ceil(n / b),
            "pairs_scored_per_call": sorted(st["pairs_scored"]),
            "batch_size_passed": sorted(st["batch_sizes_seen"]),
            "samples": len(st["predict_ms"]),
        })

    return {
        "data_source": fixture["source"],
        "fixture_path": fixture["path"],
        "note": SYNTHETIC_NOTE if fixture["source"] == "synthetic" else "captured from a real retrieval",
        "model": model_name,
        "device": "cpu",
        "thread_report": format_thread_report(thread_report),
        "candidate_count": n,
        "repeats": repeats,
        "model_load_ms": load_ms,
        "top_k": TOP_K,
        "top_k_ids_identical_across_batch_sizes": ids_identical,
        "top_k_max_score_abs_diff": max_score_diff,
        "results": rows,
    }


def print_report(report: dict) -> None:
    print("Reranker batch-size benchmark (CPU)")
    print(f"  data        : {report['data_source']}"
          + (f" ({report['fixture_path']})" if report["fixture_path"] else ""))
    if report["data_source"] == "synthetic":
        print(f"  NOTE        : {report['note']}")
    print(f"  candidates  : {report['candidate_count']} (all scored on every call)")
    print(f"  model       : {report['model']}")
    print(f"  threads     : {report['thread_report']}")
    print(f"  repeats     : {report['repeats']} timed per batch size, after 1 warm-up call")
    print(f"  model load  : {report['model_load_ms']:.0f} ms (not included below)")
    print()
    header = (f"{'batch':>6} {'predict med ms':>15} {'predict min':>12} {'predict max':>12} "
              f"{'total med ms':>13} {'batches/call':>13}")
    print(header)
    print("-" * len(header))
    for r in report["results"]:
        print(f"{r['batch_size']:>6} {r['predict_ms_median']:>15.1f} {r['predict_ms_min']:>12.1f} "
              f"{r['predict_ms_max']:>12.1f} {r['total_ms_median']:>13.1f} "
              f"{r['predict_batches_per_call']:>13}")
    print()
    print(f"top-{report['top_k']} ids identical across batch sizes: "
          f"{report['top_k_ids_identical_across_batch_sizes']}")
    print(f"top-{report['top_k']} max score difference:             "
          f"{report['top_k_max_score_abs_diff']:.3g}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="save real candidate texts from one retrieval")
    cap.add_argument("--question", required=True, help="a representative audit question")
    cap.add_argument("--out", required=True, type=Path, help="fixture path (keep out of git)")

    bench = sub.add_parser("bench", help="time batch sizes on a fixed fixture")
    bench.add_argument("--fixture", type=Path, default=None,
                       help="captured fixture JSON. Omit to use the SYNTHETIC fixture.")
    bench.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    bench.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    bench.add_argument("--model", default=None, help="default: the server's resolved reranker model")
    bench.add_argument("--json-out", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "capture":
        capture_fixture(args.out, args.question)
        return 0

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    fixture = load_fixture(args.fixture) if args.fixture else synthetic_fixture()
    if args.model:
        model_name = args.model
    else:
        model_name = _load_reranker_module().resolve_rerank_model()
    report = run_benchmark(fixture, model_name, batch_sizes, args.repeats)
    print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
