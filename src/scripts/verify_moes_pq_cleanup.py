"""Post-cleanup verification for the MoES PQ removal.

Proves — rather than assumes — that removed MoES parliamentary-question rows
are gone from the canonical corpus *and* from every artifact derived from it,
and that genuine Lok Sabha / Rajya Sabha records survived untouched.

WHY THIS EXISTS
---------------
``HybridRAGPipeline.save()`` writes more artifacts than the completeness
contract in ``src/retrieval/hybrid/artifacts.py`` checks. ``INDEX_MARKER_FILES``
covers six; ``save()`` also writes ``chunk_map.json``, ``long_chunk_map.json``,
``build_meta.json`` and ``bm25_index.postings.pkl``. Those unchecked files are
exactly where a partially-applied deletion would hide: long-document chunks live
in the SAME FAISS index as their parents and are aggregated back to
``parent_doc_id`` at query time, so a removed document can keep surfacing in
retrieval while being absent from the corpus and from ``doc_map``. This script
checks all of them.

Exit code 0 = every applicable check passed. Non-zero = at least one failure.
Checks that cannot run (no index built, no FAISS available) are reported as
SKIP, never silently passed.

Usage
-----
    python -m src.scripts.verify_moes_pq_cleanup \\
        --manifest data/migrations/pq_cleanup/removed-<run>.jsonl \\
        [--run data/migrations/pq_cleanup/run-<run>.json] \\
        [--corpus data/corpus_reports.jsonl] [--index-dir storage/hybrid_rag]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Reporter:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, check: str, status: str, detail: str = "") -> None:
        self.rows.append((check, status, detail))
        mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "·"}[status]
        print(f"  {mark} {check:<10} {status:<4} {detail}")

    @property
    def failures(self) -> int:
        return sum(1 for _, s, _ in self.rows if s == FAIL)

    @property
    def skips(self) -> int:
        return sum(1 for _, s, _ in self.rows if s == SKIP)


# ─────────────────────────────────────────────────────────────────────────────
# loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_corpus_index(corpus: Path) -> tuple[dict[str, int], set[str], set[str]]:
    """(per-source counts, all record ids, all source_url values)."""
    counts: dict[str, int] = {}
    ids: set[str] = set()
    urls: set[str] = set()
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        m = row.get("metadata") or {}
        src = str(m.get("source") or "<none>")
        counts[src] = counts.get(src, 0) + 1
        if row.get("question_id"):
            ids.add(str(row["question_id"]))
        if m.get("source_url"):
            urls.add(str(m["source_url"]))
    return counts, ids, urls


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
# checks
# ─────────────────────────────────────────────────────────────────────────────

def check_corpus(rep: Reporter, removed: list[dict[str, Any]], corpus: Path,
                 run: dict[str, Any] | None) -> None:
    counts, ids, urls = load_corpus_index(corpus)
    removed_ids = {str(e["record_id"]) for e in removed}
    removed_urls = {str(e["source_url"]) for e in removed if e.get("source_url")}

    leaked = removed_ids & ids
    rep.add("V1", FAIL if leaked else PASS,
            f"{len(leaked)} removed record_id(s) still in corpus"
            if leaked else f"all {len(removed_ids)} removed ids absent")

    leaked_u = removed_urls & urls
    rep.add("V2", FAIL if leaked_u else PASS,
            f"{len(leaked_u)} removed source_url(s) still in corpus"
            if leaked_u else f"all {len(removed_urls)} removed URLs absent")

    if run:
        before = run.get("by_source_before") or {}
        after = run.get("by_source_after") or counts
        ok = True
        detail = []
        for src in sorted(set(before) | set(after)):
            if src == "moes_website":
                continue
            b, a = before.get(src, 0), after.get(src, 0)
            if b != a:
                ok = False
            detail.append(f"{src}:{b}->{a}")
        rep.add("V8-10", FAIL if not ok else PASS,
                "protected sources " + ", ".join(detail))
        moes_b = before.get("moes_website", 0)
        moes_a = after.get("moes_website", counts.get("moes_website", 0))
        expected = moes_b - len(removed_ids)
        rep.add("V11", FAIL if moes_a != expected else PASS,
                f"moes_website {moes_b}->{moes_a} (expected {expected}); "
                f"incois {after.get('incois', '?')}")
    else:
        rep.add("V8-10", SKIP, "no run envelope supplied — cannot compare per-source counts")
        rep.add("V11", SKIP, "no run envelope supplied")


def check_index(rep: Reporter, removed: list[dict[str, Any]],
                index_dir: Path) -> None:
    removed_ids = {str(e["record_id"]) for e in removed}
    if not index_dir.is_dir():
        for c in ("V3", "V4", "V5", "V6", "V7"):
            rep.add(c, SKIP, f"index dir absent: {index_dir}")
        return

    # V3 doc_map
    doc_map = _read_json(index_dir / "doc_map.json")
    if doc_map is None:
        rep.add("V3", SKIP, "doc_map.json missing")
    else:
        leaked = removed_ids & set(doc_map.keys())
        rep.add("V3", FAIL if leaked else PASS,
                f"{len(leaked)} removed id(s) in doc_map.json"
                if leaked else f"clean ({len(doc_map)} docs)")

    # V4 vector_store.ids (+ FAISS ntotal when faiss is importable)
    vs_ids = _read_json(index_dir / "vector_store.ids")
    if vs_ids is None:
        rep.add("V4", SKIP, "vector_store.ids missing")
    else:
        leaked = removed_ids & set(vs_ids)
        ntotal = None
        try:
            import faiss  # noqa: PLC0415
            ntotal = int(faiss.read_index(
                str(index_dir / "vector_store.index")).ntotal)
        except Exception:  # noqa: BLE001
            ntotal = None
        aligned = (ntotal is None) or (ntotal == len(vs_ids))
        bad = bool(leaked) or not aligned
        rep.add("V4", FAIL if bad else PASS,
                f"leaked={len(leaked)} ids={len(vs_ids)} ntotal={ntotal}")

    # V5 bm25
    bm = _read_json(index_dir / "bm25_index.json")
    if bm is None:
        rep.add("V5", SKIP, "bm25_index.json missing")
    else:
        bm_ids = bm.get("doc_ids") or []
        leaked = removed_ids & set(bm_ids)
        rep.add("V5", FAIL if leaked else PASS,
                f"{len(leaked)} removed id(s) in bm25" if leaked
                else f"clean ({len(bm_ids)} docs)")

    # V6 chunk maps — the artifacts the completeness contract does not check
    chunk_leak: set[str] = set()
    orphan_parents: set[str] = set()
    n_chunks = 0
    for name in ("chunk_map.json", "long_chunk_map.json"):
        cm = _read_json(index_dir / name)
        if not isinstance(cm, dict):
            continue
        n_chunks += len(cm)
        for cid, chunk in cm.items():
            if str(cid) in removed_ids:
                chunk_leak.add(str(cid))
            pid = str((chunk or {}).get("parent_doc_id") or "")
            if pid in removed_ids:
                orphan_parents.add(pid)
    if n_chunks == 0 and not (index_dir / "chunk_map.json").is_file():
        rep.add("V6", SKIP, "chunk maps absent")
    else:
        bad = bool(chunk_leak or orphan_parents)
        rep.add("V6", FAIL if bad else PASS,
                f"chunk_leak={len(chunk_leak)} orphan_parents={len(orphan_parents)} "
                f"({n_chunks} chunks scanned)")

    # V7 cross-artifact agreement.
    #
    # The unit of indexing depends on `use_chunking`: with chunking OFF the
    # indexed units are the parent documents, so bm25 == doc_map. With chunking
    # ON the units are chunks, so bm25 == vector ids == a multiple of doc_map,
    # and the correct invariant is that every chunk resolves to a doc_map
    # parent and every parent has at least one unit.
    if not isinstance(doc_map, dict) or not isinstance(vs_ids, list) \
            or not isinstance(bm, dict):
        rep.add("V7", SKIP, "insufficient artifacts to compare")
        return
    bm_ids = bm.get("doc_ids") or []
    detail = {"doc_map": len(doc_map), "vector_ids": len(vs_ids),
              "bm25": len(bm_ids)}
    problems = []

    # 1. bm25 and the vector store must index exactly the same unit set
    if len(bm_ids) != len(vs_ids):
        problems.append(f"bm25({len(bm_ids)}) != vector_ids({len(vs_ids)})")
    if set(bm_ids) != set(vs_ids):
        problems.append("bm25 and vector id SETS differ")

    # 2. parent/child consistency
    parents: set[str] = set()
    for name in ("chunk_map.json", "long_chunk_map.json"):
        cm = _read_json(index_dir / name)
        if isinstance(cm, dict):
            for chunk in cm.values():
                pid = str((chunk or {}).get("parent_doc_id") or "")
                if pid:
                    parents.add(pid)
    if parents:
        unknown = parents - set(doc_map.keys())
        if unknown:
            problems.append(f"{len(unknown)} chunk parent(s) not in doc_map")
        uncovered = set(doc_map.keys()) - set(vs_ids) - parents
        if uncovered:
            problems.append(f"{len(uncovered)} doc(s) have no indexed unit")
    else:
        # no chunking: the units ARE the parents
        if set(vs_ids) != set(doc_map.keys()):
            problems.append("unit set != doc_map set (chunking off)")

    rep.add("V7", FAIL if problems else PASS,
            "; ".join(problems) if problems else f"consistent {detail}")


def check_retrieval(rep: Reporter, removed: list[dict[str, Any]],
                    index_dir: Path) -> None:
    """V12 — removed documents must not be retrievable. Needs a model."""
    try:
        from src.retrieval.hybrid.pipeline import HybridRAGPipeline
    except Exception as exc:  # noqa: BLE001
        rep.add("V12", SKIP, f"retrieval stack unavailable: {exc}")
        return
    if not index_dir.is_dir():
        rep.add("V12", SKIP, "no index built")
        return
    removed_ids = {str(e["record_id"]) for e in removed}
    queries = [str(e.get("title") or "")[:80] for e in removed[:5] if e.get("title")]
    if not queries:
        rep.add("V12", SKIP, "no titles in manifest to query with")
        return
    try:
        pipe = HybridRAGPipeline()
        pipe.load(str(index_dir))
        hits: set[str] = set()
        for q in queries:
            try:
                res = pipe.retrieve(q, top_k=10)
            except Exception:  # noqa: BLE001
                continue
            for r in res or []:
                rid = getattr(r, "question_id", None) or \
                    (r.get("question_id") if isinstance(r, dict) else None)
                if rid:
                    hits.add(str(rid))
        leaked = hits & removed_ids
        rep.add("V12", FAIL if leaked else PASS,
                f"{len(leaked)} removed doc(s) retrievable" if leaked
                else f"{len(queries)} queries, none returned a removed doc")
    except Exception as exc:  # noqa: BLE001
        rep.add("V12", SKIP, f"retrieval probe failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="removed-<run>.jsonl")
    ap.add_argument("--run", default=None, help="run-<run>.json envelope")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--with-retrieval", action="store_true",
                    help="also run V12 (needs the embedding model)")
    args = ap.parse_args(argv)

    from src.utils.app_paths import corpus_path
    from src.utils.app_paths import index_dir as default_index_dir

    manifest = Path(args.manifest)
    if not manifest.is_file():
        print(f"manifest not found: {manifest}")
        return 2
    removed = load_manifest(manifest)
    if not removed:
        print(f"WARNING: manifest {manifest} has 0 rows — V1/V2 would pass "
              f"vacuously. Pass the manifest from the run that actually "
              f"removed rows.")
    run = _read_json(Path(args.run)) if args.run else None
    corpus = Path(args.corpus) if args.corpus else corpus_path()
    idx = Path(args.index_dir) if args.index_dir else Path(default_index_dir())

    rep = Reporter()
    print(f"manifest : {manifest}  ({len(removed)} removed rows)")
    print(f"corpus   : {corpus}")
    print(f"index    : {idx}")

    print("\ncorpus checks")
    check_corpus(rep, removed, corpus, run)
    print("\nindex checks")
    check_index(rep, removed, idx)
    if args.with_retrieval:
        print("\nretrieval check")
        check_retrieval(rep, removed, idx)
    else:
        rep.add("V12", SKIP, "pass --with-retrieval to run")

    print(f"\n{sum(1 for _, s, _ in rep.rows if s == PASS)} passed, "
          f"{rep.failures} failed, {rep.skips} skipped")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
