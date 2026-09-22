"""Three-tier matching of a user query against saved knowledge questions.

Cascade
-------
1. **Exact normalised-question match** — cheap, deterministic, no embedding work.
2. **Semantic match over SAVED QUESTIONS** using the existing BGE-M3 embedder.
   Matching is on the *question*, not the answer: the saved question is what
   expresses the information need the current user is asking about.
3. **Ambiguity gate** — a match is only surfaced when it clears an absolute
   threshold *and* leads the runner-up by a margin.

The gate is the safety mechanism. A high absolute score with a small margin
means two different saved answers both plausibly fit, and picking one is a
guess. In that case we suppress the panel entirely and the user simply gets the
normal RAG answer — when uncertain, say nothing rather than mislead.

``difflib.SequenceMatcher`` is retained only as a fallback for when semantic
matching is unavailable (model failed to load, ``KNOWLEDGE_SEMANTIC=0``). It is
a character-level metric and must not be the primary decision mechanism.
"""
from __future__ import annotations

import difflib
import os
from collections.abc import Callable, Sequence

from src.retrieval.knowledge.store import KnowledgeStore, normalize_question

__all__ = [
    "knowledge_config",
    "match_knowledge",
    "sync_question_embeddings",
]


def _float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def knowledge_config() -> dict:
    """All tuning knobs, read fresh so they can be changed without a rebuild.

    ``sim_threshold`` / ``sim_margin`` defaults are starting points, not
    measured optima — calibrate against real questions before trusting them.
    """
    return {
        "sim_threshold": min(max(_float_env("KNOWLEDGE_SIM_THRESHOLD", 0.80), 0.0), 1.0),
        "sim_margin": max(_float_env("KNOWLEDGE_SIM_MARGIN", 0.05), 0.0),
        "top_k": _int_env("KNOWLEDGE_TOP_K", 5),
        "semantic": _bool_env("KNOWLEDGE_SEMANTIC", True),
        "fuzzy_threshold": min(max(_float_env("KNOWLEDGE_FUZZY_THRESHOLD", 0.85), 0.0), 1.0),
    }


def _result(matches: list, tier, ambiguous: bool, diagnostics: dict) -> dict:
    """One shape for every return path.

    ``found`` is a real key, not a property: callers (and the wire format) do
    ``result["found"]``, and a dict lookup never consults a property.
    """
    return {"matches": matches, "tier": tier, "ambiguous": bool(ambiguous),
            "found": bool(matches), "diagnostics": diagnostics or {}}


def sync_question_embeddings(store: KnowledgeStore,
                             embed_batch: Callable[[Sequence[str]], object]) -> int:
    """(Re)build ``questions.f32`` so every active record has a saved-question vector.

    Rebuilds wholesale rather than appending: the matrix rows must line up with
    the index, and a partial update is the kind of thing that silently drifts.
    At the expected scale (hundreds to low thousands of records) a batched
    rebuild is cheap and always correct.

    Returns the number of rows written.
    """
    index = store.load_index()
    if not index:
        return 0
    questions = [e.get("q_norm") or "" for e in index]
    matrix = embed_batch(questions)
    if matrix is None:
        return 0
    store.save_embeddings(matrix, index)
    return len(index)


def _group_by_content(records: list[dict], scores: dict[str, float],
                      tier: str, top_k: int) -> list[dict]:
    """Collapse records sharing a content_hash into display cards, ranked by score."""
    groups: dict[str, dict] = {}
    for rec in records:
        key = rec.get("content_hash") or rec["knowledge_id"]
        score = scores.get(rec["knowledge_id"], 0.0)
        g = groups.get(key)
        if g is None:
            groups[key] = g = {
                "content_hash": key,
                "question": rec.get("question", ""),
                "question_normalized": rec.get("question_normalized", ""),
                "answer": rec.get("answer", ""),
                "sources": rec.get("sources", []),
                "knowledge_ids": [],
                "contributors": [],
                "created_at": rec.get("created_at", ""),
                "updated_at": rec.get("updated_at", ""),
                "score": score,
                "tier": tier,
            }
        g["knowledge_ids"].append(rec["knowledge_id"])
        name = rec.get("owner_name") or "unknown"
        if name not in g["contributors"]:
            g["contributors"].append(name)
        # Sorted so the displayed list is identical on every request. Records
        # created in the same second tie on created_at and would otherwise fall
        # back to random UUID order, making "Saved by …" flicker.
        g["contributors"] = sorted(g["contributors"], key=str.lower)
        g["score"] = max(g["score"], score)
        g["updated_at"] = max(g["updated_at"] or "", rec.get("updated_at") or "")
    ranked = sorted(groups.values(), key=lambda g: (-g["score"], g["updated_at"] or ""))
    return ranked[:top_k]


def _exact_tier(store: KnowledgeStore, q_norm: str, top_k: int) -> list[dict] | None:
    hits = [e for e in store.load_index() if e.get("q_norm") == q_norm
            and e.get("lifecycle", "active") == "active"]
    if not hits:
        return None
    ids = {e["knowledge_id"] for e in hits}
    records = store.records_for_ids(ids)
    scores = {r["knowledge_id"]: 1.0 for r in records}
    return _group_by_content(records, scores, "exact", top_k)


def _semantic_tier(store: KnowledgeStore, query_vector, top_k: int,
                   threshold: float, margin: float) -> tuple[list[dict] | None, dict]:
    """Cosine similarity against saved-question embeddings, with the ambiguity gate."""
    import numpy as np

    diag: dict = {"semantic": True}
    matrix = store.load_embeddings()
    index = [e for e in store.load_index() if e.get("lifecycle", "active") == "active"]
    if matrix is None or matrix.shape[0] != len(index) or len(index) == 0:
        diag["reason"] = "no usable embedding matrix"
        return None, diag

    q = np.asarray(query_vector, dtype="float32").reshape(-1)
    if q.size != matrix.shape[1]:
        diag["reason"] = f"dimension mismatch: query {q.size} vs index {matrix.shape[1]}"
        return None, diag

    # Embeddings are L2-normalised by the embedder, so the dot product is cosine.
    sims = matrix @ q

    # Only rows with a CURRENT embedding participate. An edited record has
    # has_embedding=False until its question is re-embedded; its old row still
    # physically exists in the matrix but now describes a different question,
    # so matching on it would be matching on stale text.
    participating = [(row, entry) for row, entry in enumerate(index)
                     if entry.get("has_embedding")]

    best_per_group: dict[str, tuple[float, dict]] = {}
    for row, entry in participating:
        gid = entry.get("content_hash") or entry["knowledge_id"]
        score = float(sims[row])
        cur = best_per_group.get(gid)
        if cur is None or score > cur[0]:
            best_per_group[gid] = (score, entry)

    ordered = sorted(best_per_group.values(), key=lambda t: -t[0])
    if not ordered:
        diag["reason"] = "no records with current embeddings"
        return None, diag

    best_score = ordered[0][0]
    second_score = ordered[1][0] if len(ordered) > 1 else -1.0
    diag.update({
        "best": round(best_score, 4),
        "second": round(second_score, 4),
        "gap": round(best_score - second_score, 4),
        "threshold": threshold,
        "margin": margin,
    })

    if best_score < threshold:
        diag["reason"] = "below threshold"
        return None, diag
    if (best_score - second_score) < margin:
        diag["reason"] = "ambiguous: margin too small"
        return None, diag

    wanted = {(t[1].get("content_hash") or t[1]["knowledge_id"]) for t in ordered[:top_k]}
    ids = {e["knowledge_id"] for e in index
           if (e.get("content_hash") or e["knowledge_id"]) in wanted}
    records = store.records_for_ids(ids)
    scores = {e["knowledge_id"]: float(sims[i]) for i, e in enumerate(index)}
    return _group_by_content(records, scores, "semantic", top_k), diag


def _fuzzy_tier(store: KnowledgeStore, q_norm: str, top_k: int,
                threshold: float) -> tuple[list[dict] | None, dict]:
    """Fallback only: character-level SequenceMatcher over saved questions."""
    diag: dict = {"semantic": False, "threshold": threshold}
    index = [e for e in store.load_index() if e.get("lifecycle", "active") == "active"]
    if not index:
        return None, diag
    scored = []
    for e in index:
        ratio = difflib.SequenceMatcher(None, q_norm, e.get("q_norm") or "").ratio()
        scored.append((ratio, e))
    scored.sort(key=lambda t: -t[0])
    best = scored[0][0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    diag.update({"best": round(best, 4), "second": round(second, 4)})
    if best < threshold:
        diag["reason"] = "below fuzzy threshold"
        return None, diag
    # A LIST, not a set: the entries are dicts and dicts are unhashable, so a
    # set comprehension here raises TypeError — which the caller's blanket
    # handler would swallow, silently disabling the fuzzy fallback.
    keep = [(r, e) for r, e in scored if r >= threshold]
    ids = {e["knowledge_id"] for _, e in keep}
    records = store.records_for_ids(ids)
    scores = {e["knowledge_id"]: r for r, e in keep}
    return _group_by_content(records, scores, "fuzzy", top_k), diag


def match_knowledge(store: KnowledgeStore, query: str, *,
                    query_vector=None,
                    config: dict | None = None) -> dict:
    """Find saved answers relevant to ``query``.

    ``query_vector`` SHOULD be supplied: it is the same BGE-M3 vector dense
    retrieval already computed for this request. Re-embedding the query here
    would double the most expensive part of every request, so the caller is
    expected to thread it through (see ``pipeline.retrieve``).

    Never raises: knowledge lookup must not be able to break a real query.
    """
    cfg = config or knowledge_config()
    empty = _result([], None, False, {})
    try:
        q = (query or "").strip()
        if not q:
            return empty
        q_norm = normalize_question(q)
        top_k = cfg["top_k"]

        exact = _exact_tier(store, q_norm, top_k)
        if exact:
            return _result(exact, "exact", False, {"tier": "exact"})

        semantic_diag: dict = {}
        if cfg["semantic"] and query_vector is not None:
            try:
                matches, diag = _semantic_tier(
                    store, query_vector, top_k,
                    cfg["sim_threshold"], cfg["sim_margin"])
                if matches:
                    return _result(matches, "semantic", False, diag)
                if diag.get("reason", "").startswith("ambiguous"):
                    return _result([], None, True, diag)
                semantic_diag = diag
            except Exception as e:  # noqa: BLE001 — fall through to fuzzy
                semantic_diag = {"semantic": True, "error": f"{type(e).__name__}: {e}"}

        matches, diag = _fuzzy_tier(store, q_norm, top_k, cfg["fuzzy_threshold"])
        # Keep the semantic tier's verdict visible: it is what you need in order
        # to calibrate the threshold, and burying it under the fuzzy fallback
        # makes a suppressed match impossible to diagnose.
        if semantic_diag:
            diag = {"semantic_tier": semantic_diag, **diag}
        if matches:
            return _result(matches, "fuzzy", False, diag)
        return _result([], None, False, diag)
    except Exception as e:  # noqa: BLE001 — never break the caller
        # Reported, not silent: a lookup that always returns nothing is
        # indistinguishable from "no saved knowledge" unless this is visible.
        print(f"[knowledge] match failed: {type(e).__name__}: {e}")
        return _result([], None, False, {"error": f"{type(e).__name__}: {e}"})
