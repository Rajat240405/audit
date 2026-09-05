"""
BM25 lexical retrieval index using rank_bm25.

Design Decisions
----------------
1. We use `rank_bm25` which implements Okapi BM25 — the same algorithm
   used by Elasticsearch and Solr. It's fast, well-tested, and in-memory.

2. We apply tokenization: lowercase + stopword removal using a native, fast
   regex-based word tokenizer. This completely removes the NLTK dependency,
   eliminating the need for slow/unreliable dataset downloads and resolving
   Windows-specific import issues.

3. We store the original texts alongside the tokenized corpus so that
   after BM25 returns doc_ids, we can fetch the full structured record.

4. The BM25 parameters: k1=1.5, b=0.75.
   These are the standard values established in the original BM25 paper
   and used as defaults in most IR systems.

5. The index is built once (offline) and queried many times (online).
   This is a standard IR pattern.

6. Index is saved as two files:
   - {path}.pkl  — pickled BM25Plus or BM25Okapi object
   - {path}.json — doc_ids + original texts mapping

7. SCORING (postings fast path, added after long-document chunking blew up
   the unit count): ``rank_bm25``'s ``get_scores`` scores **every** document
   for **every** query term — a full-corpus Python list comprehension per
   term. At ~41k units that dominates query latency even though most units
   contain none of the query terms. We therefore build an inverted index
   (term -> postings) once and score only the documents that actually contain
   a query term.

   This is a PURE OPTIMIZATION, not a new ranking: the BM25Okapi object stays
   the single source of truth for tokenization, idf (including the
   ``epsilon * average_idf`` floor), doc lengths and avgdl, and the fast path
   applies the IDENTICAL per-document formula in the identical order. A
   document with zero matching terms scores exactly 0.0 either way and can
   never become a BM25 top result. Verified bit-for-bit against
   ``BM25Okapi.get_scores`` by tests/test_bm25_postings.py; the reference
   implementation remains reachable via ``use_postings=False`` /
   BM25_DISABLE_POSTINGS=1.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.result import RetrievedResult

# Escape hatch: force the (exact, slower) rank_bm25 reference scorer.
BM25_DISABLE_POSTINGS_ENV = "BM25_DISABLE_POSTINGS"

# Standard English stopwords (derived from NLTK/scikit-learn)
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}


class BM25Index:
    """
    BM25 Okapi lexical retrieval index.

    Provides fast keyword-based retrieval complementing dense vector search.

    Usage
    -----
    ```python
    index = BM25Index()

    # Build from documents (once)
    docs = [
        ("18-1", "Question text 1", "Answer text 1"),
        ("18-2", "Question text 2", "Answer text 2"),
    ]
    index.build(docs)

    # Search (many times)
    results = index.search("malaria health", k=5)
    ```
    """

    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        stopwords_lang: str = "english",
        use_postings: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        k1 : float
            BM25 term frequency saturation parameter.
            Typical range: 1.2–2.0. Higher = less saturation effect.
            Default 1.5 is standard.
        b : float
            BM25 document length normalization. Typical range: 0.5–0.75.
            Default 0.75 is standard.
        stopwords_lang : str
            Language for stopword removal (kept for API compatibility, default english only).
        use_postings : bool | None
            Score only documents containing a query term (inverted index).
            Semantically identical to rank_bm25 — see module note 7.
            None (default) = enabled unless BM25_DISABLE_POSTINGS=1.
        """
        self.k1 = k1
        self.b = b
        self.stopwords_lang = stopwords_lang
        self._index: Optional[BM25Okapi] = None
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._tokenized_corpus: Optional[list[list[str]]] = None
        if use_postings is None:
            use_postings = os.environ.get(BM25_DISABLE_POSTINGS_ENV, "") not in ("1", "true", "True")
        self.use_postings = bool(use_postings)
        # Inverted index (flat CSR-style arrays): term -> (start, end) slice
        # into _postings_idx / _postings_tf. None until built.
        self._postings_terms: Optional[dict[str, tuple[int, int]]] = None
        self._postings_idx: Optional["np.ndarray"] = None
        self._postings_tf: Optional["np.ndarray"] = None
        self._doc_len_arr: Optional["np.ndarray"] = None
        self._postings_built_for: Optional[int] = None  # corpus_size they were built for

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text: lowercase + stopword removal + alphanumeric only.

        Fix (retrieval/HPC patch #3): meaningful numerics are now KEPT.
        The audit corpus is built out of years, amounts and question numbers;
        the old alpha-only rule made BM25 blind to them ("2022-23",
        "3035", "2000 crore" all vanished from index AND query). We keep
        3–4 digit tokens (years, amounts, question numbers) while still
        dropping 1–2 digit junk ("1", "35") that would swamp row lists.

        Returns
        -------
        list[str]
            List of tokens.
        """
        # Split into lowercase alphanumeric word tokens
        tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())

        # Filter: keep alphabetic tokens (len > 2, no stopwords) OR
        # meaningful numerics (3–4 digits).
        filtered = [
            t for t in tokens
            if t not in ENGLISH_STOPWORDS and (
                (t.isalpha() and len(t) > 2)
                or (t.isdigit() and 3 <= len(t) <= 4)
            )
        ]
        return filtered

    def build(
        self,
        docs: list[tuple[str, str, str]],
    ) -> None:
        """
        Build the BM25 index from documents.

        Parameters
        ----------
        docs : list[tuple[str, str, str]]
            List of (doc_id, question_text, answer_text).
            The concatenation of question + answer is indexed.

        Note
        ----
        The BM25 index uses the concatenated "question + answer" text
        for keyword matching, just like the dense vector index.
        """
        if not docs:
            raise ValueError("Cannot build BM25 index from empty document list.")

        self._doc_ids = [doc[0] for doc in docs]
        self._doc_texts = [doc[1] + " " + doc[2] for doc in docs]  # Q + A

        # Tokenize all documents
        self._tokenized_corpus = [self._tokenize(text) for text in self._doc_texts]

        # Build BM25 index
        self._index = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
        if self.use_postings:
            self._build_postings()

    # ── inverted index (pure optimization — see module note 7) ──────────────

    def _build_postings(self) -> None:
        """Build term -> postings from the BM25Okapi object's own statistics.

        Reads ONLY ``doc_freqs`` (already computed by BM25Okapi at build time,
        and present in every pickled index), so the same code path serves a
        fresh ``build()`` and a ``load()`` of an older index. Two passes keep
        peak memory to the final arrays (no per-term Python lists of millions
        of ints).
        """
        index = self._index
        if index is None:
            return
        doc_freqs = getattr(index, "doc_freqs", None)
        if not doc_freqs:
            self._postings_terms = None
            self._postings_idx = None
            self._postings_tf = None
            self._doc_len_arr = None
            self._postings_built_for = None
            return

        started = time.perf_counter()
        counts: dict[str, int] = {}
        for freqs in doc_freqs:                      # pass 1: term -> df
            for term in freqs:
                counts[term] = counts.get(term, 0) + 1

        total = sum(counts.values())
        flat_idx = np.empty(total, dtype=np.int32)
        flat_tf = np.empty(total, dtype=np.int32)
        start_of: dict[str, int] = {}
        cursor: dict[str, int] = {}
        pos = 0
        for term in sorted(counts):                  # stable, deterministic layout
            start_of[term] = pos
            cursor[term] = pos
            pos += counts[term]

        for d_i, freqs in enumerate(doc_freqs):      # pass 2: fill
            for term, tf in freqs.items():
                p = cursor[term]
                flat_idx[p] = d_i
                flat_tf[p] = tf
                cursor[term] = p + 1

        self._postings_terms = {
            term: (start_of[term], start_of[term] + counts[term]) for term in counts
        }
        self._postings_idx = flat_idx
        self._postings_tf = flat_tf
        self._doc_len_arr = np.asarray(index.doc_len, dtype=np.int64)
        self._postings_built_for = int(index.corpus_size)
        print(
            f"[bm25] postings built: {len(counts):,} terms / {total:,} entries "
            f"in {time.perf_counter() - started:.1f}s"
        )

    def _postings_ready(self) -> bool:
        return (
            self.use_postings
            and self._postings_terms is not None
            and self._index is not None
            and self._postings_built_for == int(self._index.corpus_size)
        )

    def _scores(self, query_tokens: list[str]) -> "np.ndarray":
        """BM25 score for EVERY document — bit-identical to the reference.

        Fast path scores only postings; every other document keeps its 0.0
        (rank_bm25 computes ``idf * 0 / denom == 0.0`` for them, and adding
        0.0 changes nothing). Falls back to ``BM25Okapi.get_scores`` whenever
        the postings are unavailable or the corpus is degenerate (avgdl <= 0,
        where the reference produces inf/nan and we refuse to guess).
        """
        index = self._index
        if index is None:
            raise RuntimeError("BM25 index not built. Call build() first.")
        avgdl = getattr(index, "avgdl", 0.0)
        if (
            not self._postings_ready()
            or not avgdl
            or not math.isfinite(avgdl)
            or avgdl <= 0
        ):
            return index.get_scores(query_tokens)

        terms = self._postings_terms or {}
        flat_idx = self._postings_idx
        flat_tf = self._postings_tf
        doc_len = self._doc_len_arr
        k1 = float(index.k1)
        b = float(index.b)
        scores = np.zeros(int(index.corpus_size), dtype=np.float64)

        # rank_bm25 iterates the query IN ORDER and repeats duplicates, so we
        # do exactly the same — float accumulation order is part of equality.
        for q in query_tokens:
            span = terms.get(q)
            if span is None:
                continue                      # term absent from the corpus
            idf = index.idf.get(q) or 0
            if not idf:
                continue                      # idf == 0 -> adds 0.0 everywhere
            s, e = span
            idx = flat_idx[s:e]
            tf = flat_tf[s:e]
            denom = tf + k1 * (1.0 - b + b * doc_len[idx] / avgdl)
            scores[idx] += idf * (tf * (k1 + 1.0) / denom)
        return scores

    def _topk_indices(self, scores: "np.ndarray", k: int) -> "np.ndarray":
        """Indices of the top-k scores, with rank_bm25's exact tie-breaking.

        ``sorted(zip(doc_ids, scores), key=score, reverse=True)`` is a STABLE
        descending sort: equal scores keep ascending document order.
        ``np.argsort(-scores, kind="stable")`` is the same ordering, faster.
        """
        if k is None or k <= 0:
            return np.empty(0, dtype=np.int64)
        return np.argsort(-scores, kind="stable")[: int(k)]

    def postings_stats(self) -> dict:
        """Diagnostics: whether the fast path is live and how big it is."""
        return {
            "use_postings": self.use_postings,
            "ready": self._postings_ready(),
            "terms": len(self._postings_terms or {}),
            "entries": int(self._postings_idx.size) if self._postings_idx is not None else 0,
            "corpus_size": int(self._index.corpus_size) if self._index is not None else 0,
        }

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Retrieve top-K documents by BM25 score.

        Parameters
        ----------
        query : str
            Raw query text (not pre-processed — we tokenize it here).
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (doc_id, bm25_score) sorted by score descending.
        """
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        # Tokenize the query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # BM25 scores for every document (postings fast path when available;
        # identical values, identical ordering — see module note 7).
        scores = self._scores(query_tokens)

        # Pair with doc_ids and sort by score descending (stable: ties keep
        # document order, exactly like the previous `sorted(...)`).
        order = self._topk_indices(scores, k)

        return [(self._doc_ids[i], float(scores[i])) for i in order]

    def get_text(self, doc_id: str) -> Optional[str]:
        """
        Retrieve the indexed text for a doc_id (for context/debugging).
        """
        try:
            idx = self._doc_ids.index(doc_id)
            return self._doc_texts[idx]
        except (ValueError, IndexError):
            return None

    def save(self, path: str | Path) -> None:
        """
        Save the BM25 index to disk.

        Parameters
        ----------
        path : str | Path
            Base path. Two files are written:
            - {path}.pkl   — pickled BM25 index
            - {path}.json  — doc_ids + doc_texts
        """
        if self._index is None:
            raise RuntimeError("Cannot save empty index.")

        path = Path(path)
        from src.utils.atomic_io import write_bytes_atomic, write_text_atomic

        write_bytes_atomic(path.with_suffix(".pkl"), pickle.dumps(self._index))
        write_text_atomic(
            path.with_suffix(".json"),
            json.dumps({
                "doc_ids": self._doc_ids,
                "doc_texts": self._doc_texts,
                "k1": self.k1,
                "b": self.b,
            }),
        )
        # Inverted index is derived data: saving it turns the ~10s rebuild
        # after a restart into a fast numpy load. Missing/unsaveable is fine —
        # load() rebuilds it from doc_freqs.
        if self._postings_ready():
            try:
                write_bytes_atomic(
                    path.with_name(path.name + ".postings.pkl"),
                    pickle.dumps(
                        {
                            "version": 1,
                            "corpus_size": int(self._index.corpus_size),
                            "terms": self._postings_terms,
                            "idx": self._postings_idx,
                            "tf": self._postings_tf,
                            "doc_len": self._doc_len_arr,
                        },
                        protocol=5,
                    ),
                )
            except Exception as exc:  # pragma: no cover - disk/permission only
                print(f"[bm25] postings not persisted ({exc}); will rebuild on load")

    def load(self, path: str | Path) -> None:
        """
        Load the BM25 index from disk.

        Parameters
        ----------
        path : str | Path
            Base path (without extension).
        """
        path = Path(path)
        pkl_file = path.with_suffix(".pkl")
        json_file = path.with_suffix(".json")

        if not pkl_file.exists() or not json_file.exists():
            raise FileNotFoundError(f"BM25 files not found at {path}")

        with open(pkl_file, "rb") as f:
            self._index = pickle.load(f)
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
            self._doc_ids = data["doc_ids"]
            self._doc_texts = data["doc_texts"]
            self.k1 = data.get("k1", self.DEFAULT_K1)
            self.b = data.get("b", self.DEFAULT_B)

        # Inverted index: reuse the saved one when it matches this corpus,
        # otherwise rebuild from the BM25Okapi statistics (backward compatible
        # with indexes saved before postings existed).
        self._postings_terms = None
        self._postings_idx = None
        self._postings_tf = None
        self._doc_len_arr = None
        self._postings_built_for = None
        postings_file = path.with_name(path.name + ".postings.pkl")
        if self.use_postings and postings_file.exists():
            try:
                with open(postings_file, "rb") as f:
                    p = pickle.load(f)
                if (
                    p.get("version") == 1
                    and int(p.get("corpus_size", -1)) == int(self._index.corpus_size)
                    and len(p.get("doc_len") or []) == int(self._index.corpus_size)
                ):
                    self._postings_terms = p["terms"]
                    self._postings_idx = p["idx"]
                    self._postings_tf = p["tf"]
                    self._doc_len_arr = p["doc_len"]
                    self._postings_built_for = int(self._index.corpus_size)
            except Exception as exc:
                print(f"[bm25] saved postings unusable ({exc}); rebuilding")
        if self.use_postings and not self._postings_ready():
            self._build_postings()

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self._doc_ids)

    def __repr__(self) -> str:
        return f"BM25Index(k1={self.k1}, b={self.b}, n_docs={len(self._doc_ids)})"
