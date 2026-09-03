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
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from src.retrieval.result import RetrievedResult

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
        """
        self.k1 = k1
        self.b = b
        self.stopwords_lang = stopwords_lang
        self._index: Optional[BM25Okapi] = None
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._tokenized_corpus: Optional[list[list[str]]] = None

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

        # Get BM25 scores for all documents
        scores = self._index.get_scores(query_tokens)

        # Pair with doc_ids and sort by score descending
        scored_docs = sorted(
            zip(self._doc_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(doc_id, float(score)) for doc_id, score in scored_docs[:k]]

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

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self._doc_ids)

    def __repr__(self) -> str:
        return f"BM25Index(k1={self.k1}, b={self.b}, n_docs={len(self._doc_ids)})"
