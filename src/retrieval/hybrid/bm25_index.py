"""
BM25 lexical retrieval index using rank_bm25.

Design Decisions
----------------
1. We use `rank_bm25` which implements Okapi BM25 — the same algorithm
   used by Elasticsearch and Solr. It's fast, well-tested, and in-memory.

2. We apply tokenization: lowercase + stopword removal using NLTK's
   English stopword list. This is the standard BM25 preprocessing.

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
from pathlib import Path
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Try to download stopwords; if already present, this is a no-op
try:
    _ = stopwords.words("english")
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass  # Non-fatal; we'll handle missing stopwords gracefully


from rank_bm25 import BM25Okapi

from src.retrieval.result import RetrievedResult


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
            Language for stopword removal.
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

        Parameters
        ----------
        text : str
            Text to tokenize.

        Returns
        -------
        list[str]
            List of tokens.
        """
        # Lowercase
        text = text.lower()
        # Word tokenize
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to simple split if NLTK tokenizer fails
            tokens = text.split()

        # Filter: keep only alphabetic tokens, remove stopwords
        try:
            stop_words = set(stopwords.words(self.stopwords_lang))
        except Exception:
            stop_words = set()

        filtered = [
            t for t in tokens
            if t.isalpha() and t not in stop_words and len(t) > 2
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
        # Save BM25 object
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self._index, f)
        # Save metadata
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({
                "doc_ids": self._doc_ids,
                "doc_texts": self._doc_texts,
                "k1": self.k1,
                "b": self.b,
            }, f)

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
