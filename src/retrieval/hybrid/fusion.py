"""
Reciprocal Rank Fusion (RRF) for combining multiple retrieval methods.

Design Decisions
----------------
1. RRF is used because it is parameter-light, rank-based (not score-based),
   and proven in production at Bing and Google. It requires no score
   normalization across different retrieval systems.

2. RRF Formula: score(d) = Σ 1 / (k + rank_i(d))
   - k=60 is the standard value. Higher k reduces the influence of
     rank differences between systems.
   - Lower k (e.g., k=1) makes the top-ranked result dominant.

3. We use a dict[doc_id] for accumulation (O(1) lookups) and sort at the end.
   This is more efficient than maintaining a sorted list during fusion.

4. RRF is applied after both dense and BM25 have returned their ranked lists.
   The fused list re-ranks documents based on their position in BOTH lists.

5. All input lists must be pre-sorted by their respective scores (descending).
   The rank is simply the position in the list (0-indexed in our implementation).

References
----------
Craswell, N. (2009). "Experimental evidence for Optmal Rank Fusion." TREC.
Voorhees, E. (2005). "The TREC 9 RDR Track." TREC.
"""

from __future__ import annotations

from collections import defaultdict

from src.retrieval.result import RetrievedResult


class RRF:
    """
    Reciprocal Rank Fusion for combining ranked retrieval results.

    Combines multiple ranked result sets into a single fused ranking
    using the RRF formula.

    Usage
    -----
    ```python
    dense_results = [("doc1", 0.95), ("doc2", 0.87), ("doc3", 0.82)]
    bm25_results = [("doc3", 5.2), ("doc1", 4.8), ("doc4", 3.1)]

    fused = RRF.fuse([dense_results, bm25_results], k=60, top_k=20)
    # Returns: [("doc3", combined_rrf_score), ("doc1", ...), ...]
    ```
    """

    DEFAULT_K = 60

    @staticmethod
    def fuse(
        result_lists: list[list[tuple[str, float]]],
        k: int = DEFAULT_K,
        top_k: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Combine ranked result lists using Reciprocal Rank Fusion.

        Parameters
        ----------
        result_lists : list[list[tuple[str, float]]]
            List of ranked result lists. Each list should be pre-sorted
            by score descending (best first). Each entry is (doc_id, score).
        k : int
            RRF smoothing parameter. Higher k = more equal weight across systems.
            Default k=60 is standard. Range: 1-120.
        top_k : int, optional
            Return only the top-K fused results. If None, return all.

        Returns
        -------
        list[tuple[str, float]]
            Fused results as (doc_id, combined_rrf_score), sorted descending.
            The RRF score is the sum of 1/(k+rank) across all result lists.

        Raises
        ------
        ValueError
            If k <= 0 or result_lists is empty.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not result_lists:
            return []

        # Accumulate RRF scores
        # score[doc_id] = sum of 1/(k + rank) across all lists
        doc_scores: dict[str, float] = defaultdict(float)

        for result_list in result_lists:
            for rank, (doc_id, _score) in enumerate(result_list):
                # rank is 0-indexed; RRF uses 1-indexed, so add 1
                rrf_contribution = 1.0 / (k + rank + 1)
                doc_scores[doc_id] += rrf_contribution

        # Sort by fused RRF score descending
        sorted_results = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k is not None:
            sorted_results = sorted_results[:top_k]

        return sorted_results

    @staticmethod
    def fuse_with_metadata(
        result_lists: list[list[tuple[str, float, dict]]],
        k: int = DEFAULT_K,
        top_k: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        RRF fusion when results carry additional metadata (scores per system).

        Parameters
        ----------
        result_lists : list[list[tuple[str, float, dict]]]
            Each entry is (doc_id, system_score, metadata_dict).
            metadata_dict can contain per-system scores for later analysis.

        Returns
        -------
        list[tuple[str, float]]
            Fused results. Note: metadata is lost in this simplified version.
            Use the full pipeline version if metadata is needed.
        """
        # Strip metadata
        stripped_lists: list[list[tuple[str, float]]] = [
            [(doc_id, score) for doc_id, score, *_ in result_list]
            for result_list in result_lists
        ]
        return RRF.fuse(stripped_lists, k=k, top_k=top_k)


# Re-export Optional from typing for the class
from typing import Optional
