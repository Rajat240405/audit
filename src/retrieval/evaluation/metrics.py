"""
Retrieval metrics computation for Phase 4 Evaluation.
Computes Recall@K, Hit Rate, MRR, nDCG, and Average Rank.
"""

from __future__ import annotations

import math
from typing import Sequence


class RetrievalMetrics:
    """
    Computes scientific information retrieval (IR) metrics over a batch of queries.
    """

    @staticmethod
    def compute_recall(retrieved_ids: Sequence[str], expected_id: str, k: int) -> float:
        """
        Recall@K is 1.0 if the expected_id is in the top-K retrieved, else 0.0.
        (Since there is exactly one expected document per query).
        """
        return 1.0 if expected_id in retrieved_ids[:k] else 0.0

    @staticmethod
    def compute_mrr(retrieved_ids: Sequence[str], expected_id: str) -> float:
        """
        Mean Reciprocal Rank (MRR) is 1 / rank of the expected_id in the retrieved list.
        Returns 0.0 if not found.
        """
        try:
            rank = retrieved_ids.index(expected_id) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0

    @staticmethod
    def compute_ndcg(retrieved_ids: Sequence[str], expected_id: str, k: int) -> float:
        """
        Normalized Discounted Cumulative Gain (nDCG@K).
        With binary relevance (relevance = 1 for expected_id, 0 otherwise):
        IDCG is always 1.0 (the best rank is rank 1, so ideal DCG is 1/log2(2) = 1.0).
        DCG@K is 1 / log2(rank + 1) if expected_id is in top-K at 'rank' (1-indexed).
        """
        try:
            rank = retrieved_ids[:k].index(expected_id) + 1
            # DCG = 1 / log2(rank + 1)
            dcg = 1.0 / math.log2(rank + 1)
            # IDCG is 1.0 since ideal ranking has expected_id at Rank 1
            return dcg
        except ValueError:
            return 0.0

    @staticmethod
    def compute_average_rank(retrieved_ids: Sequence[str], expected_id: str, max_depth: int = 100) -> float:
        """
        Returns the 1-indexed rank of the expected document.
        If not found, returns max_depth (used for penalty baseline).
        """
        try:
            return float(retrieved_ids.index(expected_id) + 1)
        except ValueError:
            return float(max_depth)
