import numpy as np
from typing import List, Tuple

def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.
    `relevances` should be ordered by predicted ranking (highest first).
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / np.log2(i + 2)
    # Ideal DCG
    ideal = sorted(relevances, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k]):
        idcg += (2 ** rel - 1) / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(relevances: List[int], k: int) -> float:
    """Recall@k where relevances are binary (1 relevant, 0 not)."""
    relevant_retrieved = sum(relevances[:k])
    total_relevant = sum(relevances)
    return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

def mrr_at_k(relevances: List[int], k: int) -> float:
    """Mean Reciprocal Rank at k.
    `relevances` is a list where the first relevant item has value 1.
    """
    for i, rel in enumerate(relevances[:k]):
        if rel:
            return 1.0 / (i + 1)
    return 0.0
