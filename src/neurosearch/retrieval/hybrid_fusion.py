import numpy as np
from typing import List, Tuple

def reciprocal_rank_fusion(rankings: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    """Combine multiple ranking lists using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    rankings: List of ranking lists. Each ranking list is a list of tuples (doc_id, score).
               The order of the list is assumed to be descending relevance.
    k: The constant used in the RRF formula (default 60 as per literature).

    Returns
    -------
    A combined ranking list of (doc_id, rrf_score) sorted by descending score.
    """
    rrf_scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
    # Convert to list and sort
    combined = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return combined
