import numpy as np
from typing import List, Tuple
from ..retrieval.dense_retriever import DenseRetriever
from ..retrieval.generative_retriever import GenerativeRetriever
from ..retrieval.hybrid_fusion import reciprocal_rank_fusion

from ..retrieval.sparse_retriever import SparseRetriever

class ContextBuilder:
    """Combine dense, sparse, and generative retrieval results to build context snippets.
    """

    def __init__(self, dense_retriever: DenseRetriever = None, sparse_retriever: SparseRetriever = None, generative_retriever: GenerativeRetriever = None, top_k: int = 10):
        self.dense = dense_retriever or DenseRetriever()
        self.sparse = sparse_retriever or SparseRetriever()
        self.generative = generative_retriever or GenerativeRetriever()
        self.top_k = top_k

    async def get_context(self, query: str) -> List[Tuple[int, float, str]]:
        """Return top‑N product snippets for the given query asynchronously.
        Returns a list of (doc_id, combined_score, text).
        """
        import asyncio
        
        # Run retrievers concurrently
        dense_task = self.dense.search(query, top_k=self.top_k)
        sparse_task = self.sparse.search(query, top_k=self.top_k)
        gen_task = self.generative.generate_ids(query, num_return_sequences=self.top_k)
        
        dense_results, sparse_results, gen_ids = await asyncio.gather(dense_task, sparse_task, gen_task)
        
        # For illustration, treat each generated ID as a doc_id placeholder
        gen_results = [(i, 1.0, f"Generated ID: {gid}") for i, gid in enumerate(gen_ids)]
        
        # Apply RRF on doc_id scores (ignore text for ranking)
        combined = reciprocal_rank_fusion([
            [(doc_id, score) for doc_id, score, _ in dense_results],
            [(doc_id, score) for doc_id, score, _ in sparse_results],
            [(doc_id, score) for doc_id, score, _ in gen_results]
        ])
        
        # Merge back with texts (simple lookup)
        id_to_text = {doc_id: text for doc_id, _, text in dense_results}
        id_to_text.update({doc_id: text for doc_id, _, text in sparse_results})
        id_to_text.update({doc_id: text for doc_id, _, text in gen_results})
        
        return [(doc_id, rrf_score, id_to_text.get(doc_id, "")) for doc_id, rrf_score in combined[:self.top_k]]
