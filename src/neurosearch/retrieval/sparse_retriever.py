import pickle
import asyncio
from typing import List, Tuple
from rank_bm25 import BM25Okapi

class SparseRetriever:
    """Sparse retriever using BM25.
    """

    def __init__(self, index_path: str = None):
        self.bm25 = None
        self.corpus = []
        if index_path:
            self.load_index(index_path)

    def build_index(self, corpus: List[str]):
        """Build BM25 index from a list of documents.
        Parameters
        ----------
        corpus: List of document strings.
        """
        self.corpus = corpus
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return self

    def save_index(self, path: str):
        if self.bm25 is None:
            raise RuntimeError("Index not built yet.")
        with open(path, "wb") as f:
            pickle.dump((self.bm25, self.corpus), f)

    def load_index(self, path: str):
        with open(path, "rb") as f:
            self.bm25, self.corpus = pickle.load(f)

    async def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """Retrieve top_k documents for a query string asynchronously.
        Returns a list of (doc_id, score, text).
        """
        if self.bm25 is None:
            return []
            
        loop = asyncio.get_running_loop()
        
        def _search():
            tokenized_query = query.split(" ")
            scores = self.bm25.get_scores(tokenized_query)
            # Get top_k indices
            top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = []
            for idx in top_n:
                results.append((idx, float(scores[idx]), self.corpus[idx]))
            return results

        return await loop.run_in_executor(None, _search)
