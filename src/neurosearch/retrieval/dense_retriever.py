import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    """Dense retriever using a SentenceTransformer model and FAISS index.
    This is a simplified stub – replace with actual model loading and indexing.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", index_path: str = None):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.id_to_text = {}
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)

    def build_index(self, texts: list[str]):
        """Encode texts and build a FAISS index.
        Parameters
        ----------
        texts: List of document strings.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product for cosine similarity
        self.index.add(embeddings.astype('float32'))
        self.id_to_text = {i: txt for i, txt in enumerate(texts)}
        return self.index

    def save_index(self, path: str):
        if self.index is None:
            raise RuntimeError("Index not built yet.")
        faiss.write_index(self.index, path)
        # Save mapping separately
        import json
        with open(path + "_mapping.json", "w", encoding="utf-8") as f:
            json.dump(self.id_to_text, f)

    def load_index(self, path: str):
        self.index = faiss.read_index(path)
        import json
        with open(path + "_mapping.json", "r", encoding="utf-8") as f:
            self.id_to_text = json.load(f)

    async def search(self, query: str, top_k: int = 10):
        """Retrieve top_k documents for a query string asynchronously.
        Returns a list of (doc_id, score, text).
        """
        # Note: FAISS is CPU-bound and releases GIL, but for true async we might want run_in_executor
        # For this prototype, we'll keep it simple but async-def
        import asyncio
        loop = asyncio.get_running_loop()
        
        def _search():
            q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            distances, indices = self.index.search(q_emb.astype('float32'), top_k)
            results = []
            for idx, score in zip(indices[0], distances[0]):
                text = self.id_to_text.get(str(idx), "")
                results.append((idx, float(score), text))
            return results

        return await loop.run_in_executor(None, _search)
