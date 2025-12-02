import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import faiss
from .esci_loader import ESCILoader
from .semantic_id_builder import SemanticIDBuilder

class IndexingPipeline:
    """Pre‑compute product embeddings, assign semantic IDs, and persist.
    This is a stub implementation – actual embedding model loading is omitted.
    """

    def __init__(self, data_dir: str = "data", output_dir: str = "output"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.loader = ESCILoader(self.data_dir)
        self.id_builder = SemanticIDBuilder()

    def compute_embeddings(self, texts: list[str]):
        """Compute embeddings for a list of product texts.
        Placeholder – replace with actual model inference.
        """
        # Dummy embeddings: random vectors
        dim = 768
        return np.random.rand(len(texts), dim).astype("float32")

    def run(self, csv_path: str, batch_size: int = 32):
        """Run the indexing pipeline: load data, compute embeddings, build index, and save.
        """
        from tqdm import tqdm
        
        print(f"Loading data from {csv_path}...")
        df = self.loader.load_data(csv_path)
        
        # Ensure we have a 'text' column - simplistic assumption for now
        if 'product_title' in df.columns:
            texts = df['product_title'].fillna("").tolist()
        elif 'text' in df.columns:
            texts = df['text'].fillna("").tolist()
        else:
            raise ValueError("DataFrame must contain 'product_title' or 'text' column.")
            
        print(f"Computing embeddings for {len(texts)} products...")
        all_embeddings = []
        
        # Batch processing
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i : i + batch_size]
            batch_embs = self.compute_embeddings(batch_texts)
            all_embeddings.append(batch_embs)
            
        embeddings = np.vstack(all_embeddings)
        
        # Save embeddings to Parquet
        print("Saving embeddings to Parquet...")
        table = pa.Table.from_pandas(pd.DataFrame(embeddings))
        pq.write_table(table, os.path.join(self.output_dir, "embeddings.parquet"))
        
        # Build Semantic IDs
        print("Building Semantic IDs...")
        ids, id_strings = self.id_builder.fit_transform(embeddings)
        
        # Save IDs
        df['semantic_id'] = id_strings
        df.to_parquet(os.path.join(self.output_dir, "products_with_ids.parquet"))
        
        # Build FAISS index (using DenseRetriever logic ideally, but here inline for simplicity)
        print("Building FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, os.path.join(self.output_dir, "dense_index.faiss"))
        
        print("Indexing pipeline complete.")
