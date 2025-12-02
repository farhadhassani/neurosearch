import numpy as np
from sklearn.cluster import KMeans

class SemanticIDBuilder:
    """Build hierarchical semantic IDs using k‑means clustering.
    Three levels, each with K=10 clusters.
    """

    def __init__(self, n_levels: int = 3, n_clusters: int = 10, random_state: int = 42):
        self.n_levels = n_levels
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.clusterers = []

    def fit(self, embeddings: np.ndarray):
        """Fit hierarchical k‑means on the given embeddings.
        Parameters
        ----------
        embeddings: np.ndarray of shape (n_samples, dim)
        """
        data = embeddings
        for level in range(self.n_levels):
            km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            km.fit(data)
            self.clusterers.append(km)
            # Use cluster centers as new data for next level
            data = km.cluster_centers_[km.labels_]
        return self

    def transform(self, embeddings: np.ndarray):
        """Assign semantic ID tuples to each embedding.
        Returns a list of tuples and a string representation like "3 9 1".
        """
        ids = []
        for emb in embeddings:
            path = []
            data = emb.reshape(1, -1)
            for km in self.clusterers:
                label = km.predict(data)[0]
                path.append(label)
                # Move to the centroid of the assigned cluster for next level
                data = km.cluster_centers_[label].reshape(1, -1)
            ids.append(tuple(path))
        # Convert to string format
        id_strings = [" ".join(map(str, tup)) for tup in ids]
        return ids, id_strings

    def fit_transform(self, embeddings: np.ndarray):
        self.fit(embeddings)
        return self.transform(embeddings)
