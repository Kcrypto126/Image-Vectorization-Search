import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dim=512):
        # Inner product = cosine similarity if vectors are normalized
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []  # Keep track of mapping: index → image ID

    def add(self, vectors: np.ndarray, ids: list):
        faiss.normalize_L2(vectors)  # Normalize for cosine similarity
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, query_vector: np.ndarray, k=10):
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            results.append({
                "id": self.ids[idx],
                "score": float(distances[0][rank])
            })
        return results

    def save(self, path="faiss.index"):
        faiss.write_index(self.index, path)

    def load(self, path="faiss.index"):
        self.index = faiss.read_index(path)
