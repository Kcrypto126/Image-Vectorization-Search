# app/index_adapter.py
import os
import numpy as np
import logging
import faiss

logger = logging.getLogger(__name__)

class IndexAdapter:
    def __init__(self, backend: str = "faiss", dim: int = 512, faiss_index_path: str = "/data/faiss.index"):
        """
        backend: currently only 'faiss' is supported. Other values are placeholders.
        dim: embedding dimension
        faiss_index_path: file path to persist FAISS index
        """
        self.backend = backend
        self.faiss = faiss
        self.dim = dim
        self.faiss_index_path = faiss_index_path
        # Default: HNSW for good balance (you can change config)
        if self.backend == "faiss":
            self.index = self.faiss.IndexHNSWFlat(dim, 32)  # M=32
        else:
            # Fallback to FAISS for now
            self.index = self.faiss.IndexHNSWFlat(dim, 32)
        self.ids = []  # mapping idx -> image_id

    def exists(self) -> bool:
        return os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_index_path + ".ids")

    def load_if_exists(self):
        if os.path.exists(self.faiss_index_path):
            try:
                self.index = self.faiss.read_index(self.faiss_index_path)
                # load ids
                ids_path = self.faiss_index_path + ".ids"
                if os.path.exists(ids_path):
                    with open(ids_path, "r") as f:
                        self.ids = [l.strip() for l in f.readlines()]
                    logger.info("Loaded FAISS index and ids")
                return True
            except Exception as e:
                logger.exception("Failed loading faiss index")
        return False

    def save(self):
        # ensure directory exists
        dir_path = os.path.dirname(self.faiss_index_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.faiss.write_index(self.index, self.faiss_index_path)
        with open(self.faiss_index_path + ".ids", "w") as f:
            f.writelines([f"{i}\n" for i in self.ids])

    def reset(self):
        if self.backend == "faiss":
            self.index = self.faiss.IndexHNSWFlat(self.dim, 32)
        else:
            self.index = self.faiss.IndexHNSWFlat(self.dim, 32)
        self.ids = []

    def add(self, vectors: np.ndarray, ids: list):
        vectors = vectors.astype("float32")
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, qvec: np.ndarray, top_k=10):
        qvec = qvec.astype("float32")
        if hasattr(self.index, 'ntotal') and getattr(self.index, 'ntotal') == 0:
            return []
        distances, indices = self.index.search(qvec, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            results.append({"id": self.ids[idx], "score": float(distances[0][i])})
        return results
