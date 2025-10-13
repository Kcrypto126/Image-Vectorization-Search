# app/index_adapter.py
import os
import numpy as np
import logging
import faiss

logger = logging.getLogger(__name__)

class IndexAdapter:
    def __init__(self, backend: str = "faiss", dim: int = 512, faiss_index_path: str = "data/faiss.index"):
        """
        backend: currently only 'faiss' is supported. Other values are placeholders.
        dim: embedding dimension
        faiss_index_path: file path to persist FAISS index
        """
        self.backend = backend
        self.faiss = faiss
        self.dim = dim
        self.faiss_index_path = faiss_index_path
        # Inner product = cosine similarity if vectors are normalized
        self.index = faiss.IndexFlatIP(dim)
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
        # Inner product = cosine similarity if vectors are normalized
        self.index = faiss.IndexFlatIP(self.dim)
        self.ids = []
    
    def add(self, vectors: np.ndarray, ids: list):
        faiss.normalize_L2(vectors)  # Normalize for cosine similarity
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, qvec: np.ndarray, top_k=10):
        faiss.normalize_L2(qvec)
        distances, indices = self.index.search(qvec, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            results.append({
                "id": self.ids[idx],
                "score": float(distances[0][rank])
            })
        return results
    
    def get_vector_by_id(self, image_id: str) -> np.ndarray:
        """
        Get exact vector by image ID from the FAISS index.
        Returns None if image_id not found.
        """
        try:
            # Find the index of the image_id in our ids list
            if image_id not in self.ids:
                return None
            
            idx = self.ids.index(image_id)
            
            # Reconstruct the vector from the FAISS index
            # FAISS doesn't have a direct method to get by index, so we need to reconstruct
            # Get all vectors and return the one at the specific index
            vectors = self.index.reconstruct_batch([idx])
            return vectors[0] if len(vectors) > 0 else None
            
        except Exception as e:
            logger.warning(f"Failed to get vector for image_id {image_id}: {e}")
            return None

