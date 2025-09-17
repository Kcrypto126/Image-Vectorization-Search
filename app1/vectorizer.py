# app/vectorizer.py
import torch
from PIL import Image
import clip
import numpy as np

class Vectorizer:
    def __init__(self, model_name="ViT-B/32", device="cpu"):
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def image_to_vector(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(tensor)
        embedding = embedding.cpu().numpy().astype("float32")
        # Normalize to unit length (important for cosine/inner product similarity)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / (norms + 1e-10)
        return embedding
