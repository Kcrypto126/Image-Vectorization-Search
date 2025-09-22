# app/vectorizer.py
import torch
import clip
from PIL import Image
import numpy as np

class Vectorizer:
    def __init__(self, model_name="ViT-B/32", device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def image_to_vector(self, image_path: str):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = self.model.encode_image(image)
        return vector.cpu().numpy().astype("float32")  # FAISS expects float32

    def text_to_vector(self, text: str):
        """Convert text to vector using CLIP's text encoder"""
        with torch.no_grad():
            # Tokenize the text
            text_tokens = clip.tokenize([text]).to(self.device)
            # Encode text to vector
            vector = self.model.encode_text(text_tokens)
        return vector.cpu().numpy().astype("float32")  # FAISS expects float32

    def combine_vectors(self, text_vector: np.ndarray = None, image_vector: np.ndarray = None, text_weight: float = 0.5):
        """
        Combine text and image vectors using weighted average.
        If only one vector is provided, return it directly.
        """
        import numpy as np
        
        if text_vector is not None and image_vector is not None:
            # Both vectors provided - combine them
            image_weight = 1.0 - text_weight
            combined = text_weight * text_vector + image_weight * image_vector
            return combined
        elif text_vector is not None:
            # Only text vector
            return text_vector
        elif image_vector is not None:
            # Only image vector
            return image_vector
        else:
            raise ValueError("At least one vector (text or image) must be provided")