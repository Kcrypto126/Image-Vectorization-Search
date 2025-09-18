# app/vectorizer.py
import torch
import clip
from PIL import Image

class Vectorizer:
    def __init__(self, model_name="ViT-B/32", device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def image_to_vector(self, image_path: str):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = self.model.encode_image(image)
        return vector.cpu().numpy().astype("float32")  # FAISS expects float32
