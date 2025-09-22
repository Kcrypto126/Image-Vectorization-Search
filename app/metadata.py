import os
import math
import logging
from typing import Dict, List, Tuple, Optional

from colorthief import ColorThief
from PIL import Image

_logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO  # YOLOv8
    _YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    _YOLO_AVAILABLE = False


# Basic color name map (RGB) for coarse naming
_BASIC_COLORS = {
    "red": (220, 20, 60),
    "orange": (255, 140, 0),
    "yellow": (255, 215, 0),
    "green": (60, 179, 113),
    "blue": (30, 144, 255),
    "purple": (138, 43, 226),
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "cyan": (0, 206, 209),
    "magenta": (255, 0, 255)
}


def _rgb_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _nearest_color_name(rgb: Tuple[int, int, int]) -> str:
    best_name = "unknown"
    best_dist = float("inf")
    for name, ref in _BASIC_COLORS.items():
        d = _rgb_distance(rgb, ref)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


_STYLE_PROMPTS = [
    "modern", "minimalist", "vintage", "retro", "3d", "flat", "cartoon",
    "photorealistic", "sketch", "logo", "icon", "illustration", "abstract", "geometric"
]

_STYLE_PROMPT_EMBEDS = None  # type: Optional[List]


def _ensure_style_prompt_vectors(vectorizer) -> List:
    global _STYLE_PROMPT_EMBEDS
    if _STYLE_PROMPT_EMBEDS is None:
        embeds = []
        for prompt in _STYLE_PROMPTS:
            try:
                v = vectorizer.text_to_vector(prompt)
                embeds.append(v[0])
            except Exception:
                embeds.append(None)
        _STYLE_PROMPT_EMBEDS = embeds
    return _STYLE_PROMPT_EMBEDS


_YOLO_MODEL = None


def _ensure_yolo_model(model_name: str = None):
    global _YOLO_MODEL
    if not _YOLO_AVAILABLE:
        return None
    if _YOLO_MODEL is None:
        try:
            # Default lightweight model
            _YOLO_MODEL = YOLO(model_name or os.getenv("YOLO_MODEL", "yolov8n.pt"))
        except Exception as e:
            _logger.warning(f"Failed to load YOLO model: {e}")
            _YOLO_MODEL = None
    return _YOLO_MODEL


def extract_colors(image_path: str, palette_size: int = 6) -> Dict:
    try:
        ct = ColorThief(image_path)
        palette = ct.get_palette(color_count=palette_size)
        names = []
        for rgb in palette:
            names.append(_nearest_color_name((rgb[0], rgb[1], rgb[2])))
        # Deduplicate keeping order
        seen = set()
        ordered_unique = []
        for n in names:
            if n not in seen:
                seen.add(n)
                ordered_unique.append(n)
        return {"palette_rgb": palette, "colors": ordered_unique}
    except Exception as e:
        _logger.warning(f"Color extraction failed: {e}")
        return {"palette_rgb": [], "colors": []}


def detect_objects(image_path: str, conf: float = 0.25, max_labels: int = 10) -> List[str]:
    model = _ensure_yolo_model()
    if model is None:
        return []
    try:
        results = model.predict(source=image_path, conf=conf, verbose=False)
        labels = []
        for r in results:
            names = r.names
            for cls in r.boxes.cls.tolist():
                idx = int(cls)
                label = names.get(idx) if isinstance(names, dict) else names[idx]
                labels.append(label)
        # Return unique labels by frequency
        freq = {}
        for l in labels:
            freq[l] = freq.get(l, 0) + 1
        sorted_labels = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [l for l, _ in sorted_labels[:max_labels]]
    except Exception as e:
        _logger.warning(f"YOLO detection failed: {e}")
        return []


def infer_styles_from_clip(image_vec, vectorizer, top_k: int = 5) -> List[Dict]:
    try:
        prompts = _STYLE_PROMPTS
        prompt_vecs = _ensure_style_prompt_vectors(vectorizer)
        # Normalize to unit vectors
        import numpy as np
        img = image_vec.astype("float32")
        img = img / (np.linalg.norm(img) + 1e-8)
        scores = []
        for prompt, pv in zip(prompts, prompt_vecs):
            if pv is None:
                continue
            v = pv.astype("float32")
            v = v / (np.linalg.norm(v) + 1e-8)
            s = float((img @ v.T).item())
            scores.append({"style": prompt, "score": s})
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]
    except Exception as e:
        _logger.warning(f"Style inference failed: {e}")
        return []


def extract_image_metadata(image_path: str, vectorizer, image_vec_row) -> Dict:
    """
    image_vec_row: 1D numpy array (embedding) corresponding to this image.
    Returns a dict suitable for storing in ImageMetadata.extra_metadata JSON.
    """
    colors = extract_colors(image_path)
    objects = detect_objects(image_path)
    styles = infer_styles_from_clip(image_vec_row, vectorizer, top_k=5)
    style_tags = [s["style"] for s in styles]
    return {
        "colors": colors.get("colors", []),
        "palette_rgb": colors.get("palette_rgb", []),
        "objects": objects,
        "styles": styles,
        "style_tags": style_tags
    }


_STOPWORDS = {
    "a","an","the","of","and","or","with","for","to","on","in","at","by","from","is","are","be","this","that","these","those","as","over","under","near","into","out","about","between","across"
}


def detect_filters_from_text(text: str):
    """
    Infer desired colors, objects, and styles from a free-form query text.
    - Colors: match known basic color names present in text
    - Styles: match known style prompts present in text
    - Objects: remaining tokens (minus stopwords, colors, styles) as candidates
    Returns: (wanted_colors:set, wanted_objects:set, wanted_styles:set)
    """
    if not text:
        return set(), set(), set()
    t = text.lower()
    import re
    tokens = re.findall(r"[a-z0-9]+", t)
    token_set = set(tokens)

    color_names = set(_BASIC_COLORS.keys())
    style_names = set(_STYLE_PROMPTS)

    wanted_colors = token_set & color_names
    wanted_styles = token_set & style_names

    residual = token_set - wanted_colors - wanted_styles - _STOPWORDS
    # Keep tokens length >= 3 as object candidates
    wanted_objects = set([w for w in residual if len(w) >= 3])

    return wanted_colors, wanted_objects, wanted_styles


