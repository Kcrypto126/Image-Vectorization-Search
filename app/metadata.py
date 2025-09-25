# app/metadata.py
import os
import math
import logging
from typing import Dict, List, Tuple, Optional
import json
import re
import cv2
import layoutparser as lp
import pytesseract
from openai import OpenAI
from ultralytics import YOLO

from colorthief import ColorThief
from PIL import Image
try:
    import cv2  # type: ignore
    _CV_AVAILABLE = True
except Exception:
    cv2 = None
    _CV_AVAILABLE = False

try:
    import layoutparser as lp  # type: ignore
    _LP_AVAILABLE = True
except Exception:
    lp = None
    _LP_AVAILABLE = False

try:
    import pytesseract  # type: ignore
    _TESS_AVAILABLE = True
except Exception:
    pytesseract = None
    _TESS_AVAILABLE = False

try:
    from openai import OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    _OPENAI_AVAILABLE = False

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
        # Convert to HEX
        hex_palette = ["#%02x%02x%02x" % (r, g, b) for (r, g, b) in (palette or [])]
        return {"palette_rgb": palette, "palette_hex": hex_palette, "colors": ordered_unique}
    except Exception as e:
        _logger.warning(f"Color extraction failed: {e}")
        return {"palette_rgb": [], "palette_hex": [], "colors": []}


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


def detect_layout(image_path: str) -> Dict:
    """
    Use LayoutParser (if available) to get a coarse layout categorization and components.
    Returns: {layout: one of [one column, card grid, split] or None, components: [strings]}
    """
    layout = None
    components: List[str] = []
    if not _LP_AVAILABLE:
        return {"layout": layout, "components": components}
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = None
        if _CV_AVAILABLE:
            image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        model = lp.Detectron2LayoutModel(
            config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
        )
        layout_result = model.detect(image_np if image_np is not None else image)
        # Heuristic layout classification
        counts = {"text": 0, "title": 0, "list": 0, "table": 0, "figure": 0}
        for b in layout_result:
            lbl = b.type if hasattr(b, 'type') else getattr(b, 'label', None)
            if lbl in counts:
                counts[lbl] += 1
        # Simple rules
        if counts["figure"] >= 3 or counts["table"] >= 2:
            layout = "card grid"
        elif counts["text"] > 0 and counts["figure"] > 0:
            layout = "split"
        else:
            layout = "one column"
        # Components from detected types
        if counts["table"]:
            components.append("table")
        if counts["list"]:
            components.append("sidebar")
        if counts["title"]:
            components.append("breadcrumbs")
    except Exception as e:
        _logger.warning(f"Layout detection failed: {e}")
    return {"layout": layout, "components": list(sorted(set(components)))}


def extract_ocr_text(image_path: str) -> str:
    if not _TESS_AVAILABLE:
        return ""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        _logger.warning(f"OCR failed: {e}")
        return ""


def _gpt_generate_metadata(objects: List[str], colors_hex: List[str], layout: Optional[str], style_tags: List[str], ocr_text: str) -> Optional[Dict]:
    """
    Calls OpenAI (if available) to generate expert metadata JSON per spec.
    Falls back to heuristic if API not configured.
    """
    if not _OPENAI_AVAILABLE:
        return None
    try:
        client = OpenAI()
        prompt = {
            "role": "user",
            "content": (
                "Given: objects=" + json.dumps(objects) + ", colors=" + json.dumps(colors_hex) + 
                ", layout=" + json.dumps(layout) + ", styles=" + json.dumps(style_tags) + 
                ", ocr_text=" + json.dumps(ocr_text) + 
                ". Generate expert branding/UI metadata as concise JSON with these keys if applicable: "
                "Categories, Block, Device, Logo elements, Main components, Content hints, Layout, Visual style, "
                "Palette, Typography, Icons / Illustrations, Complexity, Data density, Verdict, Keywords, Description, Mode. "
                "Categories must be 5-6 tags chosen from the provided allowed list in the system. Palette must be 3 HEX + theme entries."
            )
        }
        sys = {"role": "system", "content": (
            "You are a senior brand and product design analyst. Respond ONLY with strict JSON, no markdown, no comments."
        )}
        resp = client.chat.completions.create(
            model=os.getenv("GEN_METADATA_MODEL", "gpt-4o-mini"),
            messages=[sys, prompt],
            temperature=0.3,
            max_tokens=600
        )
        txt = resp.choices[0].message.content if resp and resp.choices else None
        if not txt:
            return None
        # Attempt to extract JSON
        txt = txt.strip()
        # Remove triple backticks if present
        if txt.startswith("```"):
            txt = re.sub(r"^```(json)?", "", txt).strip()
            if txt.endswith("```"):
                txt = txt[:-3].strip()
        data = json.loads(txt)
        return data if isinstance(data, dict) else None
    except Exception as e:
        _logger.warning(f"GPT metadata generation failed: {e}")
        return None


def extract_image_metadata(image_path: str, vectorizer, image_vec_row) -> Dict:
    """
    image_vec_row: 1D numpy array (embedding) corresponding to this image.
    Returns dicts for legacy extra_metadata, raw_detections, and generated_metadata.
    """
    colors = extract_colors(image_path)
    objects = detect_objects(image_path)
    styles = infer_styles_from_clip(image_vec_row, vectorizer, top_k=5)
    style_tags = [s["style"] for s in styles]
    layout_info = detect_layout(image_path)
    ocr_text = extract_ocr_text(image_path)

    raw_detections = {
        "colors_named": colors.get("colors", []),
        "palette_rgb": colors.get("palette_rgb", []),
        "palette_hex": colors.get("palette_hex", []),
        "objects": objects,
        "styles_clip": styles,
        "style_tags": style_tags,
        "layout": layout_info.get("layout"),
        "components": layout_info.get("components", []),
        "ocr_text": ocr_text,
    }

    # Legacy extra_metadata for backward compatibility
    extra_metadata = {
        "colors": colors.get("colors", []),
        "palette_rgb": colors.get("palette_rgb", []),
        "objects": objects,
        "styles": styles,
        "style_tags": style_tags,
    }

    generated = _gpt_generate_metadata(
        objects=objects,
        colors_hex=colors.get("palette_hex", []),
        layout=layout_info.get("layout"),
        style_tags=style_tags,
        ocr_text=ocr_text,
    )
    if generated is None:
        # Fallback heuristic
        generated = {
            "Categories": ["#brand_identity", "#visual_identity", "#color_palette", "#typography_system", "#web_branding"],
            "Layout": layout_info.get("layout") or "one column",
            "Visual style": list(style_tags)[:3] or ["minimal"],
            "Palette": [{"hex": h, "theme": "primary" if i == 0 else ("secondary" if i == 1 else "accent")} for i, h in enumerate(colors.get("palette_hex", [])[:3])],
            "Main components": layout_info.get("components", []),
            "Keywords": list(filter(None, objects + style_tags))[:12],
            "Mode": "dark" if "black" in colors.get("colors", []) else "light",
            "Verdict": "Suitable for modern interfaces; heuristic metadata; confidence 0.4",
            "Description": "Automatically inferred branding attributes using heuristics in absence of LLM response.",
        }

    return {
        "extra_metadata": extra_metadata,
        "raw_detections": raw_detections,
        "generated_metadata": generated,
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


