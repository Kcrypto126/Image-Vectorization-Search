# app/metadata.py
import os
import math
import logging
from typing import Dict, List, Tuple, Optional
import json
import re

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
import base64
import io

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
    Prefer LayoutParser Detectron2 if available; otherwise fall back to a fast OpenCV heuristic.
    Returns: {layout: one of [one column, card grid, split] or None, components: [strings]}
    """
    # Try LayoutParser if present
    if _LP_AVAILABLE and hasattr(lp, 'Detectron2LayoutModel'):
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
            counts = {"text": 0, "title": 0, "list": 0, "table": 0, "figure": 0}
            for b in layout_result:
                lbl = getattr(b, 'type', None) or getattr(b, 'label', None)
                if lbl in counts:
                    counts[lbl] += 1
            if counts["figure"] >= 3 or counts["table"] >= 2:
                layout = "card grid"
            elif counts["text"] > 0 and counts["figure"] > 0:
                layout = "split"
            else:
                layout = "one column"
            components: List[str] = []
            if counts["table"]:
                components.append("table")
            if counts["list"]:
                components.append("sidebar")
            if counts["title"]:
                components.append("breadcrumbs")
            return {"layout": layout, "components": list(sorted(set(components)))}
        except Exception:
            pass
    # OpenCV heuristic fallback
    if not _CV_AVAILABLE:
        return {"layout": None, "components": []}
    try:
        print("----> opencv")
        img = cv2.imread(image_path)
        if img is None:
            return {"layout": None, "components": []}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dil = cv2.dilate(thr, kernel, iterations=2)
        contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            if area < (w * h) * 0.005:
                continue
            rects.append((x, y, ww, hh))
        rects = sorted(rects, key=lambda r: (r[1], r[0]))
        num_blocks = len(rects)
        cols_est = 1
        if num_blocks >= 2:
            xs = sorted([x for (x, _, ww, _) in rects] + [x+ww for (x, _, ww, _) in rects])
            gaps = [xs[i+1]-xs[i] for i in range(0, len(xs)-1, 2)]
            large_gaps = [g for g in gaps if g > w * 0.12]
            if len(large_gaps) >= 1:
                cols_est = 2
        if num_blocks >= 6:
            layout = "card grid"
        elif cols_est >= 2:
            layout = "split"
        else:
            layout = "one column"
        components: List[str] = []

        # Navbar: full-width, short bar near top
        for (x, y, ww, hh) in rects:
            if y < h * 0.15 and ww > w * 0.85 and hh < h * 0.18:
                components.append("navbar")
                break

        # Sidebar: multiple columns or large tall block on left/right
        if cols_est >= 2 or any((x < w*0.15 or x+ww > w*0.85) and hh > h*0.5 for (x, y, ww, hh) in rects):
            components.append("sidebar")

        # Table: large wide block; later upgrade with grid-line detection
        if any(ww > w * 0.45 and hh > h * 0.18 for (_, _, ww, hh) in rects):
            components.append("table")

        # Tabs: 3+ small/medium boxes aligned horizontally near top third
        top_band = [r for r in rects if r[1] < h * 0.35 and r[3] < h * 0.18]
        top_band.sort(key=lambda r: r[0])
        seq = 0
        last_x = -1
        for (x, y, ww, hh) in top_band:
            if ww > w*0.09 and ww < w*0.28:
                if last_x >= 0 and x - last_x < w * 0.2:
                    seq += 1
                else:
                    seq = 1
                last_x = x
                if seq >= 3:
                    components.append("tabs")
                    break

        # Search bar: wide, short input-like box near top
        if any(y < h*0.3 and ww > w*0.35 and hh < 70 for (x, y, ww, hh) in rects):
            components.append("search_bar")

        # Forms: multiple input-like boxes
        input_like = [(x, y, ww, hh) for (x, y, ww, hh) in rects if ww > w*0.3 and 28 <= hh <= 90]
        if len(input_like) >= 2:
            components.append("form")

        # Charts: line segments with diverse angles
        try:
            import numpy as np
            edges = cv2.Canny(gray, 100, 200)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=int(w*0.1), maxLineGap=10)
            angle_bins = set()
            if lines is not None:
                for l in lines[:200]:
                    x1,y1,x2,y2 = l[0]
                    ang = abs(math.atan2(y2-y1, x2-x1))
                    # bucket into coarse bins
                    angle_bins.add(int(ang*10))
            if lines is not None and len(lines) > 20 and len(angle_bins) >= 4:
                components.append("chart")
        except Exception:
            pass

        # Sliders: long thin bar with small circle nearby
        try:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=80, param2=30, minRadius=6, maxRadius=20)
            has_slider_track = any(ww > w*0.3 and hh < 20 for (_, _, ww, hh) in rects)
            if has_slider_track and circles is not None:
                components.append("slider")
        except Exception:
            pass

        # Video: triangle-like contour near center or OCR keyword
        try:
            cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                x, y, ww, hh = cv2.boundingRect(c)
                if len(approx) == 3 and abs((x+ww/2) - w/2) < w*0.3 and abs((y+hh/2) - h/2) < h*0.3:
                    components.append("video")
                    break
        except Exception:
            pass

        # OCR-assisted hints
        text = extract_ocr_text(image_path)
        t = text.lower()
        if t:
            if ("search" in t or "find" in t) and "search_bar" not in components:
                components.append("search_bar")
            if any(k in t for k in ["email", "password", "username", "login", "sign in", "sign up", "submit"]):
                if "form" not in components:
                    components.append("form")
            if any(k in t for k in ["home >", "/", "home >", "breadcrumbs"]):
                components.append("breadcrumbs")
            if any(k in t for k in ["star", "rating", "★", "☆"]):
                components.append("rating")
            if any(k in t.split() for k in ["new", "sale", "beta", "pro"]):
                components.append("badges")
            if any(k in t for k in ["prev", "next"]) or re.search(r"\b1\s+2\s+3\b", t):
                components.append("pagination")
            if any(k in t for k in ["map", "satellite", "terrain"]):
                components.append("map")
            if "video" in t or "play" in t:
                components.append("video")

        return {"layout": layout, "components": list(sorted(set(components)))}
    except Exception:
        return {"layout": None, "components": []}


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


def _image_to_data_url(path: str) -> Optional[str]:
    try:
        with Image.open(path) as im:
            fmt = 'PNG'
            buf = io.BytesIO()
            im.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/{fmt.lower()};base64,{b64}"
    except Exception:
        return None


_ALLOWED_CATEGORIES = [
    "logo","logo_variations","logomark","lettermark","monogram","symbol_design","typographic_logo","wordmark","brand_identity","visual_identity","brand_strategy","verbal_identity","color_palette","typography_system","photography_style","icon_system","illustration_style","motion_branding","brand_voice","tone_of_voice","grid_system","layout_principles","brand_guidelines","brand_assets","brand_kit","stationery_design","packaging_design","social_branding","web_branding","mobile_branding","presentation_design","brand_collateral","brand_merchandise","print_assets","digital_branding","rebranding","naming","tagline","slogan","startup_branding","luxury_branding","tech_branding","fashion_branding","beauty_branding","ecommerce_branding","fintech_branding","healthcare_branding","real_estate_branding","hospitality_branding","education_branding","crypto_branding","gaming_branding","eco_branding","minimal_branding","bold_branding","colorful_identity","blackandwhite_identity","dynamic_identity","static_identity","responsive_logo","geometric_logo","organic_logo","3d_logo","flat_logo","animated_logo","modern_branding","classic_branding","retro_branding","playful_identity","professional_identity","friendly_branding","serious_tone","abstract_logo","mascot_logo","symbolic_logo","custom_typography","brand_positioning","target_audience","personas","brand_values","brand_promise","brand_story","brand_architecture","subbrands","co_branded_identity","personal_branding","product_branding","corporate_branding","internal_branding","employer_branding","event_branding","brand_consistency","branding_for_pitch","packaging_system","icon_set_design","brand_photography","mockup_presentation"
]
_ALLOWED_BLOCKS = ["logo","identity","safe_zone","color_system","typography","visual_language","applications","brand_strategy","tone_of_voice","documentation"]
_ALLOWED_DEVICE = ["mobile","desktop","responsive"]
_ALLOWED_LOGO_ELEMENTS = ["metaphor","letter","combination of different ideas into one object"]
_ALLOWED_COMPONENTS = ["navbar","breadcrumbs","sidebar","tabs","table","chart","form","search_bar","rating","badges","pagination","slider","video","map"]
_ALLOWED_CONTENT_HINTS = ["logo variation","brand asset usage rules","color palette","typography","idea construction","stylistic direction","mockups"]
_ALLOWED_LAYOUT = ["one column","card grid","split"]
_ALLOWED_VISUAL_STYLE = ["minimal","brutalist","glassmorphism","neumorphism","gradient","3d","illustration","photography","dark","light"]


def _gpt_generate_metadata(objects: List[str], colors_hex: List[str], layout: Optional[str], style_tags: List[str], ocr_text: str, components: List[str], image_path: Optional[str]) -> Optional[Dict]:
    """
    Calls OpenAI (if available) to generate expert metadata JSON per spec.
    Falls back to heuristic if API not configured.
    """
    if not _OPENAI_AVAILABLE:
        return None
    try:
        client = OpenAI()
        sys = {"role": "system", "content": (
            "You are a senior brand and product design analyst. Respond ONLY with strict JSON (no markdown, no comments). "
            "Select values only from provided allowed lists where specified. Categories must NOT include leading #. "
            "Palette must be exactly 3 entries, each an object {hex, theme}. Typography must include number_of_fonts, headings, text. "
            "Mode must be one of: dark, light, mixed. Keywords must have 10–25 items (popular search queries). "
            "Verdict must be a single sentence describing suitability + 'Confidence at X.X' where X in [0,1]. "
            "Description must be a coherent, detailed, expert-level English paragraph."
        )}

        user_text = (
            "Analyze the provided image and detections to produce expert metadata. "
            "Allowed values:\n" 
            f"Categories: {json.dumps(_ALLOWED_CATEGORIES)}\n"
            f"Block: {json.dumps(_ALLOWED_BLOCKS)}\n"
            f"Device: {json.dumps(_ALLOWED_DEVICE)}\n"
            f"Logo elements: {json.dumps(_ALLOWED_LOGO_ELEMENTS)}\n"
            f"Main components: {json.dumps(_ALLOWED_COMPONENTS)}\n"
            f"Content hints: {json.dumps(_ALLOWED_CONTENT_HINTS)}\n"
            f"Layout: {json.dumps(_ALLOWED_LAYOUT)}\n"
            f"Visual style: {json.dumps(_ALLOWED_VISUAL_STYLE)}\n"
            "Return strict JSON with keys (omit any that are not applicable): \n"
            "Categories, Block, Device, Logo elements, Main components, Content hints, Layout, Visual style, "
            "Palette, Typography, Icons / Illustrations, Complexity, Data density, Verdict, Keywords, Description, Mode.\n"
            "Rules: Categories 5–6 items from list (no leading #). Mode in {dark, light, mixed}. Keywords 10–25 items. "
            "Verdict format example: 'Ideal for X because Y. Confidence at 0.87'. Description: expert-level, cohesive English.\n"
            "Detections: " + json.dumps({
                "objects": objects,
                "colors_hex": colors_hex,
                "layout": layout,
                "styles": style_tags,
                "ocr_text": ocr_text,
                "components": components,
            })
        )

        messages = [sys]
        img_url = _image_to_data_url(image_path) if image_path else None
        if img_url:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            })
        else:
            messages.append({"role": "user", "content": user_text})

        resp = client.chat.completions.create(
            model=os.getenv("GEN_METADATA_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.2,
            max_tokens=900
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
        return _normalize_generated(data) if isinstance(data, dict) else None
    except Exception as e:
        _logger.warning(f"GPT metadata generation failed: {e}")
        return None


def _normalize_generated(data: Dict) -> Dict:
    # Normalize Categories: remove # and restrict to allowed set
    cats = data.get("Categories") or []
    cats_clean = []
    for c in cats if isinstance(cats, list) else []:
        if isinstance(c, str):
            c2 = c.lstrip('#').strip()
            if c2 in _ALLOWED_CATEGORIES and c2 not in cats_clean:
                cats_clean.append(c2)
    if cats_clean:
        data["Categories"] = cats_clean[:6]

    # Mode normalization
    mode = data.get("Mode")
    if isinstance(mode, str):
        m = mode.lower()
        if m not in {"dark","light","mixed"}:
            # try infer from colors
            m = "dark" if any(h.lower() in {"#000000","#111111"} for h in (data.get("Palette") or [])) else "light"
        data["Mode"] = m

    # Keywords length constraint 10–25
    kws = data.get("Keywords") or []
    if isinstance(kws, list):
        kws = [k for k in kws if isinstance(k, str) and k.strip()]
        if len(kws) < 10:
            # Attempt to enrich from detections if present in payload mirror
            pass
        data["Keywords"] = kws[:25]

    # Palette ensure exactly 3 objects {hex, theme}
    pal = data.get("Palette") or []
    norm_pal = []
    for i, p in enumerate(pal if isinstance(pal, list) else []):
        if isinstance(p, dict) and p.get("hex"):
            theme = p.get("theme") or ("primary" if i == 0 else ("secondary" if i == 1 else "accent"))
            norm_pal.append({"hex": p.get("hex"), "theme": theme})
    data["Palette"] = norm_pal[:3]

    # Typography keys rename if needed
    typo = data.get("Typography")
    if isinstance(typo, dict):
        if "Number of fonts" in typo:
            typo["number_of_fonts"] = typo.pop("Number of fonts")
        if "Headings" in typo:
            typo["headings"] = typo.pop("Headings")
        if "Text" in typo:
            typo["text"] = typo.pop("Text")
        data["Typography"] = typo

    # Verdict format enforcement (best effort)
    v = data.get("Verdict")
    if isinstance(v, str) and "Confidence" not in v and "confidence" not in v:
        data["Verdict"] = v.rstrip('.') + ". Confidence at 0.8"

    # Ensure Description is a string
    desc = data.get("Description")
    if not isinstance(desc, str):
        data["Description"] = ""

    # Main components restriction
    comps = data.get("Main components") or []
    if isinstance(comps, list):
        comps = [c for c in comps if isinstance(c, str) and c in _ALLOWED_COMPONENTS]
        data["Main components"] = list(dict.fromkeys(comps))

    return data


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
        components=layout_info.get("components", []),
        image_path=image_path,
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


