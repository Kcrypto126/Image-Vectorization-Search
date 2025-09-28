# app/main.py
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from .vectorizer import Vectorizer
from .index_adapter import IndexAdapter
from .db import get_session, init_db, ImageMetadata
from .schemas import SearchResultSchema
from .storage import save_image
from .tasks import full_reindex, ingest_local_images, ingest_online_images
from .db import ImageMetadata
from .metadata import extract_image_metadata, detect_filters_from_text
import tempfile
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('image_search_requests_total', 'Total search requests', ['endpoint', 'method', 'status'])
REQUEST_LATENCY = Histogram('image_search_request_latency_seconds', 'Latency in seconds', ['endpoint'])

# Config
MODEL_NAME = os.getenv("CLIP_MODEL", "ViT-B/32")
DIM = int(os.getenv("VECTOR_DIM", "512"))
INDEX_BACKEND = os.getenv("INDEX_BACKEND", "faiss")  # or 'qdrant'

app = FastAPI(title="Image Search Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the 'data/images' directory on the '/data/images' URL path
app.mount("/data/images", StaticFiles(directory="data/images"), name="images")

# init DB & index
init_db(os.getenv("DATABASE_URL"))

vectorizer = Vectorizer(model_name=MODEL_NAME, device=os.getenv("DEVICE", "cpu"))
index = IndexAdapter(backend=INDEX_BACKEND, dim=DIM)

@app.on_event("startup")
async def on_startup():
    logger.info("Starting service")
    loaded = index.load_if_exists()
    # If index doesn't exist, build from DB and persist
    if not loaded:
        logger.info("No existing index found. Running full reindex...?")

@app.get("/")
async def root():
    return {"message": "Image Search Backend running"}


@app.post("/search", response_model=SearchResultSchema)
async def search(
    query: str = Query(None, description="Text query to search for similar images"),
    file: UploadFile = File(None, description="Target image"),
    top_k: int = Query(10, ge=1, le=100)
):
    """
    Enhanced search endpoint that accepts text query, image file, or both inputs.
    - If only text is provided: searches using text vector
    - If only image is provided: searches using image vector  
    - If both are provided: combines both vectors using weighted average
    - At least one input (text or image) is required
    - Enhanced metadata filtering and ranking based on all available metadata
    """
    # Validate that at least one input is provided
    if not query and not file:
        raise HTTPException(status_code=400, detail="At least one input (query or file) is required")
    
    # Validate text input
    if query and not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    # Validate image content type if provided
    if file and file.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    # validate if faiss.index exists
    loaded = index.load_if_exists()
    if not loaded:
        raise HTTPException(status_code=404, detail="Index not found")
    
    text_vector = None
    image_vector = None
    
    # Process text input
    if query and query.strip():
        try:
            text_vector = vectorizer.text_to_vector(query.strip())
        except Exception as e:
            logger.exception("Text vectorization failed")
            raise HTTPException(status_code=500, detail="Text vectorization error")
    
    # Process image input
    if file:
        # save temp file (cross-platform)
        tmp_dir = os.getenv("TMP_DIR", tempfile.gettempdir())
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_basename = f"{uuid.uuid4().hex}_{file.filename}"
        tmp_path = os.path.join(tmp_dir, tmp_basename)
        
        try:
            with open(tmp_path, "wb") as f:
                f.write(await file.read())
            
            try:
                image_vector = vectorizer.image_to_vector(tmp_path)
            except Exception as e:
                logger.exception("Image vectorization failed")
                raise HTTPException(status_code=500, detail="Image vectorization error")
        finally:
            # Clean up temp file
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                logger.warning(f"Failed to remove temp file: {tmp_path}")
    
    # Determine text_weight automatically
    if text_vector is not None and image_vector is not None:
        weight = 0.5
    elif text_vector is not None and image_vector is None:
        weight = 1.0
    elif text_vector is None and image_vector is not None:
        weight = 0.0

    # Combine vectors if both are available
    try:
        qvec = vectorizer.combine_vectors(text_vector, image_vector, weight)
    except Exception as e:
        logger.exception("Vector combination failed")
        raise HTTPException(status_code=500, detail="Vector combination error")

    # search initial candidates from FAISS (retrieve more for filtering headroom)
    initial_k = max(top_k * 3, 100)  # Get more candidates for better filtering
    results = index.search(qvec, top_k=initial_k)

    # Enhanced metadata analysis from query
    query_lower = (query or "").lower()
    
    # enrich metadata from DB and filter/re-rank
    session = get_session()
    filtered = []
    for r in results:
        meta = session.get_image_metadata(r["id"])  # type: ImageMetadata
        if not meta:
            continue
        
        # Get all metadata
        extra = meta.extra_metadata or {}
        raw = meta.raw_detections or {}
        generated = meta.generated_metadata or {}
        
        # Extract metadata fields for matching
        colors = set([c.lower() for c in (extra.get("colors") or [])])
        objects = set([o.lower() for o in (extra.get("objects") or [])])
        style_tags = set([t.lower() for t in (extra.get("style_tags") or [])])
        ocr_text = (raw.get("ocr_text") or "").lower()
        categories = set([c.lower() for c in (generated.get("Categories") or [])])
        visual_style = set([s.lower() for s in (generated.get("Visual style") or [])])
        keywords = set([k.lower() for k in (generated.get("Keywords") or [])])
        description = (generated.get("Description") or "").lower()
        layout = (generated.get("Layout") or "").lower()
        mode = (generated.get("Mode") or "").lower()
        components = set([c.lower() for c in (raw.get("components") or [])])
        
        # Enhanced re-ranking with comprehensive metadata matching
        score = float(r["score"]) if r.get("score") is not None else 0.0
        bonus = 0.0
        
        # Query tokenization for better matching
        import re
        query_tokens = set(re.findall(r"[a-z0-9]+", query_lower))
        
        # Color matching (higher weight for exact matches)
        if query_tokens & colors:
            bonus += 0.05 * len(query_tokens & colors)
        
        # Object matching (high weight for detected objects)
        if query_tokens & objects:
            bonus += 0.08 * len(query_tokens & objects)
        
        # Style matching
        if query_tokens & style_tags:
            bonus += 0.06 * len(query_tokens & style_tags)
        if query_tokens & visual_style:
            bonus += 0.07 * len(query_tokens & visual_style)
        
        # OCR text matching (very high weight for text content)
        if query_tokens and any(token in ocr_text for token in query_tokens):
            ocr_matches = sum(1 for token in query_tokens if token in ocr_text)
            bonus += 0.15 * ocr_matches
        
        # Category matching
        if query_tokens & categories:
            bonus += 0.06 * len(query_tokens & categories)
        
        # Keyword matching
        if query_tokens & keywords:
            bonus += 0.05 * len(query_tokens & keywords)
        
        # Description matching
        if query_tokens and any(token in description for token in query_tokens):
            desc_matches = sum(1 for token in query_tokens if token in description)
            bonus += 0.10 * desc_matches
        
        # Layout and mode matching
        if any(token in layout for token in query_tokens):
            bonus += 0.03
        if any(token in mode for token in query_tokens):
            bonus += 0.03
        
        # Component matching
        if query_tokens & components:
            bonus += 0.04 * len(query_tokens & components)
        
        # Exact phrase matching in OCR text (highest weight)
        if query_lower in ocr_text:
            bonus += 0.20
        
        # Exact phrase matching in description
        if query_lower in description:
            bonus += 0.15
        
        score += bonus

        filtered.append({
            "id": r["id"],
            "content_type": meta.content_type,
            "image_url": meta.image_url,
            "score": score
        })

    # Sort by updated score desc and trim to top_k
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return {"results": filtered[:top_k]}




@app.post("/admin/index")
async def reindex(background_tasks: BackgroundTasks):
    background_tasks.add_task(full_reindex)
    logger.info("Reindex started...loading...")
    return {"status": "reindex started"}

@app.post("/admin/ingest-local")
async def ingest_local(background_tasks: BackgroundTasks):
    """
    Scan data/images directory and store metadata for all images into DB
    """
    background_tasks.add_task(ingest_local_images)
    logger.info("Ingest started...loading...")
    return {"status": "ingest started"}

@app.post("/admin/ingest-online")
async def ingest_from_online(background_tasks: BackgroundTasks):
    """
    Parse app/data.json and ingest images using link + media[0].url
    """
    background_tasks.add_task(ingest_online_images)
    logger.info("Ingest from api started...loading...")
    return {"status": "ingest started"}

@app.post("/uploads")
async def upload_image(file: UploadFile = File(...)):
    logger.info("uploading...")

    if file.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    tmp_dir = os.getenv("TMP_DIR", tempfile.gettempdir())
    os.makedirs(tmp_dir, exist_ok=True)
    image_id = str(uuid.uuid4())
    tmp_path = os.path.join(tmp_dir, f"{image_id}_{file.filename}")

    try:
        # Write uploaded file to temp path
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        # Try to store file (local or s3) using app.storage
        try:
            with open(tmp_path, "rb") as rf:
                stored_url = save_image(file.filename, rf.read())
        except Exception as storage_exc:
            logger.exception("Failed to store image in storage backend")
            raise HTTPException(status_code=500, detail=f"Failed to store image: {storage_exc}")
        # Record in DB
        session = get_session()
        # Vectorize first to also compute metadata
        vec = None
        try:
            vec = vectorizer.image_to_vector(tmp_path)
        except Exception:
            logger.exception("Vectorization failed during upload; record saved without index update")
        extra_meta = None
        raw_det = None
        gen_meta = None
        try:
            if vec is not None and len(vec) > 0:
                meta_bundle = extract_image_metadata(tmp_path, vectorizer, vec[0]) or {}
                extra_meta = meta_bundle.get("extra_metadata")
                raw_det = meta_bundle.get("raw_detections")
                gen_meta = meta_bundle.get("generated_metadata")
        except Exception:
            pass
        session.add_image(id_=image_id, content_type=file.content_type, image_url=stored_url, metadata=extra_meta, raw_detections=raw_det, generated_metadata=gen_meta)
        # Vectorize and add to index incrementally
        try:
            if vec is not None and len(vec) > 0:
                index.add(vec.astype("float32"), [image_id])
                index.save()
        except Exception:
            logger.exception("Index update failed during upload")
        return {"id": image_id, "image_url": stored_url}
    finally:
        # Clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logger.warning(f"Failed to remove temp file: {tmp_path}")
