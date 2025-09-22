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

@app.post("/search-by-image", response_model=SearchResultSchema)
async def search_image(file: UploadFile = File(...), top_k: int = Query(10, ge=1, le=100)):
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    # validate if faiss.index exists
    loaded = index.load_if_exists()
    if not loaded:
        raise HTTPException(status_code=404, detail="Index not found")
    
    # save temp file (cross-platform)
    tmp_dir = os.getenv("TMP_DIR", tempfile.gettempdir())
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_basename = f"{uuid.uuid4().hex}_{file.filename}"
    tmp_path = os.path.join(tmp_dir, tmp_basename)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # vectorize
    try:
        qvec = vectorizer.image_to_vector(tmp_path)
    except Exception as e:
        logger.exception("Vectorization failed")
        raise HTTPException(status_code=500, detail="Vectorization error")

    # search
    results = index.search(qvec, top_k=top_k)

    # enrich metadata from DB
    session = get_session()
    enriched = []
    for r in results:
        meta = session.get_image_metadata(r["id"])
        enriched.append({
            "id": r["id"],
            "content_type": meta.content_type if meta else None,
            "image_url": meta.image_url if meta else None,
            "score": r["score"]
        })
    return {"results": enriched}

@app.post("/search", response_model=SearchResultSchema)
async def unified_search(
    file: UploadFile = File(None, description="Target image"),
    query_text: str = Query(None, description="Text query to search for similar images"),
    top_k: int = Query(10, ge=1, le=100)
):
    """
    Unified search endpoint that accepts text, image, or both inputs.
    - If only text is provided: searches using text vector
    - If only image is provided: searches using image vector  
    - If both are provided: combines both vectors using weighted average
    - At least one input (text or image) is required
    """
    # Validate that at least one input is provided
    if not query_text and not file:
        raise HTTPException(status_code=400, detail="At least one input (text query or image file) is required")
    
    # Validate text input
    if query_text and not query_text.strip():
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
    if query_text and query_text.strip():
        try:
            text_vector = vectorizer.text_to_vector(query_text.strip())
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
    initial_k = max(top_k, 50)
    results = index.search(qvec, top_k=initial_k)

    # Infer filters from query_text
    wanted_colors, wanted_objects, wanted_styles = detect_filters_from_text(query_text or "")
    print("wanted_colors:", wanted_colors)
    print("wanted_objects:", wanted_objects)
    print("wanted_styles:", wanted_styles)

    # enrich metadata from DB and filter/re-rank
    session = get_session()
    filtered = []
    for r in results:
        meta = session.get_image_metadata(r["id"])  # type: ImageMetadata
        if not meta:
            continue
        extra = meta.extra_metadata or {}
        colors = set([c.lower() for c in (extra.get("colors") or [])])
        objects = set([o.lower() for o in (extra.get("objects") or [])])
        style_tags = set([t.lower() for t in (extra.get("style_tags") or [])])

        # Apply inclusive filters if provided
        # if wanted_colors and not (wanted_colors & colors):
        #     continue
        # if wanted_objects and not (wanted_objects & objects):
        #     continue
        # if wanted_styles and not (wanted_styles & style_tags):
        #     continue

        # Re-ranking: always apply bonus for metadata matches
        score = float(r["score"]) if r.get("score") is not None else 0.0
        bonus = 0.0
        # if wanted_colors:
        #     bonus += 0.02 * len(wanted_colors & colors)
        # if wanted_objects:
        #     bonus += 0.03 * len(wanted_objects & objects)
        # if wanted_styles:
        #     bonus += 0.04 * len(wanted_styles & style_tags)
        # score += bonus

        filtered.append({
            "id": r["id"],
            "content_type": meta.content_type,
            "image_url": meta.image_url,
            "score": score
        })

    # Sort by updated score desc and trim to top_k
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return {"results": filtered[:top_k]}

@app.post("/search-by-text", response_model=SearchResultSchema)
async def search_by_text(query: str = Query(..., description="Text query to search for similar images"), top_k: int = Query(10, ge=1, le=100)):
    """
    Search for similar images using text query instead of image upload.
    Uses CLIP's text encoder to convert text to vector and search the FAISS index.
    """
    # Validate query
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    # validate if faiss.index exists
    loaded = index.load_if_exists()
    if not loaded:
        raise HTTPException(status_code=404, detail="Index not found")
    
    # vectorize text
    try:
        qvec = vectorizer.text_to_vector(query.strip())
    except Exception as e:
        logger.exception("Text vectorization failed")
        raise HTTPException(status_code=500, detail="Text vectorization error")

    # search
    results = index.search(qvec, top_k=top_k)

    # enrich metadata from DB
    session = get_session()
    enriched = []
    for r in results:
        meta = session.get_image_metadata(r["id"])
        enriched.append({
            "id": r["id"],
            "content_type": meta.content_type if meta else None,
            "image_url": meta.image_url if meta else None,
            "score": r["score"]
        })
    return {"results": enriched}

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
        try:
            if vec is not None and len(vec) > 0:
                extra_meta = extract_image_metadata(tmp_path, vectorizer, vec[0])
        except Exception:
            pass
        session.add_image(id_=image_id, content_type=file.content_type, image_url=stored_url, metadata=extra_meta)
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
