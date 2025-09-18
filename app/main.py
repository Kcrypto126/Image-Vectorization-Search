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
from .tasks import full_reindex
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
    print(loaded)
    # If index doesn't exist, build from DB and persist
    # if not loaded:
    #     logger.info("No existing index found. Running full reindex...")
    #     return index.run_full_reindex()

@app.get("/")
async def root():
    return {"message": "Image Search Backend running"}

@app.post("/search", response_model=SearchResultSchema)
async def search_image(file: UploadFile = File(...), top_k: int = Query(10, ge=1, le=100)):
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
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

@app.post("/admin/index")
async def reindex(background_tasks: BackgroundTasks):
    background_tasks.add_task(full_reindex)
    return {"status": "reindex started"}

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
                stored_url = save_image(image_id, file.filename, rf.read())
        except Exception as storage_exc:
            logger.exception("Failed to store image in storage backend")
            raise HTTPException(status_code=500, detail=f"Failed to store image: {storage_exc}")
        # Record in DB
        session = get_session()
        session.add_image(id_=image_id, content_type=file.content_type, image_url=stored_url, metadata=None)
        # Vectorize and add to index incrementally
        try:
            vec = vectorizer.image_to_vector(tmp_path)
            index.add(vec.astype("float32"), [image_id])
            index.save()
        except Exception:
            logger.exception("Vectorization failed during upload; record saved without index update")
        return {"id": image_id, "image_url": stored_url}
    finally:
        # Clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logger.warning(f"Failed to remove temp file: {tmp_path}")
