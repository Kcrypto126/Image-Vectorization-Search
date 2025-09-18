# app/main.py
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from .vectorizer import Vectorizer
from .index_adapter import IndexAdapter
from .db import get_session, init_db, ImageMetadata
from .schemas import SearchResultSchema
from .utils.storage import upload_file
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

# init DB & index
init_db(os.getenv("DATABASE_URL", "postgresql://postgres:Roastery818@localhost:5432/imagedb1"))

vectorizer = Vectorizer(model_name=MODEL_NAME, device=os.getenv("DEVICE", "cpu"))
index = IndexAdapter(backend=INDEX_BACKEND, dim=DIM)

@app.on_event("startup")
async def on_startup():
    logger.info("Starting service")
    loaded = index.load_if_exists()
    # If index doesn't exist, build from DB and persist
    if not loaded:
        logger.info("No existing index found. Running full reindex...")
        session = get_session()
        # fetch all images
        images = session.session.query(ImageMetadata).all()
        import numpy as np
        batch_size = int(os.getenv("REINDEX_BATCH", "128"))
        total = 0
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            ids = []
            vecs = []
            for img in batch:
                try:
                    v = vectorizer.image_to_vector(img.image_url)
                except Exception:
                    continue
                vecs.append(v[0])
                ids.append(img.id)
            if vecs:
                vecs = np.vstack(vecs).astype("float32")
                index.add(vecs, ids)
                total += len(ids)
        index.save()
        logger.info(f"Full reindex complete, vectors added: {total}")

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def record_metrics(endpoint: str):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with REQUEST_LATENCY.labels(endpoint).time():
                try:
                    res = await func(*args, **kwargs)
                    REQUEST_COUNT.labels(endpoint, "POST", "200").inc()
                    return res
                except HTTPException as e:
                    REQUEST_COUNT.labels(endpoint, "POST", str(e.status_code)).inc()
                    raise
        return wrapper
    return decorator

@app.post("/search", response_model=SearchResultSchema)
@record_metrics("/search")
async def search_image(file: UploadFile = File(...), top_k: int = Query(10, ge=1, le=100)):
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
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
async def admin_full_reindex():
    # Rebuild index from scratch from DB
    session = get_session()
    images = session.session.query(ImageMetadata).all()
    import numpy as np
    index.reset()
    batch_size = int(os.getenv("REINDEX_BATCH", "128"))
    total = 0
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        ids = []
        vecs = []
        for img in batch:
            try:
                v = vectorizer.image_to_vector(img.image_url)
            except Exception:
                continue
            vecs.append(v[0])
            ids.append(img.id)
        if vecs:
            vecs = np.vstack(vecs).astype("float32")
            index.add(vecs, ids)
            total += len(ids)
    index.save()
    return {"status": "ok", "indexed": total}

@app.post("/uploads")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    tmp_dir = os.getenv("TMP_DIR", tempfile.gettempdir())
    os.makedirs(tmp_dir, exist_ok=True)
    image_id = str(uuid.uuid4())
    tmp_path = os.path.join(tmp_dir, f"{image_id}_{file.filename}")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    # store file (local or s3)
    key = f"{image_id}.jpg"
    stored_url = upload_file(tmp_path, key)
    # record in DB
    session = get_session()
    session.add_image(id_=image_id, content_type=file.content_type, image_url=stored_url, metadata=None)
    # vectorize and add to index incrementally
    try:
        vec = vectorizer.image_to_vector(tmp_path)
        index.add(vec.astype("float32"), [image_id])
        index.save()
    except Exception:
        logger.exception("Vectorization failed during upload; record saved without index update")
    return {"id": image_id, "image_url": stored_url}
