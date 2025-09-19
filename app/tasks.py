import os
import tempfile
import requests
from urllib.parse import urlparse
from celery import Celery
from dotenv import load_dotenv
import numpy as np
import uuid
import logging

from .db import get_session, ImageMetadata
from .vectorizer import Vectorizer
from .index_adapter import IndexAdapter

# Load environment variables from .env file
load_dotenv()
logger = logging.getLogger(__name__)

CELERY_BROKER = os.getenv("CELERY_BROKER", "redis://redis:6379/0")
celery = Celery('tasks', broker=CELERY_BROKER)

def _guess_content_type_from_ext(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"

def _get_local_path_from_url(image_url):
    """
    If image_url is a local file path, return it.
    If it's an http(s) URL, download to a temp file and return the temp file path.
    """
    parsed = urlparse(image_url)
    if parsed.scheme in ("http", "https"):
        # Download the image to a temp file
        try:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            suffix = os.path.splitext(parsed.path)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpf:
                tmpf.write(resp.content)
                return tmpf.name
        except Exception as e:
            raise RuntimeError(f"Failed to download image from {image_url}: {e}")
    elif parsed.scheme == "" or parsed.scheme == "file":
        # Local file path
        if parsed.scheme == "file":
            path = parsed.path
        else:
            path = image_url
        if os.path.exists(path):
            return path
        else:
            raise RuntimeError(f"Local image file does not exist: {path}")
    else:
        raise RuntimeError(f"Unsupported image_url scheme: {image_url}")

@celery.task(bind=True)
def full_reindex(self):
    session = get_session()
    try:
        index = IndexAdapter(
            backend=os.getenv("INDEX_BACKEND", "faiss"),
            dim=int(os.getenv("VECTOR_DIM", "512"))
        )
        vectorizer = Vectorizer(
            model_name=os.getenv("CLIP_MODEL", "ViT-B/32"),
            device=os.getenv("DEVICE", "cpu")
        )
        # fetch all images
        images = session.session.query(ImageMetadata).all()
        if not images:
            logger.info("No images found from DB")
            return {"status": "done", "count": 0}

        chunk_size = 128
        for i in range(0, len(images), chunk_size):
            batch = images[i:i+chunk_size]
            ids = []
            vecs = []
            for img in batch:
                temp_path = None
                try:
                    # Determine if image_url is a local file or a URL
                    temp_path = _get_local_path_from_url(img.image_url)
                    v = vectorizer.image_to_vector(temp_path)
                    if v is not None and len(v) > 0:
                        vecs.append(v[0])
                        ids.append(img.id)
                except Exception as e:
                    print(f"Error vectorizing image {img.id}: {e}")
                finally:
                    # Clean up temp file if we downloaded it
                    if temp_path and temp_path != img.image_url and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
            if vecs and ids:
                vecs_np = np.vstack(vecs).astype("float32")
                index.add(vecs_np, ids)
        index.save()
        logger.info("Successfully reindexed!!!")
        return {"status": "done", "count": len(images)}
    except Exception as e:
        print(f"Error in full_reindex: {e}")
        return {"status": "error", "error": str(e)}


@celery.task(bind=True)
def ingest_local_images(self):
    """
    Scan data/images directory and store metadata for all images into DB.
    Creates new UUID ids, infers content_type from file extension, sets image_url
    to FastAPI static path, keeps metadata as None, created_at via DB default.
    Skips files already present (matched by image_url).
    """
    images_dir = os.path.join("data", "images")
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    processed = 0
    skipped = 0
    errors = 0
    session = get_session()
    # Prepare vectorizer and index
    index = IndexAdapter(
        backend=os.getenv("INDEX_BACKEND", "faiss"),
        dim=int(os.getenv("VECTOR_DIM", "512"))
    )
    vectorizer = Vectorizer(
        model_name=os.getenv("CLIP_MODEL", "ViT-B/32"),
        device=os.getenv("DEVICE", "cpu")
    )
    batch_size = 128
    pending_vecs = []
    pending_ids = []
    try:
        if not os.path.isdir(images_dir):
            logger.info("No image found from data/images")
            return {"status": "done", "processed": 0, "skipped": 0, "errors": 0}

        for name in os.listdir(images_dir):
            path = os.path.join(images_dir, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in allowed_exts:
                continue

            # Build the public URL served by FastAPI static files
            image_url = f"http://127.0.0.1:8000/data/images/{name}"

            # Skip if already ingested (by URL)
            try:
                existing = session.session.query(ImageMetadata).filter(ImageMetadata.image_url == image_url).first()
                if existing:
                    skipped += 1
                    continue
            except Exception:
                # If querying fails for some reason, attempt to continue
                pass

            try:
                img_id = str(uuid.uuid4())
                content_type = _guess_content_type_from_ext(name)
                # Vectorize
                v = vectorizer.image_to_vector(path)
                if v is None or len(v) == 0:
                    errors += 1
                    continue
                # Save DB row first
                session.add_image(id_=img_id, content_type=content_type, image_url=image_url, metadata=None)
                # Queue for index add
                pending_vecs.append(v[0])
                pending_ids.append(img_id)
                processed += 1
                # Flush batch to index
                if len(pending_vecs) >= batch_size:
                    vecs_np = np.vstack(pending_vecs).astype("float32")
                    index.add(vecs_np, pending_ids)
                    pending_vecs = []
                    pending_ids = []
            except Exception:
                errors += 1
                continue

        # Flush remaining vectors
        if pending_vecs:
            try:
                vecs_np = np.vstack(pending_vecs).astype("float32")
                index.add(vecs_np, pending_ids)
            except Exception:
                errors += len(pending_vecs)
            finally:
                pending_vecs = []
                pending_ids = []

        # Persist index
        try:
            index.save()
        except Exception:
            # If saving index fails, report but keep DB inserts
            pass
        logger.info("Successfully ingested and indexed!!!")
        return {"status": "done", "processed": processed, "skipped": skipped, "errors": errors}
    except Exception as e:
        return {"status": "error", "error": str(e), "processed": processed, "skipped": skipped, "errors": errors}
