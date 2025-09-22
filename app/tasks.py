import os
import tempfile
import requests
import json
import time
from urllib.parse import urlparse, urljoin
from celery import Celery
from dotenv import load_dotenv
import numpy as np
import uuid
import logging

from .db import get_session, ImageMetadata, init_db
from .vectorizer import Vectorizer
from .index_adapter import IndexAdapter
from .metadata import extract_image_metadata

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
        # Download the image to a temp file with retries and backoff
        timeout_read = float(os.getenv("DOWNLOAD_TIMEOUT", "20"))
        retries = int(os.getenv("DOWNLOAD_RETRIES", "3"))
        headers = {"User-Agent": os.getenv("DOWNLOAD_UA", "VectorImageBot/1.0")}
        last_exc = None
        for attempt in range(retries):
            try:
                # connect timeout 5s, read timeout configurable
                resp = requests.get(image_url, timeout=(5, timeout_read), headers=headers)
                resp.raise_for_status()
                content = resp.content
                if not content:
                    raise RuntimeError("Empty response body")
                suffix = os.path.splitext(parsed.path)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpf:
                    tmpf.write(content)
                    return tmpf.name
            except Exception as e:
                last_exc = e
                # exponential backoff up to 5s
                time.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f"Failed to download image from {image_url}: {last_exc}")
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
    # Ensure DB is initialized when running under Celery worker
    init_db(os.getenv("DATABASE_URL"))
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
        
        # Load existing index if it exists
        existing_indexed_ids = set()
        if index.exists():
            logger.info("Loading existing index...")
            index.load_if_exists()
            existing_indexed_ids = set(index.ids)
            logger.info(f"Found {len(existing_indexed_ids)} already indexed images")
        
        # fetch all images from DB
        images = session.session.query(ImageMetadata).all()
        if not images:
            logger.info("No images found from DB")
            return {"status": "done", "count": 0}

        # Filter out already indexed images
        new_images = [img for img in images if img.id not in existing_indexed_ids]
        skipped_count = len(images) - len(new_images)
        
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already indexed images")
        
        if not new_images:
            logger.info("All images are already indexed")
            return {"status": "done", "count": len(images), "skipped": skipped_count, "new": 0}

        logger.info(f"Processing {len(new_images)} new images for indexing")

        chunk_size = 128
        for i in range(0, len(new_images), chunk_size):
            batch = new_images[i:i+chunk_size]
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
                        # Also backfill metadata if missing
                        if not img.extra_metadata:
                            try:
                                meta = extract_image_metadata(temp_path, vectorizer, v[0])
                                img.extra_metadata = meta
                                session.session.add(img)
                                session.session.commit()
                            except Exception:
                                pass
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
        return {"status": "done", "count": len(images), "skipped": skipped_count, "new": len(new_images)}
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
    # Ensure DB is initialized when running under Celery worker
    init_db(os.getenv("DATABASE_URL"))
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
                # Extract metadata (colors, objects, styles)
                try:
                    extra_meta = extract_image_metadata(path, vectorizer, v[0])
                except Exception:
                    extra_meta = None

                # Save DB row first
                session.add_image(id_=img_id, content_type=content_type, image_url=image_url, metadata=extra_meta)
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

@celery.task(bind=True)
def ingest_online_images(self):
    """
    Fetch feed items from remote API and ingest images whose URLs are
    built from base host + media[0].url. Store metadata in DB and vectors
    in index, skipping already ingested URLs. Images are not persisted locally.
    """
    processed = 0
    skipped = 0
    errors = 0

    # Remote API endpoint for feed data (can override via FEED_API_URL)
    api_url = os.getenv(
        "FEED_API_URL",
        "https://api.uitips.me/api/v1/posts/feed?page=1&orderBy=image&order=asc&_limit=20",
    )

    # Ensure DB is initialized when running under Celery worker
    init_db(os.getenv("DATABASE_URL"))
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

    # No persistent download/save; we'll vectorize from a temp file when needed

    try:
        # Fetch feed from remote API
        resp = requests.get(api_url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        items = payload.get("data", []) if isinstance(payload, dict) else []
        if not items:
            logger.info("No items found from remote API feed")
            return {"status": "done", "processed": 0, "skipped": 0, "errors": 0}

        for item in items:
            try:
                media = item.get("media") or []
                if not media or not isinstance(media, list) or not media[0] or not media[0].get("url"):
                    skipped += 1
                    continue

                media_entry = media[0]
                media_url = media_entry.get("url")
                base_link = "https://api.uitips.me"
                if not base_link:
                    skipped += 1
                    continue
                if media_url.startswith("/"):
                    image_url = base_link.rstrip("/") + media_url
                else:
                    image_url = base_link.rstrip("/") + "/" + media_url
                print(image_url)

                # Skip if already ingested (match by URL)
                try:
                    existing = session.session.query(ImageMetadata).filter(ImageMetadata.image_url == image_url).first()
                    if existing:
                        skipped += 1
                        logger.info(f"Skip existing image_url in DB: {image_url}")
                        continue
                except Exception as db_check_exc:
                    logger.warning(f"DB check failed for {image_url}: {db_check_exc}")

                # Determine content type
                content_type = media_entry.get("mime") or _guess_content_type_from_ext(media_url)

                # Vectorize from a local temp file if remote; do not persist the image
                temp_path = None
                extra_meta = None
                try:
                    temp_path = _get_local_path_from_url(image_url)
                    v = vectorizer.image_to_vector(temp_path)
                    if v is None or len(v) == 0:
                        logger.warning(f"Vectorization produced empty vector for {image_url}")
                        errors += 1
                        continue
                    # Extract metadata while the temp file still exists
                    try:
                        extra_meta = extract_image_metadata(temp_path, vectorizer, v[0])
                    except Exception:
                        extra_meta = None
                except Exception as e:
                    logger.warning(f"Error fetching/vectorizing {image_url}: {e}")
                    errors += 1
                    continue
                finally:
                    if temp_path and temp_path != image_url and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass

                # Save DB row first
                img_id = str(uuid.uuid4())

                try:
                    session.add_image(id_=img_id, content_type=content_type, image_url=image_url, metadata=extra_meta)
                except Exception as db_write_exc:
                    logger.warning(f"Failed to insert image metadata for {image_url}: {db_write_exc}")
                    errors += 1
                    continue

                # Queue for index add
                pending_vecs.append(v[0])
                pending_ids.append(img_id)
                processed += 1

                # Flush batch
                if len(pending_vecs) >= batch_size:
                    try:
                        vecs_np = np.vstack(pending_vecs).astype("float32")
                        index.add(vecs_np, pending_ids)
                    except Exception:
                        errors += len(pending_vecs)
                    finally:
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
            pass

        logger.info("Successfully ingested from remote API feed and indexed!!!")
        return {"status": "done", "processed": processed, "skipped": skipped, "errors": errors}
    except Exception as e:
        return {"status": "error", "error": str(e), "processed": processed, "skipped": skipped, "errors": errors}
