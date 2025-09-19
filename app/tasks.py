import os
import tempfile
import requests
from urllib.parse import urlparse
from celery import Celery
from dotenv import load_dotenv
import numpy as np

from .db import get_session, ImageMetadata
from .vectorizer import Vectorizer
from .index_adapter import IndexAdapter

# Load environment variables from .env file
load_dotenv()

CELERY_BROKER = os.getenv("CELERY_BROKER", "redis://redis:6379/0")
celery = Celery('tasks', broker=CELERY_BROKER)

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
        return {"status": "done", "count": len(images)}
    except Exception as e:
        print(f"Error in full_reindex: {e}")
        return {"status": "error", "error": str(e)}
