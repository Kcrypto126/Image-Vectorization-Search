# app/tasks.py
from celery import Celery
import os
from dotenv import load_dotenv
from .db import DBSession
from .vectorizer import Vectorizer
from .index_adapter import IndexAdapter
import numpy as np

# Load environment variables from .env file
load_dotenv()

CELERY_BROKER = os.getenv("CELERY_BROKER", "redis://redis:6379/0")
celery = Celery('tasks', broker=CELERY_BROKER)

@celery.task(bind=True)
def full_reindex(self):
    db = DBSession()
    index = IndexAdapter(backend=os.getenv("INDEX_BACKEND", "faiss"), dim=int(os.getenv("VECTOR_DIM", "512")))
    vectorizer = Vectorizer(model_name=os.getenv("CLIP_MODEL", "ViT-B/32"), device=os.getenv("DEVICE","cpu"))
    # fetch all images
    # NOTE: implement pagination for large DBs!
    images = db.session.query(db.ImageMetadata).all()
    chunk = 128
    for i in range(0, len(images), chunk):
        batch = images[i:i+chunk]
        ids = []
        vecs = []
        for img in batch:
            v = vectorizer.image_to_vector(img.image_url)  # if image_url points to S3 path, download first
            vecs.append(v[0])
            ids.append(img.id)
        vecs = np.vstack(vecs).astype("float32")
        index.add(vecs, ids)
    index.save()
    return {"status": "done", "count": len(images)}
