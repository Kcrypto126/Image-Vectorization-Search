from fastapi import FastAPI, UploadFile, Form
from .vectorizer import Vectorizer
from .index import VectorIndex
from .models import init_db, ImageMetadata
from .utils import save_upload_file

# Initialize components
app = FastAPI()
SessionLocal = init_db()
vectorizer = Vectorizer("ViT-B/32")
index = VectorIndex(dim=512)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/images")
async def add_image(file: UploadFile, content_type: str = Form("unknown")):
    session = SessionLocal()
    try:
        file_id, path = save_upload_file(file, is_target=False)

        # Vectorize
        vector = vectorizer.image_to_vector(path)

        # Add to FAISS
        index.add(vector, [file_id])
        index.save()

        # Store metadata in DB
        new_image = ImageMetadata(id=file_id, content_type=content_type, image_url=path)
        session.add(new_image)
        session.commit()
        return {"id": file_id, "url": path}
    finally:
        session.close()

@app.post("/search")
async def search_image(file: UploadFile, top_k: int = 10):
    _, path = save_upload_file(file, is_target=True)

    # Vectorize query
    query_vector = vectorizer.image_to_vector(path)

    # Search index
    results = index.search(query_vector, k=top_k)

    # Fetch metadata
    session = SessionLocal()
    try:
        enriched_results = []
        for r in results:
            meta = session.query(ImageMetadata).filter_by(id=r["id"]).first()
            enriched_results.append({
                "id": r["id"],
                "score": r["score"],
                "content_type": meta.content_type if meta else None,
                "image_url": meta.image_url if meta else None
            })
        return {"results": enriched_results}
    finally:
        session.close()
