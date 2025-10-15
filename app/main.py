# app/main.py
import os
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from .vectorizer import Vectorizer
from .index_adapter import IndexAdapter
from .db import get_session, init_db, ImageMetadata, Interaction
from .schemas import SearchResultSchema, UserTasteProfile, BoardSchema, PinBoardSchema, TasteGraphStatus, RecommendationSchema, UserBoardsSchema
from .storage import save_image
from .tasks import full_reindex, ingest_local_images, ingest_online_images, rebuild_taste_graph
from .db import ImageMetadata
from .metadata import extract_image_metadata, detect_filters_from_text
from .taste_graph import initialize_taste_graph_system, get_taste_graph_manager
from .auto_board_scheduler import initialize_auto_board_scheduler
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
# app = FastAPI(root_path="/api", title="Image Search Service")
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

# Initialize taste graph system
taste_graph_manager = None
try:
    taste_graph_manager = initialize_taste_graph_system(get_session(), vectorizer, index)
    logger.info("Taste graph system initialized successfully")
    
    # Initialize auto-board scheduler
    initialize_auto_board_scheduler(taste_graph_manager, check_interval_minutes=30)
    logger.info("Auto-board scheduler initialized successfully")
    
except Exception as e:
    logger.warning(f"Failed to initialize taste graph system: {e}")
    taste_graph_manager = None

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

async def get_taste_based_recommendations(user_id: str, top_k: int) -> SearchResultSchema:
    """
    Get taste-based recommendations for a user when no search query is provided.
    Uses the user's taste graph to find personalized image recommendations.
    """
    try:
        # Initialize user if not exists
        if taste_graph_manager:
            taste_graph_manager.initialize_user(user_id)
        
        # Get all images from the index
        session = get_session()
        all_images = session.get_all_images()
        
        if not all_images:
            return SearchResultSchema(
                results=[],
                total=0,
                query="taste-based recommendations",
                user_id=user_id,
                personalized=True
            )
        
        # Convert to search results format
        search_results = []
        for img in all_images:
            search_results.append({
                'id': img.id,
                'score': 0.5,  # Base score for taste recommendations
                'image_url': img.image_url,
                'url': img.image_url,  # Alternative field for compatibility
                'metadata': img.image_metadata or {}
            })
        
        # Apply taste-based personalization if taste graph is available
        if taste_graph_manager:
            try:
                personalized_results = taste_graph_manager.get_personalized_search_results(
                    user_id, search_results, top_k=top_k
                )
                search_results = personalized_results
            except Exception as e:
                logger.warning(f"Failed to personalize taste recommendations for user {user_id}: {e}")
                # Fall back to random selection if personalization fails
                import random
                search_results = random.sample(search_results, min(top_k, len(search_results)))
        else:
            # If no taste graph, return random selection
            import random
            search_results = random.sample(search_results, min(top_k, len(search_results)))
        
        # Limit to requested top_k
        search_results = search_results[:top_k]
        
        return SearchResultSchema(
            results=search_results,
            total=len(search_results),
            query="taste-based recommendations",
            user_id=user_id,
            personalized=True
        )
        
    except Exception as e:
        logger.exception(f"Failed to get taste-based recommendations for user {user_id}")
        # Return empty results on error
        return SearchResultSchema(
            results=[],
            total=0,
            query="taste-based recommendations",
            user_id=user_id,
            personalized=False,
            error=str(e)
        )

@app.post("/search", response_model=SearchResultSchema)
async def search(
    query: str = Query(None, description="Text query to search for similar images"),
    file: UploadFile = File(None, description="Target image"),
    top_k: int = Query(10, ge=1, le=100),
    user_id: str = Query(None, description="User id to personalize results via Taste Graph"),
):
    """
    Enhanced search endpoint that accepts text query, image file, or both inputs.
    - If only text is provided: searches using text vector
    - If only image is provided: searches using image vector
    - If both are provided: combines both vectors using weighted average
    - If no input is provided but user_id is given: returns taste-based recommendations
    - Enhanced metadata filtering and ranking based on all available metadata
    """
    # Handle empty query case - return taste-based recommendations if user_id is provided
    if not query and not file:
        if user_id:
            # Return taste-based recommendations for the user
            return await get_taste_based_recommendations(user_id, top_k)
        else:
            raise HTTPException(status_code=400, detail="At least one input (query, file, or user_id) is required")
    
    # Validate text input
    if query and not query.strip():
        if user_id:
            # Empty query with user_id - return taste-based recommendations
            return await get_taste_based_recommendations(user_id, top_k)
        else:
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
    # Taste Graph: if user_id provided, compute personalized boosts based on interactions and co-occurrences
    user_boosts = {}
    if user_id:
        try:
            interactions = session.get_user_interactions(user_id)
            # Aggregate weights per image the user interacted with
            liked = {}
            for it in interactions:
                base = 0.0
                if it.event == "like":
                    base = 1.0
                elif it.event == "click":
                    base = 0.5
                elif it.event == "save":
                    base = 1.2
                elif it.event == "view":
                    base = 0.2
                w = float(it.weight or 1.0)
                liked[it.image_id] = liked.get(it.image_id, 0.0) + base * w
            # Expand via co-occurrence neighbors
            neighbor_scores = {}
            for img_id, w in liked.items():
                try:
                    neighbors = session.get_neighbors(img_id, top_k=50)
                    for nid, ns in neighbors:
                        neighbor_scores[nid] = neighbor_scores.get(nid, 0.0) + float(ns) * w
                except Exception:
                    pass
            # Combine direct liked and neighbors
            for img_id, w in liked.items():
                user_boosts[img_id] = user_boosts.get(img_id, 0.0) + 0.5 * w
            for nid, s in neighbor_scores.items():
                user_boosts[nid] = user_boosts.get(nid, 0.0) + 0.3 * s
        except Exception:
            pass
    filtered = []
    for r in results:
        meta = session.get_image_metadata(r["id"])  # type: ImageMetadata
        if not meta:
            continue
        
        # Get all metadata
        extra = meta.image_metadata or {}
        raw = meta.raw_detections or {}
        generated = meta.generated_metadata or {}
        
        # Extract metadata fields for matching
        colors = set([c.lower() for c in (extra.get("colors") or [])])
        objects = set([o.lower() for o in (extra.get("objects") or [])])
        style_tags = set([t.lower() for t in (extra.get("style_tags") or [])])
        ocr_text = (raw.get("ocr_text") or "")
        if isinstance(ocr_text, list):
            ocr_text = " ".join(str(item) for item in ocr_text)
        ocr_text = ocr_text.lower()
        categories = set([c.lower() for c in (generated.get("Categories") or [])])
        visual_style = set([s.lower() for s in (generated.get("Visual style") or [])])
        keywords = set([k.lower() for k in (generated.get("Keywords") or [])])
        description = (generated.get("Description") or "")
        if isinstance(description, list):
            description = " ".join(str(item) for item in description)
        description = description.lower()
        layout = (generated.get("Layout") or "")
        if isinstance(layout, list):
            layout = " ".join(str(item) for item in layout)
        layout = layout.lower()
        
        mode = (generated.get("Mode") or "")
        if isinstance(mode, list):
            mode = " ".join(str(item) for item in mode)
        mode = mode.lower()
        components = set([c.lower() for c in (raw.get("components") or [])])
        
        # Additional metadata fields for enhanced matching
        blocks = set([b.lower() for b in (generated.get("Block") or [])])
        devices = set([d.lower() for d in (generated.get("Device") or [])])
        logo_elements = set([le.lower() for le in (generated.get("Logo elements") or [])])
        content_hints = set([ch.lower() for ch in (generated.get("Content hints") or [])])
        main_components = set([mc.lower() for mc in (generated.get("Main components") or [])])
        
        # Additional design metadata fields
        icons_illustrations = set([ii.lower() for ii in (generated.get("Icons / Illustrations") or [])])
        complexity = (generated.get("Complexity") or "")
        if isinstance(complexity, list):
            complexity = " ".join(str(item) for item in complexity)
        complexity = complexity.lower()
        
        data_density = (generated.get("Data density") or "")
        if isinstance(data_density, list):
            data_density = " ".join(str(item) for item in data_density)
        data_density = data_density.lower()
        
        verdict = (generated.get("Verdict") or "")
        if isinstance(verdict, list):
            verdict = " ".join(str(item) for item in verdict)
        verdict = verdict.lower()
        
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
        
        # Block matching (design system blocks)
        if query_tokens & blocks:
            bonus += 0.06 * len(query_tokens & blocks)
        
        # Device matching (mobile, desktop, responsive)
        if query_tokens & devices:
            bonus += 0.05 * len(query_tokens & devices)
        
        # Logo elements matching
        if query_tokens & logo_elements:
            bonus += 0.07 * len(query_tokens & logo_elements)
        
        # Content hints matching
        if query_tokens & content_hints:
            bonus += 0.05 * len(query_tokens & content_hints)
        
        # Main components matching (from generated metadata)
        if query_tokens & main_components:
            bonus += 0.06 * len(query_tokens & main_components)
        
        # Icons/Illustrations matching
        if query_tokens & icons_illustrations:
            bonus += 0.05 * len(query_tokens & icons_illustrations)
        
        # Complexity matching
        if any(token in complexity for token in query_tokens):
            bonus += 0.04
        
        # Data density matching
        if any(token in data_density for token in query_tokens):
            bonus += 0.04
        
        # Verdict matching (expert assessment)
        if any(token in verdict for token in query_tokens):
            bonus += 0.08
        
        # Exact phrase matching in OCR text (highest weight)
        if query_lower in ocr_text:
            bonus += 0.20
        
        # Exact phrase matching in description
        if query_lower in description:
            bonus += 0.15
        
        # Personalized boost
        # if user_id and r["id"] in user_boosts:
        #     bonus += min(0.5, 0.05 + 0.1 * (user_boosts[r["id"]] ** 0.5))

        score += bonus

        filtered.append({
            "id": r["id"],
            "content_type": meta.content_type,
            "image_url": meta.image_url,
            "url": meta.image_url,  # Add url field for schema compatibility
            "score": score
        })

    # Sort by updated score desc and trim to top_k
    filtered.sort(key=lambda x: x["score"], reverse=True)
    
    # Apply taste graph personalization if user_id provided and taste graph is available
    if user_id and taste_graph_manager:
        try:
            personalized_results = taste_graph_manager.get_personalized_search_results(
                user_id, filtered, top_k
            )
            return SearchResultSchema(
                results=personalized_results,
                total=len(personalized_results),
                query=query or "image search",
                user_id=user_id,
                personalized=True
            )
        except Exception as e:
            logger.warning(f"Taste graph personalization failed: {e}")
            # Fall back to regular results
    
    return SearchResultSchema(
        results=filtered[:top_k],
        total=len(filtered[:top_k]),
        query=query or "image search",
        user_id=user_id,
        personalized=False
    )


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

@app.post("/admin/rebuild-taste-graph")
async def admin_rebuild_taste_graph(background_tasks: BackgroundTasks):
    background_tasks.add_task(rebuild_taste_graph)
    logger.info("Rebuild taste graph started...loading...")
    return {"status": "taste graph rebuild started"}

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

# Taste Graph Management Endpoints

@app.post("/taste-graph/users/{user_id}/initialize")
async def initialize_user_taste(user_id: str):
    """Initialize a new user in the taste graph system"""
    if not taste_graph_manager:
        raise HTTPException(status_code=503, detail="Taste graph system not available")
    
    try:
        result = taste_graph_manager.initialize_user(user_id)
        return result
    except Exception as e:
        logger.exception("Failed to initialize user taste")
        raise HTTPException(status_code=500, detail="Failed to initialize user taste")

@app.get("/taste-graph/users/{user_id}/profile", response_model=UserTasteProfile)
async def get_user_taste_profile(user_id: str):
    """Get user's taste profile"""
    if not taste_graph_manager:
        raise HTTPException(status_code=503, detail="Taste graph system not available")
    
    try:
        profile = taste_graph_manager.get_user_taste_profile(user_id)
        return profile
    except Exception as e:
        logger.exception("Failed to get user taste profile")
        raise HTTPException(status_code=500, detail="Failed to get user taste profile")

@app.post("/taste-graph/boards", response_model=BoardSchema)
async def create_board(
    user_id: str = Query(..., description="User id"),
    board_id: str = Query(..., description="Board id"),
    name: str = Query(..., description="Board name"),
    description: str = Query(None, description="Board description")
):
    """Create a new board"""
    if not taste_graph_manager:
        raise HTTPException(status_code=503, detail="Taste graph system not available")
    
    try:
        result = taste_graph_manager.create_board(user_id, board_id, name, description)
        return result
    except Exception as e:
        logger.exception("Failed to create board")
        raise HTTPException(status_code=500, detail="Failed to create board")

@app.post("/taste-graph/boards/{board_id}/pins/{pin_id}", response_model=PinBoardSchema)
async def add_pin_to_board(board_id: str, pin_id: str, user_id: str = Query(..., description="User id performing the action")):
    """Add a pin to a board. Only the board owner can add pins."""
    if not taste_graph_manager:
        raise HTTPException(status_code=503, detail="Taste graph system not available")
    
    try:
        result = taste_graph_manager.add_pin_to_board(pin_id, board_id, user_id=user_id)
        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.exception("Failed to add pin to board")
        raise HTTPException(status_code=500, detail="Failed to add pin to board")

@app.delete("/taste-graph/boards/{board_id}/pins/{pin_id}", response_model=PinBoardSchema)
async def remove_pin_from_board(board_id: str, pin_id: str, user_id: str = Query(..., description="User id performing the action")):
    """Remove a pin from a board. Only the board owner can remove pins."""
    if not taste_graph_manager:
        raise HTTPException(status_code=503, detail="Taste graph system not available")
    
    try:
        result = taste_graph_manager.remove_pin_from_board(pin_id, board_id, user_id=user_id)
        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.exception("Failed to remove pin from board")
        raise HTTPException(status_code=500, detail="Failed to remove pin from board")

@app.post("/admin/taste-graph/train")
async def train_taste_graph(
    background_tasks: BackgroundTasks,
    epochs: int = Query(10, ge=1, le=100, description="Number of training epochs"),
    user_ids: str = Query(None, description="Comma-separated user IDs to train on (optional)")
):
    """Run offline PinSage training"""
    if not taste_graph_manager:
        raise HTTPException(status_code=503, detail="Taste graph system not available")
    
    user_list = None
    if user_ids:
        user_list = [uid.strip() for uid in user_ids.split(",")]
    
    background_tasks.add_task(train_taste_graph_task, epochs, user_list)
    logger.info("Taste graph training started...")
    return {"status": "training started", "epochs": epochs}

@app.post("/admin/taste-graph/rebuild")
async def rebuild_taste_graph_endpoint(background_tasks: BackgroundTasks):
    """Rebuild the taste graph from scratch"""
    if not taste_graph_manager:
        raise HTTPException(status_code=503, detail="Taste graph system not available")
    
    background_tasks.add_task(rebuild_taste_graph_task)
    logger.info("Taste graph rebuild started...")
    return {"status": "rebuild started"}

# Enhanced Board Management Endpoints

@app.get("/taste-graph/users/{user_id}/boards")
async def get_user_boards(user_id: str, board_type: str = None):
    """Get user's boards (user-created and auto-generated)"""
    if not taste_graph_manager:
        raise HTTPException(status_code=503, detail="Taste graph system not available")
    
    try:
        session = get_session()
        boards = session.get_user_boards(user_id, board_type=board_type)
        
        board_data = []
        for board in boards:
            board_data.append({
                'board_id': board.board_id,
                'name': board.name,
                'description': board.description,
                'board_type': board.board_type,
                'pin_count': board.pin_count,
                'created_at': board.created_at,
                'last_activity': board.last_activity
            })
        
        return {
            'user_id': user_id,
            'boards': board_data,
            'count': len(board_data)
        }
    except Exception as e:
        logger.exception("Failed to get user boards")
        raise HTTPException(status_code=500, detail="Failed to get user boards")

# Manual board vector update API removed - vectors are now updated automatically

@app.get("/taste-graph/admin/scheduler-status")
async def get_scheduler_status():
    """Get auto-board scheduler status"""
    from .auto_board_scheduler import get_auto_board_scheduler
    
    scheduler = get_auto_board_scheduler()
    if not scheduler:
        return {"status": "not_initialized"}
    
    return scheduler.get_scheduler_status()

@app.post("/admin/migrate-database")
async def migrate_database():
    """Force database migration to add new columns and tables"""
    from .db import force_migration
    
    try:
        success = force_migration()
        if success:
            return {"status": "success", "message": "Database migration completed"}
        else:
            return {"status": "error", "message": "Database migration failed"}
    except Exception as e:
        logger.exception("Database migration failed")
        return {"status": "error", "message": f"Migration failed: {str(e)}"}

@app.post("/taste-graph/users/{user_id}/correlations")
async def compute_user_correlations(user_id: str):
    """Compute correlations for a user"""
    if not taste_graph_manager:
        raise HTTPException(status_code=503, detail="Taste graph system not available")
    
    try:
        result = taste_graph_manager.compute_correlations_for_user(user_id)
        return result
    except Exception as e:
        logger.exception("Failed to compute correlations")
        raise HTTPException(status_code=500, detail="Failed to compute correlations")

@app.get("/taste-graph/users/{user_id}/recommendations")
async def get_user_recommendations(user_id: str, top_k: int = 20):
    """Get personalized recommendations for a user"""
    if not taste_graph_manager:
        raise HTTPException(status_code=503, detail="Taste graph system not available")
    
    try:
        recommendations = taste_graph_manager.get_recommendations_for_user(user_id, top_k)
        return {
            'user_id': user_id,
            'recommendations': recommendations,
            'count': len(recommendations)
        }
    except Exception as e:
        logger.exception("Failed to get recommendations")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")

@app.get("/recommendations", response_model=SearchResultSchema)
async def get_taste_recommendations(
    user_id: str = Query(..., description="User id to get personalized recommendations"),
    top_k: int = Query(10, ge=1, le=100, description="Number of recommendations to return")
):
    """
    Get taste-based image recommendations for a user.
    This endpoint provides personalized image recommendations based on the user's taste graph.
    """
    return await get_taste_based_recommendations(user_id, top_k)

@app.post("/events")
async def record_event(
    user_id: str = Query(..., description="User id"),
    image_id: str = Query(..., description="Image id"),
    event: str = Query(..., description="Event type: like, click, save, view")
):
    if event not in {"like", "click", "save", "view"}:
        raise HTTPException(status_code=400, detail="Invalid event type")
    # Determine weight automatically based on event type
    event_weights = {
        "like": 1.0,
        "click": 0.5,
        "save": 1.2,
        "view": 0.2
    }
    weight = event_weights.get(event, 1.0)
    try:
        # session = get_session()
        # session.record_interaction(user_id=user_id, image_id=image_id, event=event, weight=weight)
        
        # Process interaction through taste graph system
        if taste_graph_manager:
            try:
                taste_graph_manager.process_user_interaction(user_id, image_id, event, weight)
            except Exception as e:
                logger.warning(f"Taste graph processing failed: {e}")
        
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Failed to record event")
        raise HTTPException(status_code=500, detail="Failed to record event")

# Background task functions for taste graph

async def train_taste_graph_task(epochs: int, user_ids: List[str] = None):
    """Background task for offline PinSage training"""
    try:
        if taste_graph_manager:
            result = taste_graph_manager.run_offline_training(user_ids, epochs)
            logger.info(f"Taste graph training completed: {result}")
        else:
            logger.error("Taste graph manager not available for training")
    except Exception as e:
        logger.exception(f"Taste graph training failed: {e}")

async def rebuild_taste_graph_task():
    """Background task for rebuilding taste graph"""
    try:
        if taste_graph_manager:
            result = taste_graph_manager.rebuild_taste_graph()
            logger.info(f"Taste graph rebuild completed: {result}")
        else:
            logger.error("Taste graph manager not available for rebuild")
    except Exception as e:
        logger.exception(f"Taste graph rebuild failed: {e}")
