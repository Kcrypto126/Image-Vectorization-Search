## Image Search FastAPI Backend with Pinterest-Style Taste Graph

FastAPI backend to analyze images, vectorize them, generate metadata (colors, objects, styles, OCR text, and AI-enriched fields), and search for similar images using either input text or an image. The service now includes a **Pinterest-style Taste Graph** powered by PinSage-GNN for personalized image recommendations based on user preferences and behavior patterns.

### Key Features

- **Advanced Image Analysis**: CLIP-based vectorization, metadata extraction, OCR text detection
- **Pinterest-Style Taste Graph**: User taste modeling using PinSage-GNN algorithm
- **Real-time Personalization**: Immediate taste vector updates on user interactions
- **Offline Training**: Long-term taste pattern learning through graph neural networks
- **Topic-based Recommendations**: Content categorization and user-topic affinity tracking
- **Board Management**: Pinterest-style board creation and pin organization

### Prerequisites

- Python 3.9+
- Tesseract OCR (for OCR text detection)
  - Windows installer:
    https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe
  - After installing on Windows, ensure the Tesseract binary path is available on PATH (e.g. `C:\Program Files\Tesseract-OCR`). Add it environment values.

### Setup

1. Create and activate a virtual environment

   - Windows (PowerShell):
     ```bash
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables
   - Copy `.env.example` to `.env` and fill in required values, such as:
     - `OPENAI_API_KEY`
     - `DATABASE_URL` (defaults to SQLite if omitted)
     - Optional: `CLIP_MODEL`, `VECTOR_DIM`, `INDEX_BACKEND`, `DEVICE`, `YOLO_MODEL`, etc.

### Running the API

Start the FastAPI server with uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --root-path /api
```

By default, the service will attempt to load an existing vector index and expose endpoints for search and ingestion tasks.

### Basic Usage

#### Image Search

- **Text or image search**: POST `/search` with `query` (text) and/or `file` (image). Returns the most similar images ranked by vector similarity and metadata signals.
- **Personalized search**: Include `user_id` parameter for taste graph-based personalization
- **Upload an image**: POST `/uploads` to store and index a new image.

#### Taste Graph Management

- **Initialize user**: POST `/taste-graph/users/{user_id}/initialize` to create user taste profile
- **Get user profile**: GET `/taste-graph/users/{user_id}/profile` to view taste preferences
- **Create board**: POST `/taste-graph/boards` to create Pinterest-style boards
- **Add pin to board**: POST `/taste-graph/boards/{board_id}/pins/{pin_id}` to organize content

#### User Interactions

- **Record events**: POST `/events` with `user_id`, `image_id`, and `event` (like, click, save, view)
- **Real-time updates**: Taste vectors update immediately on user interactions

#### Admin Operations

- `POST /admin/index` — rebuilds the index from DB (async)
- `POST /admin/ingest-local` — ingests images from `data/images/` (async)
- `POST /admin/ingest-online` — ingests sample images from configured API feed (async)
- `POST /admin/taste-graph/train` — runs offline PinSage training (async)
- `POST /admin/taste-graph/rebuild` — rebuilds taste graph from scratch (async)

### Notes

- OCR requires Tesseract to be installed and accessible on the system PATH.
- If using GPU-dependent features (e.g., certain CV/YOLO pipelines), ensure the proper CUDA/cuDNN drivers and compatible Python packages are installed.
- The Taste Graph system requires PyTorch Geometric and NetworkX for graph neural network operations.
- User taste vectors are initialized randomly and updated through interactions and offline training.

### Taste Graph Architecture

The Pinterest-style Taste Graph implementation includes:

1. **PinSage-GNN Model**: Graph neural network for learning user taste representations
2. **Real-time Updates**: Immediate taste vector updates on user interactions (like, save, click, view)
3. **Offline Training**: Periodic retraining for long-term taste pattern learning
4. **Topic Modeling**: Automatic topic extraction from image metadata
5. **User-Topic Affinities**: Tracking user preferences across different content topics
6. **Board Management**: Pinterest-style content organization

### API Examples

#### Personalized Search

```bash
curl -X POST "http://localhost:8000/search?query=modern kitchen&user_id=user123&top_k=10"
```

#### Record User Interaction

```bash
curl -X POST "http://localhost:8000/events?user_id=user123&image_id=img456&event=save"
```

#### Initialize User Taste Profile

```bash
curl -X POST "http://localhost:8000/taste-graph/users/user123/initialize"
```

#### Create Board

```bash
curl -X POST "http://localhost:8000/taste-graph/boards?user_id=user123&board_id=kitchen_ideas&name=Kitchen Ideas"
```
