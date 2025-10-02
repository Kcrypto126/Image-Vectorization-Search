## Image Search FastAPI Backend

FastAPI backend to analyze images, vectorize them, generate metadata (colors, objects, styles, OCR text, and AI-enriched fields), and search for similar images using either input text or an image. The service returns the top_k most relevant images.

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

- Text or image search: POST `/search` with `query` (text) and/or `file` (image). Returns the most similar images ranked by vector similarity and metadata signals.
- Upload an image: POST `/uploads` to store and index a new image.
- Admin helpers:
  - `POST /admin/index` — rebuilds the index from DB (async)
  - `POST /admin/ingest-local` — ingests images from `data/images/` (async)
  - `POST /admin/ingest-online` — ingests sample images from configured API feed (async)

### Notes

- OCR requires Tesseract to be installed and accessible on the system PATH.
- If using GPU-dependent features (e.g., certain CV/YOLO pipelines), ensure the proper CUDA/cuDNN drivers and compatible Python packages are installed.
