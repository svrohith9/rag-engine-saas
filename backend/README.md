# Backend (FastAPI)

## What It Does
- Stores sessions, files, chunks, embeddings, and messages in SQLite (`DB_PATH`).
- Ingests documents:
  - Extracts text (PDF/TXT/MD/DOCX)
  - Splits into overlapping chunks
  - Attempts Gemini embeddings (if configured)
- Retrieval:
  - If embeddings exist: cosine similarity over stored vectors
  - Otherwise: BM25 lexical fallback
- Chat:
  - Retrieves top-k chunks
  - Calls Gemini with sources appended to prompt
  - Returns answer plus citations (file name, page, score, snippet)

## Env Vars
Copy `backend/.env.example` to `backend/.env`.

Required:
- `GEMINI_API_KEY`

Recommended:
- `GEMINI_MODEL=gemini-2.5-flash`
- `GEMINI_EMBED_MODEL=text-embedding-004`

CORS:
- `CORS_ALLOW_ORIGINS=http://localhost:5173` (comma-separated)

## Run
```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set GEMINI_API_KEY
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API
- `POST /api/sessions` -> `{ session_id }`
- `POST /api/sessions/{session_id}/files` (multipart) -> ingestion results
- `POST /api/sessions/{session_id}/chat` -> `{ answer, citations, used_embeddings, model }`
- `GET /api/models` -> Gemini model list for your key
