# RAG Engine SaaS (Gemini)

End-to-end RAG project:
- Backend (FastAPI): upload files, chunk + index, semantic retrieval (Gemini embeddings when available), chat Q&A with citations
- Frontend (React + Mantine): smooth UI for upload, document library, and chat

## Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set GEMINI_API_KEY
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the frontend dev URL, set backend URL to `http://localhost:8000`.
