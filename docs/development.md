# Development Guide

## Prereqs
- macOS/Linux
- Python 3.11
- Node 18+

## Repo Layout
- `backend/`: FastAPI service + SQLite
- `frontend/`: React UI (Vite)
- `docs/`: architecture and runbooks

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

Health:
- `GET http://localhost:8000/health`

## Frontend
```bash
cd frontend
npm install
npm run dev
```

Open the dev URL and ensure Backend URL is `http://localhost:8000`.

## Typical Flow
1. Create a session
2. Upload a PDF/DOCX/TXT
3. Ask a question
4. Inspect citations

## Troubleshooting
See `docs/troubleshooting.md`.
