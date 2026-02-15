# API

Base URL: `http://localhost:8000`

## Sessions
- `POST /api/sessions` -> `{ "session_id": "..." }`

## Files
- `GET /api/sessions/{session_id}/files`
- `POST /api/sessions/{session_id}/files`
  - multipart: `files` (one or many)

## Chat
- `POST /api/sessions/{session_id}/chat`
  - body: `{ "message": "...", "top_k": 8, "use_images": true }`
  - response: `{ answer, citations, used_embeddings, model }`

## Gemini Models
- `GET /api/models` -> model ids available for your key
