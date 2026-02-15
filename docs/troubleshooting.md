# Troubleshooting

## 404 from Gemini model
- Call `GET /api/models` to see which model ids your key supports.
- Set `GEMINI_MODEL` to an available id.

## Embeddings not used
- If `used_embeddings=false`, the backend could not create a query embedding.
- Ensure `GEMINI_EMBED_MODEL=text-embedding-004` (or another supported embed model).

## CORS errors in the browser
- Set `CORS_ALLOW_ORIGINS=http://localhost:5173` in `backend/.env`.

## “No sources” answers
- Upload a document with extractable text (PDF text layer, DOCX, TXT).
- Scanned PDFs require OCR (not implemented yet).
