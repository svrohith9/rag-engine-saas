# Architecture

## Ingestion
1. Upload file
2. Store file metadata in SQLite
3. Extract text (PDF/TXT/MD/DOCX)
4. Chunk with overlap
5. Best-effort embeddings (Gemini `embedContent`)

## Retrieval
- Preferred: cosine similarity between query embedding and chunk embeddings
- Fallback: BM25 lexical scoring

## Generation
- Prompt includes: question + top-k sources
- Model: `GEMINI_MODEL` (default `gemini-2.5-flash`)
- Answer should cite file name and page when applicable

## Storage
- SQLite tables: sessions, files, chunks, embeddings, messages
- File bytes stored on disk under `UPLOAD_DIR/<session_id>/...`
