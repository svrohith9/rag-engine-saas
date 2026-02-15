from __future__ import annotations

import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path

import sqlite3

from app.chunking import chunk_pages
from app.extractors import SUPPORTED_DOC_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS, extract_document
from app.gemini_api import embed_text
from app.vector import to_blob


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sessions(id, created_at) VALUES (?, ?)",
        (session_id, _now_iso()),
    )
    conn.commit()


def ingest_file(
    conn: sqlite3.Connection,
    session_id: str,
    upload_dir: Path,
    file_name: str,
    content_bytes: bytes,
    gemini_api_key: str,
    gemini_embed_model: str,
) -> dict:
    ensure_session(conn, session_id)

    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_DOC_EXTENSIONS and ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return {"ok": False, "reason": "unsupported extension"}

    session_dir = upload_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    destination = session_dir / file_name
    if destination.exists():
        destination = session_dir / f"{file_id}-{file_name}"

    destination.write_bytes(content_bytes)

    mime = mimetypes.guess_type(destination.name)[0] or "application/octet-stream"
    size_bytes = destination.stat().st_size

    conn.execute(
        """
        INSERT INTO files(id, session_id, name, path, mime, size_bytes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (file_id, session_id, file_name, str(destination), mime, int(size_bytes), _now_iso()),
    )

    added_chunks = 0
    added_embeddings = 0

    # Only text-based files are chunked/embedded for RAG.
    if ext in SUPPORTED_DOC_EXTENSIONS:
        pages = extract_document(destination)
        chunks = chunk_pages([(p.page, p.text) for p in pages])

        for ch in chunks:
            chunk_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO chunks(id, file_id, idx, page, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, file_id, int(ch.idx), ch.page, ch.text, _now_iso()),
            )
            added_chunks += 1

            # Best effort embedding: if it fails, we still keep chunks and fall back to BM25.
            if gemini_api_key:
                try:
                    vec = embed_text(gemini_api_key, gemini_embed_model, ch.text)
                    conn.execute(
                        "INSERT INTO embeddings(chunk_id, dim, vector, created_at) VALUES (?, ?, ?, ?)",
                        (chunk_id, len(vec), to_blob(vec), _now_iso()),
                    )
                    added_embeddings += 1
                except Exception:
                    pass

    conn.commit()

    return {
        "ok": True,
        "file_id": file_id,
        "stored_as": str(destination),
        "mime": mime,
        "size_bytes": int(size_bytes),
        "added_chunks": added_chunks,
        "added_embeddings": added_embeddings,
    }
