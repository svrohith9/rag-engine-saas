from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from pydantic import BaseModel

from app import db
from app.gemini_api import generate_content, list_models, embed_text
from app.ingest import ingest_file, ensure_session
from app.retrieval import retrieve
from app.settings import load_settings

settings = load_settings()

app = FastAPI(title="RAG Engine SaaS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.upload_dir.mkdir(parents=True, exist_ok=True)

@contextmanager
def get_conn():
    conn = db.connect(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def conn_dep():
    with get_conn() as conn:
        yield conn


@app.on_event("startup")
def _startup() -> None:
    with get_conn() as conn:
        db.init_db(conn)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CreateSessionResponse(BaseModel):
    session_id: str


class FileInfo(BaseModel):
    id: str
    name: str
    mime: str
    size_bytes: int
    created_at: str


class UploadResponse(BaseModel):
    session_id: str
    results: list[dict]


class ChatRequest(BaseModel):
    message: str
    use_images: bool = True
    top_k: int = 8


class Citation(BaseModel):
    chunk_id: str
    file_name: str
    page: Optional[int] = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    used_embeddings: bool
    model: str


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "provider": settings.llm_provider,
        "model": settings.gemini_model,
        "embed_model": settings.gemini_embed_model,
    }


@app.get("/api/models")
def api_models() -> dict:
    if settings.llm_provider != "gemini":
        return {"provider": settings.llm_provider, "models": []}
    if not settings.gemini_api_key:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY not set")
    return {"provider": "gemini", "models": list_models(settings.gemini_api_key)}


@app.post("/api/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    session_id = str(uuid.uuid4())
    ensure_session(conn, session_id)
    return CreateSessionResponse(session_id=session_id)


@app.get("/api/sessions/{session_id}/files", response_model=list[FileInfo])
def list_files(session_id: str, conn=Depends(conn_dep)) -> list[FileInfo]:
    rows = conn.execute(
        "SELECT id, name, mime, size_bytes, created_at FROM files WHERE session_id=? ORDER BY created_at DESC",
        (session_id,),
    ).fetchall()
    return [FileInfo(**dict(r)) for r in rows]


@app.post("/api/sessions/{session_id}/files", response_model=UploadResponse)
async def upload_files(
    session_id: str,
    files: List[UploadFile] = File(...),
    conn=Depends(conn_dep),
) -> UploadResponse:
    ensure_session(conn, session_id)

    results = []
    for f in files:
        name = f.filename or f"upload-{uuid.uuid4()}"
        data = await f.read()
        results.append(
            ingest_file(
                conn=conn,
                session_id=session_id,
                upload_dir=settings.upload_dir,
                file_name=name,
                content_bytes=data,
                gemini_api_key=settings.gemini_api_key,
                gemini_embed_model=settings.gemini_embed_model,
            )
        )

    return UploadResponse(session_id=session_id, results=results)


@app.get("/api/sessions/{session_id}/messages")
def list_messages(session_id: str, conn=Depends(conn_dep)) -> list[dict]:
    rows = conn.execute(
        "SELECT id, role, content, metadata, created_at FROM messages WHERE session_id=? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        if item.get("metadata"):
            try:
                item["metadata"] = json.loads(item["metadata"])
            except Exception:
                item["metadata"] = None
        out.append(item)
    return out


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, conn=Depends(conn_dep)) -> dict:
    # Delete DB rows first, then best-effort delete files on disk.
    file_rows = conn.execute("SELECT path FROM files WHERE session_id=?", (session_id,)).fetchall()
    conn.execute("DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id IN (SELECT id FROM files WHERE session_id=?))", (session_id,))
    conn.execute("DELETE FROM chunks WHERE file_id IN (SELECT id FROM files WHERE session_id=?)", (session_id,))
    conn.execute("DELETE FROM files WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()

    for r in file_rows:
        try:
            import os
            os.remove(r["path"])
        except Exception:
            pass
    # Best-effort remove session directory if empty
    try:
        session_dir = settings.upload_dir / session_id
        if session_dir.exists():
            for p in session_dir.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
            session_dir.rmdir()
    except Exception:
        pass

    return {"ok": True}


@app.delete("/api/sessions/{session_id}/files/{file_id}")
def delete_file(session_id: str, file_id: str, conn=Depends(conn_dep)) -> dict:
    row = conn.execute("SELECT path FROM files WHERE id=? AND session_id=?", (file_id, session_id)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="File not found")

    conn.execute("DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id=?)", (file_id,))
    conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
    conn.execute("DELETE FROM files WHERE id=? AND session_id=?", (file_id, session_id))
    conn.commit()

    try:
        import os
        os.remove(row["path"])
    except Exception:
        pass

    return {"ok": True}


@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, payload: ChatRequest, conn=Depends(conn_dep)) -> ChatResponse:
    if settings.llm_provider != "gemini":
        raise HTTPException(status_code=400, detail="Only Gemini provider is implemented")
    if not settings.gemini_api_key:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY not set")

    ensure_session(conn, session_id)

    conn.execute(
        "INSERT INTO messages(id, session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), session_id, "user", payload.message, None, _now_iso()),
    )
    conn.commit()

    # Retrieve context
    query_embedding = None
    used_embeddings = False
    if settings.gemini_api_key and settings.gemini_embed_model:
        try:
            query_embedding = embed_text(settings.gemini_api_key, settings.gemini_embed_model, payload.message)
            used_embeddings = True
        except Exception:
            query_embedding = None
            used_embeddings = False

    retrieved = retrieve(conn, session_id, payload.message, query_embedding, top_k=payload.top_k)

    context_blocks = []
    citations: list[Citation] = []
    for r in retrieved:
        snippet = (r.text[:240] + "...") if len(r.text) > 240 else r.text
        citations.append(
            Citation(
                chunk_id=r.chunk_id,
                file_name=r.file_name,
                page=r.page,
                score=float(r.score),
                snippet=snippet,
            )
        )
        page_note = f" (page {r.page})" if r.page else ""
        context_blocks.append(f"SOURCE: {r.file_name}{page_note}\n{r.text}")

    context = "\n\n---\n\n".join(context_blocks)

    # Add up to 1 image file for multimodal Q&A if enabled
    image_paths: list[str] = []
    if payload.use_images:
        row = conn.execute(
            "SELECT path FROM files WHERE session_id=? AND mime LIKE 'image/%' ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row:
            image_paths = [row["path"]]

    system_text = (
        "You are a RAG assistant. Answer using the provided sources. "
        "If the sources do not contain the answer, say you don't know. "
        "When you use a source, cite it by file name and page if present."
    )

    user_text = f"Question: {payload.message}\n\nSources:\n{context}" if context else f"Question: {payload.message}"

    try:
        answer = generate_content(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            system_text=system_text,
            user_text=user_text,
            image_paths=image_paths,
            temperature=0.2,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {exc}") from exc

    assistant_metadata = json.dumps(
        {
            "citations": [c.model_dump() for c in citations],
            "used_embeddings": used_embeddings,
            "model": settings.gemini_model,
        }
    )
    conn.execute(
        "INSERT INTO messages(id, session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), session_id, "assistant", answer, assistant_metadata, _now_iso()),
    )
    conn.commit()

    return ChatResponse(
        answer=answer,
        citations=citations,
        used_embeddings=used_embeddings,
        model=settings.gemini_model,
    )
