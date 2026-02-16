from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional, AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import db
from app.gemini_api import generate_content, list_models, embed_text
from app.ingest import ingest_file, ensure_session
from app.retrieval import retrieve
from app.settings import load_settings
from app.llm_providers import get_provider

settings = load_settings()

app = FastAPI(title="RAG Engine SaaS", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.upload_dir.mkdir(parents=True, exist_ok=True)

# Initialize LLM provider
llm_provider = get_provider(settings)


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
    temperature: float = 0.2
    stream: bool = False


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
        "model": _get_current_model(),
        "embed_model": _get_current_embed_model(),
        "streaming": settings.enable_streaming,
        "features": {
            "streaming": settings.enable_streaming,
            "reranking": settings.enable_reranking,
            "web_search": settings.enable_web_search,
        },
    }


def _get_current_model() -> str:
    """Get the current LLM model based on provider"""
    if settings.llm_provider == "gemini":
        return settings.gemini_model
    elif settings.llm_provider == "openai":
        return settings.openai_model
    elif settings.llm_provider == "anthropic":
        return settings.anthropic_model
    elif settings.llm_provider == "ollama":
        return settings.ollama_model
    return settings.gemini_model


def _get_current_embed_model() -> str:
    """Get the current embedding model"""
    if settings.llm_provider == "gemini":
        return settings.gemini_embed_model
    elif settings.llm_provider == "openai":
        return settings.openai_embed_model
    return settings.gemini_embed_model


@app.get("/api/models")
def api_models() -> dict:
    """List available models for all providers"""
    models_info = {
        "provider": settings.llm_provider,
        "current_model": _get_current_model(),
        "available_providers": [],
    }
    
    # Gemini models
    if settings.gemini_api_key:
        try:
            gemini_models = list_models(settings.gemini_api_key)
            models_info["available_providers"].append({
                "name": "gemini",
                "models": gemini_models,
                "embedding_model": settings.gemini_embed_model,
            })
        except Exception:
            pass
    
    # OpenAI models
    if settings.openai_api_key:
        models_info["available_providers"].append({
            "name": "openai",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "embedding_model": settings.openai_embed_model,
        })
    
    # Anthropic models
    if settings.anthropic_api_key:
        models_info["available_providers"].append({
            "name": "anthropic",
            "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        })
    
    # Ollama models
    if settings.ollama_base_url:
        try:
            import requests
            r = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
            if r.ok:
                ollama_models = [m["name"] for m in r.json().get("models", [])]
                models_info["available_providers"].append({
                    "name": "ollama",
                    "models": ollama_models,
                })
        except Exception:
            pass
    
    return models_info


@app.post("/api/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    session_id = str(uuid.uuid4())
    with get_conn() as conn:
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
    """Non-streaming chat endpoint"""
    
    # Check if streaming requested
    if payload.stream and settings.enable_streaming:
        # Redirect to streaming endpoint
        raise HTTPException(
            status_code=400,
            detail="Use /api/sessions/{session_id}/chat/stream for streaming"
        )
    
    return _do_chat(session_id, payload, conn)


@app.post("/api/sessions/{session_id}/chat/stream")
def chat_stream(session_id: str, payload: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint using Server-Sent Events"""
    
    if not settings.enable_streaming:
        raise HTTPException(
            status_code=501,
            detail="Streaming is not enabled. Set ENABLE_STREAMING=true"
        )
    
    async def event_generator():
        try:
            # Get connection for retrieval
            with get_conn() as conn:
                ensure_session(conn, session_id)
                
                # Store user message
                conn.execute(
                    "INSERT INTO messages(id, session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), session_id, "user", payload.message, None, _now_iso()),
                )
                conn.commit()
                
                # Retrieve context
                query_embedding = None
                used_embeddings = False
                
                # Try to get embeddings
                try:
                    query_embedding = llm_provider.embed_text(payload.message)
                    used_embeddings = True
                except Exception:
                    query_embedding = None
                    used_embeddings = False
                
                retrieved = retrieve(conn, session_id, payload.message, query_embedding, top_k=payload.top_k)
                
                # Build context
                context_blocks = []
                citations = []
                for r in retrieved:
                    snippet = (r.text[:240] + "...") if len(r.text) > 240 else r.text
                    citations.append({
                        "chunk_id": r.chunk_id,
                        "file_name": r.file_name,
                        "page": r.page,
                        "score": float(r.score),
                        "snippet": snippet,
                    })
                    page_note = f" (page {r.page})" if r.page else ""
                    context_blocks.append(f"SOURCE: {r.file_name}{page_note}\n{r.text}")
                
                context = "\n\n---\n\n".join(context_blocks)
                
                # Get images if needed
                image_paths = []
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
                
                # Stream the response
                full_answer = ""
                
                for chunk in llm_provider.generate_stream(
                    system_text=system_text,
                    user_text=user_text,
                    temperature=payload.temperature,
                ):
                    if chunk.delta:
                        full_answer += chunk.delta
                        # Send chunk via SSE
                        data = json.dumps({
                            "delta": chunk.delta,
                            "model": chunk.model,
                        })
                        yield f"data: {data}\n\n"
                    
                    if chunk.done:
                        break
                
                # Store assistant message
                assistant_metadata = json.dumps({
                    "citations": citations,
                    "used_embeddings": used_embeddings,
                    "model": _get_current_model(),
                })
                conn.execute(
                    "INSERT INTO messages(id, session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), session_id, "assistant", full_answer, assistant_metadata, _now_iso()),
                )
                conn.commit()
                
                # Send final message with citations
                final_data = json.dumps({
                    "done": True,
                    "citations": citations,
                    "used_embeddings": used_embeddings,
                    "model": _get_current_model(),
                })
                yield f"data: {final_data}\n\n"
        
        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _do_chat(session_id: str, payload: ChatRequest, conn) -> ChatResponse:
    """Execute the chat logic"""
    
    if settings.llm_provider not in ["gemini", "openai", "anthropic", "ollama"]:
        raise HTTPException(status_code=400, detail=f"Provider {settings.llm_provider} not supported")
    
    if settings.llm_provider == "gemini" and not settings.gemini_api_key:
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
    if settings.gemini_api_key or settings.openai_api_key:
        try:
            query_embedding = llm_provider.embed_text(payload.message)
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
        response = llm_provider.generate(
            api_key=getattr(settings, f"{settings.llm_provider}_api_key", None),
            model=_get_current_model(),
            system_text=system_text,
            user_text=user_text,
            image_paths=image_paths,
            temperature=payload.temperature,
        )
        answer = response.content
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {exc}") from exc

    assistant_metadata = json.dumps(
        {
            "citations": [c.model_dump() for c in citations],
            "used_embeddings": used_embeddings,
            "model": _get_current_model(),
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
        model=_get_current_model(),
    )
