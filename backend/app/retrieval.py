from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from app.bm25 import BM25Index
from app.vector import cosine_similarity, from_blob


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    file_id: str
    file_name: str
    page: int | None
    text: str
    score: float


def _fetch_chunks(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT c.id as chunk_id, c.file_id as file_id, f.name as file_name, c.page as page, c.text as text
        FROM chunks c
        JOIN files f ON f.id = c.file_id
        WHERE f.session_id = ?
        ORDER BY f.created_at ASC, c.idx ASC
        """,
        (session_id,),
    ).fetchall()


def _fetch_embeddings(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, tuple[int, bytes]]:
    if not chunk_ids:
        return {}
    qmarks = ",".join(["?"] * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, dim, vector FROM embeddings WHERE chunk_id IN ({qmarks})",
        chunk_ids,
    ).fetchall()
    out: dict[str, tuple[int, bytes]] = {}
    for r in rows:
        out[r["chunk_id"]] = (int(r["dim"]), r["vector"])
    return out


def retrieve(
    conn: sqlite3.Connection,
    session_id: str,
    query: str,
    query_embedding: Optional[List[float]],
    top_k: int = 8,
) -> list[RetrievedChunk]:
    chunks = _fetch_chunks(conn, session_id)
    if not chunks:
        return []

    # Hybrid retrieval:
    # - Prefer vectors when available
    # - Fill remaining slots with BM25 (and also cover chunks without embeddings)
    out: list[RetrievedChunk] = []
    seen: set[str] = set()

    if query_embedding is not None:
        emb_map = _fetch_embeddings(conn, [r["chunk_id"] for r in chunks])
        v_scored: list[RetrievedChunk] = []
        for r in chunks:
            item = emb_map.get(r["chunk_id"])
            if not item:
                continue
            dim, blob = item
            vec = from_blob(blob, dim)
            score = cosine_similarity(query_embedding, vec)
            v_scored.append(
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    file_id=r["file_id"],
                    file_name=r["file_name"],
                    page=r["page"],
                    text=r["text"],
                    score=score,
                )
            )
        v_scored.sort(key=lambda x: x.score, reverse=True)
        for item in v_scored[:top_k]:
            out.append(item)
            seen.add(item.chunk_id)

    if len(out) < top_k:
        docs = [r["text"] for r in chunks]
        index = BM25Index.build(docs)
        bm_scored: list[RetrievedChunk] = []
        for i, r in enumerate(chunks):
            score = index.score(query, i)
            if score <= 0:
                continue
            bm_scored.append(
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    file_id=r["file_id"],
                    file_name=r["file_name"],
                    page=r["page"],
                    text=r["text"],
                    score=score,
                )
            )
        bm_scored.sort(key=lambda x: x.score, reverse=True)
        for item in bm_scored:
            if item.chunk_id in seen:
                continue
            out.append(item)
            if len(out) >= top_k:
                break

    return out[:top_k]
