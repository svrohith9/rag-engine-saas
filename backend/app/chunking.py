from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class Chunk:
    idx: int
    page: int | None
    text: str


def chunk_pages(pages: Iterable[tuple[int | None, str]], chunk_size: int = 1200, overlap: int = 200) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx = 0

    for page, text in pages:
        t = (text or "").strip()
        if not t:
            continue
        start = 0
        while start < len(t):
            end = min(start + chunk_size, len(t))
            chunk_text = t[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(idx=idx, page=page, text=chunk_text))
                idx += 1
            if end >= len(t):
                break
            start = max(0, end - overlap)

    return chunks
