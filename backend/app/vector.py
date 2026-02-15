from __future__ import annotations

import math
import struct
from typing import Iterable, List


def to_blob(vec: List[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def from_blob(blob: bytes, dim: int) -> List[float]:
    return list(struct.unpack(f"<{dim}f", blob))


def l2_norm(vec: Iterable[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (na * nb)
