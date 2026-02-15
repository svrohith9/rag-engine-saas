from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence


_WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


@dataclass
class BM25Index:
    docs_tokens: List[List[str]]
    doc_freq: Counter
    avgdl: float
    k1: float = 1.2
    b: float = 0.75

    @classmethod
    def build(cls, docs: Sequence[str]) -> "BM25Index":
        docs_tokens = [_tokenize(d) for d in docs]
        doc_freq: Counter = Counter()
        for toks in docs_tokens:
            doc_freq.update(set(toks))
        avgdl = (sum(len(t) for t in docs_tokens) / max(1, len(docs_tokens)))
        return cls(docs_tokens=docs_tokens, doc_freq=doc_freq, avgdl=avgdl)

    def score(self, query: str, doc_idx: int) -> float:
        q = _tokenize(query)
        if not q:
            return 0.0
        toks = self.docs_tokens[doc_idx]
        if not toks:
            return 0.0
        tf = Counter(toks)
        dl = len(toks)
        N = len(self.docs_tokens)

        score = 0.0
        for term in q:
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            f = tf.get(term, 0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            score += idf * (f * (self.k1 + 1)) / (denom or 1.0)
        return score
