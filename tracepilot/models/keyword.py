"""BM25 keyword retrieval.

Embeddings blur exact strings: "allow_mutation", "ImportError" or an error code
may land near loosely related text. BM25 scores a chunk by how many query tokens
it contains, weighted by how rare each token is (IDF) and normalised by chunk
length. Implemented here in ~40 lines so the formula is visible.

    score(q, d) = sum_t IDF(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

from .documents import Chunk, SearchHit, SourceDocument
from .retrieval import validate_query
from .versions import scope_matches

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lower-case identifier-like tokens. `allow_mutation` stays one token; `pydantic-settings` -> 2."""
    return _TOKEN.findall(text.lower())


class KeywordRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._chunks: Dict[str, Chunk] = {}
        self._documents: Dict[str, SourceDocument] = {}
        self._tf: Dict[str, Counter] = {}
        self._length: Dict[str, int] = {}

    def index(self, documents: Iterable[SourceDocument], chunks: Iterable[Chunk]) -> None:
        for doc in documents:
            self._documents[doc.document_id] = doc
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                continue
            doc = self._documents[chunk.document_id]
            tokens = tokenize(" ".join((doc.title,) + chunk.heading_path) + "\n" + chunk.content)
            self._chunks[chunk.chunk_id] = chunk
            self._tf[chunk.chunk_id] = Counter(tokens)
            self._length[chunk.chunk_id] = len(tokens)

    def search(self, query: str, k: int, version: Optional[str] = None) -> List[SearchHit]:
        validate_query(query, k)
        terms = set(tokenize(query))
        candidates = [cid for cid, c in self._chunks.items() if version is None or scope_matches(c.version_scope, version)]
        if not terms or not candidates:
            return []
        # Statistics over the filtered collection, so another version's vocabulary does not skew IDF.
        n = len(candidates)
        avgdl = sum(self._length[cid] for cid in candidates) / n
        df = Counter(t for cid in candidates for t in terms if t in self._tf[cid])

        scores = {}
        for cid in candidates:
            tf, length, score = self._tf[cid], self._length[cid], 0.0
            for t in terms:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * length / avgdl))
            if score > 0:
                scores[cid] = score
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
        return [
            SearchHit(rank, scores[cid], self._chunks[cid], self._documents[self._chunks[cid].document_id],
                      (f"bm25#{rank}",))
            for rank, cid in enumerate(ranked, start=1)
        ]
