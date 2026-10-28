"""Retrieval over chunks. Three interchangeable implementations of `Retriever`:

- VectorRetriever:  embeddings + cosine similarity (LangChain InMemoryVectorStore)
- KeywordRetriever: BM25 over exact tokens, in models/keyword.py
- HybridRetriever:  runs both, fuses their rankings with Reciprocal Rank Fusion

All of them apply the version filter BEFORE ranking. Indexes live in memory.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .documents import Chunk, SearchHit, SourceDocument
from .versions import scope_matches


def embedding_text(chunk: Chunk, doc: SourceDocument) -> str:
    """What actually gets embedded: the chunk plus where it sits in the document.

    The heading path gives a short chunk ("Use `frozen=True`...") the context
    that it belongs to e.g. "Models > Faux immutability".
    """
    return f"{location_text(chunk, doc)}\n\n{chunk.content}"


def location_text(chunk: Chunk, doc: SourceDocument) -> str:
    """'Title > heading > subheading', without repeating the title when it is also the H1."""
    path = chunk.heading_path
    if path and path[0] == doc.title:
        path = path[1:]
    return " > ".join((doc.title,) + path)


def diversify(hits: List[SearchHit], k: int, per_document: int) -> List[SearchHit]:
    """Keep ranking order but allow at most `per_document` chunks from one document, then re-rank 1..k.

    Without this, one long page (or an issue whose title equals the error text) can fill
    every context slot and crowd out the documentation that actually explains the change.
    """
    taken: Dict[str, int] = {}
    kept: List[SearchHit] = []
    for hit in hits:
        doc_id = hit.chunk.document_id
        if taken.get(doc_id, 0) >= per_document:
            continue
        taken[doc_id] = taken.get(doc_id, 0) + 1
        kept.append(hit)
        if len(kept) == k:
            break
    return [SearchHit(rank, h.score, h.chunk, h.document, h.channels) for rank, h in enumerate(kept, start=1)]


class Retriever(Protocol):
    """What the pipeline depends on. Introduced in U06, when a second implementation appeared."""

    def search(self, query: str, k: int, version: Optional[str] = None) -> List[SearchHit]: ...


def validate_query(query: str, k: int) -> None:
    if not query.strip():
        raise ValueError("query is empty")
    if k < 1:
        raise ValueError("k must be >= 1")


class VectorRetriever:
    def __init__(self, embeddings: Embeddings):
        self._store = InMemoryVectorStore(embedding=embeddings)
        self._chunks: Dict[str, Chunk] = {}
        self._documents: Dict[str, SourceDocument] = {}

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def index(self, documents: Iterable[SourceDocument], chunks: Iterable[Chunk]) -> None:
        for doc in documents:
            self._documents[doc.document_id] = doc
        batch = []
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                continue  # same chunk indexed twice -> keep one copy
            self._chunks[chunk.chunk_id] = chunk
            batch.append(
                Document(
                    id=chunk.chunk_id,
                    page_content=embedding_text(chunk, self._documents[chunk.document_id]),
                    metadata={"chunk_id": chunk.chunk_id, "version_scope": chunk.version_scope},
                )
            )
        if batch:
            self._store.add_documents(batch)

    def search(self, query: str, k: int, version: Optional[str] = None) -> List[SearchHit]:
        """Top-k chunks. With `version`, non-matching chunks are removed BEFORE ranking (PRD 6.4 rule 4)."""
        validate_query(query, k)
        if not self._chunks:
            return []
        version_filter = None
        if version is not None:
            version_filter = lambda doc: scope_matches(doc.metadata["version_scope"], version)  # noqa: E731
        results = self._store.similarity_search_with_score(query, k=k, filter=version_filter)
        hits = []
        for rank, (document, score) in enumerate(results, start=1):
            chunk = self._chunks[document.metadata["chunk_id"]]
            hits.append(SearchHit(rank, float(score), chunk, self._documents[chunk.document_id], (f"vector#{rank}",)))
        return hits


class HybridRetriever:
    """Reciprocal Rank Fusion: score(chunk) = sum over retrievers of 1 / (rrf_k + rank).

    RRF only uses ranks, so it does not need cosine and BM25 scores to be on the
    same scale. A chunk ranked well by both retrievers rises to the top; a chunk
    found by only one can still make it in.
    """

    def __init__(self, retrievers: Dict[str, Retriever], candidates: int = 20, rrf_k: int = 60):
        self.retrievers = retrievers
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(self, query: str, k: int, version: Optional[str] = None) -> List[SearchHit]:
        validate_query(query, k)
        fused: Dict[str, float] = {}
        first_seen: Dict[str, SearchHit] = {}
        channels: Dict[str, List[str]] = {}
        for name, retriever in self.retrievers.items():
            for hit in retriever.search(query, max(k, self.candidates), version=version):
                cid = hit.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + hit.rank)
                first_seen.setdefault(cid, hit)
                channels.setdefault(cid, []).append(f"{name}#{hit.rank}")
        ranked = sorted(fused, key=lambda cid: (-fused[cid], cid))[:k]
        return [
            SearchHit(rank, fused[cid], first_seen[cid].chunk, first_seen[cid].document, tuple(channels[cid]))
            for rank, cid in enumerate(ranked, start=1)
        ]
