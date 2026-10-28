"""In-memory vector retrieval over chunks.

index():  chunk -> embedding text -> vector, stored in LangChain's InMemoryVectorStore
search(): query -> vector -> cosine similarity against every chunk -> top-k hits

The store lives only in this process. Nothing is written to disk (PRD U02).
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .documents import Chunk, SearchHit, SourceDocument


def embedding_text(chunk: Chunk, doc: SourceDocument) -> str:
    """What actually gets embedded: the chunk plus where it sits in the document.

    The heading path gives a short chunk ("Use `frozen=True`...") the context
    that it belongs to e.g. "Models > Faux immutability".
    """
    location = " > ".join((doc.title,) + chunk.heading_path)
    return f"{location}\n\n{chunk.content}"


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
                    metadata={"chunk_id": chunk.chunk_id},
                )
            )
        if batch:
            self._store.add_documents(batch)

    def search(self, query: str, k: int) -> List[SearchHit]:
        if not query.strip():
            raise ValueError("query is empty")
        if k < 1:
            raise ValueError("k must be >= 1")
        if not self._chunks:
            return []
        results = self._store.similarity_search_with_score(query, k=k)
        hits = []
        for rank, (document, score) in enumerate(results, start=1):
            chunk = self._chunks[document.metadata["chunk_id"]]
            hits.append(SearchHit(rank, float(score), chunk, self._documents[chunk.document_id]))
        return hits
