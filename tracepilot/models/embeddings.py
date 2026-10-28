"""Embedding adapters.

Retrieval code depends on LangChain's `Embeddings` interface (embed_documents /
embed_query), not on a concrete library. This file adapts the local `fastembed`
library to that interface; tests use LangChain's DeterministicFakeEmbedding.
Swapping in an API-based embedding later means adding a class, not editing the
retriever.
"""

from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    """Runs an ONNX sentence-embedding model locally (CPU). First use downloads the model."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from fastembed import TextEmbedding  # imported lazily: heavy, and unused by tests

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self._model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
