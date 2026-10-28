"""Composition root: the only place that turns Settings into concrete objects.

Controllers ask this module for a ChatModel or a ready retriever and never pick
implementations themselves (dependency inversion, PRD 10.3).
"""

from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from .config import Settings
from .models.chunking import MarkdownChunker
from .models.embeddings import FastEmbedEmbeddings
from .models.llm import ChatModel, MockClient, OpenAICompatibleClient
from .models.retrieval import VectorRetriever
from .models.sources import load_snapshot


def build_chat_model(settings: Settings) -> ChatModel:
    if settings.provider == "mock":
        return MockClient(settings)
    return OpenAICompatibleClient(settings)


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding == "fake":
        # Random-but-repeatable vectors: exercises the pipeline, carries no meaning.
        return DeterministicFakeEmbedding(size=384)
    return FastEmbedEmbeddings(settings.embedding_model)


def build_retriever(settings: Settings, embeddings: Embeddings | None = None) -> VectorRetriever:
    documents = load_snapshot(settings.manifest_path)
    chunker = MarkdownChunker(chunk_size=settings.chunk_size)
    chunks = [chunk for doc in documents for chunk in chunker.split(doc)]
    retriever = VectorRetriever(embeddings or build_embeddings(settings))
    retriever.index(documents, chunks)
    return retriever
