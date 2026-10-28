"""Composition root: the only place that turns Settings into concrete objects.

Controllers ask this module for a ChatModel or a ready retriever and never pick
implementations themselves (dependency inversion, PRD 10.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings
from .models.chunking import MarkdownChunker
from .models.embeddings import FastEmbedEmbeddings
from .models.llm import ChatModel, MockClient, OpenAICompatibleClient
from .models.mock_chat import ExtractiveMockChatModel
from .models.rag import RagPipeline
from .models.keyword import KeywordRetriever
from .models.retrieval import HybridRetriever, Retriever, VectorRetriever
from .models.runlog import RunLog, answer_record, workflow_extra
from .models.runs import RunExecutor, RunRegistry
from .models.workflow import Budget, SupportWorkflow
from .models.conversation import ConversationService
from .models.sessions import InMemorySessionStore, SessionStore
from .models.sources import Project, load_project


def build_chat_model(settings: Settings) -> ChatModel:
    if settings.provider == "mock":
        return MockClient(settings)
    return OpenAICompatibleClient(settings)


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding == "fake":
        # Random-but-repeatable vectors: exercises the pipeline, carries no meaning.
        return DeterministicFakeEmbedding(size=384)
    return FastEmbedEmbeddings(settings.embedding_model)


def build_project(settings: Settings) -> Project:
    return load_project(settings.project_path)


@dataclass(frozen=True)
class KnowledgeIndex:
    project: Project
    retriever: Retriever
    chunk_count: int
    mode: str


def build_index(settings: Settings, embeddings: Embeddings | None = None,
                project: Project | None = None) -> KnowledgeIndex:
    project = project or build_project(settings)
    chunker = MarkdownChunker(chunk_size=settings.chunk_size)
    chunks = [chunk for doc in project.documents for chunk in chunker.split(doc)]

    channels: Dict[str, Retriever] = {}
    if settings.retrieval in ("vector", "hybrid"):
        vector = VectorRetriever(embeddings or build_embeddings(settings))
        vector.index(project.documents, chunks)
        channels["vector"] = vector
    if settings.retrieval in ("keyword", "hybrid"):
        keyword = KeywordRetriever()
        keyword.index(project.documents, chunks)
        channels["bm25"] = keyword

    if settings.retrieval == "hybrid":
        retriever: Retriever = HybridRetriever(channels, candidates=settings.candidates)
    else:
        retriever = next(iter(channels.values()))
    return KnowledgeIndex(project, retriever, len({c.chunk_id for c in chunks}), settings.retrieval)


def build_retriever(settings: Settings, embeddings: Embeddings | None = None,
                    project: Project | None = None) -> Retriever:
    return build_index(settings, embeddings, project).retriever


def build_langchain_chat(settings: Settings) -> BaseChatModel:
    if settings.provider == "mock":
        return ExtractiveMockChatModel()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_s,
        max_retries=0,  # retries are a workflow decision (PRD 9.3), not hidden in the client
    )


def build_rag(settings: Settings, retriever: Retriever | None = None,
              project: Project | None = None) -> RagPipeline:
    project = project or build_project(settings)
    return RagPipeline(retriever or build_retriever(settings, project=project), build_langchain_chat(settings),
                       covered_versions=project.versions, k=settings.top_k, per_document=settings.per_document)


def build_run_log() -> RunLog:
    return RunLog(Path("var/runs"))


def build_workflow(settings: Settings, pipeline: RagPipeline) -> SupportWorkflow:
    budget = Budget(max_retrieval_rounds=settings.max_retrieval_rounds, max_model_calls=settings.max_model_calls,
                    deadline_s=settings.deadline_s)
    return SupportWorkflow(pipeline, budget)


def build_conversation(settings: Settings, store: SessionStore | None = None,
                       index: KnowledgeIndex | None = None) -> ConversationService:
    index = index or build_index(settings)
    pipeline = build_rag(settings, index.retriever, index.project)
    return ConversationService(store or InMemorySessionStore(), build_workflow(settings, pipeline),
                               index.project.versions, index.project.project_id)


@dataclass
class ApiContext:
    project: Project
    service: ConversationService
    registry: RunRegistry
    executor: RunExecutor


def build_api_context(settings: Settings, index: KnowledgeIndex | None = None,
                      store: SessionStore | None = None) -> ApiContext:
    index = index or build_index(settings)
    service = build_conversation(settings, store, index)
    registry = RunRegistry()
    run_log = build_run_log()

    def log_run(record):
        result = record.result
        if result and result.answer:
            data = answer_record(record.run_id, result.answer, workflow_extra(result.state) if result.state else {})
        else:
            data = {"run_id": record.run_id, "status": record.status, "error_code": record.error_code}
        data.update({"session_id": record.session_id, "episode_id": record.episode_id, "status": record.status})
        run_log.save(data)

    return ApiContext(index.project, service, registry, RunExecutor(service, registry, on_done=log_run))
