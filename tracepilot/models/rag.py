"""Basic RAG pipeline (U03): retrieve once, generate once, check citations."""

from __future__ import annotations

from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .answering import Answer, attach_citations, build_messages, label_evidence, parse_draft
from .chat import invoke_chat
from .retrieval import VectorRetriever


class RagPipeline:
    def __init__(self, retriever: VectorRetriever, chat_model: BaseChatModel, k: int = 5):
        self.retriever = retriever
        self.chat_model = chat_model
        self.k = k

    def prepare(self, question: str):
        """Retrieval + prompt, without calling the model (used by --show-context)."""
        evidence = label_evidence(self.retriever.search(question, self.k))
        return evidence, build_messages(question, evidence)

    def answer(self, question: str) -> Answer:
        evidence, messages = self.prepare(question)
        return self.generate(question, evidence, messages)

    def generate(self, question: str, evidence, messages: List[BaseMessage]) -> Answer:
        call = invoke_chat(self.chat_model, messages)
        draft, citations, invalid = attach_citations(parse_draft(call.text), evidence)
        return Answer(
            question=question, draft=draft, citations=citations, invalid_citation_ids=invalid,
            evidence=evidence, model=call.model, latency_ms=call.latency_ms, usage=call.usage,
        )
