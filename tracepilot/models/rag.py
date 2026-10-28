"""Basic RAG pipeline (U03): retrieve once, generate once, check citations."""

from __future__ import annotations

from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .answering import Answer, AnswerDraft, attach_citations, build_messages, label_evidence, parse_draft
from .chat import invoke_chat
from .guards import enforce_outcome, redact_draft, suspicious_evidence
from .llm import Usage
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
        if not evidence:
            # Nothing retrieved: answering would only be the model's own knowledge. Skip the call.
            draft = AnswerDraft(outcome="insufficient_evidence", summary="No documentation was retrieved for this question.",
                                missing_information=["Documentation that covers this question"])
            return Answer(question=question, draft=draft, citations=[], invalid_citation_ids=[], evidence={},
                          model="(not called)", latency_ms=0, usage=Usage())

        call = invoke_chat(self.chat_model, messages)
        draft, citations, invalid = attach_citations(redact_draft(parse_draft(call.text)), evidence)
        draft, unsupported = enforce_outcome(draft)
        cited = {cid for f in draft.findings for cid in f.citation_ids}
        citations = [c for c in citations if c.id in cited]
        answer = Answer(
            question=question, draft=draft, citations=citations, invalid_citation_ids=invalid,
            evidence=evidence, model=call.model, latency_ms=call.latency_ms, usage=call.usage,
            unsupported_claims=unsupported, suspicious_evidence_ids=suspicious_evidence(evidence),
        )
        if answer.suspicious_evidence_ids:
            answer.notes.append(
                "Evidence " + ", ".join(answer.suspicious_evidence_ids)
                + " contains instruction-like text; it was treated as data, not followed."
            )
        return answer
