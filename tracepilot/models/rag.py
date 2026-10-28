"""Basic RAG pipeline: version gate (U05) -> retrieve once -> generate once -> check citations (U03/U04)."""

from __future__ import annotations

from typing import List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .answering import Answer, AnswerDraft, attach_citations, build_messages, label_evidence, parse_draft
from .chat import invoke_chat
from .guards import enforce_outcome, redact_draft, suspicious_evidence
from .llm import Usage
from .retrieval import VectorRetriever
from .versions import VersionChoice, parse_version


def no_call_answer(question: str, outcome: str, summary: str, missing: List[str],
                   version: Optional[str] = None) -> Answer:
    """An answer decided by rules alone; the model is not called and nothing is billed."""
    draft = AnswerDraft(outcome=outcome, summary=summary, missing_information=missing)
    return Answer(question=question, draft=draft, citations=[], invalid_citation_ids=[], evidence={},
                  model="(not called)", latency_ms=0, usage=Usage(), target_version=version)


class RagPipeline:
    def __init__(self, retriever: VectorRetriever, chat_model: BaseChatModel,
                 covered_versions: Sequence[str] = (), k: int = 5):
        self.retriever = retriever
        self.chat_model = chat_model
        self.covered_versions = tuple(covered_versions)
        self.k = k

    def resolve_version(self, raw_version: str) -> VersionChoice:
        return parse_version(raw_version, self.covered_versions)

    def version_gate(self, question: str, choice: VersionChoice) -> Optional[Answer]:
        """Stop before retrieval when the version is unusable (PRD 6.4 rules 1 and 7)."""
        covered = ", ".join(self.covered_versions)
        if choice.kind == "unsupported":
            return no_call_answer(question, "unsupported_version", f"{choice.reason}. Covered versions: {covered}.",
                                  [f"Use one of the covered versions: {covered}"])
        if choice.kind == "unknown":
            return no_call_answer(question, "insufficient_context",
                                  f"{choice.reason}. The answer depends on your Pydantic version.",
                                  [f"Your installed Pydantic version (covered: {covered}); "
                                   "run `python -c \"import pydantic; print(pydantic.VERSION)\"`"])
        return None

    def prepare(self, question: str, version: Optional[str]):
        """Version-filtered retrieval + prompt, without calling the model (used by --show-context)."""
        evidence = label_evidence(self.retriever.search(question, self.k, version=version))
        return evidence, build_messages(question, evidence)

    def answer(self, question: str, raw_version: str) -> Answer:
        choice = self.resolve_version(raw_version)
        gated = self.version_gate(question, choice)
        if gated:
            return gated
        evidence, messages = self.prepare(question, choice.version)
        return self.generate(question, evidence, messages, choice.version)

    def generate(self, question: str, evidence, messages: List[BaseMessage], version: Optional[str] = None) -> Answer:
        if not evidence:
            # Nothing retrieved: answering would only be the model's own knowledge. Skip the call.
            return no_call_answer(question, "insufficient_evidence", "No documentation was retrieved for this question.",
                                  ["Documentation that covers this question"], version)

        call = invoke_chat(self.chat_model, messages)
        draft, citations, invalid = attach_citations(redact_draft(parse_draft(call.text)), evidence)
        draft, unsupported = enforce_outcome(draft)
        cited = {cid for f in draft.findings for cid in f.citation_ids}
        citations = [c for c in citations if c.id in cited]
        answer = Answer(
            question=question, draft=draft, citations=citations, invalid_citation_ids=invalid,
            evidence=evidence, model=call.model, latency_ms=call.latency_ms, usage=call.usage,
            unsupported_claims=unsupported, suspicious_evidence_ids=suspicious_evidence(evidence),
            target_version=version,
        )
        if answer.suspicious_evidence_ids:
            answer.notes.append(
                "Evidence " + ", ".join(answer.suspicious_evidence_ids)
                + " contains instruction-like text; it was treated as data, not followed."
            )
        return answer
