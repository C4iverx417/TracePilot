"""RAG pipeline without a workflow engine.

    answer():  version gate (U05) -> retrieve once -> generate once -> checks (U03/U04/U07)
    compare(): both versions gated -> retrieve each version SEPARATELY -> grouped prompt (U07)

The deterministic checks after generation, in order:
    attach_citations   drop citation ids that were not provided
    enforce_outcome    uncited findings are not findings; downgrade the outcome
    split_historical   issue-only findings become "historical cases"
    find_conflicts     curator-marked issue/doc contradictions present in the evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .answering import Answer, AnswerDraft, attach_citations, build_messages, label_evidence, parse_draft
from .chat import ChatCall, invoke_chat
from .documents import SearchHit
from .guards import enforce_outcome, find_conflicts, redact_draft, split_historical, suspicious_evidence
from .llm import Usage
from .retrieval import Retriever, diversify
from .versions import VersionChoice, parse_version


@dataclass
class Prepared:
    """Everything decided before the model call: what it will see and why."""

    evidence: Dict[str, SearchHit]
    messages: List[BaseMessage]
    version: Optional[str]
    groups: Dict[str, str] = field(default_factory=dict)
    compare: Optional[Tuple[str, str]] = None


def no_call_answer(question: str, outcome: str, summary: str, missing: List[str],
                   version: Optional[str] = None) -> Answer:
    """An answer decided by rules alone; the model is not called and nothing is billed."""
    draft = AnswerDraft(outcome=outcome, summary=summary, missing_information=missing)
    return Answer(question=question, draft=draft, citations=[], invalid_citation_ids=[], evidence={},
                  model="(not called)", latency_ms=0, usage=Usage(), target_version=version)


class RagPipeline:
    def __init__(self, retriever: Retriever, chat_model: BaseChatModel,
                 covered_versions: Sequence[str] = (), k: int = 8, per_document: int = 2):
        self.retriever = retriever
        self.chat_model = chat_model
        self.covered_versions = tuple(covered_versions)
        self.k = k
        self.per_document = per_document

    def _select(self, question: str, k: int, version: Optional[str]) -> List[SearchHit]:
        """Over-fetch, then keep at most `per_document` chunks per source document."""
        return diversify(self.retriever.search(question, k * 3, version=version), k, self.per_document)

    # ------------------------------------------------------------ version handling

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

    # ------------------------------------------------------------ single version

    def prepare(self, question: str, version: Optional[str]) -> Prepared:
        """Version-filtered retrieval + prompt, without calling the model (used by --show-context)."""
        evidence = label_evidence(self._select(question, self.k, version))
        return Prepared(evidence, build_messages(question, evidence), version)

    def answer(self, question: str, raw_version: str) -> Answer:
        choice = self.resolve_version(raw_version)
        return self.version_gate(question, choice) or self.generate(question, self.prepare(question, choice.version))

    # ------------------------------------------------------------ upgrade comparison

    def compare_gate(self, question: str, before: VersionChoice, after: VersionChoice) -> Optional[Answer]:
        for choice in (before, after):
            gated = self.version_gate(question, choice)
            if gated:
                return gated
        if before.version == after.version:
            return no_call_answer(question, "insufficient_context", "The two versions are the same.",
                                  ["The version you upgraded FROM and the version you upgraded TO"])
        return None

    def prepare_compare(self, question: str, before: str, after: str) -> Prepared:
        """Retrieve each version separately (PRD 6.4 rule 6) and keep the groups apart."""
        per_side = max(1, self.k // 2)
        before_hits = self._select(question, per_side, before)
        after_hits = self._select(question, per_side, after)
        evidence = label_evidence(before_hits + after_hits)  # c1..cN: before group first, then after
        groups = {cid: (before if i < len(before_hits) else after) for i, cid in enumerate(evidence)}
        messages = build_messages(question, evidence, groups=groups, compare=(before, after))
        return Prepared(evidence, messages, after, groups, (before, after))

    def compare(self, question: str, raw_before: str, raw_after: str) -> Answer:
        before, after = self.resolve_version(raw_before), self.resolve_version(raw_after)
        gated = self.compare_gate(question, before, after)
        if gated:
            return gated
        return self.generate(question, self.prepare_compare(question, before.version, after.version))

    # ------------------------------------------------------------ model call + checks

    def generate(self, question: str, prepared: Prepared) -> Answer:
        if not prepared.evidence:
            return empty_evidence_answer(question, prepared)
        call = invoke_chat(self.chat_model, prepared.messages)
        return finish_answer(question, prepared, parse_draft(call.text), call)


def empty_evidence_answer(question: str, prepared: Prepared) -> Answer:
    # Nothing retrieved: answering would only be the model's own knowledge. Skip the call.
    return no_call_answer(question, "insufficient_evidence", "No documentation was retrieved for this question.",
                          ["Documentation that covers this question"], prepared.version)


def finish_answer(question: str, prepared: Prepared, draft: AnswerDraft, call: ChatCall) -> Answer:
    """Deterministic checks on a parsed model draft. Shared by the plain pipeline and the LangGraph workflow."""
    evidence = prepared.evidence
    draft, citations, invalid = attach_citations(redact_draft(draft), evidence, prepared.groups)
    draft, unsupported = enforce_outcome(draft)
    draft, historical = split_historical(draft, evidence)

    conflicts = find_conflicts(evidence)
    cited = {cid for f in draft.findings + historical for cid in f.citation_ids}
    if any(a in cited and b in cited for a, b in conflicts):
        draft = draft.model_copy(update={"outcome": "conflicting_evidence"})

    answer = Answer(
        question=question, draft=draft, citations=[c for c in citations if c.id in cited],
        invalid_citation_ids=invalid, evidence=evidence, model=call.model, latency_ms=call.latency_ms,
        usage=call.usage, target_version=prepared.version, compare_versions=prepared.compare,
        groups=prepared.groups, historical_cases=historical, unsupported_claims=unsupported,
        suspicious_evidence_ids=suspicious_evidence(evidence),
        conflicts=[f"{a} contradicts {b} (marked by curator)" for a, b in conflicts],
    )
    if historical and not draft.findings:
        answer.notes.append("Only similar issues support these points; the cause of YOUR problem is not confirmed.")
    if answer.suspicious_evidence_ids:
        answer.notes.append(
            "Evidence " + ", ".join(answer.suspicious_evidence_ids)
            + " contains instruction-like text; it was treated as data, not followed."
        )
    return answer
