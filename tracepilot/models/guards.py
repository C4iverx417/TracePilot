"""Deterministic checks applied around the model (PRD FR-06, FR-08, NFR-05).

The model *proposes* an outcome; these rules decide what we are willing to show:
- a finding without a valid citation is not shown as a finding;
- "answered" with unsupported parts becomes "partially_answered";
- nothing supported at all becomes "insufficient_evidence".
They also flag evidence that looks like it is giving instructions, and redact
anything shaped like a secret before output.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .answering import AnswerDraft, Finding
from .documents import SearchHit

_INSTRUCTION_LIKE = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions"
    r"|disregard (the )?(system|previous) prompt"
    r"|you are now\b"
    r"|(reveal|print|show|send) (the |your )?(api[ _-]?key|system prompt|secrets?|password)"
    r"|run (this|the following) command",
    re.I,
)
_SECRET = re.compile(r"\b(sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b")


def suspicious_evidence(evidence: Dict[str, SearchHit]) -> List[str]:
    return [cid for cid, hit in evidence.items() if _INSTRUCTION_LIKE.search(hit.chunk.content)]


def redact_secrets(text: str) -> str:
    return _SECRET.sub("[REDACTED]", text)


def redact_draft(draft: AnswerDraft) -> AnswerDraft:
    data = draft.model_dump()
    redacted = {
        "summary": redact_secrets(data["summary"]),
        "findings": [{**f, "text": redact_secrets(f["text"])} for f in data["findings"]],
        **{key: [redact_secrets(x) for x in data[key]] for key in ("hypotheses", "steps", "missing_information")},
    }
    return draft.model_validate({**data, **redacted})


def enforce_outcome(draft: AnswerDraft) -> Tuple[AnswerDraft, List[str]]:
    """Return (draft with only cited findings and a justified outcome, unsupported claim texts)."""
    supported = [f for f in draft.findings if f.citation_ids]
    unsupported = [f.text for f in draft.findings if not f.citation_ids]
    outcome = draft.outcome
    if outcome not in ("answered", "partially_answered", "insufficient_evidence"):
        outcome = "partially_answered" if supported else "insufficient_evidence"  # server-only outcomes
    if outcome in ("answered", "partially_answered") and not supported:
        outcome = "insufficient_evidence"
    elif outcome == "answered" and unsupported:
        outcome = "partially_answered"
    return draft.model_copy(update={"findings": supported, "outcome": outcome}), unsupported


def split_historical(draft: AnswerDraft, evidence: Dict[str, SearchHit]) -> Tuple[AnswerDraft, List[Finding]]:
    """Findings backed ONLY by issues are historical cases, not documented facts (PRD FR-06, AT-10).

    If nothing documented remains, an "answered" draft becomes "partially_answered": a similar
    case is a lead to verify, not a confirmed root cause.
    """
    def documented(f: Finding) -> bool:
        return any(evidence[cid].document.source_type != "issue" for cid in f.citation_ids)

    facts = [f for f in draft.findings if documented(f)]
    cases = [f for f in draft.findings if not documented(f)]
    outcome = draft.outcome
    if cases and not facts and outcome == "answered":
        outcome = "partially_answered"
    return draft.model_copy(update={"findings": facts, "outcome": outcome}), cases


def find_conflicts(evidence: Dict[str, SearchHit]) -> List[Tuple[str, str]]:
    """(issue evidence id, document evidence id) pairs that a curator marked as contradicting."""
    by_doc: Dict[str, List[str]] = {}
    for cid, hit in evidence.items():
        by_doc.setdefault(hit.document.document_id, []).append(cid)
    pairs = []
    for cid, hit in evidence.items():
        issue = hit.document.issue
        for doc_id in issue.conflicts_with if issue else ():
            for other in by_doc.get(doc_id, []):
                pairs.append((cid, other))
    return pairs
