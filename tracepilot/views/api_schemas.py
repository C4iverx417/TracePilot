"""API view layer: Pydantic response/request schemas and the functions that fill them.

Controllers never return domain objects directly; they pass them through these
functions, so the HTTP contract (PRD 11.2-11.4) is defined in one place.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from ..models.answering import Answer, Citation
from ..models.runs import RunRecord
from ..models.sessions import Episode, Session
from ..models.sources import Project

MAX_MESSAGE_CHARS = 8000
MAX_LOG_CHARS = 20000


# ---------------------------------------------------------------- requests

class SessionCreate(BaseModel):
    version: Optional[str] = Field(None, description="Covered version, 'unknown', or omitted")


class RunCreate(BaseModel):
    episode_id: str
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    version: Optional[str] = None
    error_log: Optional[str] = Field(None, max_length=MAX_LOG_CHARS)


# ---------------------------------------------------------------- responses

class ErrorBody(BaseModel):
    code: str
    message: str
    run_id: Optional[str] = None


class ErrorOut(BaseModel):
    error: ErrorBody


class ProjectOut(BaseModel):
    project_id: str
    display_name: str
    versions: List[str]
    snapshot_ids: List[str]
    knowledge_updated_at: str


class EpisodeOut(BaseModel):
    episode_id: str
    question: str
    confirmed: dict
    clarification_rounds: int
    version_history: List[str]


class MessageOut(BaseModel):
    role: str
    content: str
    run_id: Optional[str]
    outcome: Optional[str]
    created_at: str


class SessionOut(BaseModel):
    session_id: str
    retained_version: Optional[str]
    current_episode: EpisodeOut
    episodes: int
    messages: List[MessageOut]


class FindingOut(BaseModel):
    text: str
    citation_ids: List[str]


class AnswerOut(BaseModel):
    summary: str
    findings: List[FindingOut]
    historical_cases: List[FindingOut]
    hypotheses: List[str]
    steps: List[str]
    missing_information: List[str]
    unsupported_claims: List[str]
    conflicts: List[str]
    notes: List[str]


class CitationOut(BaseModel):
    id: str
    chunk_id: str
    title: str
    source_type: str
    heading_path: List[str]
    version_scope: str
    group: Optional[str]
    url: str
    locator: str
    snapshot_id: str
    snippet: str
    issue_status: Optional[str] = None
    issue_resolved_by: Optional[str] = None
    issue_resolution: Optional[str] = None
    issue_resolution_url: Optional[str] = None


class CitationDetailOut(CitationOut):
    content: str = Field(description="The full chunk text exactly as it was shown to the model")
    document_fetched_at: str
    version_evidence: str


class UsageOut(BaseModel):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    source: Optional[str]
    estimated_cost_usd: Optional[float] = None
    retrieval_rounds: Optional[int] = None
    model_calls: Optional[int] = None


class RunAccepted(BaseModel):
    run_id: str
    status: str
    status_url: str


class RunOut(BaseModel):
    run_id: str
    session_id: str
    episode_id: str
    status: str
    stage: str
    outcome: Optional[str] = None
    target_version: Optional[str] = None
    asked: List[str] = []
    answer: Optional[AnswerOut] = None
    citations: List[CitationOut] = []
    usage: Optional[UsageOut] = None
    error: Optional[ErrorBody] = None
    created_at: str
    updated_at: str


# ---------------------------------------------------------------- mapping functions

def project_out(project: Project) -> ProjectOut:
    return ProjectOut(project_id=project.project_id, display_name=project.display_name,
                      versions=list(project.versions), snapshot_ids=list(project.snapshot_ids),
                      knowledge_updated_at=max(d.fetched_at for d in project.documents))


def episode_out(ep: Episode) -> EpisodeOut:
    return EpisodeOut(episode_id=ep.episode_id, question=ep.question, confirmed=ep.confirmed.as_dict(),
                      clarification_rounds=ep.clarification_rounds, version_history=ep.version_history)


def session_out(session: Session) -> SessionOut:
    messages = [MessageOut(**m.__dict__) for ep in session.episodes for m in ep.messages]
    return SessionOut(session_id=session.session_id, retained_version=session.retained_version,
                      current_episode=episode_out(session.current), episodes=len(session.episodes),
                      messages=messages)


def citation_out(c: Citation) -> CitationOut:
    return CitationOut(**{**c.__dict__, "heading_path": list(c.heading_path)})


def answer_out(a: Answer) -> AnswerOut:
    d = a.draft
    return AnswerOut(
        summary=d.summary, findings=[FindingOut(**f.model_dump()) for f in d.findings],
        historical_cases=[FindingOut(**f.model_dump()) for f in a.historical_cases], hypotheses=d.hypotheses,
        steps=d.steps, missing_information=d.missing_information, unsupported_claims=a.unsupported_claims,
        conflicts=a.conflicts, notes=a.notes,
    )


def run_out(record: RunRecord) -> RunOut:
    result = record.result
    answer = result.answer if result else None
    state = result.state if result else {}
    error = ErrorBody(code=record.error_code, message=record.error or "", run_id=record.run_id) \
        if record.error_code else None
    usage = None
    if answer:
        usage = UsageOut(prompt_tokens=answer.usage.prompt_tokens, completion_tokens=answer.usage.completion_tokens,
                         source=answer.usage.source, retrieval_rounds=state.get("retrieval_rounds"),
                         model_calls=state.get("model_calls"))
    return RunOut(
        run_id=record.run_id, session_id=record.session_id, episode_id=record.episode_id, status=record.status,
        stage=record.stage, outcome=answer.outcome if answer else None,
        target_version=answer.target_version if answer else None, asked=result.asked if result else [],
        answer=answer_out(answer) if answer else None,
        citations=[citation_out(c) for c in answer.citations] if answer else [],
        usage=usage, error=error, created_at=record.created_at, updated_at=record.updated_at,
    )


def citation_detail_out(answer: Answer, citation: Citation) -> CitationDetailOut:
    hit = answer.evidence[citation.id]
    return CitationDetailOut(**citation_out(citation).model_dump(), content=hit.chunk.content,
                             document_fetched_at=hit.document.fetched_at,
                             version_evidence=hit.document.version_evidence)
