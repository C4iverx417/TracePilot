"""Conversation state that lives in the APPLICATION, not in the model's context (PRD 9.2, FR-09).

    Session  one conversation; may keep a version the user chose to retain
      Episode  one diagnostic problem; owns confirmed facts and the clarification count
        Run    one execution of the workflow (U08) for this episode

The model only ever sees a compact summary built from these records, never the
whole history. U09 keeps them in memory; U13 swaps in a database store behind
the same `SessionStore` interface.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class ConfirmedContext:
    """Facts the user stated or selected. They always beat anything a model might infer."""

    version: Optional[str] = None
    error_text: Optional[str] = None
    environment: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {"version": self.version, "error_text": self.error_text, "environment": self.environment}


@dataclass
class Message:
    role: str  # user | assistant
    content: str
    run_id: Optional[str] = None
    outcome: Optional[str] = None
    created_at: str = field(default_factory=_now)


@dataclass
class Episode:
    episode_id: str = field(default_factory=lambda: _id("ep"))
    question: str = ""  # the first message that opened this problem
    details: List[str] = field(default_factory=list)  # later user supplements
    confirmed: ConfirmedContext = field(default_factory=ConfirmedContext)
    clarification_rounds: int = 0
    version_history: List[str] = field(default_factory=list)
    run_ids: List[str] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    created_at: str = field(default_factory=_now)


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: _id("ses"))
    project_id: str = "pydantic"
    retained_version: Optional[str] = None  # carried into new episodes when the user chose a version
    episodes: List[Episode] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    @property
    def current(self) -> Episode:
        if not self.episodes:
            self.episodes.append(Episode(confirmed=ConfirmedContext(version=self.retained_version)))
        return self.episodes[-1]


class SessionNotFound(KeyError):
    pass


class SessionStore(Protocol):
    def create(self, project_id: str) -> Session: ...

    def get(self, session_id: str) -> Session: ...

    def save(self, session: Session) -> None: ...

    def list_ids(self) -> List[str]: ...


class InMemorySessionStore:
    """Lost when the process exits (stated to the user by the CLI)."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create(self, project_id: str) -> Session:
        session = Session(project_id=project_id)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFound(session_id) from None

    def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    def list_ids(self) -> List[str]:
        return list(self._sessions)
