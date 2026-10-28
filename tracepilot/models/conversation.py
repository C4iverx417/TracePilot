"""Multi-turn diagnosis (U09): decide whether to ask, what to ask, and when to stop asking.

Per user message:
  1. Record it in the current episode (first message = the problem, later ones = details).
  2. Update confirmed facts. Precedence: explicit selection > what the user wrote > what we had.
  3. Find missing critical facts (version; error text when the problem is vague).
  4. If something is missing and fewer than 2 clarification rounds were used, ask
     (<= 3 fields) and stop: the run ends "awaiting_input", no model call is made.
  5. Otherwise run the workflow on a compact query built from the episode, never
     the full transcript. A version change simply means a fresh, filtered retrieval;
     nothing retrieved for the old version is reused.

Clarification rounds belong to the EPISODE, so they add up across runs. Only an
explicit "new problem" (new episode) or a new session resets them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .answering import Answer
from .rag import no_call_answer
from .sessions import ConfirmedContext, Episode, Message, Session, SessionStore, _id
from .versions import parse_version
from .workflow import SupportWorkflow

MAX_CLARIFICATION_ROUNDS = 2  # PRD FR-03
MAX_FIELDS_PER_ASK = 3
QUERY_CHAR_BUDGET = 2000  # what the workflow receives, however long the conversation gets

_STATED_VERSION = re.compile(
    r"(?:pydantic|version|版本|用的是|using|on|使用)\s*(?:==|=|:|是|为)?\s*v?(\d+\.\d+(?:\.\d+)?)", re.I)
_ERROR_LINE = re.compile(r"^[^\n]*\b[A-Z][A-Za-z]*(?:Error|Exception)\b[^\n]*$", re.M)
_VAGUE = re.compile(r"doesn'?t work|not working|\bfail(s|ed)?\b|broken|crash|启动失败|报错|不行|无法|出错|坏了|有问题|失败",
                    re.I)

QUESTIONS = {
    "version": "Which Pydantic version are you using? (run: python -c \"import pydantic; print(pydantic.VERSION)\")",
    "error_text": "Please paste the full error message or traceback.",
}


def extract_conditions(text: str) -> Dict[str, Optional[str]]:
    """Only explicit statements count: a version inside a URL or traceback is not 'the user's version'."""
    version = _STATED_VERSION.search(text)
    error = _ERROR_LINE.search(text)
    return {"version": version.group(1) if version else None, "error_text": error.group(0).strip() if error else None}


def is_vague(text: str) -> bool:
    """Short, generic failure statements with no error text ("it fails to start")."""
    short = len(text.split()) <= 8 if text.isascii() else len(text) <= 20
    return short and bool(_VAGUE.search(text))


def compose_query(episode: Episode) -> str:
    """Question + confirmed error text + the latest details, with version statements removed
    (the version is a filter, not search words). Capped at QUERY_CHAR_BUDGET."""
    parts = [episode.question]
    if episode.confirmed.error_text and episode.confirmed.error_text not in episode.question:
        parts.append(episode.confirmed.error_text)
    for detail in episode.details[-2:]:
        cleaned = _STATED_VERSION.sub("", detail).strip(" ,.;:，。")
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    return "\n".join(parts)[:QUERY_CHAR_BUDGET]


@dataclass
class TurnResult:
    session_id: str
    episode_id: str
    run_id: str
    status: str  # completed | awaiting_input | failed
    answer: Optional[Answer] = None
    asked: List[str] = field(default_factory=list)
    error_code: Optional[str] = None
    error: Optional[str] = None
    state: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class ConversationService:
    def __init__(self, store: SessionStore, workflow: SupportWorkflow, covered_versions: Sequence[str],
                 project_id: str = "pydantic"):
        self.store = store
        self.workflow = workflow
        self.covered_versions = tuple(covered_versions)
        self.project_id = project_id

    # ------------------------------------------------------------ sessions and episodes

    def resolve_version(self, raw: str):
        return parse_version(raw, self.covered_versions)

    def new_session(self, version: Optional[str] = None) -> Session:
        session = self.store.create(self.project_id)
        if version:
            parse_version(version, self.covered_versions)  # reject malformed input early
            session.retained_version = version
        session.current  # noqa: B018 - opens the first episode
        self.store.save(session)
        return session

    def new_episode(self, session_id: str) -> Episode:
        """'Start a new problem': clears the fault summary and clarification count, keeps the chosen version."""
        session = self.store.get(session_id)
        episode = Episode(confirmed=ConfirmedContext(version=session.retained_version))
        session.episodes.append(episode)
        self.store.save(session)
        return episode

    # ------------------------------------------------------------ one user message

    def submit(self, session_id: str, text: str, version: Optional[str] = None,
               run_id: Optional[str] = None, on_progress: Optional[Callable[[str], None]] = None) -> TurnResult:
        session = self.store.get(session_id)
        episode = session.current
        run_id = run_id or _id("run")
        notes: List[str] = []

        if not episode.question:
            episode.question = text
        else:
            episode.details.append(text)
        episode.messages.append(Message("user", text, run_id=run_id))

        stated = extract_conditions(text)
        new_version = version or stated["version"]  # explicit selection beats what was written
        if version:
            parse_version(version, self.covered_versions)  # malformed -> InvalidVersionError to the caller
            session.retained_version = version
        if new_version:
            if episode.confirmed.version and new_version != episode.confirmed.version:
                notes.append(f"Version changed from {episode.confirmed.version} to {new_version}; "
                             "evidence is retrieved again for the new version.")
            episode.confirmed.version = new_version
            episode.version_history.append(new_version)
        if stated["error_text"]:
            episode.confirmed.error_text = stated["error_text"]

        missing = self._missing(episode)
        if missing and episode.clarification_rounds < MAX_CLARIFICATION_ROUNDS:
            episode.clarification_rounds += 1
            asked = missing[:MAX_FIELDS_PER_ASK]
            answer = no_call_answer(episode.question, "insufficient_context",
                                    "I need a bit more information before I can narrow this down.",
                                    [QUESTIONS[f] for f in asked], episode.confirmed.version)
            return self._close(session, episode, TurnResult(session.session_id, episode.episode_id, run_id,
                                                            "awaiting_input", answer, asked, notes=notes))

        if "version" in missing:  # still unknown after two rounds: stop, list the gap
            answer = no_call_answer(episode.question, "insufficient_context",
                                    "Without your Pydantic version I cannot pick the right documentation.",
                                    [QUESTIONS["version"]], None)
            notes.append("Clarification limit reached; add the version or start a new problem.")
            return self._close(session, episode, TurnResult(session.session_id, episode.episode_id, run_id,
                                                            "completed", answer, notes=notes))
        if "error_text" in missing:
            notes.append("Answering without the exact error text; results may be broad.")

        state = self.workflow.run(compose_query(episode), episode.confirmed.version or "unknown", on_progress)
        if state.get("error_code"):
            result = TurnResult(session.session_id, episode.episode_id, run_id, "failed",
                                error_code=state["error_code"], error=state["error"], state=state, notes=notes)
        else:
            result = TurnResult(session.session_id, episode.episode_id, run_id, "completed", state["answer"],
                                state=state, notes=notes)
        return self._close(session, episode, result)

    # ------------------------------------------------------------ helpers

    def _missing(self, episode: Episode) -> List[str]:
        missing = []
        if parse_version(episode.confirmed.version or "unknown", self.covered_versions).kind == "unknown":
            missing.append("version")
        if not episode.confirmed.error_text and is_vague(" ".join([episode.question] + episode.details)):
            missing.append("error_text")
        return missing

    def _close(self, session: Session, episode: Episode, result: TurnResult) -> TurnResult:
        episode.run_ids.append(result.run_id)
        if result.answer:
            if result.notes:
                result.answer.notes.extend(result.notes)
            episode.messages.append(Message("assistant", result.answer.draft.summary, result.run_id,
                                            result.answer.outcome))
        else:
            episode.messages.append(Message("assistant", f"[{result.error_code}] {result.error}", result.run_id,
                                            None))
        self.store.save(session)
        return result
