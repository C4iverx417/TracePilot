"""Background execution of runs for the API (U10).

POST returns at once with a run_id; the workflow runs on a small thread pool and
writes progress into the RunRegistry. Polling only reads the registry under a
lock, so it is never blocked by a slow model call (PRD NFR-02).

Rules enforced here:
- one active (queued/running) run per session -> RunInProgress (HTTP 409)
- the run must target the session's CURRENT episode -> EpisodeMismatch (HTTP 409)
- at most `max_workers` runs execute at the same time
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from .conversation import ConversationService, TurnResult
from .sessions import _id

# Node name -> what the user is told. Real backend progress, not model reasoning (PRD FR-10).
STAGES = {
    "queued": "waiting to start",
    "check_input": "checking conditions",
    "retrieve": "retrieving documentation",
    "assess": "verifying evidence",
    "generate": "generating answer",
    "finish": "saving result",
}
ACTIVE = ("queued", "running")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunInProgress(Exception):
    def __init__(self, run_id: str):
        super().__init__(f"Session already has an active run: {run_id}")
        self.run_id = run_id


class EpisodeMismatch(Exception):
    pass


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    session_id: str
    episode_id: str
    status: str = "queued"  # queued | running | awaiting_input | completed | failed
    stage: str = "queued"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    result: Optional[TurnResult] = None
    error_code: Optional[str] = None
    error: Optional[str] = None


class RunRegistry:
    def __init__(self):
        self._runs: Dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, episode_id: str) -> RunRecord:
        with self._lock:
            for run in self._runs.values():
                if run.session_id == session_id and run.status in ACTIVE:
                    raise RunInProgress(run.run_id)
            record = RunRecord(_id("run"), session_id, episode_id)
            self._runs[record.run_id] = record
            return record

    def update(self, run_id: str, **changes) -> RunRecord:
        with self._lock:
            record = replace(self._runs[run_id], updated_at=_now(), **changes)
            self._runs[run_id] = record
            return record

    def get(self, run_id: str) -> RunRecord:
        with self._lock:
            return self._runs[run_id]


class RunExecutor:
    def __init__(self, service: ConversationService, registry: RunRegistry, max_workers: int = 3,
                 on_done: Optional[Callable[[RunRecord], None]] = None):
        self.service = service
        self.registry = registry
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="run")
        self.on_done = on_done

    def submit(self, session_id: str, episode_id: str, text: str, version: Optional[str]) -> RunRecord:
        session = self.service.store.get(session_id)  # SessionNotFound -> 404
        if session.current.episode_id != episode_id:
            raise EpisodeMismatch(f"episode {episode_id} is not the current problem ({session.current.episode_id})")
        record = self.registry.create(session_id, episode_id)
        self.pool.submit(self._execute, record.run_id, session_id, text, version)
        return record

    def _execute(self, run_id: str, session_id: str, text: str, version: Optional[str]) -> None:
        self.registry.update(run_id, status="running", stage=STAGES["check_input"])
        try:
            result = self.service.submit(session_id, text, version, run_id=run_id,
                                         on_progress=lambda node: self.registry.update(
                                             run_id, stage=STAGES.get(node, node)))
            record = self.registry.update(run_id, status=result.status, stage="done", result=result,
                                          error_code=result.error_code, error=result.error)
        except Exception as exc:  # noqa: BLE001 - a crashed run must still reach a terminal state
            code = getattr(exc, "code", "INTERNAL_ERROR")
            record = self.registry.update(run_id, status="failed", stage="done", error_code=code,
                                          error="The run failed unexpectedly." if code == "INTERNAL_ERROR"
                                          else str(exc))
        if self.on_done:
            self.on_done(record)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)
