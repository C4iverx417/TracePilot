"""HTTP controller (U10): FastAPI routes. Each route validates input, calls the model
layer, and returns a view schema. No retrieval or prompt logic lives here.

Local use only by default (PRD NFR-06): session ids are not authentication.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..models.runs import EpisodeMismatch, RunInProgress
from ..models.sessions import SessionNotFound
from ..models.versions import InvalidVersionError
from ..views import api_schemas as v

API = "/api/v1"


def _error(status: int, code: str, message: str, run_id: str | None = None) -> JSONResponse:
    body = v.ErrorOut(error=v.ErrorBody(code=code, message=message, run_id=run_id))
    return JSONResponse(status_code=status, content=body.model_dump())


def create_app(context_factory: Callable[[], "ApiContext"]) -> FastAPI:  # noqa: F821 - see app.ApiContext
    """`context_factory` builds the project, index and services once at startup (slow: embeddings)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ctx = context_factory()
        yield
        app.state.ctx.executor.shutdown()

    app = FastAPI(title="TracePilot", version="0.1.0", lifespan=lifespan,
                  description="Version-aware, citation-backed support copilot for Pydantic.")

    # ------------------------------------------------------------ error mapping

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"] if p != "body")
        return _error(422, "INVALID_INPUT", f"{where}: {first['msg']}")

    @app.exception_handler(InvalidVersionError)
    async def invalid_version(_: Request, exc: InvalidVersionError):
        return _error(422, "INVALID_INPUT", str(exc))

    @app.exception_handler(SessionNotFound)
    async def missing_session(_: Request, exc: SessionNotFound):
        return _error(404, "NOT_FOUND", f"Session {exc.args[0]} not found")

    @app.exception_handler(RunInProgress)
    async def run_in_progress(_: Request, exc: RunInProgress):
        return _error(409, "RUN_IN_PROGRESS", str(exc), exc.run_id)

    @app.exception_handler(EpisodeMismatch)
    async def episode_mismatch(_: Request, exc: EpisodeMismatch):
        return _error(409, "EPISODE_MISMATCH", str(exc))

    # ------------------------------------------------------------ routes

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get(f"{API}/projects/current", response_model=v.ProjectOut)
    def current_project(request: Request):
        return v.project_out(request.app.state.ctx.project)

    @app.post(f"{API}/sessions", status_code=201, response_model=v.SessionOut)
    def create_session(request: Request, body: v.SessionCreate | None = None):
        session = request.app.state.ctx.service.new_session((body.version if body else None))
        return v.session_out(session)

    @app.get(f"{API}/sessions/{{session_id}}", response_model=v.SessionOut)
    def get_session(request: Request, session_id: str):
        return v.session_out(request.app.state.ctx.service.store.get(session_id))

    @app.post(f"{API}/sessions/{{session_id}}/episodes", status_code=201, response_model=v.EpisodeOut)
    def new_episode(request: Request, session_id: str):
        return v.episode_out(request.app.state.ctx.service.new_episode(session_id))

    @app.post(f"{API}/sessions/{{session_id}}/runs", status_code=202, response_model=v.RunAccepted)
    def create_run(request: Request, session_id: str, body: v.RunCreate):
        ctx = request.app.state.ctx
        if body.version:
            ctx.service.resolve_version(body.version)  # malformed -> 422 before anything is queued
        text = body.message.strip()
        if not text:
            return _error(422, "INVALID_INPUT", "message: must not be blank")
        if body.error_log and body.error_log.strip():
            text = f"{text}\n{body.error_log.strip()}"
        record = ctx.executor.submit(session_id, body.episode_id, text, body.version)
        return v.RunAccepted(run_id=record.run_id, status=record.status, status_url=f"{API}/runs/{record.run_id}")

    @app.get(f"{API}/runs/{{run_id}}", response_model=v.RunOut)
    def get_run(request: Request, run_id: str):
        try:
            return v.run_out(request.app.state.ctx.registry.get(run_id))
        except KeyError:
            return _error(404, "NOT_FOUND", f"Run {run_id} not found")

    @app.get(f"{API}/runs/{{run_id}}/citations/{{citation_id}}", response_model=v.CitationDetailOut)
    def get_citation(request: Request, run_id: str, citation_id: str):
        try:
            record = request.app.state.ctx.registry.get(run_id)
        except KeyError:
            return _error(404, "NOT_FOUND", f"Run {run_id} not found")
        answer = record.result.answer if record.result else None
        cited = {c.id: c for c in answer.citations} if answer else {}
        if citation_id not in cited:  # only ids validated for THIS run are real citations (PRD FR-07)
            return _error(404, "NOT_FOUND", f"Citation {citation_id} is not part of run {run_id}")
        return v.citation_detail_out(answer, cited[citation_id])

    return app
