"""U10 tests: HTTP contract, 202 + polling, one active run per session, error mapping.

A gate-controlled model blocks inside the workflow so we can observe a run that
is genuinely in progress and check that polling still answers immediately.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Optional

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from tracepilot.app import ApiContext
from tracepilot.controllers.api import create_app
from tracepilot.models.chunking import MarkdownChunker
from tracepilot.models.conversation import ConversationService
from tracepilot.models.llm import ModelAuthError
from tracepilot.models.mock_chat import ExtractiveMockChatModel
from tracepilot.models.rag import RagPipeline
from tracepilot.models.retrieval import VectorRetriever
from tracepilot.models.runs import RunExecutor, RunRegistry
from tracepilot.models.sessions import InMemorySessionStore
from tracepilot.models.sources import Project
from tracepilot.models.workflow import SupportWorkflow
from tests.test_u05 import COVERED, V1, V2

API = "/api/v1"


class GatedMock(ExtractiveMockChatModel):
    """Waits for `gate` before answering; raises `error` if set."""

    gate: Optional[threading.Event] = None
    error: Optional[Exception] = None

    model_config = {"arbitrary_types_allowed": True}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.gate.wait(5)
        if self.error:
            raise self.error
        return super()._generate(messages, stop, run_manager, **kwargs)


def make_client(gate: threading.Event, error: Exception | None = None) -> TestClient:
    def factory() -> ApiContext:
        r = VectorRetriever(DeterministicFakeEmbedding(size=16))
        r.index([V1, V2], [c for d in (V1, V2) for c in MarkdownChunker().split(d)])
        model = GatedMock(gate=gate, error=error)
        service = ConversationService(InMemorySessionStore(),
                                      SupportWorkflow(RagPipeline(r, model, covered_versions=COVERED, k=4)), COVERED)
        registry = RunRegistry()
        project = Project("pydantic", "Pydantic", COVERED, (V1, V2), ("s1",))
        return ApiContext(project, service, registry, RunExecutor(service, registry))

    return TestClient(create_app(factory))


def wait_for(client, run_id, statuses=("completed", "failed", "awaiting_input"), timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"{API}/runs/{run_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish: {body}")


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.gate = threading.Event()
        self.ctx = make_client(self.gate)
        self.client = self.ctx.__enter__()  # runs the lifespan (startup)

    def tearDown(self):
        self.gate.set()
        self.ctx.__exit__(None, None, None)

    def new_session(self, **body):
        data = self.client.post(f"{API}/sessions", json=body).json()
        return data["session_id"], data["current_episode"]["episode_id"]

    def submit(self, sid, eid, message, **extra):
        return self.client.post(f"{API}/sessions/{sid}/runs", json={"episode_id": eid, "message": message, **extra})

    def test_health_and_project(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get(f"{API}/projects/current").json()["versions"], list(COVERED))

    def test_accepted_then_polled_without_blocking(self):
        sid, eid = self.new_session(version="2.14.1")
        accepted = self.submit(sid, eid, "make models immutable")
        self.assertEqual(accepted.status_code, 202)
        run_id = accepted.json()["run_id"]

        start = time.monotonic()
        running = self.client.get(f"{API}/runs/{run_id}").json()  # the model is blocked on the gate
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertIn(running["status"], ("queued", "running"))

        self.gate.set()
        done = wait_for(self.client, run_id)
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["outcome"], "answered")
        cid = done["citations"][0]["id"]
        detail = self.client.get(f"{API}/runs/{run_id}/citations/{cid}").json()
        self.assertIn("frozen", detail["content"])
        self.assertEqual(self.client.get(f"{API}/runs/{run_id}/citations/c99").status_code, 404)

    def test_one_active_run_per_session(self):
        sid, eid = self.new_session(version="2.14.1")
        first = self.submit(sid, eid, "make models immutable").json()["run_id"]
        second = self.submit(sid, eid, "again")
        self.assertEqual(second.status_code, 409)
        self.assertEqual(second.json()["error"]["code"], "RUN_IN_PROGRESS")
        self.assertEqual(second.json()["error"]["run_id"], first)
        other_sid, other_eid = self.new_session(version="1.10")  # other sessions are not blocked
        self.assertEqual(self.submit(other_sid, other_eid, "immutable").status_code, 202)

    def test_clarification_over_http(self):
        self.gate.set()
        sid, eid = self.new_session()
        run = wait_for(self.client, self.submit(sid, eid, "it fails").json()["run_id"])
        self.assertEqual((run["status"], run["outcome"]), ("awaiting_input", "insufficient_context"))
        self.assertEqual(run["asked"], ["version", "error_text"])
        session = self.client.get(f"{API}/sessions/{sid}").json()
        self.assertEqual(session["current_episode"]["clarification_rounds"], 1)
        self.assertEqual([m["role"] for m in session["messages"]], ["user", "assistant"])

    def test_new_episode_and_stale_episode(self):
        self.gate.set()
        sid, eid = self.new_session(version="2.14.1")
        new_eid = self.client.post(f"{API}/sessions/{sid}/episodes").json()["episode_id"]
        stale = self.submit(sid, eid, "old problem")
        self.assertEqual((stale.status_code, stale.json()["error"]["code"]), (409, "EPISODE_MISMATCH"))
        self.assertEqual(self.submit(sid, new_eid, "immutable").status_code, 202)

    def test_input_errors(self):
        sid, eid = self.new_session()
        cases = [
            self.submit(sid, eid, ""),
            self.submit(sid, eid, "   "),
            self.submit(sid, eid, "x" * 8001),
            self.submit(sid, eid, "q", version="abc"),
            self.submit(sid, eid, "q", error_log="x" * 20001),
        ]
        for response in cases:
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["error"]["code"], "INVALID_INPUT")
        self.assertEqual(self.client.get(f"{API}/sessions/nope").status_code, 404)
        self.assertEqual(self.client.get(f"{API}/runs/nope").status_code, 404)


class FailedRunTest(unittest.TestCase):
    def test_model_failure_is_a_failed_run_not_an_empty_answer(self):
        gate = threading.Event()
        gate.set()
        with make_client(gate, ModelAuthError("bad key")) as client:
            sid = client.post(f"{API}/sessions", json={"version": "2.14.1"}).json()
            run_id = client.post(f"{API}/sessions/{sid['session_id']}/runs", json={
                "episode_id": sid["current_episode"]["episode_id"], "message": "immutable"}).json()["run_id"]
            run = wait_for(client, run_id)
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["error"]["code"], "MODEL_AUTH_FAILED")
        self.assertIsNone(run["outcome"])


if __name__ == "__main__":
    unittest.main()
