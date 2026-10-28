"""U08 tests: the LangGraph workflow's routing, budgets, repair and retries.

A scripted chat model returns (or raises) a fixed sequence, so every branch is
deterministic. A fake clock makes the deadline testable without waiting.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tracepilot.models.chat import ModelRateLimitedError
from tracepilot.models.chunking import MarkdownChunker
from tracepilot.models.keyword import KeywordRetriever
from tracepilot.models.llm import ModelAuthError
from tracepilot.models.rag import RagPipeline
from tracepilot.models.workflow import Budget, SupportWorkflow
from tests.test_u02 import make_doc

NOISE = make_doc("## Install\n\nRun pip install to get started quickly.\n", "noise")
ANSWER = make_doc("## Frozen\n\nSet frozen=True so instances cannot be changed.\n", "answer")


class Scripted(BaseChatModel):
    """Returns the next item; an Exception item is raised instead."""

    script: List[Any]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=item))])


def assess(sufficient: bool, rewrite: str = "", relevant: Optional[List[str]] = None) -> str:
    return json.dumps({"sufficient": sufficient, "relevant_ids": relevant or [], "rewritten_query": rewrite})


def answer(cid: str = "c1") -> str:
    return json.dumps({"outcome": "answered", "summary": "s", "findings": [{"text": "t", "citation_ids": [cid]}]})


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def workflow(script, budget=Budget(), clock=None, k=2):
    retriever = KeywordRetriever()
    retriever.index([NOISE, ANSWER], [c for d in (NOISE, ANSWER) for c in MarkdownChunker().split(d)])
    model = Scripted(script=script)
    pipeline = RagPipeline(retriever, model, covered_versions=("1.0",), k=k)
    sleeps = []
    wf = SupportWorkflow(pipeline, budget, clock=clock or Clock(), sleep=sleeps.append)
    return wf, model, sleeps


def nodes(state):
    return [e["node"] for e in state["trace"]]


class RoutingTest(unittest.TestCase):
    def test_rewrite_finds_evidence_in_second_round(self):  # PRD AT-08
        wf, model, _ = workflow([assess(False, rewrite="frozen instances"), assess(True, relevant=["c1"]), answer()])
        state = wf.run("pip install", "1.0")  # first query only matches the install page
        self.assertEqual(state["retrieval_rounds"], 2)
        self.assertEqual(state["query_variants"], ["pip install", "frozen instances"])
        self.assertEqual(state["answer"].outcome, "answered")
        self.assertEqual(state["answer"].evidence["c1"].document.document_id, "answer")  # assessed-relevant first
        self.assertEqual(nodes(state), ["check_input", "retrieve", "assess", "retrieve", "assess", "generate", "finish"])
        self.assertEqual(state["model_calls"], 3)

    def test_retrieval_budget_exhausted_is_bounded(self):  # PRD AT-09
        script = [assess(False, rewrite="a pip"), assess(False, rewrite="b pip"), assess(False, rewrite="c pip"),
                  answer()]
        wf, model, _ = workflow(script)
        state = wf.run("pip install", "1.0")
        self.assertEqual(state["retrieval_rounds"], 3)  # never a 4th round
        self.assertEqual(state["end_reason"], "retrieval_budget_exhausted")
        self.assertEqual(state["answer"].outcome, "partially_answered")  # never confirmed sufficient
        self.assertEqual(model.calls, 4)

    def test_same_rewrite_stops_early(self):
        wf, _, _ = workflow([assess(False, rewrite="pip install"), answer()])
        self.assertEqual(wf.run("pip install", "1.0")["retrieval_rounds"], 1)

    def test_version_gate_skips_everything(self):
        wf, model, _ = workflow([])
        state = wf.run("q", "9.9")
        self.assertEqual(state["answer"].outcome, "unsupported_version")
        self.assertEqual(model.calls, 0)
        self.assertEqual(nodes(state), ["check_input", "finish"])


class FailureTest(unittest.TestCase):  # PRD AT-15
    def test_one_format_repair_then_success(self):
        wf, model, _ = workflow(["not json", assess(True, relevant=["c1"]), answer()])
        state = wf.run("frozen", "1.0")
        self.assertEqual(state["answer"].outcome, "answered")
        self.assertEqual(state["model_calls"], 3)  # the repair counts as a call

    def test_second_format_error_fails_cleanly(self):
        wf, _, _ = workflow(["not json", "still not json"])
        state = wf.run("frozen", "1.0")
        self.assertEqual(state["error_code"], "INVALID_MODEL_OUTPUT")
        self.assertIsNone(state["answer"])

    def test_transient_errors_retry_with_backoff(self):
        rate = ModelRateLimitedError("429")
        wf, model, sleeps = workflow([rate, rate, assess(True, relevant=["c1"]), answer()])
        state = wf.run("frozen", "1.0")
        self.assertEqual(state["answer"].outcome, "answered")
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(state["model_calls"], 4)

    def test_retries_are_limited(self):
        rate = ModelRateLimitedError("429")
        wf, model, _ = workflow([rate, rate, rate])
        state = wf.run("frozen", "1.0")
        self.assertEqual(state["error_code"], "MODEL_UNAVAILABLE")
        self.assertEqual(model.calls, 3)  # 1 try + 2 retries

    def test_auth_error_is_not_retried(self):
        wf, model, sleeps = workflow([ModelAuthError("bad key")])
        state = wf.run("frozen", "1.0")
        self.assertEqual(state["error_code"], "MODEL_AUTH_FAILED")
        self.assertEqual((model.calls, sleeps), (1, []))

    def test_model_call_budget(self):
        wf, model, _ = workflow([assess(False, rewrite="a pip"), assess(False, rewrite="b pip")],
                                budget=Budget(max_model_calls=2))
        state = wf.run("pip install", "1.0")
        self.assertEqual(state["error_code"], "BUDGET_EXCEEDED")
        self.assertEqual(model.calls, 2)

    def test_deadline(self):
        clock = Clock()
        wf, model, _ = workflow([assess(False, rewrite="frozen")], budget=Budget(deadline_s=10), clock=clock)
        original = wf.pipeline.retriever.search

        def slow_search(*args, **kwargs):
            clock.now += 6  # each retrieval "takes" 6 seconds
            return original(*args, **kwargs)

        wf.pipeline.retriever.search = slow_search
        state = wf.run("pip install", "1.0")
        self.assertEqual(state["error_code"], "DEADLINE_EXCEEDED")
        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
