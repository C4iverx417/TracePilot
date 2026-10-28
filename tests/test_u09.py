"""U09 tests: clarification rules, episode/session state, version changes, isolation."""

from __future__ import annotations

import unittest

from langchain_core.embeddings import DeterministicFakeEmbedding

from tracepilot.models.chunking import MarkdownChunker
from tracepilot.models.conversation import ConversationService, compose_query, extract_conditions, is_vague
from tracepilot.models.mock_chat import ExtractiveMockChatModel
from tracepilot.models.rag import RagPipeline
from tracepilot.models.retrieval import VectorRetriever
from tracepilot.models.sessions import Episode, InMemorySessionStore
from tracepilot.models.workflow import SupportWorkflow
from tests.test_u05 import V1, V2

COVERED = ("1.10.26", "2.14.1")


def service() -> ConversationService:
    r = VectorRetriever(DeterministicFakeEmbedding(size=16))
    r.index([V1, V2], [c for d in (V1, V2) for c in MarkdownChunker().split(d)])
    pipeline = RagPipeline(r, ExtractiveMockChatModel(), covered_versions=COVERED, k=4)
    return ConversationService(InMemorySessionStore(), SupportWorkflow(pipeline), COVERED)


class ExtractionTest(unittest.TestCase):
    def test_only_explicit_version_statements_count(self):
        self.assertEqual(extract_conditions("我用的是 1.10")["version"], "1.10")
        self.assertEqual(extract_conditions("pydantic==2.14.1 here")["version"], "2.14.1")
        self.assertIsNone(extract_conditions("see https://errors.pydantic.dev/2.0.2/u/x")["version"])

    def test_error_line(self):
        text = "after upgrade:\nPydanticImportError: BaseSettings has been moved\nplease help"
        self.assertEqual(extract_conditions(text)["error_text"], "PydanticImportError: BaseSettings has been moved")

    def test_vagueness(self):
        self.assertTrue(is_vague("程序启动失败"))
        self.assertTrue(is_vague("it doesn't work"))
        self.assertFalse(is_vague("How do I make a model immutable with frozen config?"))

    def test_compose_query_drops_version_words_and_is_bounded(self):
        ep = Episode(question="make models immutable", details=["我用的是 2.14.1", "x" * 5000])
        query = compose_query(ep)
        self.assertNotIn("2.14.1", query)
        self.assertLessEqual(len(query), 2000)


class ClarificationTest(unittest.TestCase):  # PRD AT-04, AT-05
    def test_vague_problem_asks_and_then_answers(self):
        svc = service()
        s = svc.new_session()
        first = svc.submit(s.session_id, "程序启动失败")
        self.assertEqual(first.status, "awaiting_input")
        self.assertEqual(first.asked, ["version", "error_text"])
        second = svc.submit(s.session_id, "我用的是 2.14.1")
        self.assertEqual(second.asked, ["error_text"])  # the version is not asked again
        third = svc.submit(s.session_id, "ValueError: immutable model frozen")
        self.assertEqual(third.status, "completed")
        self.assertEqual(svc.store.get(s.session_id).current.clarification_rounds, 2)

    def test_two_rounds_then_stop_with_gaps(self):
        svc = service()
        s = svc.new_session()
        self.assertEqual(svc.submit(s.session_id, "it fails").status, "awaiting_input")
        self.assertEqual(svc.submit(s.session_id, "still fails").status, "awaiting_input")
        third = svc.submit(s.session_id, "please help")
        self.assertEqual(third.status, "completed")
        self.assertEqual(third.answer.outcome, "insufficient_context")
        self.assertEqual(third.asked, [])  # no third round of questions

    def test_new_episode_resets_count_but_keeps_selected_version(self):
        svc = service()
        s = svc.new_session(version="2.14.1")
        svc.submit(s.session_id, "it fails")  # version known, only error text asked
        episode = svc.new_episode(s.session_id)
        self.assertEqual(episode.clarification_rounds, 0)
        self.assertEqual(episode.confirmed.version, "2.14.1")
        self.assertIsNone(episode.confirmed.error_text)


class VersionChangeTest(unittest.TestCase):  # PRD AT-12
    def test_switching_version_retrieves_again_for_the_new_version(self):
        svc = service()
        s = svc.new_session()
        a = svc.submit(s.session_id, "make models immutable", version="2.14.1")
        b = svc.submit(s.session_id, "actually I use the old one", version="1.10")
        self.assertEqual({h.document.document_id for h in a.answer.evidence.values()}, {"v2"})
        self.assertEqual({h.document.document_id for h in b.answer.evidence.values()}, {"v1"})
        self.assertTrue(any("Version changed from 2.14.1 to 1.10" in n for n in b.notes))

    def test_explicit_selection_beats_written_version(self):
        svc = service()
        s = svc.new_session()
        result = svc.submit(s.session_id, "I'm on version 1.10, make models immutable", version="2.14.1")
        self.assertEqual(result.answer.target_version, "2.14.1")


class IsolationTest(unittest.TestCase):  # PRD AT-13
    def test_two_sessions_do_not_share_state(self):
        svc = service()
        a, b = svc.new_session(), svc.new_session()
        svc.submit(a.session_id, "make models immutable", version="1.10")
        svc.submit(b.session_id, "it fails")
        sa, sb = svc.store.get(a.session_id), svc.store.get(b.session_id)
        self.assertEqual(sa.current.confirmed.version, "1.10")
        self.assertIsNone(sb.current.confirmed.version)
        self.assertEqual(sb.current.clarification_rounds, 1)
        self.assertEqual(sa.current.clarification_rounds, 0)
        self.assertNotEqual(sa.current.episode_id, sb.current.episode_id)


if __name__ == "__main__":
    unittest.main()
