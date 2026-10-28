"""U05 tests: version input rules, filter-before-rank, version gate, real two-version project."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from langchain_core.embeddings import DeterministicFakeEmbedding

from tracepilot.models.chunking import MarkdownChunker
from tracepilot.models.mock_chat import ExtractiveMockChatModel
from tracepilot.models.rag import RagPipeline
from tracepilot.models.retrieval import VectorRetriever
from tracepilot.models.sources import SourceError, load_project
from tracepilot.models.versions import InvalidVersionError, parse_version
from tests.test_u02 import make_doc

COVERED = ("1.10.26", "2.14.1")


def versioned_doc(text, doc_id, version):
    doc = make_doc(text, doc_id)
    return doc.__class__(**{**doc.__dict__, "version_scope": f"exact:{version}"})


V1 = versioned_doc("## Immutability\n\nUse Config.allow_mutation = False to make models immutable.\n", "v1", "1.10.26")
V2 = versioned_doc("## Immutability\n\nUse model_config frozen=True to make models immutable.\n", "v2", "2.14.1")


def retriever():
    r = VectorRetriever(DeterministicFakeEmbedding(size=16))
    r.index([V1, V2], [c for d in (V1, V2) for c in MarkdownChunker().split(d)])
    return r


class ParseVersionTest(unittest.TestCase):
    def test_exact_and_major_minor(self):
        self.assertEqual(parse_version("2.14.1", COVERED).version, "2.14.1")
        self.assertEqual(parse_version("v1.10", COVERED).version, "1.10.26")

    def test_unknown_and_latest_never_map_to_a_version(self):
        for raw in ("unknown", "", "latest", "最新版", "2"):
            choice = parse_version(raw, COVERED)
            self.assertEqual(choice.kind, "unknown", raw)
            self.assertIsNone(choice.version)

    def test_well_formed_but_uncovered(self):
        for raw in ("2.5", "3.0.0", "2.14.0"):
            self.assertEqual(parse_version(raw, COVERED).kind, "unsupported", raw)

    def test_malformed(self):
        for raw in ("abc", "2.x", "1.2.3.4"):
            with self.assertRaises(InvalidVersionError):
                parse_version(raw, COVERED)


class FilterTest(unittest.TestCase):
    def test_filter_happens_before_ranking(self):  # PRD AT-06
        r = retriever()
        for version, doc_id in (("1.10.26", "v1"), ("2.14.1", "v2")):
            hits = r.search("immutable", k=10, version=version)
            self.assertTrue(hits)
            self.assertEqual({h.document.document_id for h in hits}, {doc_id})

    def test_no_filter_mixes_versions(self):
        self.assertEqual({h.document.document_id for h in retriever().search("immutable", k=10)}, {"v1", "v2"})


class VersionGateTest(unittest.TestCase):
    def pipeline(self, model):
        return RagPipeline(retriever(), model, covered_versions=COVERED, k=3)

    def test_unsupported_and_unknown_skip_retrieval_and_model(self):
        model = mock.Mock()
        p = self.pipeline(model)
        with mock.patch.object(p.retriever, "search") as search:
            self.assertEqual(p.answer("immutable?", "2.5").outcome, "unsupported_version")
            self.assertEqual(p.answer("immutable?", "unknown").outcome, "insufficient_context")
            search.assert_not_called()
        model.invoke.assert_not_called()

    def test_same_question_different_versions_cite_different_sources(self):
        p = self.pipeline(ExtractiveMockChatModel())
        a1, a2 = p.answer("make models immutable", "1.10"), p.answer("make models immutable", "2.14.1")
        self.assertEqual({c.document_id for c in a1.citations}, {"v1"})
        self.assertEqual({c.document_id for c in a2.citations}, {"v2"})
        self.assertEqual((a1.target_version, a2.target_version), ("1.10.26", "2.14.1"))


class RealProjectTest(unittest.TestCase):
    def test_loads_two_versions_with_tag_evidence(self):
        project = load_project(Path("data/sources/pydantic/project.json"))
        self.assertEqual(project.versions, COVERED)
        docs = [d for d in project.documents if d.source_type != "issue"]
        self.assertEqual({d.version_scope for d in docs}, {"exact:1.10.26", "exact:2.14.1"})
        self.assertTrue(all(d.version_evidence.startswith("git_tag v") for d in docs))

    def test_missing_project_file(self):
        with self.assertRaises(SourceError):
            load_project(Path("nope/project.json"))


if __name__ == "__main__":
    unittest.main()
