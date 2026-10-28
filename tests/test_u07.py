"""U07 tests: issue snapshots, version ranges, per-version grouping, facts vs historical cases,
curator-marked conflicts and the per-document context cap."""

from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tracepilot.models.chunking import MarkdownChunker
from tracepilot.models.documents import IssueMeta
from tracepilot.models.keyword import KeywordRetriever
from tracepilot.models.rag import RagPipeline
from tracepilot.models.retrieval import HybridRetriever, VectorRetriever, diversify
from tracepilot.models.sources import load_project
from tracepilot.models.versions import in_range, scope_matches
from tests.test_u05 import V1, V2

ISSUE = dataclasses.replace(
    V2, document_id="issue-1", source_type="issue", title="Issue #1: immutable models broke",
    version_scope="range:>=2.0.0,<3.0.0", text="## Report\n\nAfter upgrading my immutable models broke.\n",
    issue=IssueMeta(1, "answered", "MEMBER", "https://example.invalid/1#c", "Use frozen=True.", "reported on 2.3",
                    conflicts_with=("v2",)),
)
COVERED = ("1.10.26", "2.14.1")


def pipeline(responses, docs=(V1, V2, ISSUE), k=4):
    chunks = [c for d in docs for c in MarkdownChunker().split(d)]
    vector = VectorRetriever(DeterministicFakeEmbedding(size=16))
    vector.index(docs, chunks)
    keyword = KeywordRetriever()
    keyword.index(docs, chunks)
    return RagPipeline(HybridRetriever({"vector": vector, "bm25": keyword}), FakeListChatModel(responses=responses),
                       covered_versions=COVERED, k=k)


def reply(findings, outcome="answered"):
    return json.dumps({"outcome": outcome, "summary": "s", "findings": findings})


def cid_of(prepared, doc_id):
    return next(cid for cid, h in prepared.evidence.items() if h.document.document_id == doc_id)


class VersionRangeTest(unittest.TestCase):
    def test_ranges(self):
        self.assertTrue(in_range("2.14.1", ">=2.0.0,<3.0.0"))
        self.assertFalse(in_range("1.10.26", ">=2.0.0,<3.0.0"))
        self.assertFalse(in_range("2.14.1", ">=2.0.0,<=2.0.2"))
        self.assertTrue(scope_matches("range:>=2.0.0,<=2.0.2", "2.0.2"))
        self.assertFalse(scope_matches("unknown", "2.14.1"))  # unknown never supports a version


class RealIssuesTest(unittest.TestCase):
    def test_issue_snapshot_has_roles_and_resolution(self):
        project = load_project(Path("data/sources/pydantic/project.json"))
        issues = {d.issue.number: d for d in project.documents if d.issue}
        self.assertEqual(set(issues), {6777, 7472, 6521, 6181, 11538})
        self.assertEqual(issues[11538].issue.resolution_status, "not_planned")  # closed != solved
        self.assertIn("[MEMBER]", issues[6521].text)  # role comes from GitHub, recorded in the snapshot
        # #6521 was only verified on 2.0.x, so it must never be evidence for 2.14.1.
        self.assertFalse(scope_matches(issues[6521].version_scope, "2.14.1"))


class CompareTest(unittest.TestCase):
    def test_versions_are_retrieved_and_labelled_separately(self):  # PRD AT-07
        p = pipeline([reply([])])
        prepared = p.prepare_compare("immutable models", "1.10.26", "2.14.1")
        by_group = {}
        for cid, hit in prepared.evidence.items():
            by_group.setdefault(prepared.groups[cid], set()).add(hit.document.document_id)
        self.assertEqual(by_group["1.10.26"], {"v1"})
        self.assertTrue(by_group["2.14.1"] <= {"v2", "issue-1"})
        prompt = prepared.messages[1].content
        self.assertIn('group="1.10.26"', prompt)
        self.assertIn("upgraded from version 1.10.26 to 2.14.1", prompt)

    def test_same_version_is_not_a_comparison(self):
        self.assertEqual(pipeline([]).compare("q", "2.14", "2.14.1").outcome, "insufficient_context")


class HistoricalAndConflictTest(unittest.TestCase):
    def test_issue_only_support_is_a_historical_case_not_a_fact(self):  # PRD AT-10
        p = pipeline([])
        prepared = p.prepare("immutable models broke", "2.14.1")
        issue_cid = cid_of(prepared, "issue-1")
        p.chat_model = FakeListChatModel(responses=[reply([{"text": "cause is X", "citation_ids": [issue_cid]}])])
        answer = p.generate("q", prepared)
        self.assertEqual(answer.draft.findings, [])
        self.assertEqual([f.text for f in answer.historical_cases], ["cause is X"])
        self.assertEqual(answer.outcome, "partially_answered")
        cite = answer.citations[0]
        self.assertEqual((cite.issue_status, cite.issue_resolved_by), ("answered", "MEMBER"))

    def test_curator_marked_conflict_when_both_sides_cited(self):
        p = pipeline([])
        prepared = p.prepare("immutable models", "2.14.1")
        a, b = cid_of(prepared, "issue-1"), cid_of(prepared, "v2")
        p.chat_model = FakeListChatModel(responses=[reply([{"text": "doc says", "citation_ids": [b]},
                                                           {"text": "issue says", "citation_ids": [a]}])])
        answer = p.generate("q", prepared)
        self.assertEqual(answer.outcome, "conflicting_evidence")
        self.assertEqual(answer.conflicts, [f"{a} contradicts {b} (marked by curator)"])


class DiversifyTest(unittest.TestCase):
    def test_cap_per_document_keeps_order_and_reranks(self):
        long_doc = dataclasses.replace(V2, text="\n\n".join(f"## S{i}\n\nimmutable {i}" for i in range(6)))
        r = KeywordRetriever()
        r.index([long_doc, V1], [c for d in (long_doc, V1) for c in MarkdownChunker().split(d)])
        hits = diversify(r.search("immutable", 10), k=3, per_document=2)
        self.assertEqual([h.rank for h in hits], [1, 2, 3])
        self.assertLessEqual(sum(h.document.document_id == "v2" for h in hits), 2)


if __name__ == "__main__":
    unittest.main()
