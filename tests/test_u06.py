"""U06 tests: BM25 behaviour, RRF fusion maths, version filter through every channel."""

from __future__ import annotations

import unittest

from langchain_core.embeddings import DeterministicFakeEmbedding

from tracepilot.config import ConfigError, load_settings
from tracepilot.models.chunking import MarkdownChunker
from tracepilot.models.documents import SearchHit
from tracepilot.models.keyword import KeywordRetriever, tokenize
from tracepilot.models.retrieval import HybridRetriever, VectorRetriever
from tests.test_u02 import make_doc
from tests.test_u05 import V1, V2

DOC = make_doc(
    "## Config\n\nSet allow_mutation to False.\n\n"
    "## Errors\n\nA ValidationError lists every problem.\n\n"
    "## Other\n\nUnrelated words about mutation and errors in general.\n",
)


def keyword(*docs):
    r = KeywordRetriever()
    r.index(docs, [c for d in docs for c in MarkdownChunker().split(d)])
    return r


class Bm25Test(unittest.TestCase):
    def test_tokenizer_keeps_identifiers_whole(self):
        self.assertEqual(tokenize("Set allow_mutation=False; ValidationError!"),
                         ["set", "allow_mutation", "false", "validationerror"])

    def test_exact_identifier_ranks_first(self):
        hits = keyword(DOC).search("allow_mutation", k=3)
        self.assertEqual(hits[0].chunk.heading_path, ("Config",))
        self.assertEqual(len(hits), 1)  # chunks without any query token are not returned

    def test_no_matching_tokens_returns_empty(self):
        self.assertEqual(keyword(DOC).search("中文问题", k=3), [])

    def test_version_filter(self):
        r = keyword(V1, V2)
        self.assertEqual({h.document.document_id for h in r.search("immutable", 5, version="2.14.1")}, {"v2"})


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.versions = hits, []

    def search(self, query, k, version=None):
        self.versions.append(version)
        return self.hits[:k]


def hits_for(chunk_ids):
    chunks = {c.chunk_id: c for c in MarkdownChunker().split(DOC)}
    return [SearchHit(rank, 1.0, chunks[cid], DOC) for rank, cid in enumerate(chunk_ids, start=1)]


class RrfTest(unittest.TestCase):
    def test_found_by_both_beats_top_of_one(self):
        a = FakeRetriever(hits_for(["doc#000", "doc#001"]))
        b = FakeRetriever(hits_for(["doc#002", "doc#001"]))
        fused = HybridRetriever({"a": a, "b": b}, candidates=5, rrf_k=60).search("q", k=3)
        self.assertEqual(fused[0].chunk.chunk_id, "doc#001")  # 2 x 1/62 > 1/61
        self.assertAlmostEqual(fused[0].score, 2 / 62)
        self.assertEqual(fused[0].channels, ("a#2", "b#2"))
        self.assertEqual(len({h.chunk.chunk_id for h in fused}), 3)  # deduplicated

    def test_version_is_passed_to_every_channel(self):
        a, b = FakeRetriever([]), FakeRetriever([])
        HybridRetriever({"a": a, "b": b}).search("q", 3, version="1.10.26")
        self.assertEqual((a.versions, b.versions), (["1.10.26"], ["1.10.26"]))

    def test_real_hybrid_respects_version(self):
        vector = VectorRetriever(DeterministicFakeEmbedding(size=16))
        chunks = [c for d in (V1, V2) for c in MarkdownChunker().split(d)]
        vector.index((V1, V2), chunks)
        hybrid = HybridRetriever({"vector": vector, "bm25": keyword(V1, V2)})
        self.assertEqual({h.document.document_id for h in hybrid.search("immutable", 5, version="1.10.26")}, {"v1"})


class ConfigTest(unittest.TestCase):
    def test_retrieval_mode_validated(self):
        self.assertEqual(load_settings({}).retrieval, "hybrid")
        with self.assertRaises(ConfigError):
            load_settings({"TRACEPILOT_RETRIEVAL": "magic"})


if __name__ == "__main__":
    unittest.main()
