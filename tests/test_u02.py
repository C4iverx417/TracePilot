"""U02 tests: snapshot loading, chunking, retrieval plumbing, dev-check scoring.

Retrieval tests use LangChain's DeterministicFakeEmbedding, so they check the
plumbing (ids, offsets, k, dedupe) without downloading a model. Whether the
*real* embedding finds the right text is checked by `tracepilot devcheck`.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from langchain_core.embeddings import DeterministicFakeEmbedding

from tracepilot.config import load_settings
from tracepilot.controllers import cli
from tracepilot.models.chunking import MarkdownChunker
from tracepilot.models.dev_eval import DevQuestion, is_relevant
from tracepilot.models.documents import SearchHit, SourceDocument, sha256
from tracepilot.models.retrieval import VectorRetriever
from tracepilot.models.sources import SourceError, load_snapshot

REAL_MANIFEST = Path("data/sources/pydantic/manifest.json")

SAMPLE = """Intro paragraph.

## Install

Run pip.

## Usage

```python
# this is a comment, not a heading
#> output line
x = 1
```

### Details

Some details here.
"""


def make_doc(text: str, doc_id: str = "doc") -> SourceDocument:
    return SourceDocument(
        document_id=doc_id, project_id="p", source_type="official_doc", title="Title",
        canonical_url="https://example.invalid/doc", snapshot_id="s1", content_hash=sha256(text),
        fetched_at="2026-01-01T00:00:00+00:00", version_scope="exact:1.0", version_evidence="test",
        status="active", text=text,
    )


class ChunkerTest(unittest.TestCase):
    def test_headings_inside_code_fences_are_ignored(self):
        chunks = MarkdownChunker(chunk_size=1000).split(make_doc(SAMPLE))
        paths = [c.heading_path for c in chunks]
        self.assertEqual(paths, [(), ("Install",), ("Usage",), ("Usage", "Details")])

    def test_offsets_point_back_to_exact_source_text(self):
        doc = make_doc(SAMPLE)
        for chunk in MarkdownChunker(chunk_size=1000).split(doc):
            self.assertEqual(doc.text[chunk.char_start:chunk.char_end], chunk.content)

    def test_long_section_is_split_within_size(self):
        text = "## Big\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(20))
        doc = make_doc(text)
        chunks = MarkdownChunker(chunk_size=300).split(doc)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.content) <= 300 for c in chunks))
        self.assertTrue(all(c.heading_path == ("Big",) for c in chunks))
        # Pieces are contiguous: nothing lost between chunks.
        self.assertEqual("".join(c.content for c in chunks), text)

    def test_real_snapshot_chunks_round_trip(self):
        for doc in load_snapshot(REAL_MANIFEST):
            for chunk in MarkdownChunker().split(doc):
                self.assertEqual(doc.text[chunk.char_start:chunk.char_end], chunk.content)
                self.assertLessEqual(len(chunk.content), 400)


class SourcesTest(unittest.TestCase):
    def test_loads_real_snapshot_with_version_evidence(self):
        docs = load_snapshot(REAL_MANIFEST)
        self.assertEqual(len(docs), 5)
        self.assertTrue(all(d.version_scope == "exact:2.14.1" for d in docs))
        self.assertTrue(all("b2212c787fb5" in d.canonical_url for d in docs))

    def test_tampered_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pydantic"
            shutil.copytree(REAL_MANIFEST.parent, copy)
            (copy / "snapshot/docs/concepts/config.md").write_text("edited", encoding="utf-8")
            with self.assertRaises(SourceError):
                load_snapshot(copy / "manifest.json")

    def test_missing_manifest(self):
        with self.assertRaises(SourceError):
            load_snapshot(Path("does/not/exist.json"))


class RetrieverTest(unittest.TestCase):
    def setUp(self):
        self.doc = make_doc(SAMPLE)
        self.chunks = MarkdownChunker(chunk_size=1000).split(self.doc)
        self.retriever = VectorRetriever(DeterministicFakeEmbedding(size=32))
        self.retriever.index([self.doc], self.chunks)

    def test_returns_k_ranked_hits_with_sources(self):
        hits = self.retriever.search("how to install", k=2)
        self.assertEqual([h.rank for h in hits], [1, 2])
        self.assertGreaterEqual(hits[0].score, hits[1].score)
        self.assertEqual(hits[0].document.canonical_url, "https://example.invalid/doc")

    def test_indexing_same_chunks_twice_does_not_duplicate(self):
        self.retriever.index([self.doc], self.chunks)
        self.assertEqual(self.retriever.size, len(self.chunks))
        self.assertEqual(len(self.retriever.search("x", k=100)), len(self.chunks))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            self.retriever.search("  ", k=3)
        with self.assertRaises(ValueError):
            self.retriever.search("x", k=0)

    def test_empty_index_returns_nothing(self):
        self.assertEqual(VectorRetriever(DeterministicFakeEmbedding(size=8)).search("x", k=3), [])


class DevEvalTest(unittest.TestCase):
    def test_relevance_needs_document_and_heading(self):
        doc = make_doc(SAMPLE)
        chunk = MarkdownChunker(chunk_size=1000).split(doc)[3]  # Usage > Details
        hit = SearchHit(1, 0.9, chunk, doc)
        case = lambda d, h: DevQuestion("q", "direct", "?", [{"document_id": d, "heading": h}], 1, "")  # noqa: E731
        self.assertTrue(is_relevant(hit, case("doc", "Usage")))  # ancestor heading counts
        self.assertFalse(is_relevant(hit, case("doc", "Install")))
        self.assertFalse(is_relevant(hit, case("other", "Usage")))


class SearchCliTest(unittest.TestCase):
    def test_search_with_fake_embedding_prints_sources(self):
        env = {"TRACEPILOT_EMBEDDING": "fake", "TRACEPILOT_TOP_K": "3"}
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("tracepilot.controllers.cli.load_settings", lambda: load_settings(env)):
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main(["search", "frozen model"])
        self.assertEqual(code, 0)
        self.assertIn("[3] score=", out.getvalue())
        self.assertIn("github.com/pydantic/pydantic/blob/", out.getvalue())
        self.assertIn("fake (no semantics)", err.getvalue())

    def test_search_rejects_empty_question_before_indexing(self):
        err = io.StringIO()
        with mock.patch("tracepilot.controllers.cli.app.build_retriever") as build, redirect_stderr(err):
            code = cli.main(["search", "   "])
        self.assertEqual(code, 2)
        build.assert_not_called()

    def test_dev_questions_file_is_well_formed(self):
        with open("data/dev_questions/u02.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), 5)
        self.assertEqual({r["kind"] for r in rows} >= {"direct", "multi_chunk", "no_answer"}, True)


if __name__ == "__main__":
    unittest.main()
