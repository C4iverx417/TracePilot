"""Domain records for the knowledge base (PRD section 6.3)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    project_id: str
    source_type: str  # official_doc | migration_guide | release_notes | issue
    title: str
    canonical_url: str
    snapshot_id: str
    content_hash: str
    fetched_at: str
    version_scope: str  # e.g. "exact:2.14.1"
    version_evidence: str  # e.g. "git_tag v2.14.1 @ b2212c7..."
    status: str  # active | withdrawn
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    heading_path: Tuple[str, ...]
    content: str  # exactly text[char_start:char_end] of the snapshot file
    char_start: int
    char_end: int
    content_hash: str
    version_scope: str
    parser_version: str

    @property
    def locator(self) -> str:
        return f"chars {self.char_start}-{self.char_end}"


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float  # meaning depends on the retriever (cosine / BM25 / RRF); higher = closer, NOT "correct"
    chunk: Chunk
    document: SourceDocument
    channels: Tuple[str, ...] = ()  # which retrievers found it, e.g. ("vector#3", "bm25#1")
