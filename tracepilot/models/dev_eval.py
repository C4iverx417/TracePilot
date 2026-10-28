"""Score retrieval against a handful of hand-labelled development questions.

A hit counts as relevant when it comes from the expected document AND the
expected heading appears somewhere in its heading path. This is a coarse,
manually defined check for U02, not the final Evidence Hit@8 metric (PRD 13.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .documents import SearchHit
from .retrieval import VectorRetriever
from .versions import scope_matches


@dataclass(frozen=True)
class DevQuestion:
    id: str
    kind: str
    question: str
    expected: List[dict]
    min_hits: int
    note: str
    version: Optional[str] = None  # if set, retrieval is filtered to this version


@dataclass(frozen=True)
class DevResult:
    case: DevQuestion
    hits: List[SearchHit]
    relevant_ranks: List[int]
    wrong_version_ranks: List[int]

    @property
    def passed(self) -> bool:
        if self.wrong_version_ranks:
            return False
        if not self.case.expected:
            return True  # no-answer cases cannot fail in U02; see view for the caveat
        return len(self.relevant_ranks) >= self.case.min_hits


def load_questions(path: Path) -> List[DevQuestion]:
    with path.open(encoding="utf-8") as f:
        return [DevQuestion(**json.loads(line)) for line in f if line.strip()]


def is_relevant(hit: SearchHit, case: DevQuestion) -> bool:
    """Same document, and (if a heading is labelled) that heading is on the chunk's heading path."""
    return any(
        hit.chunk.document_id == exp["document_id"]
        and (exp.get("heading") is None or exp["heading"] in hit.chunk.heading_path)
        for exp in case.expected
    )


def wrong_version(hit: SearchHit, case: DevQuestion) -> bool:
    return case.version is not None and not scope_matches(hit.chunk.version_scope, case.version)


def run(retriever: VectorRetriever, questions: List[DevQuestion], k: int) -> List[DevResult]:
    results = []
    for case in questions:
        hits = retriever.search(case.question, k, version=case.version)
        ranks = [hit.rank for hit in hits if is_relevant(hit, case)]
        wrong = [hit.rank for hit in hits if wrong_version(hit, case)]
        results.append(DevResult(case, hits, ranks, wrong))
    return results
