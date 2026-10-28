"""Score retrieval against a handful of hand-labelled development questions.

A hit counts as relevant when it comes from the expected document AND the
expected heading appears somewhere in its heading path. This is a coarse,
manually defined check for U02, not the final Evidence Hit@8 metric (PRD 13.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .documents import SearchHit
from .retrieval import VectorRetriever


@dataclass(frozen=True)
class DevQuestion:
    id: str
    kind: str
    question: str
    expected: List[dict]
    min_hits: int
    note: str


@dataclass(frozen=True)
class DevResult:
    case: DevQuestion
    hits: List[SearchHit]
    relevant_ranks: List[int]

    @property
    def passed(self) -> bool:
        if not self.case.expected:
            return True  # no-answer cases cannot fail in U02; see view for the caveat
        return len(self.relevant_ranks) >= self.case.min_hits


def load_questions(path: Path) -> List[DevQuestion]:
    with path.open(encoding="utf-8") as f:
        return [DevQuestion(**json.loads(line)) for line in f if line.strip()]


def is_relevant(hit: SearchHit, case: DevQuestion) -> bool:
    return any(
        hit.chunk.document_id == exp["document_id"] and exp["heading"] in hit.chunk.heading_path
        for exp in case.expected
    )


def run(retriever: VectorRetriever, questions: List[DevQuestion], k: int) -> List[DevResult]:
    results = []
    for case in questions:
        hits = retriever.search(case.question, k)
        ranks = [hit.rank for hit in hits if is_relevant(hit, case)]
        results.append(DevResult(case, hits, ranks))
    return results
