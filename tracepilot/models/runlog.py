"""Local structured run records (PRD FR-11). One JSON file per run under var/runs/.

Records hold the question, retrieved chunk ids and scores, the final answer and
usage. They never contain API keys (nothing here has access to Settings).
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .answering import Answer


def new_run_id() -> str:
    return f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"


def answer_record(run_id: str, answer: Answer, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": answer.question,
        "outcome": answer.outcome,
        "target_version": answer.target_version,
        "model": answer.model,
        "latency_ms": answer.latency_ms,
        "usage": dataclasses.asdict(answer.usage),
        "estimated_cost_usd": None,  # price unknown -> null, never 0 (PRD FR-11)
        "retrieved": [
            {"id": cid, "chunk_id": h.chunk.chunk_id, "score": round(h.score, 4), "version": h.chunk.version_scope}
            for cid, h in answer.evidence.items()
        ],
        "answer": answer.draft.model_dump(),
        "citations": [dataclasses.asdict(c) for c in answer.citations],
        "invalid_citation_ids": answer.invalid_citation_ids,
        "unsupported_claims": answer.unsupported_claims,
        "suspicious_evidence_ids": answer.suspicious_evidence_ids,
        "notes": answer.notes,
        **(extra or {}),
    }


class RunLog:
    def __init__(self, directory: Path):
        self.directory = directory

    def save(self, record: Dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{record['run_id']}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
        return path
