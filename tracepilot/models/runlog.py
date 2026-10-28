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
        "status": "completed",
        "question": answer.question,
        "outcome": answer.outcome,
        "target_version": answer.target_version,
        "compare_versions": answer.compare_versions,
        "model": answer.model,
        "latency_ms": answer.latency_ms,
        "usage": dataclasses.asdict(answer.usage),
        "estimated_cost_usd": None,  # price unknown -> null, never 0 (PRD FR-11)
        "retrieved": [
            {"id": cid, "chunk_id": h.chunk.chunk_id, "score": round(h.score, 4), "version": h.chunk.version_scope,
             "group": answer.groups.get(cid), "channels": list(h.channels)}
            for cid, h in answer.evidence.items()
        ],
        "answer": answer.draft.model_dump(),
        "citations": [dataclasses.asdict(c) for c in answer.citations],
        "invalid_citation_ids": answer.invalid_citation_ids,
        "unsupported_claims": answer.unsupported_claims,
        "historical_cases": [f.model_dump() for f in answer.historical_cases],
        "conflicts": answer.conflicts,
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


def workflow_extra(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph-specific facts for the run record: branch taken, budgets used, why it ended."""
    return {
        "engine": "graph",
        "end_reason": state.get("end_reason"),
        "retrieval_rounds": state.get("retrieval_rounds"),
        "model_calls": state.get("model_calls"),
        "query_variants": state.get("query_variants"),
        "final_evidence_chunk_ids": [h.chunk.chunk_id for h in (state.get("answer").evidence.values()
                                                                  if state.get("answer") else [])],
        "trace": state.get("trace"),
    }


def failed_record(run_id: str, question: str, state: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "failed",
        "question": question,
        "error_code": state.get("error_code"),
        "error": state.get("error"),
        "usage": {"prompt_tokens": state.get("usage_in"), "completion_tokens": state.get("usage_out")},
        "estimated_cost_usd": None,
        **extra,
    }
