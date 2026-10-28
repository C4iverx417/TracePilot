"""Read a local source snapshot (manifest + files) into SourceDocument records.

Only reads files that are listed in the manifest and were already downloaded by
scripts/fetch_snapshot.py. A file whose hash differs from fetch_log.json is
rejected, so what we index is exactly what was recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .documents import IssueMeta, SourceDocument, sha256


class SourceError(Exception):
    """The snapshot on disk is missing or inconsistent with its manifest."""


def _read_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise SourceError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _read_checked(manifest_path: Path, relative: str, doc_id: str, fetch_log: dict) -> str:
    path = manifest_path.parent / "snapshot" / relative
    if not path.is_file():
        raise SourceError(f"{doc_id}: snapshot file missing at {path}")
    text = path.read_bytes().decode("utf-8")  # bytes as fetched: no newline translation
    if doc_id not in fetch_log or fetch_log[doc_id]["sha256"] != sha256(text):
        raise SourceError(f"{doc_id}: content hash does not match fetch_log.json")
    return text


def load_issue_snapshot(manifest_path: Path) -> List[SourceDocument]:
    manifest = _read_manifest(manifest_path)
    fetch_log = _read_fetch_log(manifest_path)
    documents = []
    for entry in manifest["documents"]:
        doc_id, number = entry["document_id"], entry["number"]
        text = _read_checked(manifest_path, f"{number}.md", doc_id, fetch_log)
        res = entry["resolution"]
        documents.append(
            SourceDocument(
                document_id=doc_id, project_id=manifest["project_id"], source_type="issue",
                title=f"Issue #{number}: {fetch_log[doc_id]['title']}", canonical_url=fetch_log[doc_id]["html_url"],
                snapshot_id=manifest["snapshot_id"], content_hash=sha256(text),
                fetched_at=fetch_log[doc_id]["fetched_at"], version_scope=entry["version_scope"],
                version_evidence=f"manual: {entry['version_basis']}", status="active", text=text,
                issue=IssueMeta(
                    number=number, resolution_status=res["status"], resolved_by_role=res["by_role"],
                    resolution_url=res["evidence_url"], resolution_summary=res["summary"],
                    version_basis=entry["version_basis"], conflicts_with=tuple(entry.get("conflicts_with", ())),
                ),
            )
        )
    return documents


def _read_fetch_log(manifest_path: Path) -> dict:
    log_path = manifest_path.parent / "fetch_log.json"
    if not log_path.is_file():
        raise SourceError(f"{log_path} missing; run the fetch script first")
    return json.loads(log_path.read_text(encoding="utf-8"))


def load_snapshot(manifest_path: Path) -> List[SourceDocument]:
    manifest = _read_manifest(manifest_path)
    fetch_log = _read_fetch_log(manifest_path)
    evidence = manifest["version_evidence"]
    evidence_text = f"{evidence['kind']} {evidence['tag']} @ {evidence['commit']}"
    documents = []
    for entry in manifest["documents"]:
        doc_id = entry["document_id"]
        text = _read_checked(manifest_path, entry["repo_path"], doc_id, fetch_log)
        documents.append(
            SourceDocument(
                document_id=doc_id,
                project_id=manifest["project_id"],
                source_type=entry["source_type"],
                title=entry["title"],
                canonical_url=f"https://github.com/{evidence['repo']}/blob/{evidence['commit']}/{entry['repo_path']}",
                snapshot_id=manifest["snapshot_id"],
                content_hash=sha256(text),
                fetched_at=fetch_log[doc_id]["fetched_at"],
                version_scope=f"exact:{manifest['version']}",
                version_evidence=evidence_text,
                status="active",
                text=text,
            )
        )
    return documents


@dataclass(frozen=True)
class Project:
    project_id: str
    display_name: str
    versions: Tuple[str, ...]  # covered versions, oldest first
    documents: Tuple[SourceDocument, ...]
    snapshot_ids: Tuple[str, ...]


def load_project(project_path: Path) -> Project:
    """Load every version snapshot listed in project.json."""
    if not project_path.is_file():
        raise SourceError(f"Project file not found: {project_path}")
    config = json.loads(project_path.read_text(encoding="utf-8"))
    documents: List[SourceDocument] = []
    versions, snapshots = [], []
    for relative in config["snapshots"]:
        manifest_path = project_path.parent / relative
        manifest = _read_manifest(manifest_path)
        if manifest["project_id"] != config["project_id"]:
            raise SourceError(f"{relative}: project_id does not match {config['project_id']}")
        if manifest.get("kind") == "issues":  # issues carry their own per-item version ranges
            documents.extend(load_issue_snapshot(manifest_path))
            snapshots.append(manifest["snapshot_id"])
            continue
        docs = load_snapshot(manifest_path)
        if manifest["version"] in versions:
            raise SourceError(f"{relative}: version {manifest['version']} listed twice")
        versions.append(manifest["version"])
        snapshots.append(manifest["snapshot_id"])
        documents.extend(docs)
    return Project(config["project_id"], config["display_name"], tuple(versions), tuple(documents), tuple(snapshots))
