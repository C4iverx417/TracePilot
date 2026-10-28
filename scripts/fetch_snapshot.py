"""Download the files listed in a source manifest, pinned to the manifest's commit.

Usage: uv run python scripts/fetch_snapshot.py data/sources/pydantic/manifest.json

Only raw.githubusercontent.com for the manifest's own repo is allowed (PRD
NFR-05: no arbitrary URL fetching). Files land next to the manifest under
snapshot/, and fetched_at + content hash are written to fetch_log.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_HOST = "raw.githubusercontent.com"


def main(manifest_path: str) -> None:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    evidence = manifest["version_evidence"]
    out_dir = manifest_file.parent / "snapshot"
    log = {}

    for doc in manifest["documents"]:
        url = f"https://{ALLOWED_HOST}/{evidence['repo']}/{evidence['commit']}/{doc['repo_path']}"
        with urllib.request.urlopen(url, timeout=30) as response:
            if not response.geturl().startswith(f"https://{ALLOWED_HOST}/"):
                raise SystemExit(f"Refusing redirect to {response.geturl()}")
            body = response.read()
        target = out_dir / doc["repo_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        log[doc["document_id"]] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
        }
        print(f"ok  {doc['repo_path']}  ({len(body)} bytes)")

    (manifest_file.parent / "fetch_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
