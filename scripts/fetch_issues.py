"""Snapshot hand-picked GitHub issues (body + all comments) as Markdown.

Usage: uv run python scripts/fetch_issues.py data/sources/pydantic/issues/manifest.json

Only api.github.com/repos/<manifest repo>/ is contacted. Each comment header
records the author's GitHub author_association (OWNER/MEMBER/CONTRIBUTOR/NONE),
which is the traceable role used instead of guessing "maintainer" from tone.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.github.com"


def get(url: str):
    if not url.startswith(API + "/repos/"):
        raise SystemExit(f"Refusing to fetch {url}")
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "tracepilot"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def render(issue: dict, comments: list) -> str:
    lines = [
        f"# Issue #{issue['number']}: {issue['title']}",
        "",
        f"- url: {issue['html_url']}",
        f"- state: {issue['state']} ({issue.get('state_reason') or 'n/a'})",
        f"- opened: {issue['created_at']} by {issue['user']['login']} [{issue['author_association']}]",
        f"- closed: {issue.get('closed_at') or 'n/a'}",
        "",
        "## Report",
        "",
        (issue.get("body") or "").strip(),
        "",
    ]
    for c in comments:
        lines += [f"## Comment by {c['user']['login']} [{c['author_association']}]", "", f"<{c['html_url']}>", "",
                  (c.get("body") or "").strip(), ""]
    return "\n".join(lines)


def main(manifest_path: str) -> None:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    repo, log = manifest["repo"], {}
    for entry in manifest["documents"]:
        n = entry["number"]
        issue = get(f"{API}/repos/{repo}/issues/{n}")
        comments = get(f"{API}/repos/{repo}/issues/{n}/comments?per_page=100")
        text = render(issue, comments).replace("\r\n", "\n")
        target = manifest_file.parent / "snapshot" / f"{n}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
        log[entry["document_id"]] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": len(text.encode("utf-8")),
            "title": issue["title"],
            "html_url": issue["html_url"],
        }
        print(f"ok  #{n}  {issue['title'][:60]}  ({len(comments)} comments)")
    (manifest_file.parent / "fetch_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
