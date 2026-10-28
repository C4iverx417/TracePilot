"""Turning what the user typed as a version into a version constraint (PRD 6.4).

- "2.14.1" or "2.14" that matches a covered version -> exact
- "unknown", "", "latest", "最新版" -> unknown (never silently mapped to a version)
- well-formed but not covered, e.g. "2.5" or "3.0" -> unsupported
- major only, e.g. "2" -> unknown: too coarse to choose documentation
- anything else -> InvalidVersionError (input error, not a business outcome)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

_VERSION = re.compile(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?$")
_UNKNOWN = {"", "unknown", "不清楚", "尚不清楚", "?"}
_LATEST = {"latest", "newest", "最新", "最新版"}


class InvalidVersionError(ValueError):
    pass


@dataclass(frozen=True)
class VersionChoice:
    kind: Literal["exact", "unknown", "unsupported"]
    version: Optional[str] = None  # the covered version for "exact"
    raw: str = ""
    reason: str = ""


def parse_version(raw: str, covered: Sequence[str]) -> VersionChoice:
    text = (raw or "").strip()
    lowered = text.lower()
    if lowered in _UNKNOWN:
        return VersionChoice("unknown", raw=text, reason="No version given")
    if lowered in _LATEST:
        return VersionChoice("unknown", raw=text,
                             reason="'latest' is not a reproducible version; please pick one of the covered versions")
    match = _VERSION.match(text)
    if not match:
        raise InvalidVersionError(f"{text!r} is not a version like 2.14.1")
    major, minor, patch = match.groups()
    if minor is None:
        return VersionChoice("unknown", raw=text, reason=f"Version {major} is too coarse; give at least major.minor")
    for version in covered:
        c_major, c_minor, c_patch = (version.split(".") + ["0", "0"])[:3]
        if (major, minor) == (c_major, c_minor) and (patch is None or patch == c_patch):
            return VersionChoice("exact", version=version, raw=text)
    return VersionChoice("unsupported", raw=text, reason=f"Version {text} is not covered by the knowledge base")


def scope_matches(version_scope: str, version: str) -> bool:
    """Exact-version chunks match only their version; 'general' content was hand-marked as cross-version."""
    return version_scope == f"exact:{version}" or version_scope == "general"
