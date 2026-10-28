"""Settings loaded from environment variables (and an optional .env file).

Every tunable value lives here so the rest of the code never reads os.environ
directly. Secrets are kept out of __repr__ so they cannot leak into logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

SUPPORTED_PROVIDERS = ("mock", "openai")
SUPPORTED_EMBEDDINGS = ("fastembed", "fake")


class ConfigError(Exception):
    """Configuration is missing or invalid. Message is safe to show the user."""


def load_dotenv(path: Path) -> Dict[str, str]:
    """Parse a minimal KEY=VALUE .env file. Lines starting with # are ignored."""
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    timeout_s: float
    embedding: str
    embedding_model: str
    chunk_size: int
    top_k: int
    manifest_path: Path
    # repr=False: printing Settings never shows the key.
    api_key: Optional[str] = field(default=None, repr=False)


def _get_float(env: Dict[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a number, got {raw!r}")


def _get_positive_int(env: Dict[str, str], key: str, default: int) -> int:
    value = _get_float(env, key, default)
    if value <= 0 or value != int(value):
        raise ConfigError(f"{key} must be a positive integer, got {env.get(key)!r}")
    return int(value)


def load_settings(env: Optional[Dict[str, str]] = None, dotenv_path: Optional[Path] = None) -> Settings:
    """Build Settings. Real environment variables override values from .env."""
    if env is None:
        merged = load_dotenv(dotenv_path or Path.cwd() / ".env")
        merged.update(os.environ)
        env = merged

    provider = env.get("TRACEPILOT_PROVIDER", "mock").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"Unknown TRACEPILOT_PROVIDER={provider!r}; choose one of {', '.join(SUPPORTED_PROVIDERS)}"
        )

    api_key = env.get("OPENAI_API_KEY") or None
    if provider == "openai" and not api_key:
        raise ConfigError(
            "TRACEPILOT_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "Put it in .env (see .env.example) or export it in your shell."
        )

    embedding = env.get("TRACEPILOT_EMBEDDING", "fastembed").strip().lower()
    if embedding not in SUPPORTED_EMBEDDINGS:
        raise ConfigError(
            f"Unknown TRACEPILOT_EMBEDDING={embedding!r}; choose one of {', '.join(SUPPORTED_EMBEDDINGS)}"
        )

    max_tokens = _get_positive_int(env, "TRACEPILOT_MAX_TOKENS", 512)

    return Settings(
        provider=provider,
        model=env.get("TRACEPILOT_MODEL", "mock-echo" if provider == "mock" else "gpt-4o-mini"),
        base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        temperature=_get_float(env, "TRACEPILOT_TEMPERATURE", 0.2),
        max_tokens=max_tokens,
        timeout_s=_get_float(env, "TRACEPILOT_TIMEOUT_S", 30.0),
        embedding=embedding,
        embedding_model=env.get(
            "TRACEPILOT_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        chunk_size=_get_positive_int(env, "TRACEPILOT_CHUNK_SIZE", 400),
        top_k=_get_positive_int(env, "TRACEPILOT_TOP_K", 5),
        manifest_path=Path(env.get("TRACEPILOT_MANIFEST", "data/sources/pydantic/manifest.json")),
        api_key=api_key,
    )
