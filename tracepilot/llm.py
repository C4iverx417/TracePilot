"""Chat model clients.

U01 deliberately avoids LangChain: the goal is to see the raw request and
response of a chat-completion call. Two clients share one interface:

- MockClient: no network, deterministic output. Used until a real key exists.
- OpenAICompatibleClient: talks to any OpenAI-compatible /chat/completions
  endpoint (OpenAI, DeepSeek, DashScope, a local server...) with only stdlib.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Settings

SYSTEM_PROMPT = (
    "You are TracePilot, a technical support assistant for open-source projects. "
    "Answer concisely. If you are not sure, say so instead of guessing."
)


# ---------------------------------------------------------------- errors

class ModelError(Exception):
    """Base class. `code` maps to the error codes in PRD section 11.3."""

    code = "MODEL_UNAVAILABLE"


class ModelAuthError(ModelError):
    code = "MODEL_AUTH_FAILED"


class ModelTimeoutError(ModelError):
    code = "DEADLINE_EXCEEDED"


class InvalidModelOutputError(ModelError):
    code = "INVALID_MODEL_OUTPUT"


# ---------------------------------------------------------------- data

@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # "provider" = reported by the API; "estimated" = our rough count; None = unknown.
    source: Optional[str] = None

    @property
    def total_tokens(self) -> Optional[int]:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatResult:
    text: str
    model: str
    provider: str
    finish_reason: Optional[str]
    latency_ms: int
    usage: Usage = field(default_factory=Usage)


def build_messages(question: str) -> List[Message]:
    return [Message("system", SYSTEM_PROMPT), Message("user", question)]


def build_payload(settings: Settings, messages: List[Message]) -> Dict[str, Any]:
    """The exact JSON body sent to /chat/completions. Contains no secrets."""
    return {
        "model": settings.model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }


# ---------------------------------------------------------------- clients

class MockClient:
    """Pretends to be a model. Useful to exercise the whole CLI without a key."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def chat(self, messages: List[Message]) -> ChatResult:
        start = time.monotonic()
        question = messages[-1].content
        text = (
            "[mock] This is a placeholder answer, not a real model response.\n"
            f"You asked: {question}"
        )
        prompt_chars = sum(len(m.content) for m in messages)
        return ChatResult(
            text=text,
            model=self.settings.model,
            provider="mock",
            finish_reason="stop",
            latency_ms=int((time.monotonic() - start) * 1000),
            # ~4 chars per token is a crude English heuristic; marked as estimated.
            usage=Usage(prompt_chars // 4, len(text) // 4, source="estimated"),
        )


class OpenAICompatibleClient:
    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise ModelAuthError("No API key configured")
        self.settings = settings

    def chat(self, messages: List[Message]) -> ChatResult:
        s = self.settings
        body = json.dumps(build_payload(s, messages)).encode("utf-8")
        request = urllib.request.Request(
            f"{s.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {s.api_key}",
            },
            method="POST",
        )

        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=s.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            # Never include exc.headers or the request: they may echo the key.
            if exc.code in (401, 403):
                raise ModelAuthError(f"Provider rejected the API key (HTTP {exc.code})") from None
            raise ModelError(f"Provider returned HTTP {exc.code}") from None
        except (socket.timeout, TimeoutError):
            raise ModelTimeoutError(f"No response within {s.timeout_s:g}s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise ModelTimeoutError(f"No response within {s.timeout_s:g}s") from None
            raise ModelError(f"Could not reach {s.base_url}: {exc.reason}") from None
        latency_ms = int((time.monotonic() - start) * 1000)

        return parse_response(raw, provider=s.provider, fallback_model=s.model, latency_ms=latency_ms)


def parse_response(raw: bytes, provider: str, fallback_model: str, latency_ms: int) -> ChatResult:
    """Turn the provider's JSON into a ChatResult, validating the fields we rely on."""
    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        text = choice["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise InvalidModelOutputError("Provider response is not a valid chat completion") from None
    if not isinstance(text, str):
        raise InvalidModelOutputError("Provider response has no text content")

    usage_raw = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=usage_raw.get("prompt_tokens"),
        completion_tokens=usage_raw.get("completion_tokens"),
        source="provider" if usage_raw else None,
    )
    return ChatResult(
        text=text,
        model=data.get("model", fallback_model),
        provider=provider,
        finish_reason=choice.get("finish_reason"),
        latency_ms=latency_ms,
        usage=usage,
    )


def make_client(settings: Settings):
    if settings.provider == "mock":
        return MockClient(settings)
    return OpenAICompatibleClient(settings)
