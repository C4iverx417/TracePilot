"""Calling a LangChain chat model and translating provider failures into our error types.

Everything above this module sees only ChatCall or a ModelError subclass, never
an `openai` exception, so swapping providers does not leak into the workflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .llm import InvalidModelOutputError, ModelAuthError, ModelError, ModelTimeoutError, Usage


class ModelRateLimitedError(ModelError):
    code = "MODEL_UNAVAILABLE"
    retryable = True


@dataclass(frozen=True)
class ChatCall:
    text: str
    model: str
    latency_ms: int
    usage: Usage


def _map_provider_error(exc: Exception) -> Optional[ModelError]:
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a declared dependency
        return None
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return ModelAuthError("Provider rejected the API key")
    if isinstance(exc, openai.APITimeoutError):
        return ModelTimeoutError("Model call timed out")
    if isinstance(exc, openai.RateLimitError):
        return ModelRateLimitedError("Provider rate limit (HTTP 429)")
    if isinstance(exc, openai.APIStatusError):
        return ModelError(f"Provider returned HTTP {exc.status_code}", retryable=exc.status_code >= 500)
    if isinstance(exc, openai.APIConnectionError):
        return ModelError("Could not reach the model provider", retryable=True)
    return None


def invoke_chat(model: BaseChatModel, messages: List[BaseMessage]) -> ChatCall:
    start = time.monotonic()
    try:
        reply = model.invoke(messages)
    except ModelError:
        raise
    except Exception as exc:  # noqa: BLE001 - translated below or re-raised
        mapped = _map_provider_error(exc)
        if mapped is None:
            raise
        raise mapped from None
    latency_ms = int((time.monotonic() - start) * 1000)

    if not isinstance(reply.content, str):
        raise InvalidModelOutputError("Model returned non-text content")
    meta = getattr(reply, "usage_metadata", None) or {}
    source = (reply.response_metadata or {}).get("usage_source", "provider") if meta else None
    usage = Usage(meta.get("input_tokens"), meta.get("output_tokens"), source=source)
    model_name = (reply.response_metadata or {}).get("model_name") or getattr(model, "model_name", None) or model._llm_type
    return ChatCall(text=reply.content, model=str(model_name), latency_ms=latency_ms, usage=usage)
