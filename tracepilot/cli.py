"""Terminal entry point: python -m tracepilot "your question"."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .config import ConfigError, load_settings
from .llm import ChatResult, ModelError, build_messages, build_payload, make_client

MAX_QUESTION_CHARS = 8000  # PRD FR-02


def _fmt(value) -> str:
    return "unknown" if value is None else str(value)


def render(result: ChatResult) -> str:
    u = result.usage
    usage_note = f" ({u.source})" if u.source else ""
    return "\n".join(
        [
            result.text,
            "",
            "---",
            f"provider: {result.provider}   model: {result.model}   finish_reason: {_fmt(result.finish_reason)}",
            f"latency: {result.latency_ms} ms",
            f"tokens: prompt={_fmt(u.prompt_tokens)} completion={_fmt(u.completion_tokens)} "
            f"total={_fmt(u.total_tokens)}{usage_note}",
        ]
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tracepilot", description="Ask a model a question from the terminal.")
    parser.add_argument("question", nargs="?", help="Question text. If omitted, read from stdin.")
    parser.add_argument(
        "--show-request", action="store_true", help="Print the JSON request body (never includes the API key)."
    )
    args = parser.parse_args(argv)

    question = args.question if args.question is not None else sys.stdin.read()
    question = question.strip()
    if not question:
        print("error [INVALID_INPUT]: question is empty", file=sys.stderr)
        return 2
    if len(question) > MAX_QUESTION_CHARS:
        print(
            f"error [INVALID_INPUT]: question is {len(question)} chars; limit is {MAX_QUESTION_CHARS}",
            file=sys.stderr,
        )
        return 2

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    messages = build_messages(question)
    if args.show_request:
        print(json.dumps(build_payload(settings, messages), ensure_ascii=False, indent=2))
        print("---")

    try:
        result = make_client(settings).chat(messages)
    except ModelError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)
        return 1

    print(render(result))
    return 0
