"""CLI controller: parse input, validate it, call the model layer, hand results to the view.

    tracepilot ask "question"            U01: send one question to the chat model
    tracepilot search ["question"]       U02: show the most similar source chunks
    tracepilot devcheck                  U02: run the labelled development questions
    tracepilot answer "question"         U03: retrieve, generate, show cited answer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from .. import app
from ..config import ConfigError, Settings, load_settings
from ..models import dev_eval, runlog
from ..models.llm import ModelError, build_messages, build_payload
from ..models.retrieval import VectorRetriever
from ..models.sources import SourceError
from ..views import cli_view

MAX_QUESTION_CHARS = 8000  # PRD FR-02
DEV_QUESTIONS = Path("data/dev_questions/u02.jsonl")


class InputError(Exception):
    pass


def validate_question(raw: str) -> str:
    question = raw.strip()
    if not question:
        raise InputError("question is empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise InputError(f"question is {len(question)} chars; limit is {MAX_QUESTION_CHARS}")
    return question


# ---------------------------------------------------------------- commands

def cmd_ask(args, settings: Settings) -> int:
    messages = build_messages(validate_question(args.question if args.question is not None else sys.stdin.read()))
    if args.show_request:
        print(json.dumps(build_payload(settings, messages), ensure_ascii=False, indent=2))
        print("---")
    result = app.build_chat_model(settings).chat(messages)
    print(cli_view.render_chat_result(result))
    return 0


def _build_index(settings: Settings) -> VectorRetriever:
    print("Building in-memory index (first run downloads the embedding model)...", file=sys.stderr)
    start = time.monotonic()
    retriever = app.build_retriever(settings)
    embedding = settings.embedding_model if settings.embedding == "fastembed" else "fake (no semantics)"
    summary = cli_view.render_index_summary(
        retriever.document_count, retriever.size, time.monotonic() - start, embedding, str(settings.manifest_path)
    )
    print(summary, file=sys.stderr)
    return retriever


def cmd_search(args, settings: Settings) -> int:
    k = args.k or settings.top_k
    if args.question is not None:
        question = validate_question(args.question)  # fail fast before the slow index build
        retriever = _build_index(settings)
        print(cli_view.render_hits(question, retriever.search(question, k), args.full))
        return 0

    retriever = _build_index(settings)
    print("Type a question (empty line or Ctrl-D to quit).", file=sys.stderr)
    while True:
        try:
            line = input("\nsearch> ")
        except EOFError:
            return 0
        if not line.strip():
            return 0
        try:
            question = validate_question(line)
        except InputError as exc:
            print(f"error [INVALID_INPUT]: {exc}", file=sys.stderr)
            continue
        print(cli_view.render_hits(question, retriever.search(question, k), args.full))


def cmd_devcheck(args, settings: Settings) -> int:
    k = args.k or settings.top_k
    questions = dev_eval.load_questions(DEV_QUESTIONS)
    retriever = _build_index(settings)
    print(cli_view.render_dev_results(dev_eval.run(retriever, questions, k), k))
    return 0


def cmd_answer(args, settings: Settings) -> int:
    question = validate_question(args.question)
    if args.k:
        settings = replace(settings, top_k=args.k)
    pipeline = app.build_rag(settings, _build_index(settings))
    evidence, messages = pipeline.prepare(question)
    if args.show_context:
        print(cli_view.render_context(messages))
        print()
    answer = pipeline.generate(question, evidence, messages)
    run_id = runlog.new_run_id()
    path = app.build_run_log().save(runlog.answer_record(run_id, answer))
    print(cli_view.render_answer(answer, str(path)))
    return 0


# ---------------------------------------------------------------- entry point

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tracepilot")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask the chat model a question (no retrieval).")
    ask.add_argument("question", nargs="?", help="Question text. If omitted, read from stdin.")
    ask.add_argument("--show-request", action="store_true", help="Print the JSON request body (never the API key).")
    ask.set_defaults(handler=cmd_ask)

    search = sub.add_parser("search", help="Find the source chunks most similar to a question.")
    search.add_argument("question", nargs="?", help="Omit to enter interactive mode (index is built once).")
    search.add_argument("-k", type=int, help="How many chunks to return (default TRACEPILOT_TOP_K).")
    search.add_argument("--full", action="store_true", help="Print whole chunks instead of snippets.")
    search.set_defaults(handler=cmd_search)

    check = sub.add_parser("devcheck", help="Run the labelled U02 development questions.")
    check.add_argument("-k", type=int, help="Top-k to evaluate (default TRACEPILOT_TOP_K).")
    check.set_defaults(handler=cmd_devcheck)
    answer = sub.add_parser("answer", help="Answer a question from the documentation, with citations.")
    answer.add_argument("question")
    answer.add_argument("-k", type=int, help="How many chunks to give the model (default TRACEPILOT_TOP_K).")
    answer.add_argument("--show-context", action="store_true", help="Print the exact prompt sent to the model.")
    answer.set_defaults(handler=cmd_answer)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "k", None) is not None and args.k < 1:
        print("error [INVALID_INPUT]: -k must be >= 1", file=sys.stderr)
        return 2
    try:
        settings = load_settings()
        return args.handler(args, settings)
    except InputError as exc:
        print(f"error [INVALID_INPUT]: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, SourceError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except ModelError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)
        return 1
