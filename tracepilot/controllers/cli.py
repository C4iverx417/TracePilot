"""CLI controller: parse input, validate it, call the model layer, hand results to the view.

    tracepilot ask "question"            U01: send one question to the chat model
    tracepilot search ["question"]       U02: show the most similar source chunks
    tracepilot devcheck                  U02: run the labelled development questions
    tracepilot answer "question" --version 2.14.1   U03-U05: version-filtered retrieval + cited answer
    tracepilot upgrade "error" --from 1.10 --to 2.14  U07: per-version evidence, migration steps
    tracepilot chat                                  U09: multi-turn diagnosis with clarification
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

from .. import app
from ..config import SUPPORTED_ENGINES, SUPPORTED_RETRIEVAL, ConfigError, Settings, load_settings
from ..models import dev_eval, runlog
from ..models.llm import ModelError, build_messages, build_payload
from ..models.retrieval import Retriever
from ..models.sources import Project, SourceError
from ..models.versions import InvalidVersionError, parse_version
from ..views import cli_view

MAX_QUESTION_CHARS = 8000  # PRD FR-02
DEV_QUESTION_DIR = Path("data/dev_questions")


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


def _build_index(settings: Settings) -> Tuple[Project, Retriever]:
    print("Building in-memory index (first run downloads the embedding model)...", file=sys.stderr)
    start = time.monotonic()
    index = app.build_index(settings)
    embedding = settings.embedding_model if settings.embedding == "fastembed" else "fake (no semantics)"
    summary = cli_view.render_index_summary(
        len(index.project.documents), index.chunk_count, time.monotonic() - start,
        f"{embedding}   retrieval: {index.mode}",
        f"{index.project.display_name} {', '.join(index.project.versions)}",
    )
    print(summary, file=sys.stderr)
    return index.project, index.retriever


def _search_version(raw: Optional[str], project: Project) -> Optional[str]:
    """For `search`, --version is a plain filter: it must name a covered version."""
    if raw is None:
        return None
    try:
        choice = parse_version(raw, project.versions)
    except InvalidVersionError as exc:
        raise InputError(str(exc)) from None
    if choice.kind != "exact":
        raise InputError(f"{choice.reason}. Covered versions: {', '.join(project.versions)}")
    return choice.version


def cmd_search(args, settings: Settings) -> int:
    k = args.k or settings.top_k
    if args.question is not None:
        question = validate_question(args.question)  # fail fast before the slow index build
        project, retriever = _build_index(settings)
        version = _search_version(args.version, project)
        print(cli_view.render_hits(question, retriever.search(question, k, version=version), args.full))
        return 0

    project, retriever = _build_index(settings)
    version = _search_version(args.version, project)
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
        print(cli_view.render_hits(question, retriever.search(question, k, version=version), args.full))


def cmd_devcheck(args, settings: Settings) -> int:
    k = args.k or settings.top_k
    questions = dev_eval.load_questions(DEV_QUESTION_DIR / f"{args.set}.jsonl")
    _, retriever = _build_index(settings)
    print(cli_view.render_dev_results(dev_eval.run(retriever, questions, k), k))
    return 0


def _parse(pipeline, raw: str):
    try:
        return pipeline.resolve_version(raw)
    except InvalidVersionError as exc:
        raise InputError(str(exc)) from None


def _finish(args, pipeline, question: str, gated, prepare) -> int:
    """Shared tail of answer/upgrade: optional prompt dump, model call, run record, render."""
    answer = gated
    if answer is None:
        prepared = prepare()
        if args.show_context:
            print(cli_view.render_context(prepared.messages))
            print()
        answer = pipeline.generate(question, prepared)
    run_id = runlog.new_run_id()
    path = app.build_run_log().save(runlog.answer_record(run_id, answer))
    print(cli_view.render_answer(answer, str(path)))
    return 0


def cmd_answer(args, settings: Settings) -> int:
    question = validate_question(args.question)
    if args.k:
        settings = replace(settings, top_k=args.k)
    project, retriever = _build_index(settings)
    pipeline = app.build_rag(settings, retriever, project)
    choice = _parse(pipeline, args.version)
    if settings.engine == "rag" or args.show_context:
        return _finish(args, pipeline, question, pipeline.version_gate(question, choice),
                       lambda: pipeline.prepare(question, choice.version))

    state = app.build_workflow(settings, pipeline).run(question, args.version)
    run_id = runlog.new_run_id()
    extra = runlog.workflow_extra(state)
    if args.trace:
        print(cli_view.render_trace(state["trace"]))
        print()
    if state.get("error_code"):
        path = app.build_run_log().save(runlog.failed_record(run_id, question, state, extra))
        print(f"error [{state['error_code']}]: {state['error']}\nrun record: {path}", file=sys.stderr)
        return 1
    path = app.build_run_log().save(runlog.answer_record(run_id, state["answer"], extra))
    print(cli_view.render_answer(state["answer"], str(path)))
    return 0


def cmd_upgrade(args, settings: Settings) -> int:
    question = validate_question(args.question)
    if args.k:
        settings = replace(settings, top_k=args.k)
    project, retriever = _build_index(settings)
    pipeline = app.build_rag(settings, retriever, project)
    before, after = _parse(pipeline, args.from_version), _parse(pipeline, args.to_version)
    return _finish(args, pipeline, question, pipeline.compare_gate(question, before, after),
                   lambda: pipeline.prepare_compare(question, before.version, after.version))


CHAT_HELP = """Commands:
  /new              start a new problem (resets clarification count, keeps your chosen version)
  /version X        select a version for this and future problems, e.g. /version 2.14.1
  /session          show what the system has confirmed so far
  /newsession       open a second, independent session
  /switch ID        switch to another session in this process
  /quit             exit (history is in memory only and will be lost)"""


def cmd_chat(args, settings: Settings) -> int:
    project, retriever = _build_index(settings)
    service = app.build_conversation(settings, index=app.KnowledgeIndex(project, retriever, 0, settings.retrieval))
    session = service.new_session(args.version)
    print(f"Session {session.session_id} (memory only; lost on exit). Covered versions: "
          f"{', '.join(project.versions)}. Type /help for commands.", file=sys.stderr)
    pending_version = None
    while True:
        try:
            line = input(f"\n[{session.session_id[-4:]}] you> ").strip()
        except EOFError:
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/help":
            print(CHAT_HELP)
            continue
        if line == "/new":
            episode = service.new_episode(session.session_id)
            print(f"New problem {episode.episode_id}; version kept: {episode.confirmed.version or '-'}")
            continue
        if line.startswith("/version"):
            pending_version = line.split(maxsplit=1)[1] if " " in line else None
            print(f"Version selected: {pending_version}. It applies from your next message.")
            continue
        if line == "/session":
            print(cli_view.render_session(service.store.get(session.session_id)))
            continue
        if line == "/newsession":
            session = service.new_session()
            print(f"Switched to new session {session.session_id}")
            continue
        if line.startswith("/switch "):
            try:
                session = service.store.get(line.split(maxsplit=1)[1])
            except KeyError:
                print("No such session. Known: " + ", ".join(service.store.list_ids()))
            continue
        try:
            result = service.submit(session.session_id, validate_question(line), pending_version)
        except (InputError, InvalidVersionError) as exc:
            print(f"error [INVALID_INPUT]: {exc}", file=sys.stderr)
            continue
        pending_version = None
        if result.answer or result.state:
            record = (runlog.answer_record(result.run_id, result.answer, runlog.workflow_extra(result.state))
                      if result.answer else runlog.failed_record(result.run_id, line, result.state,
                                                                  runlog.workflow_extra(result.state)))
            record.update({"session_id": result.session_id, "episode_id": result.episode_id,
                           "status": result.status})
            app.build_run_log().save(record)
        print(cli_view.render_turn(result))


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
    search.add_argument("--version", help="Only search documentation for this covered version, e.g. 2.14.1.")
    search.add_argument("--retrieval", choices=SUPPORTED_RETRIEVAL, help="Override TRACEPILOT_RETRIEVAL.")
    search.set_defaults(handler=cmd_search)

    check = sub.add_parser("devcheck", help="Run a set of labelled development questions.")
    check.add_argument("--set", default="u02", help="Question file under data/dev_questions/ (default u02).")
    check.add_argument("-k", type=int, help="Top-k to evaluate (default TRACEPILOT_TOP_K).")
    check.add_argument("--retrieval", choices=SUPPORTED_RETRIEVAL, help="Override TRACEPILOT_RETRIEVAL.")
    check.set_defaults(handler=cmd_devcheck)
    answer = sub.add_parser("answer", help="Answer a question from the documentation, with citations.")
    answer.add_argument("question")
    answer.add_argument("--version", default="unknown",
                        help="Your Pydantic version, e.g. 2.14.1 or 1.10; 'unknown' if unsure (default).")
    answer.add_argument("-k", type=int, help="How many chunks to give the model (default TRACEPILOT_TOP_K).")
    answer.add_argument("--show-context", action="store_true",
                        help="Print the exact prompt sent to the model (uses the single-pass rag engine).")
    answer.add_argument("--engine", choices=SUPPORTED_ENGINES, help="graph = LangGraph workflow, rag = single pass.")
    answer.add_argument("--max-rounds", type=int, help="Override the retrieval round limit (graph engine).")
    answer.add_argument("--trace", action="store_true", help="Print the node-by-node workflow trace.")
    answer.add_argument("--retrieval", choices=SUPPORTED_RETRIEVAL, help="Override TRACEPILOT_RETRIEVAL.")
    answer.set_defaults(handler=cmd_answer)

    upgrade = sub.add_parser("upgrade", help="Explain what changed between two versions for a problem.")
    upgrade.add_argument("question", help="The error text or problem after upgrading.")
    upgrade.add_argument("--from", dest="from_version", required=True, help="Version before the upgrade.")
    upgrade.add_argument("--to", dest="to_version", required=True, help="Version after the upgrade.")
    upgrade.add_argument("-k", type=int, help="Total chunks given to the model, split between the versions.")
    upgrade.add_argument("--show-context", action="store_true", help="Print the exact prompt sent to the model.")
    upgrade.add_argument("--retrieval", choices=SUPPORTED_RETRIEVAL, help="Override TRACEPILOT_RETRIEVAL.")
    upgrade.set_defaults(handler=cmd_upgrade)

    chat = sub.add_parser("chat", help="Multi-turn diagnosis in the terminal (memory only).")
    chat.add_argument("--version", help="Pre-select a version for this session.")
    chat.add_argument("--retrieval", choices=SUPPORTED_RETRIEVAL, help="Override TRACEPILOT_RETRIEVAL.")
    chat.set_defaults(handler=cmd_chat)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {}
    if getattr(args, "retrieval", None):
        overrides["retrieval"] = args.retrieval
    if getattr(args, "engine", None):
        overrides["engine"] = args.engine
    if getattr(args, "max_rounds", None) is not None:
        if args.max_rounds < 1:
            print("error [INVALID_INPUT]: --max-rounds must be >= 1", file=sys.stderr)
            return 2
        overrides["max_retrieval_rounds"] = args.max_rounds
    if getattr(args, "k", None) is not None and args.k < 1:
        print("error [INVALID_INPUT]: -k must be >= 1", file=sys.stderr)
        return 2
    try:
        settings = replace(load_settings(), **overrides)
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
