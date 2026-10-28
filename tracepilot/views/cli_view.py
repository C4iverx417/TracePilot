"""Terminal rendering. Functions here take model results and return strings;
they make no decisions and call nothing outside this module."""

from __future__ import annotations

import textwrap
from typing import List

from ..models.dev_eval import DevResult
from ..models.documents import SearchHit
from ..models.llm import ChatResult

SNIPPET_CHARS = 300


def _fmt(value) -> str:
    return "unknown" if value is None else str(value)


def render_chat_result(result: ChatResult) -> str:
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


def render_index_summary(documents: int, chunks: int, seconds: float, embedding: str, snapshot: str) -> str:
    return (
        f"Indexed {chunks} chunks from {documents} documents in {seconds:.1f}s\n"
        f"snapshot: {snapshot}   embedding: {embedding}"
    )


def _snippet(text: str, full: bool) -> str:
    body = text.strip()
    if not full and len(body) > SNIPPET_CHARS:
        body = body[:SNIPPET_CHARS].rstrip() + " …"
    return textwrap.indent(body, "    │ ", predicate=lambda _: True)


def render_hits(query: str, hits: List[SearchHit], full: bool = False) -> str:
    if not hits:
        return f"No chunks indexed; nothing to search for {query!r}."
    lines = [f"Top {len(hits)} chunks for: {query}", ""]
    for hit in hits:
        c, d = hit.chunk, hit.document
        lines += [
            f"[{hit.rank}] score={hit.score:.3f}  {d.title} > {' > '.join(c.heading_path) or '(intro)'}",
            f"    chunk: {c.chunk_id}   {c.locator}   version: {c.version_scope}",
            f"    source: {d.canonical_url}",
            _snippet(c.content, full),
            "",
        ]
    lines.append("score = cosine similarity of embeddings. High score means 'similar text', not 'correct answer'.")
    return "\n".join(lines)


def render_dev_results(results: List[DevResult], k: int) -> str:
    lines = [f"Development retrieval check (top-{k})", ""]
    for r in results:
        case = r.case
        if case.expected:
            verdict = "PASS" if r.passed else "MISS"
            detail = f"relevant ranks {r.relevant_ranks or '-'} (need >= {case.min_hits})"
        else:
            verdict = "N/A "
            detail = "no relevant source exists; top-k is returned anyway (U02 cannot refuse)"
        top = r.hits[0].score if r.hits else float("nan")
        lines.append(f"{verdict} {case.id} [{case.kind}] top1={top:.3f}  {detail}")
        lines.append(f"     Q: {case.question}")
        for hit in r.hits:
            mark = "*" if hit.rank in r.relevant_ranks else " "
            path = " > ".join(hit.chunk.heading_path) or "(intro)"
            lines.append(f"     {mark}{hit.rank}. {hit.score:.3f} {hit.document.title} > {path}")
        lines.append("")
    graded = [r for r in results if r.case.expected]
    passed = sum(r.passed for r in graded)
    lines.append(f"{passed}/{len(graded)} answerable questions found their labelled section in the top-{k}.")
    lines.append("This is a 5-question sanity check, not an accuracy measurement.")
    return "\n".join(lines)
