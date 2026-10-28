"""Terminal rendering. Functions here take model results and return strings;
they make no decisions and call nothing outside this module."""

from __future__ import annotations

import textwrap
from typing import List

from ..models.answering import Answer
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


def render_index_summary(documents: int, chunks: int, seconds: float, embedding: str, coverage: str) -> str:
    return (
        f"Indexed {chunks} chunks from {documents} documents in {seconds:.1f}s\n"
        f"knowledge base: {coverage}   embedding: {embedding}"
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
            f"    chunk: {c.chunk_id}   {c.locator}   version: {c.version_scope}"
            + (f"   via: {', '.join(hit.channels)}" if hit.channels else ""),
            f"    source: {d.canonical_url}",
            _snippet(c.content, full),
            "",
        ]
    lines.append("score = cosine (vector), BM25 (keyword) or RRF (hybrid). High score means 'similar', not 'correct'.")
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
        scope = f" v{case.version}" if case.version else ""
        lines.append(f"{verdict} {case.id} [{case.kind}{scope}] top1={top:.3f}  {detail}")
        lines.append(f"     Q: {case.question}")
        for hit in r.hits:
            mark = "!" if hit.rank in r.wrong_version_ranks else "*" if hit.rank in r.relevant_ranks else " "
            path = " > ".join(hit.chunk.heading_path) or "(intro)"
            lines.append(f"     {mark}{hit.rank}. {hit.score:.3f} {hit.document.title} > {path}  [{hit.chunk.version_scope}]")
        lines.append("")
    graded = [r for r in results if r.case.expected]
    passed = sum(r.passed for r in graded)
    mixed = sum(len(r.wrong_version_ranks) for r in results)
    mrr = sum(1 / r.relevant_ranks[0] for r in graded if r.relevant_ranks) / len(graded) if graded else 0.0
    lines.append(f"{passed}/{len(graded)} answerable questions found their labelled section in the top-{k}.")
    lines.append(f"MRR (mean of 1/rank of the first relevant chunk, 0 if none): {mrr:.3f}")
    lines.append(f"Chunks from a non-matching version in any top-{k}: {mixed}")
    lines.append(f"This is a {len(results)}-question sanity check, not an accuracy measurement.")
    return "\n".join(lines)


OUTCOME_LABELS = {
    "answered": "ANSWERED",
    "partially_answered": "PARTIALLY ANSWERED",
    "insufficient_evidence": "INSUFFICIENT EVIDENCE - the knowledge base does not cover this",
    "insufficient_context": "NEED MORE INFORMATION",
    "unsupported_version": "UNSUPPORTED VERSION",
    "conflicting_evidence": "CONFLICTING EVIDENCE",
}


def render_context(messages) -> str:
    parts = []
    for m in messages:
        parts.append(f"===== {m.type.upper()} =====\n{m.content}")
    return "\n\n".join(parts)


def render_answer(answer: Answer, run_path: str | None = None) -> str:
    d = answer.draft
    if answer.compare_versions:
        version = f"   upgrade: {answer.compare_versions[0]} -> {answer.compare_versions[1]}"
    else:
        version = f"   version: {answer.target_version}" if answer.target_version else ""
    lines = [f"[{OUTCOME_LABELS.get(d.outcome, d.outcome.upper())}]{version}", "", d.summary, ""]
    if d.findings:
        lines.append("Findings (from the documentation):")
        for f in d.findings:
            refs = " ".join(f"[{c}]" for c in f.citation_ids)
            lines.append(f"  - {f.text} {refs}")
        lines.append("")
    if answer.historical_cases:
        lines.append("Similar historical cases (someone else's issue - verify before assuming the same cause):")
        for f in answer.historical_cases:
            lines.append(f"  ~ {f.text} " + " ".join(f"[{c}]" for c in f.citation_ids))
        lines.append("")
    if answer.conflicts:
        lines.append("Conflicting sources:")
        lines += [f"  x {c}" for c in answer.conflicts]
        lines.append("")
    for title, items in (("Hypotheses (unverified):", d.hypotheses), ("Steps:", d.steps),
                         ("Missing information:", d.missing_information)):
        if items:
            lines.append(title)
            lines += [f"  {i}. {item}" for i, item in enumerate(items, start=1)]
            lines.append("")
    if answer.citations:
        lines.append("Citations:")
        groups = [None] if not answer.compare_versions else list(answer.compare_versions)
        for group in groups:
            members = [c for c in answer.citations if c.group == group]
            if group and members:
                lines.append(f"  -- version {group} --")
            for c in members:
                path = c.heading_path[1:] if c.heading_path and c.heading_path[0] == c.title else c.heading_path
                lines.append(f"  [{c.id}] {' > '.join((c.title,) + path)}  ({c.version_scope})")
                lines.append(f"       {c.url}  {c.locator}")
                if c.issue_status:
                    lines.append(f"       issue resolution: {c.issue_status} by {c.issue_resolved_by} - {c.issue_resolution}")
        lines.append("")
    if answer.unsupported_claims:
        lines.append("Not supported by the evidence (shown for transparency, not as fact):")
        lines += [f"  ? {claim}" for claim in answer.unsupported_claims]
        lines.append("")
    if answer.invalid_citation_ids:
        lines.append(f"! Removed citation ids not present in the evidence: {', '.join(answer.invalid_citation_ids)}")
    for note in answer.notes:
        lines.append(f"! {note}")
    u = answer.usage
    lines += [
        "---",
        f"model: {answer.model}   latency: {answer.latency_ms} ms   evidence: {len(answer.evidence)} chunks",
        f"tokens: in={_fmt(u.prompt_tokens)} out={_fmt(u.completion_tokens)}"
        + (f" ({u.source})" if u.source else "") + "   cost: unknown",
    ]
    if run_path:
        lines.append(f"run record: {run_path}")
    return "\n".join(lines)
