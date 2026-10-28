# TracePilot

A version-aware, citation-backed technical support copilot for open-source projects
(Python · RAG · LangChain · LangGraph). Target project for the knowledge base: **Pydantic**.

> Status: **early development, built one user-facing unit at a time.**
> Nothing beyond the units marked done below exists yet. Requirements: [docs/PRD.md](docs/PRD.md).

## Progress

| Unit | What a user can do | Status |
|---|---|---|
| U01 | Ask a question in the terminal and get a model answer | Code verified (mock + OpenAI-compatible) |
| U02 | Find the most relevant source passages in a few Pydantic docs | Code verified, awaiting review |
| U03–U15 | See [docs/PRD.md §16.3](docs/PRD.md) | Not started |

Per-unit notes and learning cards: [docs/progress.md](docs/progress.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the locked dependencies):

```bash
uv sync
```

## Usage

```bash
# U01: ask the chat model (mock by default, no network)
uv run tracepilot ask "How do I pin a dependency version?"
uv run tracepilot ask --show-request "hello"

# U02: show the most similar chunks with score, location and source link
uv run tracepilot search "How do I make a model immutable?"
uv run tracepilot search -k 8 --full "如何在字段校验之前转换输入？"
uv run tracepilot search                 # interactive; index is built once

# U02: run the 5 labelled development questions
uv run tracepilot devcheck
```

The first `search` downloads the local embedding model (~220 MB from Hugging Face).
To use a real chat model, copy `.env.example` to `.env`, set `TRACEPILOT_PROVIDER=openai`
and `OPENAI_API_KEY` (any OpenAI-compatible endpoint via `OPENAI_BASE_URL`).

```bash
uv run python -m unittest discover -s tests -v
```

## Architecture (MVC, see PRD §10.3)

```
tracepilot/
  config.py            Settings from env/.env (the only place that reads env)
  app.py               composition root: Settings -> concrete chat model / embeddings / retriever
  models/              M: domain data and logic, no printing
    llm.py               ChatModel protocol, MockClient, OpenAICompatibleClient
    documents.py         SourceDocument, Chunk, SearchHit
    sources.py           load + hash-check a local source snapshot
    chunking.py          heading-aware Markdown chunker with exact char offsets
    embeddings.py        fastembed adapter for LangChain's Embeddings interface
    retrieval.py         VectorRetriever over LangChain InMemoryVectorStore
    dev_eval.py          score retrieval on labelled dev questions
  views/cli_view.py    V: turn results into terminal text
  controllers/cli.py   C: parse args, validate, call models, hand results to the view
data/sources/pydantic/ manifest, fetch log and verbatim doc snapshot (v2.14.1)
data/dev_questions/    hand-labelled development questions
scripts/               fetch_snapshot.py (allow-listed download pinned to a commit)
```
