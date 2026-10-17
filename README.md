# TracePilot

A version-aware, citation-backed technical support copilot for open-source projects
(Python · RAG · LangChain · LangGraph).

> Status: **early development, built one user-facing unit at a time.**
> Nothing beyond the units marked done below exists yet. Requirements: [docs/PRD.md](docs/PRD.md).

## Progress

| Unit | What a user can do | Status |
|---|---|---|
| U01 | Ask a question in the terminal and get a model answer | Code verified (mock + OpenAI-compatible) |
| U02 | Find relevant source passages in a few documents | Not started |
| U03–U15 | See [docs/PRD.md §16.3](docs/PRD.md) | Not started |

Per-unit notes: [docs/progress.md](docs/progress.md).

## Quick start (U01)

Requires Python ≥ 3.9. No third-party packages yet.

```bash
python3 -m tracepilot "How do I pin a dependency version?"
```

By default the provider is `mock`, which returns a placeholder answer without any network call.
To use a real model, copy `.env.example` to `.env`, set `TRACEPILOT_PROVIDER=openai` and
`OPENAI_API_KEY`. Any OpenAI-compatible endpoint works via `OPENAI_BASE_URL`.

Show the exact request body (the API key is never printed):

```bash
python3 -m tracepilot --show-request "hello"
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

## Layout

```
tracepilot/
  config.py   settings from env/.env, validation, key hidden from repr
  llm.py      Message/Usage/ChatResult, MockClient, OpenAICompatibleClient, error types
  cli.py      argument parsing, input checks, output rendering, exit codes
tests/        unit tests; a local fake HTTP server stands in for the provider
docs/         PRD and per-unit progress notes
```
