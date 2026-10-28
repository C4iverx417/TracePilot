# Pydantic source snapshot

Files under `snapshot/` are copied verbatim from
[pydantic/pydantic](https://github.com/pydantic/pydantic) at tag `v2.14.1`
(commit `b2212c787fb5c73386dc770681ec49cd09da8e73`), MIT licensed,
© Pydantic Services Inc. and contributors.

`manifest.json` lists which files are included; `fetch_log.json` records when each
was fetched and its SHA-256. Refresh with:

    uv run python scripts/fetch_snapshot.py data/sources/pydantic/manifest.json
