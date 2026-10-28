# Pydantic source snapshots

`project.json` lists one snapshot per covered version. Each `<version>/snapshot/`
holds files copied verbatim from [pydantic/pydantic](https://github.com/pydantic/pydantic),
MIT licensed (© Pydantic Services Inc., Samuel Colvin and contributors):

| Version | Tag | Commit |
|---|---|---|
| 1.10.26 | `v1.10.26` | `d771df98f3194ea2695a6a893e318b32381032c3` |
| 2.14.1 | `v2.14.1` | `b2212c787fb5c73386dc770681ec49cd09da8e73` |

Version evidence is the git tag the file was read from, not the fetch date.
`manifest.json` lists the included files; `fetch_log.json` records fetch time and SHA-256.

The v1 docs reference code examples via `{!.tmp_examples/...!}` include markers that
are rendered at docs build time; those examples are not part of the snapshot.

Refresh a snapshot with:

    uv run python scripts/fetch_snapshot.py data/sources/pydantic/<version>/manifest.json
