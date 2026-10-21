# Engram — Engineering Memory MCP Server

[![CI](https://github.com/paarths-collab/engram/actions/workflows/ci.yml/badge.svg)](https://github.com/paarths-collab/engram/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Engram is an evidence-based repository context and change-impact layer for coding agents.**
It locates exact files, symbols, tests, and definitions; expands from known files/paths/symbols
to connected code; explains every connection with the fact behind it; and returns compact,
ranked context. No LLM inside — the coding agent is the brain; Engram is the fast deterministic
memory. Works with Claude Code, Cursor, Codex, or any MCP client.

The public API is frozen in [`docs/CONTRACT.md`](docs/CONTRACT.md). Canonical tools:
`search_context(query)`, `expand_connections(paths)`, `explain_connection(source, target)`.

## Build
```bash
cargo build --release
# binary at target/release/engram
```

## Commands
- `engram [--repo <path>]` — runs the MCP stdio server for the repo (default: current directory)
- `engram ingest-github [--limit N]` — ingests GitHub PRs, commits, and review comments into the
  local store. Needs `ENGRAM_GITHUB_TOKEN` or `GITHUB_TOKEN`, plus an `origin` remote. It fetches at
  most 100 records per list, so it is not a complete history.

## Wire into an agent (Claude Code example)
```json
{
  "mcpServers": {
    "engram": { "command": "/absolute/path/to/engram", "args": ["--repo", "/absolute/path/to/your/repo"] }
  }
}
```

Use an absolute repo path, not `.`. MCP clients start the server from a working directory
that is not predictable, so a relative path may point at the wrong repository. The repo
directory must be writable: Engram caches its index in `<repo>/.engram/engram.db`.

Drop into CLAUDE.md / AGENTS.md:
```
Before implementing any task:
1. Call engram search_context with the task description to find the relevant files, symbols, and tests
2. Call engram expand_connections with the file(s) you already know you're touching, to get
   files linked by recorded facts (co-change history, import graph), not a guess
3. Call engram explain_connection on any suggested file to see the fact behind the link
4. Call engram find_existing_implementation before writing any new function/class/service
5. Call engram get_review_history on a file before changing it, to see past reviewer comments
6. Call engram get_verification_plan after making changes, before opening a PR
```

If you have no anchor file yet, `predict_impact(task)` (experimental) is a text-driven guess.

## Tools
The server exposes ten tools.

**Canonical** (the public API, see [`docs/CONTRACT.md`](docs/CONTRACT.md)):
- `search_context(query)` — hybrid (BM25 + vector + symbol) search for files, symbols, tests, and definitions relevant to a task, with past reviewer comments on matched files
- `expand_connections(paths)` — deterministic: given anchor file(s), returns only files linked to them by recorded facts (co-change graph, import graph). Prefer this whenever you have concrete anchors — see `docs/benchmarks.md` Benchmark 2
- `explain_connection(source, target)` — the recorded fact behind a connection between two files

**Supporting:**
- `find_existing_implementation(concept)` — check for existing code before writing new code, to reuse instead of duplicate
- `get_graph(paths, max_hops?, kinds?)` — the raw graph facts around some files (heuristic import edges and historical co-change rows), each with a citable `edge_id` that `expand_connections` and `explain_connection` reuse. Facts, not a ranking, and not a call graph: symbol-level call edges are not recorded
- `get_verification_plan(changed_files)` — merged checklist from YAML domain profiles, detected test commands, and tests that historically changed together with these files
- `get_review_history(path?, task?)` — raw, unsummarized past reviewer comments with PR number and merged status

**Deprecated aliases** (kept for compatibility; use the canonical name):
- `get_task_context(task)` — alias for `search_context`
- `find_connected_files(files)` — alias for `expand_connections`

**Experimental:**
- `predict_impact(task)` — text-driven impact guess; use only when there's no anchor file

The `historically_co_failing_tests` key in `get_verification_plan` output is a legacy name. It
lists tests that historically changed together with the files, not tests that failed together.

## Embeddings
The default embedder is hashed n-grams: character-trigram and token features hashed into a
fixed-size vector. It is lexical and subword-based, not a learned semantic model. It works well
on identifiers that share structure (`RetryPolicy` and `retry_policy`), but it will not match
synonyms with no shared characters.

## What's inside
- Tier-0 inventory: file walk, language + test detection (crates/repo-map/inventory.rs)
- Tier-1 symbols + imports: tree-sitter extraction — Rust/Python/TS/JS only, lazy on
  retrieval miss. Files in other languages are still fully indexed (BM25 body, embedding,
  co-change edges); they just carry no symbols or import edges. Go is recognised as a
  language but has no grammar yet, so it falls in that second group.
- Co-change graph from `git log` history + in-memory `petgraph` graph layer for
  multi-hop traversal (crates/repo-map/graph.rs) — see docs/adr/0001
- Hybrid retrieval: Tantivy BM25 + hashed-ngram vectors (lexical/subword, see Embeddings) + symbol boost + path match +
  recency + doc/changelog demotion + stopword filtering, weighted score fusion (crates/retrieval)
- Vector store persisted to SQLite (content-hash keyed; unchanged files skip re-embedding)
- Incremental reindex: file-watcher (save → reparse just that file) + git-HEAD watcher
  (new commit → refresh co-change graph), both on background threads (crates/mcp-server/watcher.rs)
- GitHub PR ingestion: PRs (with base/head SHAs and timestamps), changed files
  **with their diff hunks**, the PR's commit list in order, and review comments
  with the hunk and timestamp they were left against (crates/connectors-github).
  The timestamps are what let a push be attributed to the review comment that
  prompted it — the basis for mining a repo's house rules from its own history.
- Benchmark harness: file-prediction Recall@k and connection-recovery eval with a
  temporal co-change split (crates/evals) — see docs/benchmarks.md
- SQLite store in .engram/engram.db; background indexing on server start
- Hand-rolled MCP stdio server, ten tools (crates/mcp-server)

## Scoring config
Fusion weights, doc/changelog demotions, and recency are externalized to
`config/scoring.toml` (loaded at startup, hot-reloaded in dev when the file
changes). Tune retrieval without recompiling. Missing keys fall back to
built-in defaults (retrieval/src/weights.rs).

## Swap points (marked in code)
- `Embedder` trait (retrieval/src/embed.rs): replace HashedNgramEmbedder with
  fastembed-rs (local ONNX) or an OpenAI-compatible API — nothing else changes
- `Weights` (retrieval/src/weights.rs): tune via config/scoring.toml
- SQLite → Postgres for cloud/team mode

## Evidence and limits
Each connection names the recorded fact behind it. The exact fields are in
[`docs/CONTRACT.md`](docs/CONTRACT.md); a captured session is in
[`docs/demo-transcript.md`](docs/demo-transcript.md).

- **Import edges are heuristic.** They come from module-name matching, not compiler
  resolution, and are reported as candidates.
- **Co-change is history, not a requirement.** Two files changed in the same commits
  within a bounded window. It does not mean one must change with the other.
- **Budgets are estimates.** `budget_tokens` is checked against `ceil(bytes / 3)` of
  the returned JSON (`pack.estimator`), not against a model tokenizer.
- **Timings are not benchmarks.** The cold-start and per-call numbers in the demo
  transcript come from a small fixture on a shared machine. Retrieval quality is in
  [`docs/benchmarks.md`](docs/benchmarks.md).

## Benchmarks
See `docs/benchmarks.md`. Headline: given one anchor file, the deterministic
graph (`expand_connections`) recovers **58% of the files that actually
changed alongside it** (Recall@10) on langchain-ai/langchain. The co-change split is temporal
(leakage-free); the import graph is built at the evaluated repo's current HEAD. No text, no LLM,
no guessing.

## Roadmap (not implemented)
- Root-cause engine (`explain_failure`): combine CI failure logs + diff + history
- Real embeddings (fastembed-rs) behind the existing `Embedder` trait
- Learning loop (`submit_outcome`) + deterministic feedback scoring

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Apache-2.0 rather than MIT for the explicit patent grant in section 3: contributors
grant users a patent licence covering their contributions, and that grant terminates
for anyone who starts patent litigation over the work. MIT is silent on patents.

Dependency licences are not assumed. `cargo deny check licenses` runs on every pull
request against the allow-list in [deny.toml](deny.toml); a copyleft dependency
appearing anywhere in the tree fails the build rather than shipping quietly.
