# Engram — Engineering Memory MCP Server

[![CI](https://github.com/paarths-collab/engram/actions/workflows/ci.yml/badge.svg)](https://github.com/paarths-collab/engram/actions/workflows/ci.yml)

Headless MCP server that gives coding agents (Claude Code, Cursor, Codex, any MCP client)
persistent knowledge of a repository: what exists, what to reuse, what a task will affect.
No LLM inside — the coding agent is the brain; Engram is the fast deterministic memory.

## Build
```bash
cargo build --release
# binary at target/release/engram
```

## Wire into an agent (Claude Code example)
```json
{
  "mcpServers": {
    "engram": { "command": "/path/to/engram", "args": ["--repo", "."] }
  }
}
```

Drop into CLAUDE.md / AGENTS.md:
```
Before implementing any task:
1. Call engram get_task_context with the task description
2. If you already know which file(s) you're touching, call engram find_connected_files
   with those paths instead of predict_impact — it returns only files linked by
   recorded facts (co-change history, import graph), not a guess.
3. Otherwise call engram predict_impact before modifying files
4. Call engram find_existing_implementation before writing any new function/class/service
5. Call engram get_verification_plan after making changes, before opening a PR
6. Call engram get_review_history on a file before changing it, to see past reviewer comments
```

## Tools
- `get_task_context(task)` — hybrid (BM25+vector+symbol) evidence for a task, plus past reviewer comments on the matched files
- `find_existing_implementation(concept)` — check before writing new code, to reuse instead of duplicate
- `predict_impact(task)` — text-driven impact guess; use when there's no anchor file yet
- `find_connected_files(files)` — **deterministic, no prediction**: given anchor file(s) you already know you're changing, returns only files linked by recorded facts (co-change graph, import graph). Prefer this over `predict_impact` whenever you have concrete anchors — see `docs/benchmarks.md` Benchmark 2 for why.
- `get_verification_plan(changed_files)` — merged checklist from YAML domain profiles + detected test commands + historically co-failing tests
- `get_review_history(path, task)` — raw, unsummarized past reviewer comments with PR number and merged status

## What's inside
- Tier-0 inventory: file walk, language + test detection (crates/repo-map/inventory.rs)
- Tier-1 symbols + imports: tree-sitter extraction — Rust/Python/TS/JS, lazy on retrieval miss
- Co-change graph from `git log` history + in-memory `petgraph` graph layer for
  multi-hop traversal (crates/repo-map/graph.rs) — see docs/adr/0001
- Hybrid retrieval: Tantivy BM25 + hashed-ngram vectors + symbol boost + path match +
  recency + doc/changelog demotion + stopword filtering, weighted score fusion (crates/retrieval)
- Vector store persisted to SQLite (content-hash keyed; unchanged files skip re-embedding)
- Incremental reindex: file-watcher (save → reparse just that file) + git-HEAD watcher
  (new commit → refresh co-change graph), both on background threads (crates/mcp-server/watcher.rs)
- GitHub PR ingestion: merged PRs, changed files, review comments (crates/connectors-github)
- Benchmark harness: file-prediction Recall@k and leakage-free connection-recovery
  eval (crates/evals) — see docs/benchmarks.md
- SQLite store in .engram/engram.db; background indexing on server start
- Hand-rolled MCP stdio server, six tools (crates/mcp-server)

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

## Benchmarks
See `docs/benchmarks.md`. Headline: given one anchor file, the deterministic
graph (`find_connected_files`) recovers **58% of the files that actually
changed alongside it** (Recall@10) on langchain-ai/langchain, with a
leakage-free temporal split — no text, no LLM, no guessing.

## Next
- Root-cause engine (explain_failure): combine CI failure logs + diff + history
- Real embeddings (fastembed-rs) behind the existing `Embedder` trait
- Learning loop: submit_outcome + deterministic feedback scoring
