# Engram — Engineering Memory MCP Server

[![CI](https://github.com/paarths-collab/engram/actions/workflows/ci.yml/badge.svg)](https://github.com/paarths-collab/engram/actions/workflows/ci.yml)

Headless, MCP-only repository memory for coding agents. Engram gives Claude
Code, Cursor, Codex, and other MCP clients the smallest valid context needed for
the current task: existing code, reusable symbols, review evidence, deterministic
file connections, and verification facts.

No LLM runs inside Engram. The coding agent reasons; Engram retrieves,
validates, budgets, and cites evidence.

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
2. Once you have a selected file or diff, call engram expand_connections with those explicit paths
3. Call engram find_existing_implementation before writing any new function/class/service
```

## Core contract

- Classify each task deterministically and select an evidence profile.
- Fuse BM25, vectors, symbols, paths, recency, and repository structure.
- Remove weak candidates and enforce a task-specific context ceiling.
- Report approximate output tokens, truncation, and MCP latency.
- Expand connections only from explicit file paths or a real diff.
- Abstain when indexed evidence is weak instead of claiming new code is safe.

See [the MCP context contract](docs/context-contract.md) for routing, validity
rules, and the with/without-Engram product evaluation.

## What's inside
- Tier-0 inventory: file walk, language + test detection (crates/repo-map/inventory.rs)
- Tier-1 symbols: tree-sitter extraction — Rust/Python/TS/JS (symbols.rs), lazy on retrieval miss
- Co-change graph from `git log` history (cochange.rs)
- Hybrid retrieval: Tantivy BM25 + hashed-ngram vectors + symbol boost + path match,
  weighted score fusion (crates/retrieval)
- SQLite store in .engram/engram.db; background indexing on server start
- Task-adaptive context router with relevance and token-budget gates
- Hand-rolled MCP stdio server with task-context, code-reuse, deterministic
  connection expansion, verification, and review-history tools (crates/mcp-server)

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

## Product gates

The next milestone is evidence quality, not feature count:

1. Run pinned implementation tasks with and without the Engram skill.
2. Measure first-attempt test success, code reuse, duplicate code, iterations,
   elapsed time, and total tokens.
3. Add latency and context-budget regression gates to CI.
4. Improve retrieval only when those outcomes or the connection/review recovery
   benchmarks identify a measured weakness.
