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
2. Call engram predict_impact before modifying files
3. Call engram find_existing_implementation before writing any new function/class/service
```

## What's inside (Phase 1)
- Tier-0 inventory: file walk, language + test detection (crates/repo-map/inventory.rs)
- Tier-1 symbols: tree-sitter extraction — Rust/Python/TS/JS (symbols.rs), lazy on retrieval miss
- Co-change graph from `git log` history (cochange.rs)
- Hybrid retrieval: Tantivy BM25 + hashed-ngram vectors + symbol boost + path match,
  weighted score fusion (crates/retrieval)
- SQLite store in .engram/engram.db; background indexing on server start
- Hand-rolled MCP stdio server, three tools: get_task_context,
  find_existing_implementation, predict_impact (crates/mcp-server)

## Swap points (marked in code)
- `Embedder` trait (retrieval/src/embed.rs): replace HashedNgramEmbedder with
  fastembed-rs (local ONNX) or an OpenAI-compatible API — nothing else changes
- `Weights` (retrieval/src/lib.rs): fusion weights → move to config/scoring.toml
- SQLite → Postgres for cloud/team mode

## Next (per blueprint)
- PR history ingestion (GitHub API) → review memory
- File-watcher incremental reindex
- get_verification_plan (YAML profiles), explain_failure
- Benchmarks: file-prediction Recall@10, review recovery
