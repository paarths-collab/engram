# Changelog

All notable changes to Engram are documented here. The format is loosely based
on [Keep a Changelog](https://keepachangelog.com/); versions are the reproducible
release tags.

## [Unreleased]

### Added
- **Product contract** (`docs/CONTRACT.md`): frozen positioning, canonical tool
  names, deprecation path, and the target evidence shape.
- Canonical MCP tools `search_context`, `expand_connections`, and a new
  `explain_connection(source, target)` that returns the concrete recorded edges
  (import direction, historical co-change) linking two files — never a guess.

### Changed
- `get_task_context` → deprecated alias of `search_context` (accepts `query`).
- `find_connected_files` → deprecated alias of `expand_connections` (accepts `paths`).
- `predict_impact` marked **experimental**; prefer `expand_connections` when
  anchor files are known.

Aliases keep working with both argument names, so no client breaks.

## [0.1.0] — first reproducible release (pending tag)

The core evidence-based repository context layer for coding agents:

- Tier-0 inventory, Tier-1 tree-sitter symbols + imports (Rust/Python/TS/JS).
- Hybrid retrieval: Tantivy BM25 + hashed-ngram vectors + symbol/path boost +
  recency + doc/changelog demotion + stopword filtering, with exact-symbol
  ranking so a named identifier in the query wins.
- Deterministic connection graph: co-change (git history) + import graph
  (petgraph), multi-hop, leakage-free-benchmarked.
- Per-symbol-span embeddings cached in SQLite; incremental reindex on file save
  and new commits.
- GitHub PR ingestion (PRs, changed files, review comments) + review memory.
- Verification plans (YAML domain profiles), benchmark harness (`engram-evals`).
- Hand-rolled MCP stdio server; multi-OS CI (Linux/macOS/Windows); Apache-2.0.
