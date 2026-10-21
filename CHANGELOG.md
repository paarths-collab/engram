# Changelog

All notable changes to Engram are documented here. The format is loosely based
on [Keep a Changelog](https://keepachangelog.com/); versions are the reproducible
release tags.

## [Unreleased]

### Added
- **Product contract** (`docs/CONTRACT.md`): frozen positioning, canonical tool
  names and arguments, deprecation path, the pack object, snapshot ids, index
  states, and a roadmap of what is not implemented.
- Canonical MCP tools `search_context`, `expand_connections`, and
  `explain_connection(source, target)`.
- **Budgeted results.** `search_context` and `expand_connections` accept
  `budget_tokens`. Whole evidence entries are kept in rank order, and a response
  that does not fit can return minimal records with an `expansion_handle`.
- **Pack object.** Every budgeted response reports its estimator (`bytes3-v1`,
  an estimate and not a model tokenizer), `used_tokens`, `truncated`,
  `omitted_count`, and what was left out and why (`over_budget`, `below_cap`,
  `recommendation_cap`).
- `expand_connections` accepts `max_hops` (1 or 2, default 2).
- **Snapshot id.** Every index-backed response carries a `snapshot` id. Search
  and expansion accept a `snapshot` argument and report `snapshot_mismatch` when
  it differs from the current index.
- **Explicit index state.** The server reports `Indexing`, `Ready` or `Failed`.
  `Indexing` and `Failed` come back as tool errors with recovery guidance;
  `Failed` says to fix the cause and restart. Responses that may be stale carry
  `index_warnings` and `index_state`.
- **End-to-end stdio tests** that drive the compiled server through
  search, expansion and explanation on a fixture git repository, including a
  restart after offline edits.

### Changed
- `explain_connection` is a direct pairwise lookup. It returns every recorded
  fact linking two files: co-change rows in both directions, and import chains
  of up to two hops in both directions. Each fact is typed, with a `direction`,
  its `scope`, and a `status` of `found`, `not_found_within_scope` or
  `unknown_beyond_bound`. Import and co-change facts are reported separately.
- Import edges are labelled as heuristic module-name resolution, not compiler
  resolution. Two-hop dependents in expansion are no longer described as direct
  importers.
- **Restart freshness.** Every indexed file is fingerprinted by size, mtime and
  content hash. An edit made while the server was stopped invalidates that
  file's symbols, imports and vectors before the index becomes `Ready`. Deleted
  and renamed files are removed completely. A parser/config version is recorded,
  and a change to it discards derived data. The in-memory engine and import
  graph are built off to the side and swapped in whole, including after lazy
  extraction, so a query never mixes generations.
- **Deterministic ordering.** Search results are ordered by score, then path,
  then start line. Expansions are ordered by confidence, then path.
- **Watcher.** Watcher errors are logged and surfaced as `index_warnings`
  instead of being silent. The watcher uses the same ignore rules as the
  inventory and skips root-level events.
- `get_task_context` → deprecated alias of `search_context` (accepts `query` or `task`).
- `find_connected_files` → deprecated alias of `expand_connections` (accepts `paths` or `files`).
  Aliases keep working with both argument names, so no client breaks.
- `predict_impact` is marked **experimental**; prefer `expand_connections` when
  anchor files are known.
- Docs aligned: README and benchmark notes lead with the canonical tools.

### Behaviour changes
- `explain_connection` now returns an error when either path is not an indexed
  file, or when both paths are the same file. Callers must pass repo-relative
  paths returned by `search_context` or `expand_connections`.
- For equal confidence, `likely_files` in `expand_connections` is ordered by path.

### Fixed
- A clippy lint in the cached-vector decoder (`chunks_exact`) is fixed, so the
  code passes newer clippy with `-D warnings`.

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
