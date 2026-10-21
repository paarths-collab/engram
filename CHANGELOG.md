# Changelog

All notable changes to Engram are documented here. The format is loosely based
on [Keep a Changelog](https://keepachangelog.com/); versions are the reproducible
release tags.

## [Unreleased]

### Added
- **Citable graph facts.** Every import and co-change edge has a content-derived
  `edge_id` (`e_` plus 16 hex digits: a 64-bit FNV-1a over the snapshot id, the
  edge kind, its endpoints, and the full import chain or the co-change
  direction). Ids are stable across runs and machines for one snapshot and
  change with it; being 64-bit they are not globally unique.
  `explain_connection` reports an `edge_id` on every reason and on each hop of
  an import route (`edges[].edge_id`, the id of that single edge).
- **`get_graph(paths, max_hops?, kinds?, snapshot?)`**, a supporting tool that
  returns the raw recorded facts around anchor files: single-hop heuristic
  import edges (importer to imported) and stored co-change rows, independent of
  the recommendation caps. Deterministic order; hard caps (200 nodes, 500 edges,
  50 co-change rows per anchor) with truncation always disclosed in `limits`
  (`omitted_edges`, `omitted_nodes`, `reason`). It is not a call graph:
  symbol-level call edges are not implemented.
- **Recommendations cite their evidence.** `expand_connections` entries gain
  `evidence_ids` (the co-change row ids, or the id of every import hop back to
  an anchor) and `score_components` (the numbers behind `confidence`). Every
  cited id is an id `get_graph` emits for the same anchors in the same snapshot.
  Search evidence is unchanged: its per-signal scores are not kept, so it has no
  `score_components`.
- **Co-change completeness facts.** Each co-change reason in
  `explain_connection` (and each co-change edge in `get_graph`) carries
  `opportunities_source` / `opportunities_target` (non-bulk commits in the
  window that touched each file, the exact denominator of `strength`), and the
  `explain_connection` / `get_graph` scope carries `commits_scanned` and
  `bulk_commits_excluded`. Unknown is `null`, never `0`: an index that predates
  them, or whose history walk failed, reports `null`.
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
- **Every snapshot id changes once on upgrade.** The id now folds in the
  inventory rules fingerprint (ignored directories, extension table, extra
  suffixes, size cap, parser version) instead of the bare parser version, and a
  database holding the old value is re-extracted on first open. Edits to those
  rules likewise change the id from then on, with no manual bump.
- **Wording of the co-change `detail` in `explain_connection`.** It used to say
  "changed together in 3 of 4 commits that touched X", but the denominator
  counts only non-bulk commits (those touching at most 30 files). It now reads
  "changed together in 3 of the 4 non-bulk commits that touched X (75%)". The
  percentage is computed as an integer from the two counts (rounded to nearest)
  instead of truncating the stored float. When the denominator is not recorded
  the text says so. Clients that matched the old sentence should read `count` and
  `opportunities_*`.
- A failed history walk (not a git repository, no commits, `git` missing) no
  longer records "0 commits scanned": it stores no co-change rows and leaves the
  completeness facts unknown (`null`). The walk also pins `log.showSignature` and
  `core.quotePath` so a user's git configuration cannot change what is parsed;
  non-ASCII paths now come back unescaped and match the inventory.
- Each `edges[]` entry of an `explain_connection` import reason now carries an
  `edge_id` next to `from`/`to`. Clients that compared an entry for exact
  equality with `{from, to}` must ignore the extra key.
- `explain_connection` now returns an error when either path is not an indexed
  file, or when both paths are the same file. Callers must pass repo-relative
  paths returned by `search_context` or `expand_connections`.
- For equal confidence, `likely_files` in `expand_connections` is ordered by path.

### Fixed
- **`database is locked` during startup.** The busy timeout is now set before
  any other statement (it used to follow the journal-mode switch and schema
  creation) and raised to 10 seconds, and write transactions take the write lock
  immediately instead of upgrading from a read, which fails at once without
  waiting. Startup indexing, the watcher and request handlers no longer fail
  against each other on the shared SQLite file.
- Two connections opening the store at the same moment can no longer both wipe
  derived data on a rules-fingerprint mismatch: the reset runs under
  `BEGIN IMMEDIATE` and re-reads the stored fingerprint inside the transaction.
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
