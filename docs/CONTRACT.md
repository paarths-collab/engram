# Engram product contract (v0.1)

This is the public contract as implemented on `main`. Tool names, arguments and
output fields below are the stable surface. Output fields are additive; a field
is not removed or repurposed without a deprecation cycle. Anything not described
here is not part of the contract, including everything in the roadmap at the end.

## Positioning

> **Engram is an evidence-based repository context and change-impact layer for
> coding agents.**

Engram **does**:

1. Locate exact files, symbols, tests, and definitions.
2. Expand from known files, paths, symbols, or diffs to connected code.
3. Explain every returned connection with the concrete fact behind it.
4. Return compact, ranked context to an agent, optionally within a size budget.

Engram **does not** claim it can reliably predict affected code from a vague
natural-language task. That path exists (`predict_impact`) but is marked
**experimental** and is not part of the stable contract.

## Canonical tools

| Tool | Arguments | Purpose |
|---|---|---|
| `search_context` | `query: string`, `budget_tokens?: int >= 1`, `snapshot?: string` | Ranked, evidence-backed context for a query (hybrid retrieval plus past reviews on matched files). |
| `expand_connections` | `paths: string[]`, `budget_tokens?: int >= 1`, `max_hops?: 1..2` (default 2), `snapshot?: string` | Deterministic: files connected to known anchors by recorded facts (co-change and import graph). No guessing. |
| `explain_connection` | `source: string`, `target: string` | Every recorded fact linking two indexed files, with its direction, scope and status. A direct pairwise lookup. |

Supporting tools (stable): `find_existing_implementation(concept)`,
`get_verification_plan(changed_files)`, `get_review_history(path?, task?)`.

Paths are repo-relative. Each argument is validated: `budget_tokens` must be an
integer >= 1, `max_hops` an integer 1 or 2, and `snapshot` a string. Invalid
values are tool errors (`isError: true`).

## Deprecations

| Old | Status | Replacement |
|---|---|---|
| `get_task_context(task)` | deprecated alias | `search_context(query)`; accepts `query` or `task`, same response |
| `find_connected_files(files)` | deprecated alias | `expand_connections(paths)`; accepts `paths` or `files`, same response |
| `predict_impact(task)` | experimental | `expand_connections` when anchors are known |

Aliases keep working (same handler, both argument names accepted) so no agent
breaks; they will be removed in a future release. The aliases also accept
`budget_tokens`, `max_hops` and `snapshot` where the canonical tool does.

## Common response fields

Responses from the index-backed tools (`search_context`, `expand_connections`,
`explain_connection`, `find_existing_implementation`, `predict_impact` and the
two aliases) carry these fields when they apply:

| Field | Present | Meaning |
|---|---|---|
| `snapshot` | always, on index-backed tools | Id of the index the answer was computed from (see [Snapshot id](#snapshot-id)). |
| `snapshot_mismatch` | `search_context`, `expand_connections` and aliases, when the `snapshot` argument differs from the current one | `{ "requested": "...", "current": "..." }`. The answer still comes from the current snapshot. |
| `index_warnings` | when results may be out of date | Array of plain-text warnings (see [Index state](#index-state)). |
| `index_state` | only when results may be stale | `{ "status": "ready", "stale": true, "generation": n, "snapshot": "..." }`. |
| `pack` | `search_context`, `expand_connections` and aliases | Budget and omission report (see [The pack object](#the-pack-object)). |

`get_verification_plan` and `get_review_history` do not carry `snapshot` or
`pack`. `get_verification_plan` is still gated on indexing (see [Index state](#index-state)).

## `search_context`

Request:

```json
{ "name": "search_context", "arguments": { "query": "execute order" } }
```

Response (trimmed to one evidence entry; from `docs/demo-transcript.md`,
captured at engram `0a0b1d3`, before the `pack` object existed):

```json
{
  "evidence": [
    {
      "id": "ev_000",
      "path": "app/service.py",
      "score": 2.184487819671631,
      "signals": ["bm25", "vector", "recent"],
      "snippet": "def execute(order):\n    \"\"\"Execute an order: load its record and return a status envelope.\"\"\"\n    record = load(order[\"id\"])\n    return {\"status\": \"ok\", \"record\": record}",
      "symbol": "execute",
      "title": "execute in app/service.py:5",
      "type": "symbol"
    }
  ],
  "past_reviews": [],
  "snapshot": "2c4a4e25961017ed",
  "task": "execute order"
}
```

- `evidence[].type` is one of `existing_code`, `symbol`, `test`, `related_file`.
- `evidence[].signals` names the retrieval signals that contributed; see
  [Evidence signals](#evidence-signals).
- `past_reviews` holds raw reviewer comments on the files that were kept. A
  comment is only included when its file is in `evidence`.
- The search asks for 8 results and keeps at most 11 evidence entries after
  ranking. Anything cut by that cap is listed in `pack.omitted` with reason
  `below_cap`.
- Results are ordered by score descending, then path ascending, then start line
  ascending (`pack.ordering` = `score_desc,path_asc,line_asc`).

## `expand_connections`

Request (from the transcript):

```json
{ "name": "expand_connections", "arguments": { "paths": ["app/service.py"] } }
```

Response (from the transcript; `pack` is not in that capture, see below):

```json
{
  "cochange_expansions": [
    { "confidence": 0.800000011920929, "path": "app/api.py", "reason": "changes with app/service.py in 80% of its commits" },
    { "confidence": 0.4000000059604645, "path": "app/audit.py", "reason": "changes with app/service.py in 40% of its commits" }
  ],
  "import_expansions": [
    { "confidence": 0.6000000238418579, "path": "tests/test_service.py", "reason": "imports a likely-changed file (1 hop)" }
  ],
  "likely_files": [
    { "confidence": 1.0, "path": "app/service.py", "reason": "given as an anchor file" }
  ],
  "likely_tests": ["tests/test_service.py"],
  "snapshot": "2c4a4e25961017ed"
}
```

(Trimmed: the transcript also lists three more co-change entries at 40%.)

- `likely_files` holds the anchors that are in the index, at confidence 1.0.
  Paths not in the index are ignored; they do not produce an error.
- `cochange_expansions` are files that changed with an anchor in the recorded
  history. At most 5 co-change rows are taken per anchor.
- `import_expansions` are files that import an anchor, found within `max_hops`
  hops (default 2). At most 8 entries per kind are returned; the rest are
  reported in `pack.omitted` with reason `recommendation_cap`.
- `likely_tests` lists test files among the anchors and expansions.
- Ordering: `pack.ordering` = `confidence_desc,path_asc`. Equal-confidence
  `likely_files` are in path order.
- `max_hops` outside 1..2 is an error.

### Budgeted packs

With `budget_tokens`, whole entries are kept in priority order (anchors, then
co-change, then imports, then tests). An entry too large to fit whole may come
back as a minimal record:

```json
{ "path": "app/audit.py", "confidence": 0.4000000059604645, "minimal": true,
  "expansion_handle": { "tool": "explain_connection", "target": "app/audit.py" } }
```

Call `explain_connection` with the handle's `target` to see the facts behind it.

## `explain_connection`

Request (from the transcript):

```json
{ "name": "explain_connection", "arguments": { "source": "app/api.py", "target": "app/service.py" } }
```

Response (trimmed: one of the two co-change reasons and the scope note is
shortened; the transcript has the full text):

```json
{
  "connected": true,
  "status": "found",
  "source": "app/api.py",
  "target": "app/service.py",
  "reasons": [
    {
      "count": 8,
      "detail": "app/api.py and app/service.py changed together in 8 commits, 100% of the commits that touched app/api.py",
      "direction": "source_to_target",
      "edge_kind": "cochange",
      "strength": 1.0,
      "type": "historical_cochange",
      "weight": 1.0,
      "window": { "max_commits": 2000, "max_files_per_commit": 30, "min_count": 2 }
    },
    {
      "chain": ["app/api.py", "app/service.py"],
      "detail": "app/api.py imports app/service.py directly (heuristic module-name resolution, not compiler resolution)",
      "direction": "source_imports_target",
      "edge_kind": "import",
      "edges": [{ "from": "app/api.py", "to": "app/service.py" }],
      "hops": 1,
      "resolution": "heuristic",
      "resolution_status": "candidate",
      "resolver": "heuristic_module_segments",
      "type": "import_edge",
      "weight": 0.6000000238418579
    }
  ],
  "scope": {
    "cochange_directions_searched": ["source_to_target", "target_to_source"],
    "cochange_lookup": "exact_pair",
    "cochange_window": { "max_commits": 2000, "max_files_per_commit": 30, "min_count": 2 },
    "import_directions_searched": ["source_imports_target", "target_imports_source"],
    "import_max_hops": 2,
    "import_search_source_imports_target": "found",
    "import_search_target_imports_source": "none_within_bound",
    "note": "Import chains up to 2 hops were searched in both directions over heuristically resolved import edges. Co-change covers only the last 2000 commits, pairs sharing at least 2 commits, ignoring commits touching more than 30 files. Absence of a fact means none was recorded within this scope, not that the files are unrelated."
  },
  "snapshot": "2c4a4e25961017ed"
}
```

### Top-level fields

- `connected`: `true` when `reasons` is non-empty. `false` only means no
  recorded fact within `scope`. It is never proof that the files are unrelated.
- `status`:
  - `found`: at least one fact was recorded.
  - `not_found_within_scope`: nothing recorded, and every search finished within its bound.
  - `unknown_beyond_bound`: nothing recorded within the hop bound, but imports continue past it, so a longer chain may exist.
- `scope`: exactly what was searched.
  - `import_max_hops`: always 2 on this version.
  - `import_directions_searched`, `cochange_directions_searched`: both directions.
  - `import_search_source_imports_target`, `import_search_target_imports_source`: per-direction outcome, `found`, `none_within_bound` or `beyond_bound`.
  - `cochange_lookup`: `exact_pair`.
  - `cochange_window`: `max_commits`, `min_count`, `max_files_per_commit`. A pair outside these limits is absent from the table.
  - `note`: a plain-language statement of the scope.

### Reasons

Import and co-change facts are independent. Both can appear, and neither
implies, requires, or replaces the other. A reason with `type` `import_edge`
or `historical_cochange` has:

| Field | Import (`import_edge`) | Co-change (`historical_cochange`) |
|---|---|---|
| `detail` | yes, plain text | yes, plain text |
| `weight` | `0.6` for a direct import (1 hop), `0.35` for a 2-hop chain | equal to `strength` |
| `edge_kind` | `import` | `cochange` |
| `direction` | `source_imports_target` or `target_imports_source` | `source_to_target` or `target_to_source` |
| `hops` | 1 or 2 | not present |
| `chain` | repo-relative paths, endpoints included (`hops + 1` entries) | not present |
| `edges` | one `{from, to}` per hop, in route order | not present |
| `resolution` | `heuristic` | not present |
| `resolver` | `heuristic_module_segments` | not present |
| `resolution_status` | `candidate`: a plausible edge, not compiler-verified | not present |
| `count` | not present | commits in which both files changed |
| `strength` | not present | `count` divided by the commits that touched the row's first file |
| `window` | not present | the co-change window (see `scope.cochange_window`) |

Co-change means the two files changed in the same commits. It does not mean
one must change with the other.

### Errors

Each of these is a tool error (`isError: true`) with a message in the text:

- Either path is not an indexed text file. The message asks for a repo-relative
  path returned by `search_context` or `expand_connections`.
- `source` and `target` are the same file.
- A required argument is missing or empty.

## The pack object

`search_context`, `expand_connections` and the aliases return a `pack` object.
It describes what was kept and what was left out. Nothing is dropped silently.

```json
{
  "estimator": "bytes3-v1",
  "budget_tokens": 400,
  "used_tokens": 371,
  "truncated": true,
  "omitted_count": 3,
  "omitted": [
    { "path": "app/metrics.py", "symbol": "count_run", "reason": "over_budget" },
    { "path": "app/audit.py", "kind": "cochange", "reason": "recommendation_cap" }
  ],
  "ordering": "confidence_desc,path_asc",
  "within_budget": true
}
```

The example values are illustrative. The transcript does not capture a `pack`
object, so this shape is documented from the source (`crates/retrieval/src/budget.rs`).

| Field | Meaning |
|---|---|
| `estimator` | Always `bytes3-v1`: `estimated tokens = ceil(utf8 bytes / 3)` of the text returned to the client. |
| `budget_tokens` | The requested budget, or `null` when none was requested. |
| `used_tokens` | The estimate for the whole response, `pack` included. |
| `truncated` | `true` when anything was omitted or any unit came back minimal. |
| `omitted_count` | Count of all omitted candidates, even those not listed. |
| `omitted` | Up to a bounded list of `{path, symbol?, kind?, reason}`. Under a budget the list holds at most one entry per 250 budget tokens, and at most 20 entries. `kind` appears only for kinds other than `evidence` (for example `cochange`, `import`, `test`, `review_comment`). |
| `omitted[].reason` | `over_budget` (the budget cut it), `below_cap` (the search's own cap cut it), or `recommendation_cap` (an expansion cap cut it). |
| `ordering` | `score_desc,path_asc,line_asc` for search; `confidence_desc,path_asc` for expansion. |
| `within_budget` | `true` when the whole response is within `budget_tokens`. |

How the budget works:

- The budget is measured on the decoded `content[0].text`: the pretty-printed JSON the MCP
  layer returns for the tool result. It does not include the JSON-RPC envelope.
- Units are kept whole in priority order. A unit that does not fit whole may come
  back as a minimal record with an `expansion_handle`:
  - Search evidence: `{id, type, path, symbol, start_line, end_line, score, minimal: true, expansion_handle: {path, symbol?}}`. The snippet is removed.
  - Expansion candidates: `{path, confidence, minimal: true, expansion_handle: {tool: "explain_connection", target}}`.
  - Review comments: `{pr_number, path, line, pr_merged, minimal: true, expansion_handle: {tool: "get_review_history", path}}`.
- If no unit fits, the result keeps only the snapshot fields and the pack, so
  base fields such as `task` are absent. If even that is over budget, the pack
  alone is returned with `within_budget: false`.
- The estimator is not a model tokenizer. It errs high for JSON-heavy text
  and is deterministic. A different divisor would be a new estimator id (`bytes3-v2`).
- Without `budget_tokens`, every unit is returned subject to the existing caps,
  and the pack only reports what those caps cut.

## Evidence signals

`evidence[].signals` (search) lists the signals that contributed to a result.
Only these names are emitted:

| Signal | Meaning |
|---|---|
| `bm25` | Lexical match (Tantivy BM25). |
| `vector` | Similarity from the hashed-n-gram embedding. It is lexical and subword-based, not a learned semantic model. It matches identifiers that share characters, such as `RetryPolicy` and `retry_policy`, but not synonyms with no shared characters. |
| `path` | A query term appears in the file path. |
| `recent` | The file was committed recently (recency boost). |
| `symbol_exact` | A query identifier equals the name of a symbol exactly. |
| `symbol` | A query term matches part of a symbol name. |
| `changelog_demoted` | A changelog file; its score was reduced. |
| `doc_demoted` | A documentation file; its score was reduced. |

A result can carry several signals. Demotion signals show why a doc or changelog ranks low.

## Snapshot id

A snapshot id identifies exactly what the index was built from. It is a 16-character
hexadecimal hash (64-bit FNV). It is computed from:

1. The fixed domain tag `engram-snapshot-v1`.
2. The checked-out commit: the `HEAD` SHA, or the literal `no-git` when the repository is not
   a git repository or has no commits.
3. The parser/config version (`PARSER_CONFIG_VERSION`, currently `1`). A store
   written under a different version discards its symbols, imports and vectors
   and re-extracts them.
4. A manifest of every indexed file, sorted by path: the path and its content hash
   (FNV-1a of the file bytes).

Consequences:

- The same content at the same commit gives the same id, whatever the mtimes or the checkout location.
- Any content, path, or parser-version change gives a different id.
- File mtimes are excluded from the id. Internally a file counts as changed when
  its size, mtime, or content hash differs from the stored fingerprint, and a
  changed file's derived rows are discarded and re-extracted.
- The ignored directories are excluded from indexing and from the id: `.git`, `.engram`,
  `node_modules`, `target`, `dist`, `build`, `.venv`, `venv`, `__pycache__`, `.next`,
  `.cache`, `vendor`, `.idea`, `.vscode`. A file merely named like one of them is still indexed.

Use the id to check whether a remembered answer is still current: pass it back as
`snapshot` and look for `snapshot_mismatch`. The answer is always from the current
snapshot.

## Index state

The index is built in the background when the server starts. Each index-backed
call is gated on its state:

| State | When | Client sees |
|---|---|---|
| `Indexing` | The first index is still being built. | Tool error (`isError: true`): `Engram is still indexing this repository in the background (running for {N}s). Retry this tool call in a few seconds.` |
| `Ready` | The index is built and the server serves from it. | Normal results. |
| `Failed` | Indexing failed (an error, a panic, an unwritable cache, or an invalid `--repo`). | Tool error: `Engram failed to index the repository at {repo}: {error}. Engram keeps its cache in {repo}/.engram/engram.db, so the repository directory must be writable, and --repo must point to an existing, readable directory. Fix that and restart the Engram MCP server; calling this tool again will not help until then.` |

A server started without `--repo` that cannot resolve a repository returns this instead:
`Engram could not start: {error}. Pass --repo <path> with an existing, readable directory, then restart the Engram MCP server; calling this tool again will not help until then.`

Rebuilds after the first index are not a separate state. They happen lazily on
the next call. If a rebuild fails, the previous generation keeps serving and the
response carries an `index_warnings` entry: `rebuilding the index failed ({err});
results come from the previous snapshot and may be out of date`. The response
also gains `index_state` with `stale: true`.

Watcher problems are reported the same way, as `index_warnings` entries:

- `file watcher is not running ({err}); results may not reflect edits made since startup`
- `file watcher reported an error ({err}); some file changes may have been missed`
- `file watcher could not refresh {what} ({err}); results for it may be stale`
- `file watcher could not refresh {n} more paths`

`index_warnings` and `index_state` are absent when the answer is current. On a
restart, files edited while the server was stopped are re-fingerprinted before
the index becomes `Ready`.

## Supporting tools

- `find_existing_implementation(concept)`: candidates to reuse, with the
  recommendation "Review these candidates before implementing" or "No existing
  implementation found."
- `get_verification_plan(changed_files)`: merged checklist for the matched
  domain profiles, the repository's detected test commands, and
  `historically_co_failing_tests`. That key is a legacy name. It lists tests that
  historically changed together with the files, not tests that failed together.
- `get_review_history(path?, task?)`: raw, unsummarized reviewer comments, with
  `pr_number` and `pr_merged`, at most 10 per call. Pass `path`, `task`, or both.

## Roadmap / not implemented

Not part of this contract. Listed so the gaps are visible:

- `kinds` filter on `search_context` or `expand_connections`. No tool accepts it.
- Learned embeddings. The default embedder is hashed n-grams behind the `Embedder` trait.
- Compiler-level import resolution. Import edges stay heuristic candidates.
- Outcome feedback (`submit_outcome` and scoring from outcomes).
- GraphMesh snapshot core. It is on a separate branch and is not merged. The release-branch choice is pending.
- Full GitHub history. On `main`, `engram ingest-github` reads one page of at
  most 100 records per list, with no pagination: the PR list is limited by
  `--limit` (capped at 100), and each PR's files, commits and review comments are
  one page of up to 100. It is not a complete history.
- The target evidence shape. Search and expansion results do not carry a
  structured `reasons` array. They use `signals` (search) and the fields above
  (expansion), not `reason.type` values such as `exact_symbol` or `historical_cochange`.
