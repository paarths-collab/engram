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
`get_verification_plan(changed_files)`, `get_review_history(path?, task?)`,
`get_graph(paths, max_hops?, kinds?, max_edges?, max_nodes?, snapshot?)` (see [`get_graph`](#get_graph)).

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
`explain_connection`, `get_graph`, `find_existing_implementation`,
`predict_impact` and the two aliases) carry these fields when they apply:

| Field | Present | Meaning |
|---|---|---|
| `snapshot` | always, on index-backed tools | Id of the index the answer was computed from (see [Snapshot id](#snapshot-id)). |
| `generation` | always, on index-backed tools | Integer. The index generation every fact in the answer was read at (see [Index generations](#index-generations)). A publish counter, not a content identity: unlike `snapshot` it says how many times the index has been published, and it does not repeat. |
| `snapshot_mismatch` | `search_context`, `expand_connections`, `get_graph` and aliases, when the `snapshot` argument differs from the current one | `{ "requested": "...", "current": "..." }`. The answer still comes from the current snapshot. |
| `index_warnings` | when results may be out of date | Array of plain-text warnings (see [Index state](#index-state)). |
| `index_state` | only when results may be stale | `{ "status": "ready", "stale": true, "generation": n, "built_generation": m, "snapshot": "..." }`. `generation` is the newest generation in the cache; `built_generation` (listed only when it differs) is the one the answer was read at, which is also the response's own `generation`. They differ exactly when a newer generation was published after the answer's facts were read; the next call rebuilds first. |
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

(The capture predates edge ids: current responses also carry `evidence_ids` and
`score_components` on each `cochange_expansions` and `import_expansions` entry,
for example `"evidence_ids": ["e_<16 hex digits>"]` and
`"score_components": {"basis": "cochange_strength", "strength": 0.8, "count": 8,
"anchor": "app/service.py"}`.)

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
- Recommendations cite their evidence (see [Edge ids](#edge-ids)). Each
  `cochange_expansions` and `import_expansions` entry carries:
  - `evidence_ids`: the ids of the recorded edges the entry rests on. For a
    co-change entry, the stored row from every anchor that kept it among its top
    5; for an import entry, one id per hop of the import route back to an anchor
    (a 2-hop dependent cites both of its single-hop edges). Sorted for
    co-change, in route order for imports.
  - `score_components`: the numbers `confidence` was taken from. Co-change:
    `{"basis": "cochange_strength", "strength", "count", "anchor"}` (the
    strongest cited row). Import: `{"basis": "import_hops", "hops", "weight"}`
    (0.6 for one hop, 0.35 for two).
  - A cited id is an id [`get_graph`](#get_graph) emits for the same anchors in
    the same snapshot, but the guarantee is conditional: the id is present only
    when the graph's `limits.truncated` is `false` **and** its `max_hops` is at
    least the hop the id was cited at. `get_graph` defaults to `max_hops` 2, the
    same as this tool, so at default arguments a 2-hop dependent's outer-hop id
    is listed; with `max_hops` 1 it is not. A cap can also cut a cited edge, and
    `limits.truncated` is then `true`.
    Anchors in `likely_files` and the entries of `predict_impact`'s search hits
    carry neither field. Search evidence has no `score_components`: its
    per-signal scores are not kept, only the fused `score` and the `signals` list.
  - Minimal records under a budget omit both fields; their `expansion_handle`
    still leads to `explain_connection`.

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
      "detail": "app/api.py and app/service.py changed together in 8 of the 8 non-bulk commits that touched app/api.py (100%)",
      "direction": "source_to_target",
      "edge_kind": "cochange",
      "opportunities_source": 8,
      "opportunities_target": 10,
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
    "bulk_commits_excluded": 0,
    "cochange_directions_searched": ["source_to_target", "target_to_source"],
    "cochange_lookup": "exact_pair",
    "cochange_window": { "max_commits": 2000, "max_files_per_commit": 30, "min_count": 2 },
    "commits_scanned": 11,
    "import_directions_searched": ["source_imports_target", "target_imports_source"],
    "import_max_hops": 2,
    "import_search_source_imports_target": "found",
    "import_search_target_imports_source": "none_within_bound",
    "note": "Import chains up to 2 hops were searched in both directions over heuristically resolved import edges. Co-change covers only the last 2000 commits, pairs sharing at least 2 commits, ignoring commits touching more than 30 files (counted in bulk_commits_excluded). Absence of a fact means none was recorded within this scope, not that the files are unrelated."
  },
  "snapshot": "2c4a4e25961017ed"
}
```

(The capture predates edge ids: current responses also carry `edge_id` on each
reason and inside each `edges[]` entry, see [Reasons](#reasons) and
[Edge ids](#edge-ids). The co-change `detail`, `opportunities_*`,
`commits_scanned`, `bulk_commits_excluded` and `note` shown here follow the
current wording and fields; they are not part of the recorded capture.)

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
  - `commits_scanned`: commits the co-change walk covered (merges and empty
    commits included). `null` on an index that has not recorded it, and also
    when the history walk failed (not a repository, no commits yet, `git`
    unavailable). Unknown is `null`, never `0`.
  - `bulk_commits_excluded`: commits in that walk that touched more than
    `cochange_window.max_files_per_commit` files. They feed neither a pair's
    `count` nor any file's `opportunities_*`. `null` under the same conditions.
  - `note`: a plain-language statement of the scope. It states the co-change
    window, that bulk commits are ignored (and counted in
    `bulk_commits_excluded`), and that absence of a fact is not proof that the
    files are unrelated.

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
| `edges` | one `{from, to, edge_id}` per hop, in route order; `edge_id` is the id of that single edge | not present |
| `edge_id` | id of the whole route (equal to its only hop's id for a direct import) | id of the stored row this fact came from |
| `resolution` | `heuristic` | not present |
| `resolver` | `heuristic_module_segments` | not present |
| `resolution_status` | `candidate`: a plausible edge, not compiler-verified | not present |
| `count` | not present | commits in which both files changed |
| `strength` | not present | `count` divided by the non-bulk commits that touched the row's first file |
| `opportunities_source` | not present | non-bulk commits in the window that touched `source` (the denominator of the `source_to_target` strength); `null` when not recorded, never a silent `0` |
| `opportunities_target` | not present | the same for `target` |
| `window` | not present | the co-change window (see `scope.cochange_window`) |

Co-change means the two files changed in the same commits. It does not mean
one must change with the other.

The co-change `detail` reads, for a row `from -> to`, `"<from> and <to> changed
together in 3 of the 4 non-bulk commits that touched <from> (75%)"`. The
denominator counts only non-bulk commits, the ones touching at most
`max_files_per_commit` files: bulk commits are excluded from it, so it is not
"all commits that touched `<from>`". The percentage is computed as an integer
from the two counts and rounded to nearest, so it always agrees with them. On
an index that did not record the denominator the text is `"... changed together
in 3 commits, 75% of the non-bulk commits that touched <from> (that commit
count is not recorded in this index)"`, with the percentage taken from
`strength`, rounded. (Before this wording the text said "N of M commits that
touched X", which overstated the denominator; clients that matched on that
sentence must read the structured fields `count` and `opportunities_*`
instead.)

Edge ids ([Edge ids](#edge-ids)) are additive on this tool: `edge_id` on each
reason, and `edge_id` inside each `edges[]` entry. A hop's id equals the id
[`get_graph`](#get_graph) emits for that single import edge, so the hops of a
2-hop chain are individually citeable. A co-change reason's id belongs to the
stored row it reports, not to the direction of the question: asking
`(a, b)` or `(b, a)` yields the same two ids.

The `direction` of a reason is relative to the question (`source_to_target` is
the row from the `source` you asked about to the `target`), so the same stored
row is labelled `source_to_target` when asked `(a, b)` and `target_to_source`
when asked `(b, a)`. [`get_graph`](#get_graph) has no question: its co-change
`direction` is always the stored row direction `source_to_target`, with
`source` and `target` being the row's own `path_a` and `path_b`. The same
`edge_id` can therefore carry different `direction` labels in the two tools; the
id identifies the stored fact, the label is a reading of it.

When two import routes of equal length connect the pair, the route reported is
the one through the alphabetically first intermediate file at each step, so the
same snapshot gives the same chain (and ids) whatever order the index stored
its rows in.

### Errors

Each of these is a tool error (`isError: true`) with a message in the text:

- Either path is not an indexed text file. The message asks for a repo-relative
  path returned by `search_context` or `expand_connections`.
- `source` and `target` are the same file.
- A required argument is missing or empty.

## Edge ids

Every graph fact has a content-derived id so a recommendation can cite exactly
the fact it rests on, and a client can check the citation.

```
edge_id = "e_" + 16 lowercase hex digits
        = FNV-1a-64 over  len-prefixed( "engram-edge-v1", snapshot, kind, source, target )
                          ‖ count ‖ len-prefixed( each of chain_or_direction )
```

- Every string is encoded as its byte length (`u64`, little-endian) followed by
  its UTF-8 bytes; `count` is the number of trailing items (`u64`, little-endian).
- `kind` is `import` or `cochange`.
- Import edge: `source` is the importer, `target` the imported file, and the
  trailing items are the full chain of paths in order. A single hop is the
  two-element chain `[source, target]`, so a hop has the same id whether it is
  cited alone (`get_graph`, `evidence_ids`) or as a step of a longer route
  (`explain_connection` `edges[]`). A whole 2-hop route has its own, different id.
- Co-change edge: the id belongs to the stored row `source -> target`, with the
  single trailing item `source_to_target`. The row `target -> source` is a
  different edge with a different id, whichever way a lookup was phrased.
- Stability: the same snapshot gives the same ids on any machine and on any run.
  The id includes the [snapshot id](#snapshot-id), so any content, path,
  commit or parser-version change gives every edge a new id. Use ids within one
  snapshot; pass the `snapshot` you cited from to detect that it moved.
- Collision caveat: the id is a 64-bit hash, not a unique key. Within one
  snapshot a collision is possible in principle (about one in 2^64 per pair of
  edges); ids are not unique across snapshots or repositories and must not be
  used as global identifiers. It is a fingerprint, not a security boundary.

## `get_graph`

A supporting tool: the raw recorded facts around some files. It lists edges; it
does not rank files, and it is independent of the recommendation caps of
`expand_connections`.

What it is not: **not a call graph**. The graph facts are heuristic import edges
(module-name matching, `resolution_status: "candidate"`, never compiler-verified)
and historical co-change (files that changed in the same commits). Symbol-level
call edges are not implemented.

Request:

```json
{ "name": "get_graph", "arguments": { "paths": ["app/service.py"] } }
```

| Argument | Meaning |
|---|---|
| `paths: string[]` | Anchor files (required, non-empty). Each must be an indexed file, else a tool error in the style of `explain_connection`; this differs from `expand_connections`, which silently ignores a path it does not know. More distinct paths than `max_nodes` is an error, refused before any path is looked up. Order and repeats do not matter. |
| `max_hops?: 1..2` (default 2) | Import distance followed from the anchors. 1 returns every import edge touching an anchor; 2 also returns the edges touching those neighbours. The default matches `expand_connections`, so what an expansion cites is listed (see the citation condition under [`expand_connections`](#expand_connections)). Co-change edges are always anchor-to-partner. Outside 1..2 is an error. |
| `kinds?: ("import" \| "cochange")[]` | Which kinds to return; default both. Empty or unknown values are an error. |
| `max_edges?: 1..500` (default 500) | Return at most this many edges. It can only lower the hard cap. An integer outside 1..500, or a non-integer, is an error. |
| `max_nodes?: 1..200` (default 200) | Return at most this many nodes (anchors always stay, so it must be at least the number of distinct anchors). It can only lower the hard cap. An integer outside 1..200, or a non-integer, is an error. |
| `snapshot?: string` | As elsewhere: a different current snapshot is reported in `snapshot_mismatch`; the answer is always from the current one. |

There is no `budget_tokens`: the hard caps below are the bound.

Response (from the end-to-end test `get_graph_returns_the_api_service_facts_under_the_ids_explain_reports`
on the standard fixture; trimmed to two nodes and two of the edges):

```json
{
  "snapshot": "0a2d2ba23ebbd0db",
  "anchors": [
    "app/service.py"
  ],
  "nodes": [
    {
      "distance": 1,
      "is_test": false,
      "language": "python",
      "path": "app/api.py",
      "role": "neighbor"
    },
    {
      "distance": 0,
      "is_test": false,
      "language": "python",
      "path": "app/service.py",
      "role": "anchor"
    }
  ],
  "edges": [
    {
      "direction": "source_imports_target",
      "edge_id": "e_7d78b9c1bb72fbfe",
      "hops": 1,
      "kind": "import",
      "resolution_status": "candidate",
      "resolver": "heuristic_module_segments",
      "source": "app/api.py",
      "target": "app/service.py"
    },
    {
      "count": 8,
      "direction": "source_to_target",
      "edge_id": "e_ce59056d1a703ec0",
      "kind": "cochange",
      "opportunities_source": 8,
      "opportunities_target": 10,
      "source": "app/api.py",
      "strength": 1.0,
      "target": "app/service.py",
      "window": {
        "max_commits": 2000,
        "max_files_per_commit": 30,
        "min_count": 2
      }
    }
  ],
  "limits": {
    "cochange_per_anchor": 25,
    "max_edges": 500,
    "max_nodes": 200,
    "omitted_edges": 0,
    "omitted_nodes": 0,
    "reason": "complete: no cap cut anything",
    "truncated": false
  },
  "scope": {
    "bulk_commits_excluded": 0,
    "cochange_window": {
      "max_commits": 2000,
      "max_files_per_commit": 30,
      "min_count": 2
    },
    "commits_scanned": 11,
    "import_max_hops": 2,
    "kinds": [
      "import",
      "cochange"
    ],
    "note": "Facts only, not a ranking and not a call graph: import edges are heuristic module-name matches (single hop, either direction, from the anchors out to 2 hop(s)); co-change edges are stored git-history rows between each anchor and its strongest partners (last 2000 commits, pairs sharing at least 2 commits, commits touching more than 30 files ignored). Symbol-level call edges are not recorded. Absence of an edge means none was recorded within this scope."
  }
}
```

- `nodes`: the anchors (`role: "anchor"`, `distance` 0) and neighbours
  (`role: "neighbor"`; `distance` is the import hops from the nearest anchor,
  1 for co-change partners). `language` is absent for unsupported extensions.
  Sorted by `path`.
- `edges`, sorted by (`kind`, `source`, `target`, `direction`), each once:
  - `import`: `source` imports `target`, one hop (`hops: 1`), found in either
    direction around an anchor; `direction` is always `source_imports_target`.
    `resolver` is `heuristic_module_segments`, `resolution_status` is `candidate`.
  - `cochange`: the stored row `source -> target`: `count`, `strength` (relative
    to commits touching `source`), the build `window`, and
    `opportunities_source` / `opportunities_target` (non-bulk commits in the
    window that touched each file; `null` on an index that predates them or
    whose history walk failed, never a silent zero). For each partner of an anchor both rows are listed when both exist;
    `direction` is always the stored row direction `source_to_target`, never
    relative to a question (see [Reasons](#reasons) for how
    `explain_connection` labels the same row).
  - Import and co-change facts about one pair are separate edges.
- `limits`: `max_nodes` (200, or the lower value asked for), `max_edges` (500,
  or lower), `cochange_per_anchor` 25 (partners read per anchor, strongest
  first by strength, then count, then path; the partner's row back is read with
  its partner). `truncated` is `true` when anything was cut.
  - `omitted_edges` is exact. The unit is one edge: one directed co-change row
    or one import edge. Emitted `edges` plus `omitted_edges` equals the number
    of edges the same call returns with every cap removed. It counts co-change
    rows past the per-anchor limit (both the hidden partners' own rows and their
    rows back, since the stored table holds each pair in both directions), edges
    touching a file past the node cap, and edges past the edge cap.
  - `omitted_nodes` counts neighbouring files past the node cap. A co-change
    partner hidden by the per-anchor limit is never added as a node, so it is
    not counted in `omitted_nodes`; its rows are counted in `omitted_edges`.
  - `reason` says which caps cut what. Anchors are always kept; neighbours are
    kept nearest first then by path; edges nearest an anchor first (co-change
    rows, then imports, ties in output order). Nothing is cut silently.
  - The work is bounded as well as the answer: import edges are counted as they
    are walked, but only the edges that survive `max_edges` are built, so a file
    with thousands of importers costs a count, not thousands of edges.
- `scope`: what was searched (kinds, `import_max_hops`, the co-change window,
  `commits_scanned`, `bulk_commits_excluded`, and a note). A missing edge means
  none was recorded within it.
- Co-change partners that are not indexed files are not listed and not counted.
- Deterministic: the same snapshot and arguments give identical bytes.
- Carries `snapshot`, and `index_warnings` / `index_state` when results may be stale.

Ids in `evidence_ids` ([`expand_connections`](#expand_connections)) and on
`explain_connection` reasons are drawn from the same function, so for one
snapshot they can be looked up here. They are guaranteed to be listed only when
`limits.truncated` is `false` and `max_hops` is at least the hop the id was
cited at (a 2-hop dependent cites a second-hop edge that a `max_hops` 1 graph
does not hold).

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
3. The inventory rules fingerprint: a hash of the ignored directory names, the
   extension-to-language table, the extra document suffixes (`.md`, `.toml`,
   `.json`), the file size cap, and the hand-maintained parser/config version
   (`PARSER_CONFIG_VERSION`, currently `1`). Editing any of these rules changes
   the fingerprint with no manual bump. A store written under a different
   fingerprint discards its symbols, imports and vectors and re-extracts them,
   so the snapshot id changes with it. The check and the reset run under one
   write lock and re-read the stored value inside it, so two processes opening
   the store at the same moment cannot wipe each other's fresh data.
4. A manifest of every indexed file, sorted by path: the path and its content hash
   (FNV-1a of the file bytes).

Consequences:

- The same content at the same commit gives the same id, whatever the mtimes or the checkout location.
- Any content, path, inventory-rule, or parser-version change gives a different id.
- **Upgrade note:** the id is now derived from the rules fingerprint rather than
  the bare parser version, so every existing snapshot id changes exactly once
  on upgrade (a database holding the old value is re-extracted on first open).
  Ids recorded before the upgrade, including edge ids, will report
  `snapshot_mismatch`.
- File mtimes are excluded from the id. Internally a file counts as changed when
  its size, mtime, or content hash differs from the stored fingerprint, and a
  changed file's derived rows are discarded and re-extracted.
- The ignored directories are excluded from indexing and from the id: `.git`, `.engram`,
  `node_modules`, `target`, `dist`, `build`, `.venv`, `venv`, `__pycache__`, `.next`,
  `.cache`, `vendor`, `.idea`, `.vscode`. A file merely named like one of them is still indexed.

Use the id to check whether a remembered answer is still current: pass it back as
`snapshot` and look for `snapshot_mismatch`. The answer is always from the current
snapshot.

## Index generations

A **generation** is one atomically published state of the index. The cache keeps
a counter (`meta.generation`, an integer that starts at 0 and only goes up), and
every response from an index-backed tool carries the `generation` its facts were
read at, next to its `snapshot`.

The two answer different questions:

| | `snapshot` | `generation` |
|---|---|---|
| What it is | A hash of the indexed content (commit, rules, every file's path and content hash). | A counter of publishes to the cache. |
| Same content twice | Same id, on any machine. | Different generations, if anything was published between. |
| A publish that changes nothing | Same id. | Same generation (a no-op batch does not bump). |
| Use it to | Cite what an answer was computed from; `edge_id`s derive from it. | Tell whether the cache has moved past an answer, and order two answers from one cache. |

Generation numbers are local to one cache directory; they say nothing across
machines or after deleting `.engram`.

### What a publish is

Every change to the indexed facts follows the same two steps. The change is
**staged** first: files are read and parsed, and git history is walked, into an
in-memory batch, with no write transaction open (so a slow parse never blocks a
reader or another writer). It is then **published** by one call that applies the
whole batch in a single SQLite `IMMEDIATE` transaction: file fingerprints, the
removal of the derived rows of changed files, symbols, imports, removals of
deleted files, co-change rows and recency, and the `tier1_done` marks that go
with the symbols they describe. The generation is bumped by one inside that same
transaction, and only if some fact actually changed.

Guarantees:

- **Atomic.** An error, a panic or a crash before the commit leaves the previous
  generation exactly as it was: no half-applied file set, no new symbols beside
  old imports. A reader sees all of a publish or none of it.
- **One generation per unit of work.** A restart's whole reconcile (files edited,
  added and deleted while the server was stopped, the refreshed history, and the
  eager extraction) is one generation, and the server is `Ready` only after it
  commits, so the stale facts of an edited or deleted file are gone before the
  first answer. A burst of file events (events within a 75 ms quiet window, up
  to 2000 events or one second) is one generation. A file that is lazily
  extracted on a search hit is a generation of its own, but only if it gained
  symbols or imports.
- **Consistent reads.** An engine is built from one read transaction (a WAL
  snapshot): the file list, symbols, import edges, co-change edges, recency,
  hashes, snapshot id and cached chunk vectors are all of the generation the
  engine reports. Queries that also read the store live (the co-change rows of
  `explain_connection`, `get_graph` and `expand_connections`; the symbol lookups
  of `search_context`) read inside one transaction too.
- **Never silently mixed.** If the store has moved past an engine's generation
  when a query runs, the query is answered from the engine's own facts where it
  can be (a search skips its lazy extraction and its live symbol lookups),
  and the response is flagged: `index_state.stale` is `true` with `generation`
  (newest) and `built_generation` (the answer's). The next call rebuilds the
  engine first. If that rebuild fails, the previous engine keeps serving, still
  flagged, with the `index_warnings` entry described under [Index state](#index-state).
- **The disk can be ahead of the index.** An edit the watcher has not published
  yet is not a generation. While building an engine, a file whose content no
  longer matches the hash its symbols were extracted from has its symbols left
  out (their line spans would point at the wrong lines), and a warning says how
  many files; the watcher's publish replaces the engine and clears it.

Limits, stated plainly:

- A full reconcile is one generation up to 5000 eagerly extracted files (the
  server's own eager limit is 3000). Beyond that its extractions are published
  in further generations of up to 5000 each, to bound memory. The first carries
  the inventory, the removals and the history; every one is atomic by itself,
  but the reconcile as a whole is then not.
- Two writers publish one after the other, never merged. A staged extraction is
  a snapshot of the disk when it was read: if the same file changes between
  staging and publish, the next scan or watcher event corrects it, as before.
- Rebuilding the in-memory engine after a publish still costs a pass over the
  indexed files (cached embeddings make it cheap, not free); a publish does not
  patch the engine in place.
- Embeddings computed while building an engine are cache (`vectors`,
  `chunk_vectors`), written back after the build and dropped when a file changes.
  They are not part of the generation and do not move it.
- Pull-request and review data ingested from GitHub lives in its own tables and
  is not part of the generation.
- A watcher whose publish fails leaves the previous generation intact and is
  reported in `index_warnings` as stale; a restart reconciles the whole
  inventory again in one generation.

## Local cache and concurrency

The index lives in `<repo>/.engram/engram.db` (SQLite, write-ahead logging, so
`engram.db-wal` and `engram.db-shm` appear beside it while it is open; the whole
`.engram` directory is excluded from indexing and from the snapshot id). The
background indexer, the file watcher and request handlers each hold their own
connection to that file. Every connection waits up to 10 seconds for another
writer's lock before failing, and write transactions take the lock up front, so
startup indexing racing a watcher event or a first query does not surface
`database is locked`. Only a lock held for longer than that fails, as an
ordinary tool error or index warning.

Writers hold that lock only for the publish itself: parsing and the git walk
happen first, in memory, and the whole result is applied in one short
transaction (see [Index generations](#index-generations)). Readers never take
it. An engine is built from a single read transaction, which in WAL mode is a
snapshot, so a publish by the watcher while an engine is being built can neither
block the build nor leak into it half-way.

A failed history walk (the directory is not a git repository, it has no commits
yet, or `git` cannot be run) stores no co-change rows and leaves the
completeness facts unknown (`null` in `commits_scanned`, `bulk_commits_excluded`
and `opportunities_*`), not zero.

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
- `get_graph(paths, max_hops?, kinds?, max_edges?, max_nodes?, snapshot?)`: raw graph
  facts with citable edge ids; see [`get_graph`](#get_graph).

## Roadmap / not implemented

Not part of this contract. Listed so the gaps are visible:

- `kinds` filter on `search_context` or `expand_connections`. No tool accepts it.
- Learned embeddings. The default embedder is hashed n-grams behind the `Embedder` trait.
- Compiler-level import resolution. Import edges stay heuristic candidates.
- Symbol-level call edges. `get_graph` is not a call graph: it holds only
  heuristic import edges and historical co-change.
- Outcome feedback (`submit_outcome` and scoring from outcomes).
- GraphMesh snapshot core. It is on a separate branch and is not merged. The release-branch choice is pending.
- Full GitHub history. On `main`, `engram ingest-github` reads one page of at
  most 100 records per list, with no pagination: the PR list is limited by
  `--limit` (capped at 100), and each PR's files, commits and review comments are
  one page of up to 100. It is not a complete history.
- The target evidence shape. Search and expansion results do not carry a
  structured `reasons` array. They use `signals` (search) and the fields above
  (expansion), not `reason.type` values such as `exact_symbol` or `historical_cochange`.
