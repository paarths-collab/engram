# Engram product contract (v0.1)

This is the frozen public contract. Positioning, tool names, and output shapes
below are the stable surface; changes go through deprecation, not silent drift.

## Positioning

> **Engram is an evidence-based repository context and change-impact layer for
> coding agents.**

Engram **does**:

1. Locate exact files, symbols, tests, and definitions.
2. Expand from known files, paths, symbols, or diffs to connected code.
3. Explain every returned connection with the concrete fact behind it.
4. Return compact, ranked context to an agent.

Engram **does not** (yet) claim it can reliably predict affected code from a
vague natural-language task. That path exists (`predict_impact`) but is marked
**experimental** and is not part of the stable contract.

## Canonical tools

| Tool | Input | Purpose |
|---|---|---|
| `search_context` | `query: string` | Ranked, evidence-backed context for a query (hybrid retrieval + past reviews). |
| `expand_connections` | `paths: string[]` | Deterministic: files connected to known anchors by recorded facts (co-change + import graph). No guessing. |
| `explain_connection` | `source: string, target: string` | The concrete recorded edges linking two files (import direction, historical co-change), each weighted. Empty = no recorded connection. |

Supporting tools (stable): `find_existing_implementation`,
`get_verification_plan`, `get_review_history`.

## Deprecations

| Old | Status | Replacement |
|---|---|---|
| `get_task_context(task)` | deprecated alias | `search_context(query)` |
| `find_connected_files(files)` | deprecated alias | `expand_connections(paths)` |
| `predict_impact(task)` | experimental | `expand_connections` when anchors are known |

Aliases keep working (same handler, both argument names accepted) so no agent
breaks; they will be removed in a future release.

## Evidence shape (target)

Every returned candidate should carry structured, traceable evidence. Today
`search_context` returns a flat `signals: string[]`; `explain_connection`
already returns structured `reasons`. The **next contract step** is to converge
`search_context`/`expand_connections` results onto the same structured shape:

```json
{
  "path": "src/utils/merge.py",
  "score": 0.91,
  "reasons": [
    { "type": "exact_symbol",        "value": "merge_dicts",          "weight": 1.0  },
    { "type": "import_edge",         "source": "src/chains/base.py",  "weight": 0.6  },
    { "type": "historical_cochange", "commits": 4,                    "weight": 0.35 }
  ]
}
```

`reason.type` values: `exact_symbol`, `symbol_match`, `path_match`, `lexical`,
`semantic`, `import_edge`, `historical_cochange`, `recency`. This is the one
open item to close before calling the evidence contract fully frozen.

## Stability rules

- New tools may be added at any time.
- Canonical tool names and required arguments do not change except through a
  deprecation cycle (add alias → warn → remove).
- Output fields are additive; a field is not removed or repurposed without a
  deprecation cycle.
