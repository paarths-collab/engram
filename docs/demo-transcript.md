# Engram demo transcript

Recorded fixture output: this is not mocked. It is the captured JSON-RPC session of a real MCP client (`crates/mcp-server/tests/e2e_canonical.rs`, test `canonical_flow_search_expand_explain`) driving the compiled `engram` binary over stdio against a small real git repository built for the test.

- Engram commit: `0a0b1d3`
- Fixture: a Python repository with 11 commits (fixed identities and dates)
- Regenerate: run the test with `ENGRAM_E2E_TRANSCRIPT=<path>` and `ENGRAM_E2E_COMMIT=<short sha>` set. Timings are from one run on a shared machine (a Docker container), not a benchmark.

## Fixture history

| # | Commit | Files |
|---|---|---|
| 1 | Add order service skeleton | `app/api.py`, `app/service.py`, `app/repo.py`, `tests/test_service.py`, `app/audit.py`, `app/metrics.py`, `app/config.py`, `app/cache.py`, `app/schemas.py` |
| 2 | Audit and meter order runs | `app/api.py`, `app/service.py`, `app/audit.py`, `app/metrics.py` |
| 3 | Cache configured lookups | `app/api.py`, `app/service.py`, `app/config.py`, `app/cache.py` |
| 4 | Validate order payloads | `app/api.py`, `app/service.py`, `app/schemas.py`, `app/audit.py` |
| 5 | Meter configured runs | `app/api.py`, `app/service.py`, `app/metrics.py`, `app/config.py` |
| 6 | Cache validated payloads | `app/api.py`, `app/service.py`, `app/cache.py`, `app/schemas.py` |
| 7 | Audit configured runs | `app/api.py`, `app/service.py`, `app/audit.py`, `app/config.py` |
| 8 | Move order lookup into the repo layer | `app/service.py`, `app/repo.py` |
| 9 | Handle missing orders in the repo layer | `app/service.py`, `app/repo.py` |
| 10 | Cover metered cached runs | `app/api.py`, `app/service.py`, `app/metrics.py`, `app/cache.py`, `tests/test_service.py` |
| 11 | Tune the repo layer | `app/repo.py` |

`app/api.py` imports `app/service.py`, which imports `app/repo.py`; `app/api.py` does not import `app/repo.py`. `tests/test_service.py` imports `app/service.py`.

## Timings

| Step | Time |
|---|---|
| Cold start: process spawn to first answer (background index + first engine build, 6 polls) | 651 ms |
| initialize | 3 ms |
| tools/list | 1 ms |
| readiness probe | 140 ms |
| search_context | 10 ms |
| expand_connections | 1 ms |
| explain_connection | 1 ms |

## Session

Requests are shown as the `params` of the JSON-RPC call. For `tools/call`, the response is the tool's JSON payload exactly as returned in `result.content[0].text`.

### 1. initialize

`initialize`

Request:

```json
{
  "capabilities": {},
  "clientInfo": {
    "name": "engram-e2e",
    "version": "0.0.0"
  },
  "protocolVersion": "2024-11-05"
}
```

Response:

```json
{
  "capabilities": {
    "tools": {}
  },
  "protocolVersion": "2024-11-05",
  "serverInfo": {
    "name": "engram",
    "version": "0.1.0"
  }
}
```

### 2. tools/list

`tools/list`

Request:

```json
{}
```

Response:

```json
{
  "tools": [
    {
      "description": "ALWAYS call this FIRST, before planning or implementing any coding task. Returns ranked, evidence-backed context about this repository: existing code relevant to the query, symbols to reuse instead of reimplementing, related tests, and matched files (hybrid BM25+vector+symbol retrieval), plus any past reviewer comments on the matched files. Using this prevents duplicate implementations and wrong assumptions about the codebase.",
      "inputSchema": {
        "properties": {
          "query": {
            "description": "What you are looking for, in natural language, e.g. 'add retry logic to billing webhooks'",
            "type": "string"
          }
        },
        "required": [
          "query"
        ],
        "type": "object"
      },
      "name": "search_context"
    },
    {
      "description": "DEPRECATED alias of search_context (use `query` instead of `task`). Kept for compatibility; will be removed in a future release.",
      "inputSchema": {
        "properties": {
          "task": {
            "description": "The task description in natural language",
            "type": "string"
          }
        },
        "required": [
          "task"
        ],
        "type": "object"
      },
      "name": "get_task_context"
    },
    {
      "description": "Call this BEFORE writing any new function, class, service, or utility. Checks whether an implementation of this concept already exists in the repository so it can be reused instead of duplicated. Pass the concept you are about to implement.",
      "inputSchema": {
        "properties": {
          "concept": {
            "description": "What you are about to implement, e.g. 'audit log writer' or 'exponential backoff retry'",
            "type": "string"
          }
        },
        "required": [
          "concept"
        ],
        "type": "object"
      },
      "name": "find_existing_implementation"
    },
    {
      "description": "Call this when you already know which file(s) you are changing (e.g. the current diff) and want everything CONNECTED to them, with zero guessing. Takes exact file paths as anchors and returns only files linked by hard, deterministic facts already recorded in the store: the co-change graph (files that historically changed together in the same commits) and the import graph (files that statically import an anchor, up to 2 hops). Every result traces to a concrete recorded edge — this is fact-finding, not prediction. Prefer this over predict_impact whenever you have concrete anchor files.",
      "inputSchema": {
        "properties": {
          "paths": {
            "description": "Repo-relative paths of files you are changing or about to change, e.g. ['src/billing/cancel.rs']",
            "items": {
              "type": "string"
            },
            "type": "array"
          }
        },
        "required": [
          "paths"
        ],
        "type": "object"
      },
      "name": "expand_connections"
    },
    {
      "description": "Call this to see WHY two specific files are connected, from recorded facts only. A direct pairwise lookup (not a ranked or capped recommendation): returns every recorded edge linking them. Import edges: source imports target and/or target imports source, directly or through one intermediate file (2 hops, with the full `chain` of paths); import edges come from heuristic module-name resolution, not compiler resolution (`resolution: \"heuristic\"`). Historical co-change: each direction found, with commit `count` and `strength`. Each reason has `type`, `detail`, `weight` plus typed fields (imports: `edge_kind`, `direction`, `hops`, `chain`, per-hop `edges`, `resolver`, `resolution_status: \"candidate\"`; co-change: `edge_kind`, `direction`, `count`, `strength`, `window`); import and co-change are separate facts and neither implies the other (co-change means files changed together, not that one must change with the other). `status` is `found`, `not_found_within_scope`, or `unknown_beyond_bound` (no chain within the hop bound but imports continue past it); `scope` states exactly what was searched, so `connected: false` means no recorded fact within that scope, never proof the files are unrelated. Returns an error if either path is not an indexed file or both paths are the same; use repo-relative paths from search_context/expand_connections. Never a guess.",
      "inputSchema": {
        "properties": {
          "source": {
            "description": "Repo-relative path of the first file",
            "type": "string"
          },
          "target": {
            "description": "Repo-relative path of the second file",
            "type": "string"
          }
        },
        "required": [
          "source",
          "target"
        ],
        "type": "object"
      },
      "name": "explain_connection"
    },
    {
      "description": "DEPRECATED alias of expand_connections (use `paths` instead of `files`). Kept for compatibility; will be removed in a future release.",
      "inputSchema": {
        "properties": {
          "files": {
            "description": "Repo-relative paths of files you are changing",
            "items": {
              "type": "string"
            },
            "type": "array"
          }
        },
        "required": [
          "files"
        ],
        "type": "object"
      },
      "name": "find_connected_files"
    },
    {
      "description": "EXPERIMENTAL. Predicts which files a fuzzy natural-language task will likely affect (text retrieval + co-change/import expansion). Prefer expand_connections when you have concrete anchor files — prediction from vague tasks is not yet reliable.",
      "inputSchema": {
        "properties": {
          "task": {
            "description": "The task description",
            "type": "string"
          }
        },
        "required": [
          "task"
        ],
        "type": "object"
      },
      "name": "predict_impact"
    },
    {
      "description": "Call this AFTER making changes and BEFORE opening a PR. Given the list of changed files, returns the merged verification checklist for the engineering domains they touch (backend/frontend/database/infra), the repo's detected test commands, and tests that historically change together with these files. Run these checks before considering the change done.",
      "inputSchema": {
        "properties": {
          "changed_files": {
            "description": "Repo-relative paths of the files changed, e.g. ['src/billing/cancel.rs', 'migrations/003_add.sql']",
            "items": {
              "type": "string"
            },
            "type": "array"
          }
        },
        "required": [
          "changed_files"
        ],
        "type": "object"
      },
      "name": "get_verification_plan"
    },
    {
      "description": "Call this before changing a file to see what human reviewers said about it before. Returns RAW, unsummarized reviewer comments from past pull requests, with the PR number and whether that PR was merged (accepted). Pass a file 'path' to get comments on that file, and/or a 'task' to rank comments by relevance. Reading these avoids repeating mistakes reviewers already flagged.",
      "inputSchema": {
        "properties": {
          "path": {
            "description": "Repo-relative file path to fetch review comments for",
            "type": "string"
          },
          "task": {
            "description": "Task/concept to rank comments by relevance",
            "type": "string"
          }
        },
        "type": "object"
      },
      "name": "get_review_history"
    }
  ]
}
```

### 3. readiness probe

`tools/call:search_context`

Request:

```json
{
  "arguments": {
    "query": "execute"
  },
  "name": "search_context"
}
```

Response:

```json
{
  "evidence": [
    {
      "id": "ev_000",
      "path": "tests/test_service.py",
      "score": 1.7601288557052612,
      "signals": [
        "bm25",
        "vector",
        "recent"
      ],
      "snippet": "def test_execute_returns_ok():\n    assert execute({\"id\": 1})[\"status\"] == \"ok\"",
      "symbol": "test_execute_returns_ok",
      "title": "test_execute_returns_ok in tests/test_service.py:4",
      "type": "test"
    },
    {
      "id": "ev_001",
      "path": "app/service.py",
      "score": 1.5314418077468872,
      "signals": [
        "bm25",
        "vector",
        "recent"
      ],
      "snippet": "def execute(order):\n    \"\"\"Execute an order: load its record and return a status envelope.\"\"\"\n    record = load(order[\"id\"])\n    return {\"status\": \"ok\", \"record\": record}",
      "symbol": "execute",
      "title": "execute in app/service.py:5",
      "type": "symbol"
    },
    {
      "id": "ev_002",
      "path": "app/api.py",
      "score": 1.1218748092651367,
      "signals": [
        "bm25",
        "vector",
        "recent"
      ],
      "snippet": "def handle_run(payload):\n    \"\"\"Route handler for POST /orders/run; delegates to the service.\"\"\"\n    return execute(payload)",
      "symbol": "handle_run",
      "title": "handle_run in app/api.py:7",
      "type": "symbol"
    },
    {
      "id": "ev_003",
      "path": "app/metrics.py",
      "score": 0.11038585752248764,
      "signals": [
        "vector",
        "recent"
      ],
      "snippet": "def count_run(name):\n    return {\"counted\": name}",
      "symbol": "count_run",
      "title": "count_run in app/metrics.py:1",
      "type": "symbol"
    },
    {
      "id": "ev_004",
      "path": "app/config.py",
      "score": 0.09618426859378816,
      "signals": [
        "vector",
        "recent"
      ],
      "snippet": "def read_setting(key):\n    return {\"setting\": key}",
      "symbol": "read_setting",
      "title": "read_setting in app/config.py:1",
      "type": "symbol"
    },
    {
      "id": "ev_005",
      "path": "app/audit.py",
      "score": 0.0959389954805374,
      "signals": [
        "vector",
        "recent"
      ],
      "snippet": "def record_audit(entry):\n    return {\"audited\": entry}",
      "symbol": "record_audit",
      "title": "record_audit in app/audit.py:1",
      "type": "symbol"
    },
    {
      "id": "ev_006",
      "path": "app/cache.py",
      "score": 0.09161258488893508,
      "signals": [
        "vector",
        "recent"
      ],
      "snippet": "def cached_lookup(key):\n    return {\"cached\": key}",
      "symbol": "cached_lookup",
      "title": "cached_lookup in app/cache.py:1",
      "type": "symbol"
    },
    {
      "id": "ev_007",
      "path": "app/schemas.py",
      "score": 0.08689452707767487,
      "signals": [
        "vector",
        "recent"
      ],
      "snippet": "def validate_payload(payload):\n    return bool(payload)",
      "symbol": "validate_payload",
      "title": "validate_payload in app/schemas.py:1",
      "type": "symbol"
    }
  ],
  "past_reviews": [],
  "snapshot": "2c4a4e25961017ed",
  "task": "execute"
}
```

### 4. search_context

`tools/call:search_context`

Request:

```json
{
  "arguments": {
    "query": "execute order"
  },
  "name": "search_context"
}
```

Response:

```json
{
  "evidence": [
    {
      "id": "ev_000",
      "path": "app/service.py",
      "score": 2.184487819671631,
      "signals": [
        "bm25",
        "vector",
        "recent"
      ],
      "snippet": "def execute(order):\n    \"\"\"Execute an order: load its record and return a status envelope.\"\"\"\n    record = load(order[\"id\"])\n    return {\"status\": \"ok\", \"record\": record}",
      "symbol": "execute",
      "title": "execute in app/service.py:5",
      "type": "symbol"
    },
    {
      "id": "ev_001",
      "path": "tests/test_service.py",
      "score": 1.4811389446258545,
      "signals": [
        "bm25",
        "vector",
        "recent"
      ],
      "snippet": "def test_execute_returns_ok():\n    assert execute({\"id\": 1})[\"status\"] == \"ok\"",
      "symbol": "test_execute_returns_ok",
      "title": "test_execute_returns_ok in tests/test_service.py:4",
      "type": "test"
    },
    {
      "id": "ev_002",
      "path": "app/api.py",
      "score": 1.3956767320632937,
      "signals": [
        "bm25",
        "vector",
        "recent"
      ],
      "snippet": "\"\"\"HTTP entrypoint: routes incoming requests to the service layer.\"\"\"\nfrom app.service import execute\n\nROUTES = {\"/orders/run\": \"handle_run\"}\n\n\ndef handle_run(payload):\n    \"\"\"Route handler for POST /orders/run; delegates to the service.\"\"\"\n    return execute(payload)\n# change 1: Audit and meter order runs\n# change 2: Cache configured lookups\n# change 3: Validate order payloads\n# change 4: Meter c",
      "title": "app/api.py",
      "type": "existing_code"
    },
    {
      "id": "ev_003",
      "path": "app/repo.py",
      "score": 1.257183313369751,
      "signals": [
        "bm25",
        "vector",
        "recent"
      ],
      "snippet": "def load(order_id):\n    \"\"\"Load an order record by id.\"\"\"\n    return _STORE.get(order_id)",
      "symbol": "load",
      "title": "load in app/repo.py:6",
      "type": "symbol"
    },
    {
      "id": "ev_004",
      "path": "app/audit.py",
      "score": 0.739762544631958,
      "signals": [
        "bm25",
        "vector",
        "recent"
      ],
      "snippet": "def record_audit(entry):\n    return {\"audited\": entry}\n# change 1: Audit and meter order runs\n# change 3: Validate order payloads\n# change 6: Audit configured runs\n",
      "title": "app/audit.py",
      "type": "existing_code"
    },
    {
      "id": "ev_005",
      "path": "app/schemas.py",
      "score": 0.5185495018959045,
      "signals": [
        "bm25",
        "vector",
        "recent"
      ],
      "snippet": "def validate_payload(payload):\n    return bool(payload)\n# change 3: Validate order payloads\n# change 5: Cache validated payloads\n",
      "title": "app/schemas.py",
      "type": "existing_code"
    },
    {
      "id": "ev_006",
      "path": "app/metrics.py",
      "score": 0.4975529909133911,
      "signals": [
        "bm25",
        "vector",
        "recent"
      ],
      "snippet": "def count_run(name):\n    return {\"counted\": name}\n# change 1: Audit and meter order runs\n# change 4: Meter configured runs\n# change 9: Cover metered cached runs\n",
      "title": "app/metrics.py",
      "type": "existing_code"
    },
    {
      "id": "ev_007",
      "path": "app/config.py",
      "score": 0.06614868342876434,
      "signals": [
        "vector",
        "recent"
      ],
      "snippet": "def read_setting(key):\n    return {\"setting\": key}",
      "symbol": "read_setting",
      "title": "read_setting in app/config.py:1",
      "type": "symbol"
    }
  ],
  "past_reviews": [],
  "snapshot": "2c4a4e25961017ed",
  "task": "execute order"
}
```

### 5. expand_connections

`tools/call:expand_connections`

Request:

```json
{
  "arguments": {
    "paths": [
      "app/service.py"
    ]
  },
  "name": "expand_connections"
}
```

Response:

```json
{
  "cochange_expansions": [
    {
      "confidence": 0.800000011920929,
      "path": "app/api.py",
      "reason": "changes with app/service.py in 80% of its commits"
    },
    {
      "confidence": 0.4000000059604645,
      "path": "app/audit.py",
      "reason": "changes with app/service.py in 40% of its commits"
    },
    {
      "confidence": 0.4000000059604645,
      "path": "app/metrics.py",
      "reason": "changes with app/service.py in 40% of its commits"
    },
    {
      "confidence": 0.4000000059604645,
      "path": "app/config.py",
      "reason": "changes with app/service.py in 40% of its commits"
    },
    {
      "confidence": 0.4000000059604645,
      "path": "app/cache.py",
      "reason": "changes with app/service.py in 40% of its commits"
    }
  ],
  "import_expansions": [
    {
      "confidence": 0.6000000238418579,
      "path": "tests/test_service.py",
      "reason": "imports a likely-changed file (1 hop)"
    }
  ],
  "likely_files": [
    {
      "confidence": 1.0,
      "path": "app/service.py",
      "reason": "given as an anchor file"
    }
  ],
  "likely_tests": [
    "tests/test_service.py"
  ],
  "snapshot": "2c4a4e25961017ed"
}
```

### 6. explain_connection

`tools/call:explain_connection`

Request:

```json
{
  "arguments": {
    "source": "app/api.py",
    "target": "app/service.py"
  },
  "name": "explain_connection"
}
```

Response:

```json
{
  "connected": true,
  "reasons": [
    {
      "count": 8,
      "detail": "app/api.py and app/service.py changed together in 8 commits, 100% of the commits that touched app/api.py",
      "direction": "source_to_target",
      "edge_kind": "cochange",
      "strength": 1.0,
      "type": "historical_cochange",
      "weight": 1.0,
      "window": {
        "max_commits": 2000,
        "max_files_per_commit": 30,
        "min_count": 2
      }
    },
    {
      "count": 8,
      "detail": "app/service.py and app/api.py changed together in 8 commits, 80% of the commits that touched app/service.py",
      "direction": "target_to_source",
      "edge_kind": "cochange",
      "strength": 0.800000011920929,
      "type": "historical_cochange",
      "weight": 0.800000011920929,
      "window": {
        "max_commits": 2000,
        "max_files_per_commit": 30,
        "min_count": 2
      }
    },
    {
      "chain": [
        "app/api.py",
        "app/service.py"
      ],
      "detail": "app/api.py imports app/service.py directly (heuristic module-name resolution, not compiler resolution)",
      "direction": "source_imports_target",
      "edge_kind": "import",
      "edges": [
        {
          "from": "app/api.py",
          "to": "app/service.py"
        }
      ],
      "hops": 1,
      "resolution": "heuristic",
      "resolution_status": "candidate",
      "resolver": "heuristic_module_segments",
      "type": "import_edge",
      "weight": 0.6000000238418579
    }
  ],
  "scope": {
    "cochange_directions_searched": [
      "source_to_target",
      "target_to_source"
    ],
    "cochange_lookup": "exact_pair",
    "cochange_window": {
      "max_commits": 2000,
      "max_files_per_commit": 30,
      "min_count": 2
    },
    "import_directions_searched": [
      "source_imports_target",
      "target_imports_source"
    ],
    "import_max_hops": 2,
    "import_search_source_imports_target": "found",
    "import_search_target_imports_source": "none_within_bound",
    "note": "Import chains up to 2 hops were searched in both directions over heuristically resolved import edges. Co-change covers only the last 2000 commits, pairs sharing at least 2 commits, ignoring commits touching more than 30 files. Absence of a fact means none was recorded within this scope, not that the files are unrelated."
  },
  "snapshot": "2c4a4e25961017ed",
  "source": "app/api.py",
  "status": "found",
  "target": "app/service.py"
}
```

