# ADR 0001 — How Engram stores and traverses code relationships

- Status: Accepted
- Date: 2026-07-20
- Context: retrieval and impact analysis over code graphs (imports, calls, co-change)

## Question

Engram's value depends on relationships: which files import which, which change
together, which symbols call which. As repositories grow and we extract deeper
relationships, those graphs get large and dense. Do we need a graph database
(Neo4j, Dgraph) — or GraphQL — for this?

## Decision

**No graph database. No GraphQL for storage.** Store relationships as **edge
rows** in the relational store, and do multi-hop traversal by loading the
relevant edges into an **in-memory graph** (`petgraph`) in the Rust worker.

### 1. Storage: nodes + edge tables

A graph is nodes + edges. Nodes already exist (`files`, `symbols`). Relationships
are edge tables — we already ship two:

- `cochange(path_a, path_b, count, strength)` — historical co-change
- `file_imports(path, target)` — import targets

As we extract more relationship kinds (calls, references — the `symbol_edges`
table in the blueprint's §17), we add **one generic table** rather than a table
per relationship:

```sql
edges(src_id, dst_id, kind, weight)   -- kind = 'imports' | 'calls' | 'cochange' | 'reference'
CREATE INDEX idx_edges_src ON edges(src_id, kind);
CREATE INDEX idx_edges_dst ON edges(dst_id, kind);
```

A dense graph of tens of millions of edges is a few GB, fully indexed. This is
the same node+edge representation a graph database uses internally; we lose
nothing on storage by using rows, and we keep the single-binary story.

### 2. Query pattern decides the engine, not graph density

Storage is easy anywhere; the engine choice is driven by the *query*:

| Query pattern | Example | Engine |
| --- | --- | --- |
| Bounded hop (1–3) | callers of X; importers of Y; tests co-changing with a diff | Indexed SQL joins — sub-millisecond even on tens of millions of edges. ~95% of Engram's queries. |
| Deep / variable-depth traversal | transitive closure, shortest dependency path, cycles | **Not the database.** Load edges into an in-memory `petgraph` / CSR adjacency and traverse in Rust. |
| Ad-hoc graph-native queries | exploratory Cypher, PageRank over the call graph | Embedded graph DB (KùzuDB) — still one binary. Evaluate; don't assume. |
| Multi-tenant deep serving at scale | graph exceeds one machine's RAM; concurrent deep queries across orgs | A server graph DB (Neo4j). Enterprise-tier; gate on a real number. |

For a single repository the whole edge set fits in RAM, so in-memory traversal
in Rust is faster than any out-of-process graph DB (no network hop, no planner)
and needs no new infrastructure.

### 3. "The model should understand relationships"

A model reasoning over graph *structure* (a GNN reranker, blueprint §14 Stage 2)
is trained offline in Python on exported edges and served in Rust via ONNX
(`ort`). The storage engine is irrelevant to whether the model understands
structure — it just needs clean, exportable edge tables. This is a reason to keep
tidy edge tables, not to adopt a graph database.

### 4. The real scaling wall is vectors, not the graph

"Index everything on a large repo" hits brute-force vector search long before it
hits any graph-traversal limit. The fix is an ANN index (`pgvector` HNSW,
embedded `usearch`/`hnsw_rs`, or Qdrant) — **not** a graph database.

## Scaling path

```
Local / single repo (now):  SQLite edge tables + in-memory petgraph for deep hops
Cloud / team:               Postgres edge tables + pgvector (ANN) + in-memory
                            subgraph materialization in the worker
Graph-native ergonomics:    embedded KùzuDB (single binary) — evaluate, don't assume
Server graph DB (Neo4j):    only when the graph outgrows one machine's RAM AND
                            multi-tenant concurrent deep serving is required
```

## Consequences

- Relationships stay behind the `Store` abstraction (`cochange_for`,
  `importers_of`, future `edges`), so SQLite → Postgres is an additive swap with
  no caller changes.
- Deep traversal will live in a future `repo-map/graph.rs` that builds a
  `petgraph` from the edge tables — additive, no datastore change.
- We do **not** adopt a graph database now; doing so would cost the single-binary
  distribution and the "feels instant" latency that are the product's first wow.
- GraphQL, if ever adopted, is an API-layer choice and unrelated to this decision.
