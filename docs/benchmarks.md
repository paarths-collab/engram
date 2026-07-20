# Benchmarks

Run with the `engram-evals` crate:

```bash
cargo run --release -p engram-evals -- --repo <path> [--max-commits N]
```

## Benchmark 1 — file prediction

Each git commit is treated as a `(message → changed files)` sample. The changed
files are hidden; the commit subject is fed to each ranking strategy and we
measure **Recall@k** (fraction of the actually-changed files found in the top k)
and **MRR** (mean reciprocal rank of the first correct file). This tests the core
thesis: does hybrid retrieval beat BM25-only?

Baselines: **bm25-only** (Tantivy lexical), **vector-only** (hashed-ngram
cosine), **random** (deterministic floor).

### Result on this repository

`engram` itself · 30 files · 9 eligible commits:

| strategy | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---|---|---|---|
| hybrid | 0.662 | 0.796 | 0.936 | 0.806 |
| bm25-only | 0.590 | 0.729 | 0.845 | 0.833 |
| vector-only | 0.462 | 0.625 | 0.888 | 0.731 |
| random | 0.170 | 0.329 | 0.553 | 0.346 |

**Read:** hybrid beats bm25-only on Recall@5/10/20, and every method crushes
random. On this tiny sample bm25-only edges MRR by a hair (it places the first
correct file marginally higher), while hybrid recovers more of the changed set
overall.

### Caveat: sample size

This repo is young, so only 9 commits qualify (non-merge, 1–8 changed files that
still exist). That is far too small to be conclusive — the numbers are noisy and
meant to demonstrate the harness and methodology, not to be a headline metric.
The blueprint's plan is to run this across large external repositories
(golang/go, microsoft/TypeScript, django/django, rails/rails); this sandbox
scopes GitHub/network access to this repo only, so those runs are pending an
environment with clone access. The harness is repo-agnostic: point `--repo` at
any git checkout and it produces the same table.
