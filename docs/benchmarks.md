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

### Headline result — langchain-ai/langchain

A real, large codebase: **2,639 files · 200 commit samples** (recent 400 commits,
keeping non-merge commits that touch 1–8 still-existing files).

| strategy | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---|---|---|---|
| **hybrid** | **0.313** | **0.402** | **0.440** | **0.317** |
| bm25-only | 0.030 | 0.035 | 0.035 | 0.017 |
| vector-only | 0.234 | 0.288 | 0.409 | 0.289 |
| random | 0.000 | 0.000 | 0.001 | 0.000 |

**Read:** hybrid beats **bm25-only by ~10×** and **vector-only by ~35%** at
Recall@5, and everything crushes random. This clears the MVP acceptance
criterion ("hybrid retrieval beats vector-only") decisively, at scale.

Why BM25-only collapses here: LangChain's recent history is dominated by
`chore: bump <dep> …` commits whose changed file is a lockfile/manifest that the
title barely describes lexically. Hybrid's *path* and *vector* signals recover
those files — a concrete illustration of why multi-signal fusion wins over any
single signal.

Two honest notes on the absolute numbers (the *relative* comparison is the
thesis, and it holds):

- The vector backend is still the placeholder hashed-ngram embedder. Swapping in
  real embeddings (fastembed-rs ONNX, behind the existing `Embedder` trait)
  should lift vector and hybrid further.
- Dependency-bump/chore commits are intrinsically hard (the message says little
  about code), which caps the ceiling for every method.

### Secondary result — this repository

`engram` itself · 30 files · 9 eligible commits (kept for reference; too small to
be conclusive):

| strategy | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---|---|---|---|
| hybrid | 0.662 | 0.796 | 0.936 | 0.806 |
| bm25-only | 0.590 | 0.729 | 0.845 | 0.833 |
| vector-only | 0.462 | 0.625 | 0.888 | 0.731 |
| random | 0.170 | 0.329 | 0.553 | 0.346 |

### Reproducing

The harness is repo-agnostic — point `--repo` at any git checkout:

```bash
git clone --depth 400 https://github.com/langchain-ai/langchain.git /tmp/langchain
cargo run --release -p engram-evals -- --repo /tmp/langchain --max-commits 400
```
