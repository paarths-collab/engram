# Benchmarks

## Connection recovery — the product benchmark

Engram does not claim to infer a changed-file set from a natural-language task.
Its core contract is narrower and testable: given an explicit file anchor from
the editor or a diff, it returns relationships proved by static imports and
observed git history.

For each historical commit:

1. Build the graph only from commits older than the commit under evaluation.
2. Provide one changed source file as the anchor.
3. Hide the remaining changed files and tests.
4. Run `expand_connections`.
5. Measure recovery of the hidden files, separately by evidence type.

```bash
cargo run --release -p engram-evals -- --repo /path/to/repo --max-commits 300
```

## Temporal isolation, and the leak that remains

Step 1 is the whole benchmark. A co-change edge is a direct function of the
commits that produced it, so a graph built from the full log already contains
the commit being predicted; scoring against it measures leakage, not retrieval.
The harness therefore rebuilds co-change evidence per sample from commits
strictly older than the commit under evaluation
(`cochange::build_from_records` with a cutoff, covered by unit tests).

One leak is not yet closed, and the numbers must be read with it in mind: the
**static import graph is built from working-tree contents at HEAD**, not
reconstructed per commit. It is a weaker leak than co-change, because current
imports are not derived from any single commit's file set, but a commit that
introduced an import is still visible to its own evaluation. Until per-commit
checkout is added, treat the `static import` and `combined` rows as upper
bounds and the `historical co-change` row as clean.

Report `Recall@5`, `Recall@10`, and MRR for:

- static import dependents;
- historical co-change connections;
- related tests;
- the union of those evidence-backed results.

Results must be split by repository, language, and commit category. Dependency
bumps and formatting-only commits are reported separately, not folded into a
headline number. The prior task-text-to-file benchmark is deliberately retired:
it measured a different, probabilistic product that Engram no longer exposes.
