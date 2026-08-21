# Benchmarks

Run with the `engram-evals` crate:

```bash
cargo run --release -p engram-evals -- --repo <path> \
  [--max-commits N] [--eval-commits N] [--history-commits N]
```

It runs two benchmarks that test two different capabilities. **Benchmark 2 is
the primary evidence for the product** — it measures the deterministic engine
(the co-change and import graphs) in isolation, with zero text prediction.
Benchmark 1 measures the fuzzy text-retrieval front door, which is optional
and only used when the agent has no concrete anchor file yet.

The default command above still runs Benchmark 1 and Benchmark 2. Honest reuse
retrieval is a separate, frozen benchmark:

```bash
cargo run --release -p engram-evals -- reuse
```

Add `--check` in CI to return a non-zero status when a launch gate is missed:

```bash
cargo run --release -p engram-evals -- reuse --check
```

## Honest reuse retrieval — labeled fixture

**The question:** does `Engine::assess_reuse` return a strong implementation
candidate only when the evidence supports it, and otherwise explicitly
abstain? This benchmark is separate from the commit-message-to-file B1 and the
connection-recovery B2 because neither measures the reuse decision contract.

The frozen labels live in `benchmarks/reuse/cases.json`, outside every indexed
directory so expected answers cannot leak into retrieval. Its `corpora` map
assigns each stratum an isolated profile under `benchmarks/reuse/corpora`.
Consequently a lookalike case cannot retrieve an exact implementation from a
different stratum. The no-match and incomplete-index strata deliberately share
the same neutral corpus: only index completeness differs between them.

The runner copies each referenced profile into a temporary repository, sets a
fixed Git author/committer timestamp, commits it, and indexes the requested
complete or deliberately incomplete view. It verifies that two index profiles
of the same corpus have the same reproducible snapshot SHA, then calls the
public reuse assessment API exactly once per case. Labels are validated before
the run: each path must stay inside its assigned corpus, and each symbol must be
defined at the exact labeled line rather than merely occurring somewhere in
the file.

The 100 cases contain ten examples in each reliability stratum:

- exact existing implementation;
- same behavior with different terminology;
- renamed copied function;
- partial duplication;
- similar vocabulary but different behavior;
- deprecated implementation;
- test helper versus production helper;
- no matching implementation;
- incomplete index; and
- conflicting approved decisions (documentation is not implementation
  evidence).

The report includes failed case IDs and these metrics:

| metric | definition | `--check` gate |
|---|---|---:|
| `reuse_likely` precision | relevant candidates emitted as `reuse_likely` / all candidates emitted as `reuse_likely` | >= 0.90 |
| Recall@3 | labeled candidate identities recovered in the first three results / all labeled identities | >= 0.80 |
| correct abstention | `no_evidence` and `index_incomplete` cases returned with the exact labeled state and no candidates | >= 0.85 |
| citation validity | returned candidates whose path, symbol, line span, snippet, and signals resolve against the frozen snapshot | 1.00 |
| decision-state accuracy | exact top-level state matches across all cases | informational |
| query p95 | wall-clock reuse-assessment latency, excluding indexing | informational |

`no_evidence` means only that no sufficiently similar current implementation
was found in indexed coverage. `index_incomplete` is distinct: the benchmark
must never turn partial symbol coverage into an apparently definitive negative.

Precision and decision-state accuracy are intentionally separate. A relevant
candidate is a precision true positive even if a conservative case label says
`possible_reuse`; that state mismatch is counted only by decision-state
accuracy.

### Current Phase 1 red gates

The isolated 100-case fixture currently reports 85.5% `reuse_likely`
precision and 80.0% correct abstention, below the 90% and 85% launch gates.
Recall@3 is 100% and citation validity is 100%. The failures are retained as
red tests: vocabulary-heavy UI/lookalike implementations still receive too
much evidence, and one deliberately unrelated query produces a spurious
candidate in both the complete and incomplete neutral profiles. `--check`
therefore exits non-zero until the classifier becomes more conservative; the
labels and thresholds are not relaxed to manufacture a pass.

To evaluate another compatible frozen manifest without changing the checked-in
labels, pass `--cases <path>`. Each value in its `corpora` map is resolved
relative to the manifest file, must stay below that directory, and is used only
by cases in the corresponding stratum.

## Benchmark 2 — connection recovery (no text, no prediction)

**The question:** given one file you already know you're changing, does
Engram's deterministic graph — the co-change table (a fact recorded from git
history) and the import graph (a fact derived from static analysis) — recover
the *other* files that actually changed with it? There is no language model,
no ranking heuristic, and no guessing involved: `find_connected_files` returns
only files linked to the anchor by a concrete, already-recorded edge. This is
fact-finding, not prediction — directly testing the concern that a
prediction-based approach could be the weak link.

**Leakage-free by construction:** for each evaluation, the co-change graph is
built *only* from the `--history-commits` commits strictly older than the
`--eval-commits` evaluation window. It is architecturally incapable of seeing
the future commit it's being asked to recover.

**Method:** for each of the newest `--eval-commits` commits (non-merge, 2–8
still-indexed changed files), take the alphabetically-first changed file as
the anchor and the rest as held-out targets. Call `find_connected_files` with
only the anchor. Score whether the returned candidates recover the targets.

### Result — langchain-ai/langchain

2,639 files · 68 eligible commits · history from the 300 commits older than
the 200-commit evaluation window (260 leakage-free co-change edges):

| source | Recall@5 |
|---|---|
| co-change only | 0.341 |
| import graph only | 0.226 |
| **combined (Recall@10)** | **0.579** |

**Hit rate (found ≥1 real connection): 67.6%** over 68 samples. On average
each commit had 1.7 held-out target files, and the tool returned 6.2
candidates.

**Read:** given one anchor file, Engram recovers **58% of the files that
actually changed alongside it**, from git-history and import-graph facts
alone — no text, no LLM, no guessing. Co-change and import graph are
complementary (roughly additive when combined), the same fusion principle as
Benchmark 1, but here it's fusing *facts* rather than fuzzy signals. This is
the number that matters most for the product: it's what `find_connected_files`
does whenever the agent already knows which file it's touching (the common
case — most tasks start from an existing file or diff, per the CLI's
`engram impact --diff current`).

### Reproducing

```bash
git clone --depth 500 https://github.com/langchain-ai/langchain.git /tmp/langchain
cargo run --release -p engram-evals -- --repo /tmp/langchain \
  --eval-commits 200 --history-commits 300
```

---

## Benchmark 1 — file prediction (text query, optional front door)

Each git commit is treated as a `(message → changed files)` sample. The
changed files are hidden; the commit subject is fed to each ranking strategy
and we measure **Recall@k** and **MRR**. This is the *text-driven* path — used
only when the agent has no anchor file yet (e.g. a brand-new task) — and tests
whether hybrid text retrieval beats BM25-only / vector-only for that narrower
use case.

Baselines: **bm25-only** (Tantivy lexical), **vector-only** (hashed-ngram
cosine), **random** (deterministic floor).

### Result — langchain-ai/langchain

2,639 files · 202 commit samples:

| strategy | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---|---|---|---|
| **hybrid** | **0.314** | **0.402** | **0.443** | **0.323** |
| bm25-only | 0.030 | 0.035 | 0.035 | 0.017 |
| vector-only | 0.234 | 0.290 | 0.412 | 0.278 |
| random | 0.000 | 0.000 | 0.001 | 0.000 |

**Read:** hybrid beats bm25-only by ~10× and vector-only by ~35% at Recall@5.
Confirms fusion beats any single text signal — but this benchmark starts from
a one-line commit subject standing in for a task description, and has known
methodology caveats (see below), so treat it as directional, not a headline
product number. Benchmark 2 is the more trustworthy evidence.

### Known caveats of Benchmark 1 (why it's secondary)

- **No temporal cutoff.** Unlike Benchmark 2, the index (embeddings, co-change
  graph feeding the fusion score) is built at current HEAD, which includes the
  commits being evaluated. This mildly flatters every method equally but isn't
  a rigorous leakage-free split.
- **Query proxy is weak.** A one-line commit subject is a poor stand-in for a
  real task description (a paragraph). Real usage should score higher.
- **Unstratified.** LangChain's history is dominated by `chore: bump <dep>`
  commits whose changed file (a lockfile) the title never mentions — no
  content-based method can predict those, and averaging them in drags every
  method toward the floor.
- The vector backend is still the placeholder hashed-ngram embedder; real
  embeddings (fastembed-rs, behind the existing `Embedder` trait) should lift
  vector and hybrid further.

### Secondary result — this repository

`engram` itself · 30 files · 9 eligible commits (kept for reference; too small
to be conclusive):

| strategy | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---|---|---|---|
| hybrid | 0.662 | 0.796 | 0.936 | 0.806 |
| bm25-only | 0.590 | 0.729 | 0.845 | 0.833 |
| vector-only | 0.462 | 0.625 | 0.888 | 0.731 |
| random | 0.170 | 0.329 | 0.553 | 0.346 |
