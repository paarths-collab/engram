# Engram evaluation plan

Status: **plan only.** This document reports no results and no scores. Any
numbers here are budgets or design parameters, not measurements.

## Protocol

### 1. Task set

- **Evaluation set:** 60 tasks over 4 public repositories, one per language in
  scope (Rust, Python, TypeScript, JavaScript), 15 tasks per repository. Each
  repository covers reuse, caller discovery, tests-to-inspect, indirect
  relationship, and negative/ambiguous cases.
- **Development set:** 20 separate tasks from other repositories, used only for
  tuning. Evaluation repositories are held out and not inspected while tuning.
- **Labelling:** two engineers independently label each task's files as
  *required*, *useful*, or *irrelevant*, and label factual relationships (type,
  direction, hops), before seeing any tool output.
- **Disagreement:** reported per task and in aggregate. Labels are never
  silently unioned; unresolved items are shown as such.

### 2. Historical-prediction hygiene

- All features come from the pre-change snapshot: file inventory, import
  resolution, chunks, and git history.
- The target fix and everything after it are excluded from history.
- Declared ablation: runs with and without removal of bot, generated, and bulk
  commits from history.
- Queries are written before the reference solution is read.
- The fix's changed-file list is not a perfect gold set. It omits files that
  should have been read and includes incidental edits. The labels in section 1
  are the gold set.
- Do not describe the whole pipeline as "leakage-free". The LangChain co-change
  split is temporal, but its import graph uses current HEAD, a known limitation
  (see `docs/benchmarks.md`). New runs resolve imports at the pre-change
  snapshot.

### 3. Offline retrieval arms

Run on identical snapshots and budgets:

1. Exact ripgrep / identifier lookup
2. BM25
3. Hashed vectors
4. Import-only expansion
5. Co-change-only expansion
6. Combined Engram
7. Budgeted-map baseline

Optional later: CodeGraphContext and the team's existing commercial tool, on the
same snapshots.

### 4. Metrics

- **Relevant-file precision and recall** at serialized budgets of 2k, 4k, and
  8k. Size is measured on the `pack` object with the `bytes3-v1` estimator
  (`ceil(utf8 bytes / 3)`, see `docs/CONTRACT.md`). It is an **estimator**, not a
  model tokenizer, and it errs high for JSON-heavy text.
- **Evidence accuracy:** type, direction, and hop accuracy against the authored
  contract suite.
- **Stale-fact rate** after restart tests.
- **Latency** (warm and cold), **memory**, **index time**, and **disk size**.
- **No single aggregate.** Report per repository and per language, with restart
  and language failures shown separately.

### 5. Patch-outcome study

- 20 tasks from the evaluation set, run with the same existing agent, fixed
  model and settings, and a fixed tool-call bound.
- Arms: no retrieval, best simple baseline, combined Engram, and an **oracle
  pack** as an upper bound.
- Measured: duplicate helpers introduced, required tests run, regressions,
  reviewer acceptance, and wall time.

### 6. Go gate

Proceed only if all hold:

- Zero false direction, hop, or type claims on the authored contract suite in
  `crates/mcp-server/tests/`.
- No stale facts after restart tests.
- Useful-context gain over the best simple baseline at equal budget.
- Improved reviewed patch outcomes.

A recall gain that adds noise or lengthens task time is a failure.

### 7. Pilot

Four weeks, six developers. Each developer voluntarily uses Engram on a second
real task. Reviewer recognition is recorded with consent.

### 8. Ethics and licensing

Telemetry is opt-in. No code or queries enter shared analytics by default.
Public repositories are used under their licences, with attribution.

## What exists today

- **End-to-end tests** in `crates/mcp-server/tests/`: `e2e_canonical.rs`
  (search, expand, explain over stdio), `e2e_budget.rs` (`budget_tokens` and
  `pack`), and `e2e_lifecycle.rs` (restart freshness, snapshot identity, index
  failure reporting).
- **Demo transcript:** `docs/demo-transcript.md`, a captured session from the
  canonical test on a small fixture. Not a benchmark.
- **LangChain harness:** `benchmarks/langchain/harness/`, four Python scripts
  that drive the `engram` binary over stdio: `retrieval_on_issue.py` (blind
  ranking of one real issue), `query_sensitivity.py` (ranks under cleaner
  queries), `bench_tokens.py` (returned size against a ripgrep-and-read
  baseline, using chars/4, not `bytes3-v1`), and `bridge_test.py` (whether
  connections link the symptom file to the fix file). The README documents the
  first three.
- **Evals crate:** `crates/evals/` (`engram-evals`): Benchmark 1 (text-to-file
  recall, hybrid vs BM25 and vector-only) and Benchmark 2 (anchor-to-file
  connection recovery with a temporal co-change split; import graph at current
  HEAD).

## What does not exist yet

- A labelled task set, gold sets, or labeller records for either set.
- A baselines runner for the arms in section 3.
- The pilot, its telemetry, or reviewer data.
- Per-language and restart-staleness reporting beyond the single fixtures.

## Proposed task record (not implemented)

```json
{
  "_status": "PROPOSED, not implemented",
  "id": "rust-reuse-003",
  "repo": "https://github.com/example/project",
  "base_commit": "<40-char SHA of pre-change snapshot>",
  "query": "<written before reading the solution>",
  "anchors": ["src/lib.rs"],
  "gold_required": ["src/util/hash.rs"],
  "gold_useful": ["tests/hash_test.rs"],
  "labeller_a": {"required": ["src/util/hash.rs"], "useful": ["tests/hash_test.rs"], "irrelevant": []},
  "labeller_b": {"required": ["src/util/hash.rs"], "useful": [], "irrelevant": ["tests/hash_test.rs"]},
  "notes": "Labellers disagree on tests/hash_test.rs; reported, not unioned."
}
```
