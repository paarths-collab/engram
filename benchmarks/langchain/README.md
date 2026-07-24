# Langchain retrieval benchmark

Retrieval quality and token efficiency measured against a real, unmodified
checkout of [`langchain-ai/langchain`](https://github.com/langchain-ai/langchain)
(2,639 files indexed, 19,455 symbol-span chunks), driving the actual `engram`
MCP binary over stdio the way a coding agent would.

This is deliberately adversarial toward Engram: the repository is one it has
never seen, the tasks are real GitHub issues, and the failure it found is
reported before the fix that closed it.

## How to reproduce

The harness drives the real binary. Each script takes the binary, a repo, and
(where relevant) an issue-text file as arguments — no hard-coded paths.

```bash
# 1. build or download the engram binary (see repo README)
# 2. shallow-clone the target repo (history depth feeds the co-change graph)
git clone --depth 400 https://github.com/langchain-ai/langchain.git

# 3. blind retrieval on a real issue: does engram surface the file the real fix touched?
python benchmarks/langchain/harness/retrieval_on_issue.py ./engram ./langchain issue.txt

# 4. is retrieval sensitive to how the issue is phrased?
python benchmarks/langchain/harness/query_sensitivity.py ./engram ./langchain issue.txt

# 5. token efficiency vs a ripgrep-and-read baseline
python benchmarks/langchain/harness/bench_tokens.py ./engram ./langchain
```

On Windows, Smart App Control may block a freshly downloaded unsigned binary by
hash. Run the Linux binary in a container instead
(`docker run --rm -v "$PWD:/work" python:3.11-slim ...`); results are identical.

## What "good" means here

Engram's job is to hand a coding agent the smallest slice of the repo that
lets it do the task. Two measurable things:

1. **Retrieval quality** — given a task, does the correct file rank at the top?
   Ground truth is the file the real merged fix actually changed.
2. **Token efficiency** — how much context does the agent ingest vs the naive
   "grep the keywords and read every match" path?

## Result 1: token efficiency (B2)

Per task, Engram returns a small ranked evidence packet; the ripgrep baseline
is the total size of every file matching the task's keywords.

| | engram tokens/task | grep-and-read tokens | ratio |
|---|--:|--:|--:|
| mean over 5 tasks | ~2,800 | ~1.15M | **~0.3%** |

Engram returns ~2,800 tokens of ranked, relevant context where reading every
grep match would be ~400x larger. **Caveat:** the grep-and-read-everything
baseline is a ceiling, not what a disciplined agent does. The honest reading is
"Engram hands you a ranked answer; grep hands you a pile to triage." The token
win only matters when the ranked answer is *correct* — which is Result 2.

## Result 2: retrieval quality on a real issue, and the failure it exposed

Task: langchain issue
[#38366](https://github.com/langchain-ai/langchain/issues/38366) —
`merge_dicts` concatenates identical string metadata across streaming chunks
(`model_name`, `finish_reason` doubled). The real fix lives in
`libs/core/langchain_core/utils/_merge.py`.

### The failure (reported, not hidden)

Feeding Engram the raw issue text, the correct file was **not in the top 11**,
across five different phrasings — including one that literally named the
function:

| query | ground-truth rank (before) |
|---|:--:|
| raw issue text | MISS |
| issue title only | MISS |
| one-line problem statement | MISS |
| "merge_dicts doubles model_name and finish_reason" | MISS |

Root cause, three compounding bugs in the ranking layer:

1. the tokenizer split on every non-alphanumeric, so `merge_dicts` shattered
   into `merge` + `dicts` and the identifier was never searched as a name;
2. the symbol lookup was a substring `LIKE` with `LIMIT 3` and no relevance
   order, so `merge_dicts` lost to `merge_content`, `merge_configs`, ...;
3. symbol hits were appended and the list truncated **without re-sorting**, so
   fusion noise permanently owned the top slots.

An earlier draft of this benchmark used hand-written queries whose words sat in
the target files (`CacheBackedEmbeddings`, `InMemoryRateLimiter`) and reported
"5/5 perfect." That was an artifact of easy queries. Real issues describe
symptoms in the vocabulary of the *callers*, and the buggy utility contained
none of it. Running a real issue is what surfaced the failure.

### The fix, verified

The ranking fix ([#23](https://github.com/paarths-collab/engram/pull/23)):
recover whole identifiers, look them up by exact name, score an exact match
above the current top, and re-sort before truncating. Re-running the identical
blind test on the fixed binary:

| query | before | after |
|---|:--:|:--:|
| raw issue text | MISS | **#1** |
| issue title only | MISS | **#1** |
| one-line problem statement | MISS | **#1** |
| "merge_dicts doubles model_name and finish_reason" | MISS | **#1** |

## What this does and does not prove

- **Proves:** Engram no longer loses to plain `ripgrep` on issues that name a
  symbol — a large fraction of real bug reports. It returns the right file at
  #1, plus ranked context, plus the deterministic connection graph on top.
- **Does not prove:** that Engram *beats* `ripgrep`. On a named-symbol issue,
  `rg merge_dicts` also finds the file instantly. On this class Engram now
  *ties* on finding and adds context; it does not leapfrog.
- **Open:** the case where Engram could genuinely beat grep is a symptom
  described with no symbol named. That depends on real semantic embeddings
  (the current `HashedNgramEmbedder` is a placeholder) and is untested here.

n is small (one issue, five phrasings). Treat this as a worked example of the
method and an honest before/after, not a headline hit-rate. A larger real-issue
run is the natural next step.
