# MCP context contract

Engram's first product promise is narrow and measurable:

> Give a coding agent the smallest valid repository context that materially
> improves implementation quality.

Engram is MCP-only. The host coding agent reasons and edits code; Engram
classifies the task, retrieves evidence, expands explicit repository
connections, and supplies verification facts.

## Adaptive routing

`get_task_context` classifies each request deterministically:

| Task profile | Evidence emphasis | Context ceiling |
|---|---|---:|
| bug fix | code, tests, review history | 7,000 chars |
| feature | reusable code, symbols, reviews | 7,500 chars |
| refactor | symbols, module structure, tests | 6,500 chars |
| test | tests, tested symbols, fixtures | 5,500 chars |
| documentation | docs and public symbols | 4,000 chars |
| investigation | broad code, tests, history | 8,500 chars |
| security | code, tests, rules, reviews | 9,000 chars |
| general | code and symbols | 5,500 chars |

The server retrieves a small candidate surplus, rejects candidates below the
profile's relative relevance threshold, deduplicates reviews, truncates long
review bodies, and stops adding evidence when the context ceiling is reached.

Every response reports:

- selected task profile and evidence focus;
- evidence paths, symbols, snippets, scores, and contributing signals;
- raw review evidence when the profile warrants it;
- approximate output tokens and truncation state;
- MCP processing latency.

## Validity rules

1. Current source and tests are evidence; inferred prose is not.
2. Natural-language retrieval may locate existing code, but it does not predict
   which files a change will touch.
3. Dependency expansion starts only from explicit paths selected by the agent
   or present in a diff.
4. Weak or absent retrieval results must cause abstention, not a claim that a
   new implementation is safe.
5. Review comments remain attributed raw evidence and are never promoted above
   current source or passing tests.
6. Every connection includes a deterministic reason.

## Product evaluation

Run the same realistic implementation tasks with and without the Engram skill
and MCP server. Pin the model, prompt, repository commit, and tool permissions.

Primary outcomes:

- first-attempt test pass rate;
- existing implementation reuse;
- duplicate code introduced;
- relevant tests and connected files inspected;
- implementation iterations and elapsed time;
- total input/output/tool-result tokens;
- unsupported claims derived from Engram context.

Engram is useful only when it improves implementation outcomes enough to repay
its own latency and token cost.
