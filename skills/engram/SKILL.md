---
name: engram
description: Use the Engram MCP server to retrieve repository evidence, discover existing implementations, and deterministically expand explicit file connections before editing code.
---

# Engram

Use Engram when it is configured for the active repository. Engram is MCP-only:
do not ask the user to run an Engram CLI command.

1. For a new task, call `get_task_context` with the user's task description.
   Read its `profile`, `budget`, and cited evidence. The server adapts retrieval
   depth to the task type and removes weak candidates automatically.
   - Treat paths, symbols, snippets, and raw review comments as evidence.
   - Inspect cited source before relying on it.
   - If Engram returns no strong evidence, continue with normal repository
     inspection; never interpret an empty result as permission to create code.
2. Before creating a function, type, service, or utility, call
   `find_existing_implementation` with the concept.
3. After selecting a file or producing a diff, call `expand_connections` with
   the explicit file paths. It returns only evidence-backed import dependents,
   historical co-change connections, and related tests. Never use natural
   language to ask Engram to guess files that will change.
4. Before handoff or PR creation, call `get_verification_plan` with all changed
   files. Call `get_review_history` separately only when a selected path is
   high-risk or the task-context packet indicates relevant historical reviews.

Engram is deliberately not an LLM. It performs retrieval and lazy structural
extraction internally; the host coding agent evaluates the returned evidence.
