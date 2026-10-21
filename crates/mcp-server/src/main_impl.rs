//! engram binary: MCP server + automatic background indexing.
//! Usage: engram --repo /path/to/repo   (defaults to cwd)

use crate::mcp::{serve, ToolHandler};
use crate::watcher::SharedHealth;
use engram_repo_map::store::Store;
use engram_retrieval::Engine;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

const EAGER_TIER1_LIMIT: usize = 3000;

/// Where the background indexing job stands. Shared between the indexing
/// thread (which writes it exactly once, on completion) and the server (which
/// reads it on every tool call), so a failed index is reported as a failure
/// instead of looking like an index that is merely slow.
#[derive(Debug, Clone)]
pub enum IndexState {
    Indexing {
        started: Instant,
    },
    Ready,
    /// The underlying error, as text. Recovery guidance is added when it is
    /// shown to the client; see [`failure_message`].
    Failed(String),
}

pub type SharedIndexState = Arc<Mutex<IndexState>>;

fn lock_state(state: &SharedIndexState) -> MutexGuard<'_, IndexState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What a client is told while the first index is still being built.
fn indexing_message(elapsed: Duration) -> String {
    format!(
        "Engram is still indexing this repository in the background (running for {}s). \
         Retry this tool call in a few seconds.",
        elapsed.as_secs()
    )
}

/// What a client is told once indexing has failed. Deliberately does not
/// suggest calling again: nothing will change until the cause is fixed and the
/// server restarted.
fn failure_message(repo_root: &Path, error: &str) -> String {
    let repo = repo_root.display();
    format!(
        "Engram failed to index the repository at {repo}: {error}. \
         Engram keeps its cache in {repo}/.engram/engram.db, so the repository directory \
         must be writable, and --repo must point to an existing, readable directory. \
         Fix that and restart the Engram MCP server; calling this tool again will not \
         help until then."
    )
}

/// What a client is told when the server could not even work out which
/// repository to serve (`--repo` with no value). There is no repository to
/// name, and the working directory is not one the user asked for.
fn no_repo_message(error: &str) -> String {
    format!(
        "Engram could not start: {error}. Pass --repo <path> with an existing, readable \
         directory, then restart the Engram MCP server; calling this tool again will not \
         help until then."
    )
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_owned())
}

/// Run an indexing job and record how it ended. Both an `Err` and a panic end
/// as `Failed`; without that, either would leave the state at `Indexing`
/// forever and the client would be told to retry something that cannot work.
fn run_index_job<F>(state: &SharedIndexState, job: F)
where
    F: FnOnce() -> anyhow::Result<()>,
{
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
    let next = match outcome {
        Ok(Ok(())) => IndexState::Ready,
        Ok(Err(e)) => {
            eprintln!("[engram] indexing failed: {e:#}");
            IndexState::Failed(format!("{e:#}"))
        }
        Err(payload) => {
            let msg = panic_message(payload.as_ref());
            eprintln!("[engram] indexing panicked: {msg}");
            IndexState::Failed(format!("the indexing thread panicked: {msg}"))
        }
    };
    *lock_state(state) = next;
}

/// The real indexing job for a repository.
fn default_index_job(root: PathBuf) -> impl FnOnce() -> anyhow::Result<()> + Send + 'static {
    move || {
        let stats = engram_repo_map::index_repo(&root, EAGER_TIER1_LIMIT)?;
        eprintln!(
            "[engram] indexed: {} files, {} cochange edges, {} tier1 files, {} pruned",
            stats.files, stats.cochange_edges, stats.tier1_files, stats.pruned_files
        );
        Ok(())
    }
}

/// Canonicalize `--repo` and check it is a readable directory. The error names
/// the path so the client can see what was wrong with it.
fn resolve_repo(path: &Path) -> Result<PathBuf, String> {
    let canonical = std::fs::canonicalize(path).map_err(|e| {
        format!(
            "the --repo path {} does not exist or cannot be resolved ({e})",
            path.display()
        )
    })?;
    if !canonical.is_dir() {
        return Err(format!(
            "the --repo path {} is not a directory",
            canonical.display()
        ));
    }
    std::fs::read_dir(&canonical).map_err(|e| {
        format!(
            "the --repo path {} is not a readable directory ({e})",
            canonical.display()
        )
    })?;
    Ok(canonical)
}

/// Tools served from the in-memory index. Their responses gain a `snapshot`
/// field, plus `index_warnings` / `index_state` when results may be stale.
const INDEXED_TOOLS: &[&str] = &[
    "search_context",
    "get_task_context",
    "find_existing_implementation",
    "predict_impact",
    "expand_connections",
    "find_connected_files",
    "explain_connection",
];

/// The `index_state` object attached to a response when the index is not
/// plainly Ready and current.
fn index_state_info(
    state: &IndexState,
    snapshot: Option<&str>,
    generation: u64,
    stale: bool,
) -> Value {
    let mut info = json!({ "generation": generation, "stale": stale });
    match state {
        IndexState::Indexing { started } => {
            info["status"] = json!("indexing");
            info["elapsed_seconds"] = json!(started.elapsed().as_secs());
        }
        IndexState::Ready => info["status"] = json!("ready"),
        IndexState::Failed(error) => {
            info["status"] = json!("failed");
            info["error"] = json!(error);
        }
    }
    if let Some(id) = snapshot {
        info["snapshot"] = json!(id);
    }
    info
}

pub struct Engram {
    repo_root: PathBuf,
    /// False when no repository was named at all, so `repo_root` is only a
    /// placeholder and must not be presented to the client as "the repository".
    repo_named: bool,
    /// The published generation: chunks, BM25 index, vectors and code graph
    /// built together. Replaced only as a whole.
    engine: Option<Engine>,
    store: Option<Store>,
    index_state: SharedIndexState,
    /// Flipped by the watcher when files or HEAD change; triggers a lazy rebuild.
    dirty: Arc<AtomicBool>,
    /// Whether live reindexing is working, for `index_warnings`.
    watcher_health: SharedHealth,
    /// How many engines have been published so far.
    generation: u64,
    /// Why the last attempt to rebuild failed, while an older generation is
    /// still being served in its place.
    rebuild_error: Option<String>,
}

impl Engram {
    pub fn start(repo: PathBuf) -> Self {
        match resolve_repo(&repo) {
            Ok(root) => {
                let job = default_index_job(root.clone());
                Self::start_with(root, true, job)
            }
            // Still start: the client should see *why* nothing works, and an
            // unusable `--repo` must never silently fall back to another
            // directory.
            Err(msg) => Self::failed(repo, msg),
        }
    }

    /// A server whose repository could not be used. It answers MCP requests,
    /// but every tool that needs the index reports `error`.
    fn failed(repo_root: PathBuf, error: String) -> Self {
        eprintln!("[engram] {error}");
        Engram {
            repo_root,
            repo_named: true,
            engine: None,
            store: None,
            index_state: Arc::new(Mutex::new(IndexState::Failed(error))),
            dirty: Arc::new(AtomicBool::new(false)),
            watcher_health: SharedHealth::default(),
            generation: 0,
            rebuild_error: None,
        }
    }

    /// A server that was never told which repository to serve. Like
    /// [`Engram::failed`], but its errors do not describe `placeholder` (the
    /// working directory) as the repository.
    fn failed_without_repo(placeholder: PathBuf, error: String) -> Self {
        Engram {
            repo_named: false,
            ..Self::failed(placeholder, error)
        }
    }

    /// Start indexing with `job` in the background; the server accepts
    /// connections immediately and tools report progress until it finishes.
    /// `watch` turns on the incremental file watcher (off in unit tests).
    fn start_with<F>(repo_root: PathBuf, watch: bool, job: F) -> Self
    where
        F: FnOnce() -> anyhow::Result<()> + Send + 'static,
    {
        let index_state: SharedIndexState = Arc::new(Mutex::new(IndexState::Indexing {
            started: Instant::now(),
        }));
        let thread_state = index_state.clone();
        let spawned = std::thread::Builder::new()
            .name("engram-index".into())
            .spawn(move || run_index_job(&thread_state, job));
        if let Err(e) = spawned {
            *lock_state(&index_state) =
                IndexState::Failed(format!("could not start the indexing thread: {e}"));
        }

        // Incremental reindex: watch for file saves and HEAD moves.
        let dirty = Arc::new(AtomicBool::new(false));
        let watcher_health = SharedHealth::default();
        if watch {
            crate::watcher::spawn(repo_root.clone(), dirty.clone(), watcher_health.clone());
        }
        Engram {
            repo_root,
            repo_named: true,
            engine: None,
            store: None,
            index_state,
            dirty,
            watcher_health,
            generation: 0,
            rebuild_error: None,
        }
    }

    #[cfg(test)]
    fn index_state(&self) -> IndexState {
        lock_state(&self.index_state).clone()
    }

    /// Err with the client-facing message unless indexing has finished.
    fn index_gate(&self) -> Result<(), String> {
        match &*lock_state(&self.index_state) {
            IndexState::Ready => Ok(()),
            IndexState::Indexing { started } => Err(indexing_message(started.elapsed())),
            IndexState::Failed(error) if !self.repo_named => Err(no_repo_message(error)),
            IndexState::Failed(error) => Err(failure_message(&self.repo_root, error)),
        }
    }

    /// Ensure the background index is ready and the store is open. Cheap tools
    /// (verification plan) need only this, not the embedding index.
    fn ensure_store(&mut self) -> Result<(), String> {
        self.index_gate()?;
        if self.store.is_none() {
            self.store = Some(Store::open(&self.repo_root).map_err(|e| {
                failure_message(&self.repo_root, &format!("could not open its cache: {e}"))
            })?);
        }
        Ok(())
    }

    fn ensure_engine(&mut self) -> Result<(), String> {
        self.ensure_store()?;
        // A watcher change invalidates the in-memory index, and so does lazy
        // extraction that wrote symbols/imports after the engine was built.
        // Rebuild lazily here (cheap: persisted vectors mean only changed
        // files re-embed).
        let watcher_dirty = self.dirty.swap(false, Ordering::SeqCst);
        let engine_stale = self.engine.as_ref().is_some_and(Engine::is_stale);
        if self.engine.is_some() && !watcher_dirty && !engine_stale {
            return Ok(());
        }
        // Build the next generation completely off to the side (chunks, BM25
        // index, vectors, and code graph together) and only then swap it in
        // with one assignment. The previous engine keeps serving until then,
        // so a query never sees new symbols with an old graph or vice versa.
        let store = self.store.as_mut().unwrap();
        match Engine::build(&self.repo_root, store) {
            Ok(next) => {
                self.engine = Some(next);
                self.generation += 1;
                self.rebuild_error = None;
                Ok(())
            }
            Err(e) if self.engine.is_some() => {
                // Keep serving the last good generation, but never silently:
                // it is out of date, and responses say so. Ask for another
                // attempt on the next call.
                eprintln!("[engram] index rebuild failed, serving the previous snapshot: {e:#}");
                self.rebuild_error = Some(format!("{e:#}"));
                self.dirty.store(true, Ordering::SeqCst);
                Ok(())
            }
            Err(e) => Err(format!("engine: {e}")),
        }
    }

    /// Everything currently making results possibly out of date, as text for
    /// the client: a degraded watcher, or a failed rebuild.
    fn index_warnings(&self) -> Vec<String> {
        let mut warnings = crate::watcher::warnings(&self.watcher_health);
        if let Some(err) = &self.rebuild_error {
            warnings.push(format!(
                "rebuilding the index failed ({err}); results come from the previous snapshot \
                 and may be out of date"
            ));
        }
        warnings
    }

    /// The `snapshot` and (when not plainly healthy) `index_state` fields for
    /// a response served by the current engine.
    fn decorate(&self, result: &mut Value) {
        let Some(obj) = result.as_object_mut() else {
            return;
        };
        let warnings = self.index_warnings();
        let engine_stale = self.engine.as_ref().is_some_and(Engine::is_stale);
        let snapshot = self.engine.as_ref().map(|e| e.snapshot_id().to_owned());
        if let Some(id) = &snapshot {
            obj.insert("snapshot".to_owned(), json!(id));
        }
        if !warnings.is_empty() {
            obj.insert("index_warnings".to_owned(), json!(warnings));
        }
        // Only reached after `index_gate` let the call through, and the state
        // never leaves Ready afterwards, so Ready is the only state to report
        // here; the response gains `index_state` exactly when it may be stale.
        let stale = engine_stale || !warnings.is_empty();
        if stale {
            obj.insert(
                "index_state".to_owned(),
                index_state_info(
                    &IndexState::Ready,
                    snapshot.as_deref(),
                    self.generation,
                    stale,
                ),
            );
        }
    }
}

impl ToolHandler for Engram {
    fn list_tools(&self) -> Value {
        json!({ "tools": [
            {
                "name": "search_context",
                "description": "ALWAYS call this FIRST, before planning or implementing any coding task. Returns ranked, evidence-backed context about this repository: existing code relevant to the query, symbols to reuse instead of reimplementing, related tests, and matched files (hybrid BM25+vector+symbol retrieval), plus any past reviewer comments on the matched files. Using this prevents duplicate implementations and wrong assumptions about the codebase.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "What you are looking for, in natural language, e.g. 'add retry logic to billing webhooks'" }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_task_context",
                "description": "DEPRECATED alias of search_context (use `query` instead of `task`). Kept for compatibility; will be removed in a future release.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": { "type": "string", "description": "The task description in natural language" }
                    },
                    "required": ["task"]
                }
            },
            {
                "name": "find_existing_implementation",
                "description": "Call this BEFORE writing any new function, class, service, or utility. Checks whether an implementation of this concept already exists in the repository so it can be reused instead of duplicated. Pass the concept you are about to implement.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "concept": { "type": "string", "description": "What you are about to implement, e.g. 'audit log writer' or 'exponential backoff retry'" }
                    },
                    "required": ["concept"]
                }
            },
            {
                "name": "expand_connections",
                "description": "Call this when you already know which file(s) you are changing (e.g. the current diff) and want everything CONNECTED to them, with zero guessing. Takes exact file paths as anchors and returns only files linked by hard, deterministic facts already recorded in the store: the co-change graph (files that historically changed together in the same commits) and the import graph (files that statically import an anchor, up to 2 hops). Every result traces to a concrete recorded edge — this is fact-finding, not prediction. Prefer this over predict_impact whenever you have concrete anchor files.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Repo-relative paths of files you are changing or about to change, e.g. ['src/billing/cancel.rs']"
                        }
                    },
                    "required": ["paths"]
                }
            },
            {
                "name": "explain_connection",
                "description": "Call this to see WHY two specific files are connected, from recorded facts only. A direct pairwise lookup (not a ranked or capped recommendation): returns every recorded edge linking them. Import edges: source imports target and/or target imports source, directly or through one intermediate file (2 hops, with the full `chain` of paths); import edges come from heuristic module-name resolution, not compiler resolution (`resolution: \"heuristic\"`). Historical co-change: each direction found, with commit `count` and `strength`. Each reason has `type`, `detail`, `weight` plus typed fields (imports: `edge_kind`, `direction`, `hops`, `chain`, per-hop `edges`, `resolver`, `resolution_status: \"candidate\"`; co-change: `edge_kind`, `direction`, `count`, `strength`, `window`); import and co-change are separate facts and neither implies the other (co-change means files changed together, not that one must change with the other). `status` is `found`, `not_found_within_scope`, or `unknown_beyond_bound` (no chain within the hop bound but imports continue past it); `scope` states exactly what was searched, so `connected: false` means no recorded fact within that scope, never proof the files are unrelated. Returns an error if either path is not an indexed file or both paths are the same; use repo-relative paths from search_context/expand_connections. Never a guess.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": { "type": "string", "description": "Repo-relative path of the first file" },
                        "target": { "type": "string", "description": "Repo-relative path of the second file" }
                    },
                    "required": ["source", "target"]
                }
            },
            {
                "name": "find_connected_files",
                "description": "DEPRECATED alias of expand_connections (use `paths` instead of `files`). Kept for compatibility; will be removed in a future release.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Repo-relative paths of files you are changing"
                        }
                    },
                    "required": ["files"]
                }
            },
            {
                "name": "predict_impact",
                "description": "EXPERIMENTAL. Predicts which files a fuzzy natural-language task will likely affect (text retrieval + co-change/import expansion). Prefer expand_connections when you have concrete anchor files — prediction from vague tasks is not yet reliable.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": { "type": "string", "description": "The task description" }
                    },
                    "required": ["task"]
                }
            },
            {
                "name": "get_verification_plan",
                "description": "Call this AFTER making changes and BEFORE opening a PR. Given the list of changed files, returns the merged verification checklist for the engineering domains they touch (backend/frontend/database/infra), the repo's detected test commands, and tests that historically change together with these files. Run these checks before considering the change done.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "changed_files": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Repo-relative paths of the files changed, e.g. ['src/billing/cancel.rs', 'migrations/003_add.sql']"
                        }
                    },
                    "required": ["changed_files"]
                }
            },
            {
                "name": "get_review_history",
                "description": "Call this before changing a file to see what human reviewers said about it before. Returns RAW, unsummarized reviewer comments from past pull requests, with the PR number and whether that PR was merged (accepted). Pass a file 'path' to get comments on that file, and/or a 'task' to rank comments by relevance. Reading these avoids repeating mistakes reviewers already flagged.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Repo-relative file path to fetch review comments for" },
                        "task": { "type": "string", "description": "Task/concept to rank comments by relevance" }
                    }
                }
            }
        ]})
    }

    fn call_tool(&mut self, name: &str, args: &Value) -> Result<Value, String> {
        let mut result = self.dispatch_tool(name, args)?;
        if INDEXED_TOOLS.contains(&name) {
            self.decorate(&mut result);
        }
        Ok(result)
    }
}

impl Engram {
    fn dispatch_tool(&mut self, name: &str, args: &Value) -> Result<Value, String> {
        match name {
            "search_context" | "get_task_context" => {
                self.ensure_engine()?;
                let engine = self.engine.as_mut().unwrap();
                let store = self.store.as_mut().unwrap();
                let task = args
                    .get("query")
                    .or_else(|| args.get("task"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if task.is_empty() {
                    return Err("missing required argument: query".into());
                }
                let packets = engine.search(store, task, 8).map_err(|e| e.to_string())?;
                // past_reviews: raw reviewer comments on the files retrieval matched.
                let mut past_reviews = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for p in &packets {
                    if seen.insert(p.path.clone()) {
                        if let Ok(cs) = store.review_comments_for_path(&p.path) {
                            past_reviews.extend(cs);
                        }
                    }
                }
                Ok(json!({ "task": task, "evidence": packets, "past_reviews": past_reviews }))
            }
            "find_existing_implementation" => {
                self.ensure_engine()?;
                let engine = self.engine.as_mut().unwrap();
                let store = self.store.as_mut().unwrap();
                let concept = args.get("concept").and_then(|v| v.as_str()).unwrap_or("");
                if concept.is_empty() {
                    return Err("missing required argument: concept".into());
                }
                let packets = engine
                    .search(store, concept, 5)
                    .map_err(|e| e.to_string())?;
                let found = !packets.is_empty();
                Ok(json!({
                    "concept": concept,
                    "existing_candidates": packets,
                    "recommendation": if found {
                        "Review these candidates before implementing. Reuse if one matches."
                    } else {
                        "No existing implementation found. Safe to implement new."
                    }
                }))
            }
            "predict_impact" => {
                self.ensure_engine()?;
                let engine = self.engine.as_mut().unwrap();
                let store = self.store.as_mut().unwrap();
                let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("");
                if task.is_empty() {
                    return Err("missing required argument: task".into());
                }
                let impact = engine
                    .predict_impact(store, task)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::to_value(impact).map_err(|e| e.to_string())?)
            }
            "expand_connections" | "find_connected_files" => {
                self.ensure_engine()?;
                let engine = self.engine.as_mut().unwrap();
                let store = self.store.as_mut().unwrap();
                let files: Vec<String> = args
                    .get("paths")
                    .or_else(|| args.get("files"))
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                if files.is_empty() {
                    return Err("missing required argument: paths (array of file paths)".into());
                }
                let impact = engine
                    .impact_from_files(store, &files)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::to_value(impact).map_err(|e| e.to_string())?)
            }
            "explain_connection" => {
                self.ensure_engine()?;
                let engine = self.engine.as_mut().unwrap();
                let store = self.store.as_mut().unwrap();
                let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("");
                let target = args.get("target").and_then(|v| v.as_str()).unwrap_or("");
                if source.is_empty() || target.is_empty() {
                    return Err("missing required arguments: source and target (file paths)".into());
                }
                let explanation = engine
                    .explain_connection(store, source, target)
                    .map_err(|e| e.to_string())?;
                Ok(json!({
                    "source": source,
                    "target": target,
                    "connected": !explanation.reasons.is_empty(),
                    "status": serde_json::to_value(explanation.status).map_err(|e| e.to_string())?,
                    "scope": serde_json::to_value(&explanation.scope).map_err(|e| e.to_string())?,
                    "reasons": serde_json::to_value(&explanation.reasons).map_err(|e| e.to_string())?,
                }))
            }
            "get_verification_plan" => {
                self.ensure_store()?;
                let changed: Vec<String> = args
                    .get("changed_files")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                if changed.is_empty() {
                    return Err("missing required argument: changed_files (array of paths)".into());
                }
                let profiles = engram_verify::load_profiles(&self.repo_root);
                let m = engram_verify::match_profiles(&profiles, &changed);
                let test_commands = engram_verify::detect_test_commands(&self.repo_root);

                // Historically co-failing tests: tests in the co-change set of changed files.
                let store = self.store.as_ref().unwrap();
                let mut co_failing: Vec<String> = Vec::new();
                for f in &changed {
                    if let Ok(edges) = store.cochange_for(f, 10) {
                        for e in edges {
                            if engram_repo_map::inventory::is_test_path(&e.path_b)
                                && !co_failing.contains(&e.path_b)
                            {
                                co_failing.push(e.path_b);
                            }
                        }
                    }
                }
                co_failing.sort();
                co_failing.dedup();

                let plan = engram_domain::VerificationPlan {
                    matched_profiles: m.matched_profiles,
                    checklist: m.checklist,
                    test_commands,
                    historically_co_failing_tests: co_failing,
                };
                Ok(serde_json::to_value(plan).map_err(|e| e.to_string())?)
            }
            "get_review_history" => {
                self.ensure_store()?;
                let store = self.store.as_ref().unwrap();
                let path = args.get("path").and_then(|v| v.as_str());
                let task = args.get("task").and_then(|v| v.as_str());
                if path.is_none() && task.is_none() {
                    return Err("provide 'path' and/or 'task'".into());
                }
                let mut comments = match path {
                    Some(p) => store
                        .review_comments_for_path(p)
                        .map_err(|e| e.to_string())?,
                    None => store.all_review_comments().map_err(|e| e.to_string())?,
                };
                if let Some(t) = task {
                    use engram_retrieval::embed::{cosine, Embedder, HashedNgramEmbedder};
                    let emb = HashedNgramEmbedder::default();
                    let qv = emb.embed(t);
                    let mut scored: Vec<(f32, engram_domain::ReviewComment)> = comments
                        .into_iter()
                        .map(|c| (cosine(&qv, &emb.embed(&c.body)), c))
                        .collect();
                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    comments = scored.into_iter().map(|(_, c)| c).collect();
                }
                comments.truncate(10);
                Ok(json!({
                    "review_history": comments,
                    "note": "Raw reviewer comments, unsummarized. pr_merged=true means that PR was accepted."
                }))
            }
            other => Err(format!("unknown tool: {other}")),
        }
    }
}

/// The repository named by `--repo`, or the current directory when the flag
/// is absent. `--repo` with no value is an error rather than a silent fall
/// back to the current directory, which would index the wrong repository.
fn repo_arg(args: &[String], cwd: PathBuf) -> Result<PathBuf, String> {
    match args.iter().position(|a| a == "--repo") {
        None => Ok(cwd),
        Some(pos) => match args.get(pos + 1) {
            Some(p) => Ok(PathBuf::from(p)),
            None => Err("--repo was given without a path".to_owned()),
        },
    }
}

pub fn run() {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let args: Vec<String> = std::env::args().collect();
    let requested = repo_arg(&args, cwd.clone());
    // Subcommand: `engram ingest-github [--limit N]` ingests PR history, then exits.
    if args.iter().any(|a| a == "ingest-github") {
        let limit = args
            .iter()
            .position(|a| a == "--limit")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(50);
        let result = requested
            .and_then(|repo| resolve_repo(&repo))
            .and_then(|repo| ingest_github(&repo, limit));
        std::process::exit(match result {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[engram] ingest-github failed: {e}");
                1
            }
        });
    }
    let mut engram = match requested {
        Ok(repo) => {
            eprintln!("[engram] starting MCP server for {}", repo.display());
            Engram::start(repo)
        }
        Err(msg) => Engram::failed_without_repo(cwd, msg),
    };
    serve(&mut engram);
}

/// Ingest pull-request history for the repo's `origin` remote into the store.
fn ingest_github(repo: &Path, limit: usize) -> Result<(), String> {
    use engram_connectors_github as gh;
    let token =
        gh::token_from_env().ok_or("no GitHub token (set ENGRAM_GITHUB_TOKEN or GITHUB_TOKEN)")?;
    let remote = std::process::Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(repo)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .ok_or("could not read git remote 'origin'")?;
    let (owner, name) = gh::parse_repo_slug(&remote)
        .ok_or_else(|| format!("cannot parse owner/repo from {remote}"))?;
    eprintln!("[engram] ingesting up to {limit} PRs from {owner}/{name}");
    let client = gh::GitHubClient::new(token, owner, name).map_err(|e| e.to_string())?;
    let mut store = Store::open(repo).map_err(|e| format!("store: {e}"))?;
    let stats = gh::ingest(&mut store, &client, limit).map_err(|e| e.to_string())?;
    eprintln!(
        "[engram] ingested: {} PRs, {} changed files ({} without a diff), \
         {} commits, {} review comments",
        stats.pull_requests,
        stats.files,
        stats.files_without_patch,
        stats.commits,
        stats.review_comments
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Throwaway repo root, removed on drop. Not `tempfile`: CI builds
    /// `--locked`, so a dev-dependency would need a Cargo.lock regeneration.
    struct TempRepo(PathBuf);

    impl TempRepo {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "engram-server-test-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create temp repo");
            TempRepo(path)
        }

        /// Make `<repo>/.engram` a regular file, so creating the cache
        /// directory fails. Unlike chmod, this fails identically when the
        /// tests run as root.
        fn block_cache_dir(&self) {
            std::fs::write(self.0.join(".engram"), "not a directory").expect("write");
        }
    }

    impl Drop for TempRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Wait for the background job to stop being `Indexing`.
    fn settled(engram: &Engram) -> IndexState {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let state = engram.index_state();
            if !matches!(state, IndexState::Indexing { .. }) || Instant::now() > deadline {
                return state;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// One call to every tool that needs the index.
    fn every_indexed_tool() -> Vec<(&'static str, Value)> {
        vec![
            ("search_context", json!({ "query": "x" })),
            ("get_task_context", json!({ "task": "x" })),
            ("find_existing_implementation", json!({ "concept": "x" })),
            ("predict_impact", json!({ "task": "x" })),
            ("expand_connections", json!({ "paths": ["a.rs"] })),
            ("find_connected_files", json!({ "files": ["a.rs"] })),
            (
                "explain_connection",
                json!({ "source": "a.rs", "target": "b.rs" }),
            ),
            (
                "get_verification_plan",
                json!({ "changed_files": ["a.rs"] }),
            ),
            ("get_review_history", json!({ "path": "a.rs" })),
        ]
    }

    fn assert_failure_guidance(err: &str, repo: &Path, cause: &str) {
        assert!(
            err.contains(&repo.display().to_string()),
            "no repo path: {err}"
        );
        assert!(err.contains(cause), "no underlying error {cause:?}: {err}");
        assert!(err.contains("writable"), "no writability hint: {err}");
        assert!(err.contains("restart"), "no restart guidance: {err}");
        assert!(
            !err.to_lowercase().contains("retry"),
            "a failure must never tell the client to retry: {err}"
        );
        assert!(!err.contains("still indexing"), "{err}");
    }

    #[test]
    fn a_failed_index_is_reported_as_failed_with_guidance() {
        let repo = TempRepo::new();
        repo.block_cache_dir();
        let mut engram =
            Engram::start_with(repo.0.clone(), false, default_index_job(repo.0.clone()));

        let IndexState::Failed(cause) = settled(&engram) else {
            panic!("expected Failed, got {:?}", engram.index_state());
        };
        assert!(!cause.is_empty());
        for (tool, args) in every_indexed_tool() {
            let err = engram
                .call_tool(tool, &args)
                .expect_err(&format!("{tool} must fail when indexing failed"));
            assert_failure_guidance(&err, &repo.0, &cause);
        }
    }

    #[test]
    fn a_failed_index_reaches_the_client_as_an_is_error_result() {
        let repo = TempRepo::new();
        repo.block_cache_dir();
        let mut engram =
            Engram::start_with(repo.0.clone(), false, default_index_job(repo.0.clone()));
        settled(&engram);

        let request = json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": { "name": "search_context", "arguments": { "query": "x" } }
        });
        let mut out = Vec::new();
        crate::mcp::serve_io(&mut engram, format!("{request}\n").as_bytes(), &mut out);
        let response: Value = serde_json::from_slice(&out).expect("one json response");
        assert_eq!(response["result"]["isError"], true, "{response}");
        let text = response["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("failed to index"), "{text}");
    }

    #[test]
    fn a_panicking_index_job_is_reported_as_failed_not_indexing() {
        let repo = TempRepo::new();
        let mut engram = Engram::start_with(repo.0.clone(), false, || panic!("kaboom in indexer"));

        let IndexState::Failed(cause) = settled(&engram) else {
            panic!("a panic must not leave the state at Indexing");
        };
        assert!(cause.contains("panicked"), "{cause}");
        assert!(cause.contains("kaboom in indexer"), "{cause}");
        let err = engram
            .call_tool("search_context", &json!({ "query": "x" }))
            .unwrap_err();
        assert_failure_guidance(&err, &repo.0, "kaboom in indexer");
    }

    #[test]
    fn an_erroring_index_job_is_reported_as_failed() {
        let repo = TempRepo::new();
        let engram = Engram::start_with(repo.0.clone(), false, || {
            Err(anyhow::anyhow!("disk on fire").context("opening cache"))
        });
        let IndexState::Failed(cause) = settled(&engram) else {
            panic!("expected Failed");
        };
        // The whole error chain, not just the outermost context.
        assert!(cause.contains("opening cache"), "{cause}");
        assert!(cause.contains("disk on fire"), "{cause}");
    }

    #[test]
    fn a_nonexistent_repo_starts_the_server_in_the_failed_state() {
        let missing = std::env::temp_dir().join(format!(
            "engram-no-such-repo-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let mut engram = Engram::start(missing.clone());

        let IndexState::Failed(cause) = engram.index_state() else {
            panic!("a bad --repo must fail immediately, not hang or index cwd");
        };
        assert!(cause.contains("does not exist"), "{cause}");
        // It must not have created the directory as a side effect either.
        assert!(!missing.exists());
        let err = engram
            .call_tool("search_context", &json!({ "query": "x" }))
            .unwrap_err();
        assert_failure_guidance(&err, &missing, "does not exist");
    }

    #[test]
    fn a_repo_path_that_is_a_file_fails_clearly() {
        let repo = TempRepo::new();
        let file = repo.0.join("not-a-dir.txt");
        std::fs::write(&file, "x").unwrap();
        let engram = Engram::start(file);
        let IndexState::Failed(cause) = engram.index_state() else {
            panic!("expected Failed");
        };
        assert!(cause.contains("not a directory"), "{cause}");
    }

    #[test]
    fn repo_is_canonicalized() {
        let repo = TempRepo::new();
        let dotted = repo.0.join(".");
        let resolved = resolve_repo(&dotted).unwrap();
        assert_eq!(resolved, std::fs::canonicalize(&repo.0).unwrap());
    }

    #[test]
    fn repo_flag_without_a_path_is_an_error_not_cwd() {
        let cwd = PathBuf::from("/some/cwd");
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(repo_arg(&args(&["engram"]), cwd.clone()).unwrap(), cwd);
        assert_eq!(
            repo_arg(&args(&["engram", "--repo", "/r"]), cwd.clone()).unwrap(),
            PathBuf::from("/r")
        );
        assert!(repo_arg(&args(&["engram", "--repo"]), cwd).is_err());
    }

    #[test]
    fn while_indexing_the_client_is_asked_to_retry() {
        let repo = TempRepo::new();
        let (release, gate) = std::sync::mpsc::channel::<()>();
        let mut engram = Engram::start_with(repo.0.clone(), false, move || {
            let _ = gate.recv();
            Ok(())
        });

        assert!(matches!(engram.index_state(), IndexState::Indexing { .. }));
        for (tool, args) in every_indexed_tool() {
            let err = engram.call_tool(tool, &args).unwrap_err();
            assert!(err.contains("still indexing"), "{tool}: {err}");
            assert!(err.contains("Retry this tool call"), "{tool}: {err}");
            assert!(!err.contains("failed"), "{tool}: {err}");
        }

        release.send(()).unwrap();
        assert!(matches!(settled(&engram), IndexState::Ready));
        // Once ready, the cheap tools work without ever having been "retried"
        // through a failure path.
        engram
            .call_tool(
                "get_verification_plan",
                &json!({ "changed_files": ["a.rs"] }),
            )
            .expect("ready");
    }

    #[test]
    fn index_warnings_appear_only_when_there_is_something_to_say() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        let mut engram =
            Engram::start_with(repo.0.clone(), false, default_index_job(repo.0.clone()));
        assert!(matches!(settled(&engram), IndexState::Ready));

        let args = json!({ "paths": ["a.rs"] });
        let healthy = engram.call_tool("expand_connections", &args).unwrap();
        assert!(healthy.get("index_warnings").is_none(), "{healthy}");

        engram
            .watcher_health
            .lock()
            .unwrap()
            .mark_not_running("boom");
        let degraded = engram.call_tool("expand_connections", &args).unwrap();
        assert_eq!(
            degraded["index_warnings"],
            json!([
                "file watcher is not running (boom); results may not reflect edits made since startup"
            ])
        );
        // Existing fields are untouched.
        for key in healthy.as_object().unwrap().keys() {
            assert_eq!(healthy[key], degraded[key], "{key} changed");
        }
        let search = engram
            .call_tool("search_context", &json!({ "query": "a" }))
            .unwrap();
        assert!(search["index_warnings"].is_array(), "{search}");
        assert!(search.get("evidence").is_some() && search.get("task").is_some());
    }

    #[test]
    fn index_state_info_describes_each_state() {
        let started = Instant::now();
        let indexing = index_state_info(&IndexState::Indexing { started }, None, 0, false);
        assert_eq!(indexing["status"], "indexing");
        assert!(indexing.get("snapshot").is_none());
        let failed = index_state_info(&IndexState::Failed("boom".into()), None, 0, false);
        assert_eq!(failed["status"], "failed");
        assert_eq!(failed["error"], "boom");
        let ready = index_state_info(&IndexState::Ready, Some("abc"), 3, true);
        assert_eq!(ready["status"], "ready");
        assert_eq!(ready["snapshot"], "abc");
        assert_eq!(ready["generation"], 3);
        assert_eq!(ready["stale"], true);
    }

    fn ready_engram(repo: &TempRepo) -> Engram {
        let engram = Engram::start_with(repo.0.clone(), false, default_index_job(repo.0.clone()));
        assert!(matches!(settled(&engram), IndexState::Ready));
        engram
    }

    #[test]
    fn responses_carry_a_snapshot_id_that_follows_the_content() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        // explain_connection needs both endpoints to be indexed files.
        std::fs::write(repo.0.join("b.rs"), "fn b() {}\n").unwrap();
        let mut engram = ready_engram(&repo);
        let args = json!({ "paths": ["a.rs"] });

        let first = engram.call_tool("expand_connections", &args).unwrap();
        let snapshot = first["snapshot"]
            .as_str()
            .expect("snapshot field")
            .to_owned();
        assert!(!snapshot.is_empty());
        // Healthy and current: no index_state noise.
        assert!(first.get("index_state").is_none(), "{first}");
        for (tool, args) in [
            ("search_context", json!({ "query": "a" })),
            (
                "explain_connection",
                json!({ "source": "a.rs", "target": "b.rs" }),
            ),
        ] {
            let r = engram.call_tool(tool, &args).unwrap();
            assert_eq!(r["snapshot"], snapshot.as_str(), "{tool}");
        }

        // Edit on disk, reindex as the watcher would, and the next call
        // publishes a new generation with a different snapshot.
        std::fs::write(repo.0.join("a.rs"), "fn a_renamed() {}\n").unwrap();
        let mut store = Store::open(&repo.0).unwrap();
        engram_repo_map::reindex_file(&mut store, &repo.0, "a.rs").unwrap();
        engram.dirty.store(true, Ordering::SeqCst);
        let second = engram.call_tool("expand_connections", &args).unwrap();
        assert_ne!(second["snapshot"], snapshot.as_str());
        assert_eq!(engram.generation, 2);
        let explained = engram
            .call_tool(
                "explain_connection",
                &json!({ "source": "a.rs", "target": "b.rs" }),
            )
            .unwrap();
        assert_eq!(explained["snapshot"], second["snapshot"]);
    }

    #[test]
    fn explain_connection_with_an_unindexed_path_is_an_error() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        let mut engram = ready_engram(&repo);
        let err = engram
            .call_tool(
                "explain_connection",
                &json!({ "source": "a.rs", "target": "missing.rs" }),
            )
            .expect_err("an unindexed path must be an error, not 'not connected'");
        assert!(err.contains("missing.rs"), "{err}");
        assert!(err.contains("indexed text file"), "{err}");
    }

    #[test]
    fn a_repo_flag_without_a_value_does_not_call_the_cwd_the_repository() {
        let mut engram = Engram::failed_without_repo(
            PathBuf::from("/some/cwd"),
            "--repo was given without a path".to_owned(),
        );
        let err = engram
            .call_tool("search_context", &json!({ "query": "x" }))
            .unwrap_err();
        assert!(err.contains("--repo was given without a path"), "{err}");
        assert!(!err.contains("/some/cwd"), "{err}");
        assert!(!err.contains("the repository at"), "{err}");
        assert!(!err.to_lowercase().contains("retry"), "{err}");
        assert!(err.contains("restart"), "{err}");
    }

    #[test]
    fn a_failed_rebuild_keeps_serving_the_old_generation_and_says_so() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        let mut engram = ready_engram(&repo);
        let args = json!({ "paths": ["a.rs"] });
        let before = engram.call_tool("expand_connections", &args).unwrap();

        // Break the store under the rebuild, then ask for one.
        engram
            .store
            .as_ref()
            .unwrap()
            .conn
            .execute_batch("ALTER TABLE files RENAME TO files_x;")
            .unwrap();
        engram.dirty.store(true, Ordering::SeqCst);
        let during = engram
            .call_tool("expand_connections", &args)
            .expect("the previous generation must keep serving");
        assert_eq!(during["snapshot"], before["snapshot"]);
        let warnings = during["index_warnings"].as_array().expect("warnings");
        assert!(
            warnings[0]
                .as_str()
                .unwrap()
                .contains("rebuilding the index failed"),
            "{warnings:?}"
        );
        assert_eq!(during["index_state"]["stale"], true);
        assert_eq!(engram.generation, 1, "no new generation was published");

        // Repair it; the retry succeeds and the warning goes away.
        engram
            .store
            .as_ref()
            .unwrap()
            .conn
            .execute_batch("ALTER TABLE files_x RENAME TO files;")
            .unwrap();
        let after = engram.call_tool("expand_connections", &args).unwrap();
        assert!(after.get("index_warnings").is_none(), "{after}");
        assert_eq!(engram.generation, 2);
    }

    #[test]
    fn a_degraded_watcher_marks_the_generation_stale() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        let mut engram = ready_engram(&repo);
        engram
            .watcher_health
            .lock()
            .unwrap()
            .mark_not_running("boom");
        let r = engram
            .call_tool("expand_connections", &json!({ "paths": ["a.rs"] }))
            .unwrap();
        assert_eq!(r["index_state"]["status"], "ready");
        assert_eq!(r["index_state"]["stale"], true);
        assert!(r["index_state"]["snapshot"].is_string());
    }

    #[test]
    fn restart_reconciles_offline_edits_before_the_state_becomes_ready() {
        let repo = TempRepo::new();
        std::fs::write(
            repo.0.join("a.py"),
            "import helpers\n\ndef user():\n    pass\n",
        )
        .unwrap();
        std::fs::write(repo.0.join("helpers.py"), "def thing():\n    pass\n").unwrap();
        std::fs::write(repo.0.join("gone.py"), "def ghost():\n    pass\n").unwrap();
        drop(ready_engram(&repo));

        // While "stopped": change the import, rename the module, delete a file.
        std::fs::write(
            repo.0.join("a.py"),
            "import renamed\n\ndef user():\n    pass\n",
        )
        .unwrap();
        std::fs::rename(repo.0.join("helpers.py"), repo.0.join("renamed.py")).unwrap();
        std::fs::remove_file(repo.0.join("gone.py")).unwrap();

        let mut engram =
            Engram::start_with(repo.0.clone(), false, default_index_job(repo.0.clone()));
        assert!(matches!(settled(&engram), IndexState::Ready));
        // The instant Ready is observable, the store is already reconciled.
        let store = Store::open(&repo.0).unwrap();
        let paths: Vec<String> = store
            .all_files()
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert!(!paths.contains(&"gone.py".to_owned()), "{paths:?}");
        assert!(!paths.contains(&"helpers.py".to_owned()), "{paths:?}");
        assert!(store.symbols_matching("ghost", 5).unwrap().is_empty());
        assert_eq!(
            store.all_imports().unwrap(),
            vec![("a.py".to_owned(), "renamed".to_owned())]
        );

        // And the graph the tools use sees the new edge, not the old one.
        let to_renamed = engram
            .call_tool("expand_connections", &json!({ "paths": ["renamed.py"] }))
            .unwrap()
            .to_string();
        assert!(to_renamed.contains("a.py"), "{to_renamed}");
        let to_old = engram
            .call_tool("expand_connections", &json!({ "paths": ["helpers.py"] }))
            .unwrap()
            .to_string();
        assert!(!to_old.contains("a.py"), "{to_old}");
    }
}
