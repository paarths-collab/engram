//! engram binary: MCP server + automatic background indexing.
//! Usage: engram --repo /path/to/repo   (defaults to cwd)

use crate::mcp::{serve, ToolHandler};
use crate::watcher::{Gate, SharedHealth};
use engram_repo_map::store::Store;
use engram_retrieval::budget::{self, PackPlan};
use engram_retrieval::{Engine, GraphCaps, GraphKinds, GRAPH_MAX_EDGES, GRAPH_MAX_NODES};
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

/// The watcher's gate on this state: closed while the startup index job runs,
/// open once it has ended either way (`Ready` or `Failed`; the job records one
/// of the two even when it panics, see [`run_index_job`]). The watcher holds
/// its events back until then, so it never publishes into the middle of the
/// job's own staged publish.
fn index_settled_gate(state: &SharedIndexState) -> Gate {
    let state = state.clone();
    Arc::new(move || !matches!(*lock_state(&state), IndexState::Indexing { .. }))
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
            "[engram] indexed: {} files, {} cochange edges, {} tier1 files, {} pruned, \
             {} superseded by newer edits",
            stats.files,
            stats.cochange_edges,
            stats.tier1_files,
            stats.pruned_files,
            stats.superseded
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
    "get_graph",
];

/// What to tell a client when the cache database could not be put in WAL mode
/// (a network or otherwise unusual filesystem). Readers and the writer then
/// block one another: a rebuild that reads the database holds off the watcher's
/// publishes, and a publish holds off reads, until the busy timeout. Results
/// are still consistent, but live refreshes may lag or fail.
fn journal_mode_warning(mode: &str) -> Option<String> {
    (mode != "wal").then(|| {
        format!(
            "the index cache runs in journal mode `{mode}`, not `wal` (typical of a network \
             filesystem): reads and index updates block one another, so live refreshes may \
             lag or fail"
        )
    })
}

/// The `index_state` object attached to a response when the index is not
/// plainly Ready and current.
///
/// `generation` is the store's publish counter as last read, `built_generation`
/// the one the serving engine's facts were read at; the latter is only listed
/// when they differ, which is exactly "the store has moved past this answer".
fn index_state_info(
    state: &IndexState,
    snapshot: Option<&str>,
    generation: u64,
    built_generation: u64,
    stale: bool,
) -> Value {
    let mut info = json!({ "generation": generation, "stale": stale });
    if built_generation != generation {
        info["built_generation"] = json!(built_generation);
    }
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
    /// How many engines this process has built and swapped in so far. Not the
    /// store's generation (the published-batch counter in `meta`): many batches
    /// can land between two engine builds, and one build reads only one.
    engine_builds: u64,
    /// Why the last attempt to rebuild failed, while an older generation is
    /// still being served in its place.
    rebuild_error: Option<String>,
    /// Test hook: runs once at the end of the next `ensure_engine`, i.e. in the
    /// window between the engine being current and the query being answered.
    #[cfg(test)]
    after_ensure: Option<Box<dyn FnOnce() + Send>>,
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
            engine_builds: 0,
            rebuild_error: None,
            #[cfg(test)]
            after_ensure: None,
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
            crate::watcher::spawn(
                repo_root.clone(),
                dirty.clone(),
                watcher_health.clone(),
                index_settled_gate(&index_state),
            );
        }
        Engram {
            repo_root,
            repo_named: true,
            engine: None,
            store: None,
            index_state,
            dirty,
            watcher_health,
            engine_builds: 0,
            rebuild_error: None,
            #[cfg(test)]
            after_ensure: None,
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
        let outcome = self.ensure_engine_current();
        #[cfg(test)]
        if let Some(hook) = self.after_ensure.take() {
            hook();
        }
        outcome
    }

    fn ensure_engine_current(&mut self) -> Result<(), String> {
        self.ensure_store()?;
        // A watcher change invalidates the in-memory index, and so does lazy
        // extraction that wrote symbols/imports after the engine was built, and
        // so does ANY published generation the engine has not seen (another
        // process, a watcher batch that landed before its dirty flag was
        // noticed): the store's generation is read (one cheap meta row) and a
        // mismatch with the engine's counts as dirty.
        // Rebuild lazily here (cheap: persisted vectors mean only changed
        // files re-embed).
        let watcher_dirty = self.dirty.swap(false, Ordering::SeqCst);
        let engine_stale = match (&self.engine, &self.store) {
            (Some(engine), Some(store)) => engine.is_stale_in(store),
            _ => false,
        };
        if self.engine.is_some() && !watcher_dirty && !engine_stale {
            return Ok(());
        }
        // Build the next generation completely off to the side (chunks, BM25
        // index, vectors, and code graph together, all read from one store
        // generation) and only then swap it in with one assignment. The
        // previous engine keeps serving until then, so a query never sees new
        // symbols with an old graph or vice versa.
        let store = self.store.as_mut().unwrap();
        match Engine::build(&self.repo_root, store) {
            Ok(next) => {
                self.engine = Some(next);
                self.engine_builds += 1;
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
    /// the client: a degraded watcher, a failed rebuild, or indexed files that
    /// changed on disk before the refresh was published.
    fn index_warnings(&self) -> Vec<String> {
        let mut warnings = crate::watcher::warnings(&self.watcher_health);
        if let Some(warning) = self
            .store
            .as_ref()
            .and_then(|store| journal_mode_warning(store.journal_mode()))
        {
            warnings.push(warning);
        }
        if let Some(err) = &self.rebuild_error {
            warnings.push(format!(
                "rebuilding the index failed ({err}); results come from the previous snapshot \
                 and may be out of date"
            ));
        }
        if let Some(ahead) = self
            .engine
            .as_ref()
            .map(Engine::files_ahead_of_index)
            .filter(|n| *n > 0)
        {
            warnings.push(format!(
                "{ahead} indexed file(s) changed on disk before their refresh was published; \
                 their symbols are left out of these results until it is"
            ));
        }
        warnings
    }

    /// The `snapshot`, `generation` and (when not plainly healthy)
    /// `index_state` fields for a response served by the current engine.
    ///
    /// `generation` is the store generation the engine's facts were read at;
    /// `index_state.generation` is the store's generation now, so the two
    /// differ exactly when a publish landed after the engine was built (in
    /// which case `built_generation` is listed too and the response is stale).
    fn decorate(&self, result: &mut Value) {
        let Some(obj) = result.as_object_mut() else {
            return;
        };
        let warnings = self.index_warnings();
        // Checked here, after the query ran, so a publish that landed while it
        // was being answered is reported too.
        let engine_stale = match (&self.engine, &self.store) {
            (Some(engine), Some(store)) => engine.is_stale_in(store),
            (Some(engine), None) => engine.is_stale(),
            _ => false,
        };
        let snapshot = self.engine.as_ref().map(|e| e.snapshot_id().to_owned());
        let built_generation = self.engine.as_ref().map(Engine::generation);
        if let Some(id) = &snapshot {
            obj.insert("snapshot".to_owned(), json!(id));
        }
        if let Some(generation) = built_generation {
            obj.insert("generation".to_owned(), json!(generation));
        }
        if !warnings.is_empty() {
            obj.insert("index_warnings".to_owned(), json!(warnings));
        }
        // Only reached after `index_gate` let the call through, and the state
        // never leaves Ready afterwards, so Ready is the only state to report
        // here; the response gains `index_state` exactly when it may be stale.
        let stale = engine_stale || !warnings.is_empty();
        if stale {
            let built = built_generation.unwrap_or(0);
            let current = self
                .store
                .as_ref()
                .and_then(|store| store.generation().ok())
                .unwrap_or(built);
            obj.insert(
                "index_state".to_owned(),
                index_state_info(
                    &IndexState::Ready,
                    snapshot.as_deref(),
                    current,
                    built,
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
                "description": "ALWAYS call this FIRST, before planning or implementing any coding task. Returns ranked, evidence-backed context about this repository: existing code relevant to the query, symbols to reuse instead of reimplementing, related tests, and matched files (hybrid BM25+vector+symbol retrieval), plus any past reviewer comments on the matched files. Using this prevents duplicate implementations and wrong assumptions about the codebase. The result carries a `pack` object: the token estimator used (`pack.estimator`, an estimate from serialized bytes, not a real tokenizer), `used_tokens`, whether anything was left out (`truncated`, `omitted_count`, and a budget-dependent number of `omitted` entries naming path/symbol/reason: `over_budget` or `below_cap`), and the fixed `ordering`. With `budget_tokens`, whole evidence units are kept in rank order so the serialized result, pack included, stays within the budget (estimated); a unit too large to include whole may come back as a `minimal` record (path, symbol, line range, score, `expansion_handle`, no snippet). A budget below the size of a bare `pack` cannot be honoured: the pack is returned alone with `pack.within_budget: false`. When the answer comes from a different snapshot than requested, or the index may be stale or degraded, `pack` says so itself (`snapshot_mismatch`, `stale`, `has_warnings`, each `true` only when it applies), even when the other fields were dropped to fit the budget. Results are always ordered by score, then path, then line.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "What you are looking for, in natural language, e.g. 'add retry logic to billing webhooks'" },
                        "budget_tokens": { "type": "integer", "minimum": 1, "description": "Upper bound on the estimated tokens (ceil(bytes/3), see `pack.estimator`) of the whole serialized result, `pack` included. The budget bounds the decoded `content[0].text` the tool returns; the JSON-RPC envelope around it (string escaping, wrapper fields) is not counted. Omit for the default result with no budget truncation (existing result caps still apply and are reported in `pack`)." },
                        "snapshot": { "type": "string", "description": "A `snapshot` id from an earlier response. If it differs from the current index snapshot the answer still comes from the current snapshot and says so in `snapshot_mismatch`." }
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
                        "task": { "type": "string", "description": "The task description in natural language" },
                        "budget_tokens": { "type": "integer", "minimum": 1, "description": "Same as search_context." },
                        "snapshot": { "type": "string", "description": "Same as search_context." }
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
                "description": "Call this when you already know which file(s) you are changing (e.g. the current diff) and want everything CONNECTED to them, with zero guessing. Takes exact file paths as anchors and returns only files linked by hard, deterministic facts already recorded in the store: the co-change graph (files that historically changed together in the same commits) and the import graph (files that statically import an anchor, up to `max_hops` hops, default 2). Every result traces to a concrete recorded edge — this is fact-finding, not prediction. Prefer this over predict_impact whenever you have concrete anchor files. Results are capped (top 5 co-change rows per anchor, 8 entries per kind); the result's `pack` object discloses what the caps cut (`omitted` entries with reason `recommendation_cap`, and `omitted_count`) and what `budget_tokens` cut (`over_budget`), names the estimator (an estimate from serialized bytes, not a real tokenizer) and reports `used_tokens`. As with search_context, `pack` itself flags `snapshot_mismatch`, `stale` and `has_warnings` (each `true` only when it applies), so they survive any budget. Ordering is confidence descending, then path.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Repo-relative paths of files you are changing or about to change, e.g. ['src/billing/cancel.rs']"
                        },
                        "budget_tokens": { "type": "integer", "minimum": 1, "description": "Upper bound on the estimated tokens (ceil(bytes/3), see `pack.estimator`) of the whole serialized result, `pack` included. The budget bounds the decoded `content[0].text` the tool returns; the JSON-RPC envelope around it (string escaping, wrapper fields) is not counted. Whole entries are kept in priority order (anchors, co-change, imports, tests); an entry too large to include whole may come back as a `minimal` record with an `expansion_handle`. Omit for no budget truncation." },
                        "max_hops": { "type": "integer", "minimum": 1, "maximum": 2, "default": 2, "description": "How many import hops to follow from each anchor to find dependents (1 = direct importers only). Values outside 1..=2 are an error." },
                        "snapshot": { "type": "string", "description": "A `snapshot` id from an earlier response. If it differs from the current index snapshot the answer still comes from the current snapshot and says so in `snapshot_mismatch`." }
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
                "name": "get_graph",
                "description": "SUPPORTING tool: the raw graph FACTS around some files, for citing and for checking a recommendation. Not a ranking and not a call graph: it lists recorded edges, it does not score files. Edges are of two kinds only: `import` (single-hop, heuristic module-name resolution, `resolver: \"heuristic_module_segments\"`, `resolution_status: \"candidate\"`, never compiler-verified; always `source` = importer, `target` = imported, in either direction around an anchor, `hops: 1`) and `cochange` (a stored git-history row `source -> target` with commit `count`, `strength` relative to the commits touching `source`, the `window` it was built under, and `opportunities_source`/`opportunities_target`; the row back from a partner is its own edge with its own numbers). Symbol-level call edges are NOT recorded. Every edge has an `edge_id` (`e_` + 16 hex digits, derived from the snapshot and the edge; stable within a snapshot, changes with it, 64-bit so not globally unique): the same ids that `expand_connections` lists in `evidence_ids` and `explain_connection` reports on its reasons and per-hop `edges`. `nodes` are the anchors plus neighbours (`distance` = import hops from the nearest anchor, 1 for co-change partners). `max_hops` defaults to 2, the same as expand_connections: 1 gives every import edge touching an anchor; 2 (default) also adds the edges touching those neighbours. CITED IDS: an `edge_id` that expand_connections or explain_connection cites is guaranteed to be among these edges only when `limits.truncated` is false and `max_hops` is at least the hop it was cited at (a 2-hop dependent cites a second-hop edge that a `max_hops` 1 graph does not hold; a cap can cut a cited edge, which `limits.truncated` then reports). The `direction` of a co-change edge here is always the stored row direction (`source_to_target`), whereas the `direction` on an explain_connection reason is relative to the pair as you asked it; the same `edge_id` can therefore carry different direction labels in the two tools (the id identifies the stored fact). Unlike expand_connections, which silently ignores a path it does not know, this tool returns an error for any path that is not an indexed file. Output is deterministic: nodes by path, edges by (kind, source, target, direction). Bounded, never silently: `limits` gives `max_nodes` (default and ceiling 200), `max_edges` (default and ceiling 500; pass lower values of either to get a smaller answer), the per-anchor co-change read limit (25 partners), whether the result is `truncated`, and the exact `omitted_edges` / `omitted_nodes`. `omitted_edges` is exact: emitted edges plus `omitted_edges` equals every edge (one directed co-change row or one import edge) the same call returns with no cap, counting the reverse rows of co-change partners hidden by the per-anchor limit; `omitted_nodes` counts neighbouring files past the node cap (a partner hidden by the per-anchor limit is never added as a node). `scope` states what was searched, so a missing edge means none was recorded within it. There is no `budget_tokens`: the caps are the bound.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": { "type": "string" },
                            "minItems": 1,
                            "description": "Repo-relative paths of the anchor files, e.g. ['app/service.py']. Every path must be an indexed file."
                        },
                        "max_hops": { "type": "integer", "minimum": 1, "maximum": 2, "default": 2, "description": "Import distance to follow from the anchors (1 = edges touching an anchor, 2 = also edges touching their direct neighbours). Default 2, the same as expand_connections, so the edges an expansion cites are present (unless `limits.truncated`). Co-change edges are always anchor-to-partner. Values outside 1..=2 are an error." },
                        "max_edges": { "type": "integer", "minimum": 1, "maximum": 500, "default": 500, "description": "Return at most this many edges. Can only lower the hard cap of 500; values outside 1..=500 or non-integers are an error. Edges cut by it are counted in `limits.omitted_edges`." },
                        "max_nodes": { "type": "integer", "minimum": 1, "maximum": 200, "default": 200, "description": "Return at most this many nodes (anchors always stay, so it must be at least the number of distinct anchors). Can only lower the hard cap of 200; values outside 1..=200 or non-integers are an error. Files cut by it are counted in `limits.omitted_nodes`." },
                        "kinds": {
                            "type": "array",
                            "items": { "type": "string", "enum": ["import", "cochange"] },
                            "minItems": 1,
                            "description": "Which kinds of edge to return. Default: both."
                        },
                        "snapshot": { "type": "string", "description": "A `snapshot` id from an earlier response. If it differs from the current index snapshot the answer still comes from the current snapshot and says so in `snapshot_mismatch`." }
                    },
                    "required": ["paths"]
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
                        },
                        "budget_tokens": { "type": "integer", "minimum": 1, "description": "Same as expand_connections." },
                        "max_hops": { "type": "integer", "minimum": 1, "maximum": 2, "default": 2, "description": "Same as expand_connections." },
                        "snapshot": { "type": "string", "description": "Same as expand_connections." }
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
        let (mut result, plan) = match name {
            "search_context"
            | "get_task_context"
            | "expand_connections"
            | "find_connected_files" => {
                let (result, plan) = self.dispatch_packed(name, args)?;
                (result, Some(plan))
            }
            _ => (self.dispatch_tool(name, args)?, None),
        };
        if INDEXED_TOOLS.contains(&name) {
            self.decorate(&mut result);
        }
        // Last, so the budget covers the decorated result: the snapshot id,
        // index warnings and the pack itself are all part of what is measured.
        if let Some(plan) = plan {
            result = budget::apply(result, plan);
        }
        Ok(result)
    }
}

impl Engram {
    /// The two budgeted tools (and their deprecated aliases). Returns the
    /// undecorated result and how to pack it; [`ToolHandler::call_tool`] adds
    /// the snapshot/health fields and then applies the plan.
    fn dispatch_packed(&mut self, name: &str, args: &Value) -> Result<(Value, PackPlan), String> {
        let budget = budget_arg(args)?;
        let requested_snapshot = snapshot_arg(args)?;
        let expand = matches!(name, "expand_connections" | "find_connected_files");
        let max_hops = if expand { max_hops_arg(args)? } else { None };
        let (mut result, plan) = if expand {
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
            self.ensure_engine()?;
            let engine = self.engine.as_mut().unwrap();
            let store = self.store.as_mut().unwrap();
            let impact = engine
                .impact_from_files_with(store, &files, max_hops.unwrap_or(2))
                .map_err(|e| e.to_string())?;
            let dropped: Vec<(&'static str, String)> = impact
                .dropped
                .iter()
                .map(|d| (d.kind, d.path.clone()))
                .collect();
            (
                serde_json::to_value(&impact).map_err(|e| e.to_string())?,
                budget::expansion_plan(budget, &dropped),
            )
        } else {
            let task = args
                .get("query")
                .or_else(|| args.get("task"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if task.is_empty() {
                return Err("missing required argument: query".into());
            }
            self.ensure_engine()?;
            let engine = self.engine.as_mut().unwrap();
            let store = self.store.as_mut().unwrap();
            let outcome = engine
                .search_with_outcome(store, task, 8)
                .map_err(|e| e.to_string())?;
            // past_reviews: raw reviewer comments on the files retrieval matched.
            let mut past_reviews = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for p in &outcome.packets {
                if seen.insert(p.path.clone()) {
                    if let Ok(cs) = store.review_comments_for_path(&p.path) {
                        past_reviews.extend(cs);
                    }
                }
            }
            (
                json!({ "task": task, "evidence": outcome.packets, "past_reviews": past_reviews }),
                budget::search_plan(budget, &outcome.cap_dropped, outcome.beyond_first_cap),
            )
        };
        // Always answered from the current snapshot; a different one asked for
        // is reported, never silently served as if it were the requested one.
        if let (Some(requested), Some(engine)) = (requested_snapshot, self.engine.as_ref()) {
            if requested != engine.snapshot_id() {
                if let Some(obj) = result.as_object_mut() {
                    obj.insert(
                        "snapshot_mismatch".to_owned(),
                        json!({ "requested": requested, "current": engine.snapshot_id() }),
                    );
                }
            }
        }
        Ok((result, plan))
    }

    fn dispatch_tool(&mut self, name: &str, args: &Value) -> Result<Value, String> {
        match name {
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
            "get_graph" => {
                let requested_snapshot = snapshot_arg(args)?;
                let max_hops = max_hops_arg(args)?.unwrap_or(2);
                let kinds = kinds_arg(args)?;
                let paths = graph_paths_arg(args)?;
                let caps = GraphCaps {
                    max_nodes: bounded_arg(args, "max_nodes", GRAPH_MAX_NODES)?,
                    max_edges: bounded_arg(args, "max_edges", GRAPH_MAX_EDGES)?,
                    ..GraphCaps::default()
                };
                self.ensure_engine()?;
                let engine = self.engine.as_ref().unwrap();
                let store = self.store.as_ref().unwrap();
                let facts = engine
                    .graph_facts(store, &paths, max_hops, kinds, caps)
                    .map_err(|e| e.to_string())?;
                let mut result = serde_json::to_value(&facts).map_err(|e| e.to_string())?;
                if let Some(requested) = requested_snapshot {
                    if requested != engine.snapshot_id() {
                        if let Some(obj) = result.as_object_mut() {
                            obj.insert(
                                "snapshot_mismatch".to_owned(),
                                json!({ "requested": requested, "current": engine.snapshot_id() }),
                            );
                        }
                    }
                }
                Ok(result)
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

                // Co-changed test paths: tests in the co-change set of the changed files
                // (emitted under the legacy key `historically_co_failing_tests`).
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

/// `budget_tokens`: absent or null means no budget; otherwise an integer >= 1.
fn budget_arg(args: &Value) -> Result<Option<usize>, String> {
    match args.get("budget_tokens") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) if n >= 1 => usize::try_from(n).map(Some).map_err(|_| {
                format!(
                    "budget_tokens {n} is too large for this platform (usize); use a smaller value"
                )
            }),
            _ => Err(format!("budget_tokens must be an integer >= 1, got {v}")),
        },
    }
}

/// `max_hops`: absent or null means the default; otherwise an integer in 1..=2.
fn max_hops_arg(args: &Value) -> Result<Option<usize>, String> {
    match args.get("max_hops") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) if (1..=2).contains(&n) => usize::try_from(n)
                .map(Some)
                .map_err(|_| format!("max_hops {n} does not fit in usize")),
            _ => Err(format!(
                "max_hops must be an integer between 1 and 2, got {v}"
            )),
        },
    }
}

/// A `get_graph` bound (`max_nodes`, `max_edges`): absent or null means the
/// ceiling; otherwise an integer in `1..=ceiling`. It can lower the cap, never
/// raise it.
fn bounded_arg(args: &Value, name: &str, ceiling: usize) -> Result<usize, String> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(ceiling),
        Some(v) => match v.as_u64().and_then(|n| usize::try_from(n).ok()) {
            Some(n) if (1..=ceiling).contains(&n) => Ok(n),
            _ => Err(format!(
                "{name} must be an integer between 1 and {ceiling}, got {v}"
            )),
        },
    }
}

/// `kinds` of `get_graph`: absent or null means both; otherwise a non-empty
/// array of `"import"` / `"cochange"`.
fn kinds_arg(args: &Value) -> Result<GraphKinds, String> {
    let bad = |v: &Value| {
        format!("kinds must be a non-empty array of \"import\" and/or \"cochange\", got {v}")
    };
    match args.get("kinds") {
        None | Some(Value::Null) => Ok(GraphKinds::default()),
        Some(v) => {
            let items = v
                .as_array()
                .filter(|a| !a.is_empty())
                .ok_or_else(|| bad(v))?;
            let mut kinds = GraphKinds {
                import: false,
                cochange: false,
            };
            for item in items {
                match item.as_str() {
                    Some("import") => kinds.import = true,
                    Some("cochange") => kinds.cochange = true,
                    _ => return Err(bad(v)),
                }
            }
            Ok(kinds)
        }
    }
}

/// `paths` of `get_graph`: a non-empty array of strings.
fn graph_paths_arg(args: &Value) -> Result<Vec<String>, String> {
    let missing = || "missing required argument: paths (array of file paths)".to_owned();
    let items = args
        .get("paths")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(missing)?;
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("paths must be an array of strings, got {item}"))
        })
        .collect()
}

/// `snapshot`: absent or null means none requested; otherwise a string.
fn snapshot_arg(args: &Value) -> Result<Option<String>, String> {
    match args.get("snapshot") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) => Ok(Some(id.clone())),
        Some(v) => Err(format!("snapshot must be a string, got {v}")),
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

    #[test]
    fn the_watcher_gate_opens_exactly_when_the_index_job_has_ended_however_it_ended() {
        use std::sync::mpsc;
        for outcome in ["ready", "failed", "panicked"] {
            let state: SharedIndexState = Arc::new(Mutex::new(IndexState::Indexing {
                started: Instant::now(),
            }));
            let gate = index_settled_gate(&state);
            assert!(!gate(), "{outcome}: closed before the job starts");

            let (running_tx, running_rx) = mpsc::channel::<()>();
            let (finish_tx, finish_rx) = mpsc::channel::<()>();
            let job_state = state.clone();
            let job = std::thread::spawn(move || {
                run_index_job(&job_state, move || {
                    let _ = running_tx.send(());
                    // Bounded: a test that failed elsewhere must not hang here.
                    let _ = finish_rx.recv_timeout(Duration::from_secs(10));
                    match outcome {
                        "ready" => Ok(()),
                        "failed" => Err(anyhow::anyhow!("boom")),
                        _ => panic!("boom"),
                    }
                });
            });
            running_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("the job started");
            assert!(!gate(), "{outcome}: closed while the job runs");

            finish_tx.send(()).unwrap();
            job.join().unwrap();
            assert!(gate(), "{outcome}: open once the job has ended");
        }
    }

    #[test]
    fn a_cache_that_is_not_in_wal_mode_is_said_so_and_wal_says_nothing() {
        assert_eq!(journal_mode_warning("wal"), None);
        let warning = journal_mode_warning("delete").expect("a warning");
        assert!(
            warning.contains("`delete`") && warning.contains("`wal`"),
            "{warning}"
        );
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
            ("get_graph", json!({ "paths": ["a.rs"] })),
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
        // (`pack.used_tokens` measures the whole result, warnings included,
        // so it is the one field that legitimately grows with them.)
        for key in healthy.as_object().unwrap().keys().filter(|k| *k != "pack") {
            assert_eq!(healthy[key], degraded[key], "{key} changed");
        }
        for key in healthy["pack"]
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| *k != "used_tokens")
        {
            assert_eq!(healthy["pack"][key], degraded["pack"][key], "pack.{key}");
        }
        assert!(degraded["pack"]["used_tokens"].as_u64() > healthy["pack"]["used_tokens"].as_u64());
        let search = engram
            .call_tool("search_context", &json!({ "query": "a" }))
            .unwrap();
        assert!(search["index_warnings"].is_array(), "{search}");
        assert!(search.get("evidence").is_some() && search.get("task").is_some());
    }

    #[test]
    fn index_state_info_describes_each_state() {
        let started = Instant::now();
        let indexing = index_state_info(&IndexState::Indexing { started }, None, 0, 0, false);
        assert_eq!(indexing["status"], "indexing");
        assert!(indexing.get("snapshot").is_none());
        let failed = index_state_info(&IndexState::Failed("boom".into()), None, 0, 0, false);
        assert_eq!(failed["status"], "failed");
        assert_eq!(failed["error"], "boom");
        let ready = index_state_info(&IndexState::Ready, Some("abc"), 3, 3, true);
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
        assert_eq!(engram.engine_builds, 2);
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
        assert_eq!(engram.engine_builds, 1, "no new generation was published");

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
        assert_eq!(engram.engine_builds, 2);
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

    /// What the watcher does on a save: edit the file, then refresh it through
    /// ANOTHER connection, publishing a generation the server's engine has not
    /// seen. Deliberately does not touch the server's `dirty` flag.
    fn publish_edit(repo: &TempRepo, rel: &str, contents: &str) -> u64 {
        std::fs::write(repo.0.join(rel), contents).unwrap();
        let mut other = Store::open(&repo.0).unwrap();
        assert!(engram_repo_map::reindex_file(&mut other, &repo.0, rel).unwrap());
        other.generation().unwrap()
    }

    fn store_generation(repo: &TempRepo) -> u64 {
        Store::open(&repo.0).unwrap().generation().unwrap()
    }

    /// One call per tool that is served from the engine, with arguments that
    /// work on a repository holding `a.rs` and `b.rs`.
    fn engine_backed_calls() -> Vec<(&'static str, Value)> {
        vec![
            ("search_context", json!({ "query": "a" })),
            ("get_task_context", json!({ "task": "a" })),
            ("find_existing_implementation", json!({ "concept": "a" })),
            ("predict_impact", json!({ "task": "a" })),
            ("expand_connections", json!({ "paths": ["a.rs"] })),
            ("find_connected_files", json!({ "files": ["a.rs"] })),
            (
                "explain_connection",
                json!({ "source": "a.rs", "target": "b.rs" }),
            ),
            ("get_graph", json!({ "paths": ["a.rs"] })),
        ]
    }

    #[test]
    fn every_engine_backed_response_carries_the_generation_it_was_answered_at() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(repo.0.join("b.rs"), "fn b() {}\n").unwrap();
        let mut engram = ready_engram(&repo);
        let generation = store_generation(&repo);
        assert!(generation >= 1);

        for (tool, args) in engine_backed_calls() {
            let r = engram.call_tool(tool, &args).unwrap();
            assert_eq!(r["generation"], generation, "{tool}: {r}");
            assert!(r["snapshot"].is_string(), "{tool}");
            assert!(
                r.get("index_state").is_none(),
                "{tool}: healthy and current"
            );
        }
        assert_eq!(engram.engine_builds, 1, "nothing moved, nothing rebuilt");

        // Another connection publishes. The server's dirty flag is NOT set: the
        // generation alone makes the next call rebuild, and answer at the new one.
        let published = publish_edit(&repo, "a.rs", "fn a_renamed() {}\n");
        assert_eq!(published, generation + 1);
        let r = engram
            .call_tool("expand_connections", &json!({ "paths": ["a.rs"] }))
            .unwrap();
        assert_eq!(r["generation"], published);
        assert_eq!(engram.engine_builds, 2);
        assert!(r.get("index_state").is_none(), "{r}");
    }

    #[test]
    fn a_publish_between_the_engine_and_the_answer_is_reported_never_silently_mixed() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(repo.0.join("b.rs"), "fn b() {}\n").unwrap();
        let mut engram = ready_engram(&repo);

        for (round, (tool, args)) in engine_backed_calls().into_iter().enumerate() {
            // Make the engine current, then arrange for a publish to land in the
            // window after it is current and before the query is answered.
            engram.call_tool(tool, &args).unwrap();
            let answered_at = store_generation(&repo);
            let root = repo.0.clone();
            engram.after_ensure = Some(Box::new(move || {
                std::fs::write(root.join("b.rs"), format!("fn b_{round}() {{}}\n")).unwrap();
                let mut other = Store::open(&root).unwrap();
                engram_repo_map::reindex_file(&mut other, &root, "b.rs").unwrap();
            }));

            let r = engram.call_tool(tool, &args).unwrap();
            assert_eq!(
                r["generation"], answered_at,
                "{tool}: the facts it holds are of the generation it was built at"
            );
            let state = &r["index_state"];
            assert_eq!(state["stale"], true, "{tool}: {r}");
            assert_eq!(state["generation"], answered_at + 1, "{tool}: {r}");
            assert_eq!(state["built_generation"], answered_at, "{tool}: {r}");

            // The next call rebuilds first and is plainly current again.
            let next = engram.call_tool(tool, &args).unwrap();
            assert_eq!(next["generation"], answered_at + 1, "{tool}");
            assert!(next.get("index_state").is_none(), "{tool}: {next}");
        }
    }

    #[test]
    fn a_failed_rebuild_after_a_publish_keeps_the_old_engine_and_names_both_generations() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        let mut engram = ready_engram(&repo);
        let args = json!({ "paths": ["a.rs"] });
        let before = engram.call_tool("expand_connections", &args).unwrap();
        let old = before["generation"].as_u64().unwrap();

        let published = publish_edit(&repo, "a.rs", "fn a_renamed() {}\n");
        // Break the store under the rebuild that the new generation triggers.
        engram
            .store
            .as_ref()
            .unwrap()
            .conn
            .execute_batch("ALTER TABLE files RENAME TO files_x;")
            .unwrap();
        let during = engram
            .call_tool("expand_connections", &args)
            .expect("the previous engine must keep serving");
        assert_eq!(during["snapshot"], before["snapshot"]);
        assert_eq!(during["generation"], old);
        assert_eq!(during["index_state"]["stale"], true);
        assert_eq!(during["index_state"]["generation"], published);
        assert_eq!(during["index_state"]["built_generation"], old);
        assert!(during["index_warnings"].is_array());

        engram
            .store
            .as_ref()
            .unwrap()
            .conn
            .execute_batch("ALTER TABLE files_x RENAME TO files;")
            .unwrap();
        let after = engram.call_tool("expand_connections", &args).unwrap();
        assert_eq!(after["generation"], published);
        assert!(after.get("index_state").is_none(), "{after}");
    }

    #[test]
    fn index_state_lists_the_built_generation_only_when_the_store_has_moved_past_it() {
        let current = index_state_info(&IndexState::Ready, Some("abc"), 5, 5, true);
        assert_eq!(current["generation"], 5);
        assert!(current.get("built_generation").is_none(), "{current}");
        let behind = index_state_info(&IndexState::Ready, Some("abc"), 7, 5, true);
        assert_eq!(behind["generation"], 7);
        assert_eq!(behind["built_generation"], 5);
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

    // ---- budgeted, deterministic evidence packs ----

    /// Thirty near-duplicate modules, written in `order`, so a query matches
    /// far more than fits in a small budget.
    fn billing_repo(order: &[usize]) -> TempRepo {
        let repo = TempRepo::new();
        std::fs::create_dir_all(repo.0.join("src")).unwrap();
        for &i in order {
            let body: String = (0..14)
                .map(|n| {
                    format!("    step_{n} = attempt_{i} + {n}  # retry billing webhook delivery\n")
                })
                .collect();
            std::fs::write(
                repo.0.join(format!("src/svc{i:02}.py")),
                format!(
                    "def retry_billing_webhook_{i}(event):\n    \"\"\"Retry the billing webhook delivery.\"\"\"\n    attempt_{i} = 0\n{body}    return attempt_{i}\n"
                ),
            )
            .unwrap();
        }
        repo
    }

    const BILLING_QUERY: &str = "retry billing webhook delivery";

    fn tokens(result: &Value) -> usize {
        budget::estimate_tokens(&budget::render(result))
    }

    fn array_len(result: &Value, key: &str) -> usize {
        result[key].as_array().map_or(0, Vec::len)
    }

    #[test]
    fn a_budgeted_search_stays_within_its_serialized_budget_and_names_what_it_omitted() {
        let repo = billing_repo(&(0..30).collect::<Vec<_>>());
        let mut engram = ready_engram(&repo);
        let unbounded = engram
            .call_tool("search_context", &json!({ "query": BILLING_QUERY }))
            .unwrap();
        let all = array_len(&unbounded, "evidence");
        assert!(all >= 8, "corpus must yield many candidates, got {all}");
        assert!(tokens(&unbounded) > 2000, "{}", tokens(&unbounded));

        let result = engram
            .call_tool(
                "search_context",
                &json!({ "query": BILLING_QUERY, "budget_tokens": 2000 }),
            )
            .unwrap();
        // Measured on the exact text the MCP layer returns as the content.
        let text = budget::render(&result);
        assert!(budget::estimate_tokens(&text) <= 2000, "{}", text.len());
        let pack = &result["pack"];
        assert_eq!(pack["estimator"], "bytes3-v1");
        assert_eq!(pack["budget_tokens"], 2000);
        assert_eq!(pack["truncated"], true);
        assert_eq!(pack["within_budget"], true);
        assert_eq!(pack["ordering"], "score_desc,path_asc,line_asc");
        let used = pack["used_tokens"].as_u64().unwrap() as usize;
        assert!(used >= tokens(&result) && used <= 2000, "{used}");

        let kept = array_len(&result, "evidence");
        assert!(kept > 0 && kept < all, "kept {kept} of {all}");
        let omitted = pack["omitted"].as_array().unwrap();
        assert!(!omitted.is_empty());
        assert!(pack["omitted_count"].as_u64().unwrap() as usize >= all - kept);
        for entry in omitted {
            assert!(
                entry["path"].as_str().unwrap().starts_with("src/svc"),
                "{entry}"
            );
            assert!(
                ["over_budget", "below_cap"].contains(&entry["reason"].as_str().unwrap()),
                "{entry}"
            );
        }
        assert!(omitted.iter().any(|e| e["reason"] == "over_budget"));
        // What was kept is a prefix of the unbudgeted ranking, in the same order.
        for (got, want) in result["evidence"]
            .as_array()
            .unwrap()
            .iter()
            .zip(unbounded["evidence"].as_array().unwrap())
        {
            assert_eq!(got["id"], want["id"]);
        }
    }

    #[test]
    fn the_same_query_and_differently_ordered_inputs_give_byte_identical_output() {
        let forward: Vec<usize> = (0..30).collect();
        let shuffled: Vec<usize> = (0..30).map(|i| (i * 7) % 30).collect();
        assert_ne!(forward, shuffled);
        let call = |order: &[usize]| {
            let repo = billing_repo(order);
            let mut engram = ready_engram(&repo);
            let args = json!({ "query": BILLING_QUERY, "budget_tokens": 2000 });
            let warm = engram.call_tool("search_context", &args).unwrap();
            let first = budget::render(&engram.call_tool("search_context", &args).unwrap());
            let second = budget::render(&engram.call_tool("search_context", &args).unwrap());
            assert_eq!(first, second, "same query twice");
            assert_eq!(first, budget::render(&warm), "first call vs later calls");
            first
        };
        assert_eq!(call(&forward), call(&shuffled));
    }

    #[test]
    fn a_tiny_budget_gets_minimal_records_and_a_tinier_one_gets_only_the_pack() {
        let repo = billing_repo(&(0..30).collect::<Vec<_>>());
        let mut engram = ready_engram(&repo);
        let at = |engram: &mut Engram, budget: usize| {
            engram
                .call_tool(
                    "search_context",
                    &json!({ "query": BILLING_QUERY, "budget_tokens": budget }),
                )
                .unwrap()
        };
        let total = array_len(
            &engram
                .call_tool("search_context", &json!({ "query": BILLING_QUERY }))
                .unwrap(),
            "evidence",
        );

        // The smallest budget the pack itself fits in: nothing else does.
        let bare = (1..400)
            .find(|b| at(&mut engram, *b)["pack"]["within_budget"] == true)
            .expect("a bare pack fits in 400 tokens");
        let nothing = at(&mut engram, bare);
        assert!(tokens(&nothing) <= bare);
        assert_eq!(array_len(&nothing, "evidence"), 0, "{nothing}");
        assert_eq!(nothing["pack"]["truncated"], true);
        assert!(nothing["pack"]["omitted_count"].as_u64().unwrap() as usize >= total);

        // Below that, the budget cannot be met and the response says so.
        if bare > 1 {
            let impossible = at(&mut engram, bare - 1);
            assert_eq!(impossible["pack"]["within_budget"], false);
            assert_eq!(array_len(&impossible, "evidence"), 0);
        }

        // The first budget that admits any evidence admits a minimal record.
        let (budget, admitted) = (bare..4000)
            .step_by(4)
            .map(|b| (b, at(&mut engram, b)))
            .find(|(_, r)| array_len(r, "evidence") > 0)
            .expect("some budget admits evidence");
        assert!(tokens(&admitted) <= budget);
        let first = &admitted["evidence"][0];
        assert_eq!(first["minimal"], true, "{first}");
        assert!(first.get("snippet").is_none());
        assert!(first["path"].as_str().unwrap().starts_with("src/svc"));
        assert!(first["symbol"].is_string());
        assert!(first["start_line"].is_number() && first["end_line"].is_number());
        assert!(first["score"].is_number());
        assert_eq!(first["expansion_handle"]["path"], first["path"]);
        assert_eq!(first["expansion_handle"]["symbol"], first["symbol"]);
        assert_eq!(admitted["pack"]["truncated"], true);
    }

    #[test]
    fn without_a_budget_the_evidence_is_what_it_was_and_the_pack_only_reports() {
        let repo = billing_repo(&(0..30).collect::<Vec<_>>());
        let mut engram = ready_engram(&repo);
        let result = engram
            .call_tool("search_context", &json!({ "query": BILLING_QUERY }))
            .unwrap();
        let before = {
            let engine = engram.engine.as_mut().unwrap();
            let store = engram.store.as_mut().unwrap();
            engine.search(store, BILLING_QUERY, 8).unwrap()
        };
        assert_eq!(array_len(&result, "evidence"), before.len());
        for (got, want) in result["evidence"].as_array().unwrap().iter().zip(&before) {
            assert_eq!(got["id"], want.id.as_str());
            assert_eq!(got["path"], want.path.as_str());
            assert!(got.get("minimal").is_none());
            assert_eq!(got["snippet"], json!(want.snippet));
        }
        let pack = &result["pack"];
        assert_eq!(pack["budget_tokens"], Value::Null);
        assert_eq!(pack["estimator"], budget::TOKEN_ESTIMATOR);
        assert_eq!(pack["within_budget"], true);
        let used = pack["used_tokens"].as_u64().unwrap() as usize;
        assert!(used >= tokens(&result) && used <= tokens(&result) + 1);
        // Everything it used to return is still there, with its old fields.
        for key in ["task", "evidence", "past_reviews", "snapshot"] {
            assert!(result.get(key).is_some(), "{key} missing");
        }
        // The only omissions are the search's own caps, and they are named.
        for entry in pack["omitted"].as_array().unwrap() {
            assert_eq!(entry["reason"], "below_cap");
        }
    }

    /// `app/anchor.py` with 12 co-change partners and 12 direct importers plus
    /// one file that imports an importer (two hops away).
    fn fan_repo() -> (TempRepo, Engram) {
        let repo = TempRepo::new();
        std::fs::create_dir_all(repo.0.join("app")).unwrap();
        let mut names = vec!["anchor".to_owned(), "mid".to_owned(), "top".to_owned()];
        names.extend(["chain_a", "chain_b", "chain_c"].map(str::to_owned));
        names.extend((0..12).map(|i| format!("p{i:02}")));
        names.extend((0..11).map(|i| format!("imp{i:02}")));
        for n in &names {
            std::fs::write(
                repo.0.join(format!("app/{n}.py")),
                format!("def {n}():\n    pass\n"),
            )
            .unwrap();
        }
        let engram = ready_engram(&repo);
        let mut store = Store::open(&repo.0).unwrap();
        store
            .replace_imports_for_file("app/mid.py", &["anchor".to_owned()])
            .unwrap();
        store
            .replace_imports_for_file("app/top.py", &["mid".to_owned()])
            .unwrap();
        // chain_c imports chain_b imports chain_a: a two-hop chain no cap touches.
        store
            .replace_imports_for_file("app/chain_b.py", &["chain_a".to_owned()])
            .unwrap();
        store
            .replace_imports_for_file("app/chain_c.py", &["chain_b".to_owned()])
            .unwrap();
        for i in 0..11 {
            store
                .replace_imports_for_file(&format!("app/imp{i:02}.py"), &["anchor".to_owned()])
                .unwrap();
        }
        let edges: Vec<engram_domain::CoChange> = (0..12)
            .map(|i| engram_domain::CoChange {
                path_a: "app/anchor.py".to_owned(),
                path_b: format!("app/p{i:02}.py"),
                count: 50 - i,
                strength: 0.9 - i as f32 * 0.01,
            })
            .collect();
        store.replace_cochange(&edges).unwrap();
        engram.dirty.store(true, Ordering::SeqCst);
        (repo, engram)
    }

    #[test]
    fn expand_connections_discloses_exactly_what_its_caps_dropped() {
        let (_repo, mut engram) = fan_repo();
        let result = engram
            .call_tool("expand_connections", &json!({ "paths": ["app/anchor.py"] }))
            .unwrap();
        // Defaults unchanged: 5 co-change per anchor, 8 imports.
        assert_eq!(array_len(&result, "cochange_expansions"), 5);
        assert_eq!(array_len(&result, "import_expansions"), 8);
        // 12 partners - 5 returned; 12 direct + 1 two-hop importers - 8 returned.
        let pack = &result["pack"];
        assert_eq!(pack["omitted_count"], 7 + 5);
        assert_eq!(pack["truncated"], true);
        let omitted = pack["omitted"].as_array().unwrap();
        assert_eq!(omitted.len(), 12);
        let of = |kind: &str| -> Vec<&str> {
            omitted
                .iter()
                .filter(|e| e["kind"] == kind)
                .map(|e| e["path"].as_str().unwrap())
                .collect()
        };
        assert_eq!(of("cochange").len(), 7);
        assert_eq!(of("import").len(), 5);
        assert!(omitted.iter().all(|e| e["reason"] == "recommendation_cap"));
        // None of the disclosed paths is one that was returned.
        for (kind, key) in [
            ("cochange", "cochange_expansions"),
            ("import", "import_expansions"),
        ] {
            let returned: Vec<&str> = result[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["path"].as_str().unwrap())
                .collect();
            assert!(of(kind).iter().all(|p| !returned.contains(p)), "{kind}");
        }
        assert!(
            of("import").contains(&"app/top.py"),
            "two-hop importer is cut"
        );
    }

    #[test]
    fn expand_connections_with_a_budget_counts_both_kinds_of_omission() {
        let (_repo, mut engram) = fan_repo();
        let args = json!({ "paths": ["app/anchor.py"], "budget_tokens": 500 });
        let result = engram.call_tool("expand_connections", &args).unwrap();
        let text = budget::render(&result);
        assert!(budget::estimate_tokens(&text) <= 500, "{}", text.len());
        let pack = &result["pack"];
        assert_eq!(pack["budget_tokens"], 500);
        assert_eq!(pack["ordering"], "confidence_desc,path_asc");
        let kept: usize = [
            "likely_files",
            "cochange_expansions",
            "import_expansions",
            "likely_tests",
        ]
        .iter()
        .map(|k| array_len(&result, k))
        .sum();
        // 1 anchor + 5 co-change + 8 imports = 14 candidates before the budget;
        // each is kept (whole or minimal) or counted, and 12 were cap-cut.
        assert_eq!(
            pack["omitted_count"].as_u64().unwrap() as usize + kept,
            14 + 12,
            "{pack}"
        );
        assert!(pack["omitted"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["reason"] == "over_budget"));
        // Same call again: identical bytes.
        let again = engram.call_tool("expand_connections", &args).unwrap();
        assert_eq!(text, budget::render(&again));
    }

    #[test]
    fn expand_connections_ignores_the_order_and_repetition_of_paths() {
        let (_repo, mut engram) = fan_repo();
        let call = |engram: &mut Engram, paths: Value| {
            budget::render(
                &engram
                    .call_tool(
                        "expand_connections",
                        &json!({ "paths": paths, "budget_tokens": 600 }),
                    )
                    .unwrap(),
            )
        };
        let a = call(&mut engram, json!(["app/anchor.py", "app/mid.py"]));
        let b = call(
            &mut engram,
            json!(["app/mid.py", "app/anchor.py", "app/mid.py"]),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn max_hops_one_excludes_two_hop_dependents_and_out_of_range_is_an_error() {
        let (_repo, mut engram) = fan_repo();
        let imports = |result: &Value| -> Vec<String> {
            result["import_expansions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["path"].as_str().unwrap().to_owned())
                .collect()
        };
        // chain_b imports the anchor directly, chain_c only through chain_b.
        let chain = |engram: &mut Engram, extra: Value| {
            let mut args = json!({ "paths": ["app/chain_a.py"] });
            args.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            engram.call_tool("expand_connections", &args).unwrap()
        };
        assert_eq!(
            imports(&chain(&mut engram, json!({}))),
            ["app/chain_b.py", "app/chain_c.py"]
        );
        assert_eq!(
            imports(&chain(&mut engram, json!({ "max_hops": 2 }))),
            ["app/chain_b.py", "app/chain_c.py"]
        );
        let hop1 = chain(&mut engram, json!({ "max_hops": 1 }));
        assert_eq!(imports(&hop1), ["app/chain_b.py"]);
        assert!(!budget::render(&hop1).contains("chain_c"), "{hop1}");
        assert_eq!(hop1["pack"]["omitted_count"], 0);
        let anchor_two = engram
            .call_tool(
                "expand_connections",
                &json!({ "paths": ["app/anchor.py"], "max_hops": 2 }),
            )
            .unwrap();
        assert!(anchor_two["pack"]["omitted"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["path"] == "app/top.py"));
        let default = engram
            .call_tool("expand_connections", &json!({ "paths": ["app/anchor.py"] }))
            .unwrap();
        assert_eq!(budget::render(&default), budget::render(&anchor_two));

        let one = engram
            .call_tool(
                "expand_connections",
                &json!({ "paths": ["app/anchor.py"], "max_hops": 1 }),
            )
            .unwrap();
        // top is not merely hidden: it is neither returned nor counted.
        assert!(!budget::render(&one).contains("app/top.py"));
        assert_eq!(array_len(&one, "import_expansions"), 8);
        assert_eq!(one["pack"]["omitted_count"], 7 + 4);

        for bad in [json!(0), json!(3), json!(-1), json!("1"), json!(1.5)] {
            let err = engram
                .call_tool(
                    "expand_connections",
                    &json!({ "paths": ["app/anchor.py"], "max_hops": bad }),
                )
                .expect_err("max_hops outside 1..=2 must be an error");
            assert!(err.contains("max_hops"), "{err}");
        }
        // The deprecated alias takes the new arguments too.
        let alias = engram
            .call_tool(
                "find_connected_files",
                &json!({ "files": ["app/anchor.py"], "max_hops": 3 }),
            )
            .expect_err("alias validates max_hops too");
        assert!(alias.contains("max_hops"), "{alias}");
    }

    #[test]
    fn a_snapshot_other_than_the_current_one_is_flagged_never_silently_served() {
        let repo = billing_repo(&[0, 1, 2]);
        let mut engram = ready_engram(&repo);
        let current = engram
            .call_tool("search_context", &json!({ "query": BILLING_QUERY }))
            .unwrap()["snapshot"]
            .as_str()
            .unwrap()
            .to_owned();
        for (tool, args) in [
            ("search_context", json!({ "query": BILLING_QUERY })),
            ("get_task_context", json!({ "task": BILLING_QUERY })),
            ("expand_connections", json!({ "paths": ["src/svc00.py"] })),
            ("find_connected_files", json!({ "files": ["src/svc00.py"] })),
        ] {
            let mut stale = args.clone();
            stale["snapshot"] = json!("snap-from-yesterday");
            let r = engram.call_tool(tool, &stale).unwrap();
            assert_eq!(r["snapshot"], current.as_str(), "{tool}");
            assert_eq!(
                r["snapshot_mismatch"],
                json!({ "requested": "snap-from-yesterday", "current": current }),
                "{tool}"
            );
            let mut same = args.clone();
            same["snapshot"] = json!(current);
            let r = engram.call_tool(tool, &same).unwrap();
            assert!(r.get("snapshot_mismatch").is_none(), "{tool}: {r}");
            assert!(engram
                .call_tool(tool, &args)
                .unwrap()
                .get("snapshot_mismatch")
                .is_none());
        }
        let err = engram
            .call_tool("search_context", &json!({ "query": "x", "snapshot": 7 }))
            .expect_err("snapshot must be a string");
        assert!(err.contains("snapshot"), "{err}");
    }

    #[test]
    fn get_graph_is_advertised_as_a_supporting_tool_without_a_budget() {
        let repo = TempRepo::new();
        let engram = Engram::start_with(repo.0.clone(), false, || Ok(()));
        let tools = engram.list_tools();
        let tool = tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "get_graph")
            .expect("get_graph is listed");
        let description = tool["description"].as_str().unwrap();
        assert!(description.starts_with("SUPPORTING"), "{description}");
        assert!(description.contains("not a call graph"), "{description}");
        let props = &tool["inputSchema"]["properties"];
        assert_eq!(props["paths"]["type"], "array");
        assert_eq!(props["max_hops"]["maximum"], 2);
        assert_eq!(props["max_hops"]["default"], 2);
        assert_eq!(props["max_edges"]["maximum"], 500);
        assert_eq!(props["max_edges"]["default"], 500);
        assert_eq!(props["max_nodes"]["maximum"], 200);
        assert_eq!(props["max_nodes"]["default"], 200);
        assert!(
            description.contains("limits.truncated")
                && description.contains("is at least the hop it was cited at"),
            "{description}"
        );
        assert_eq!(
            props["kinds"]["items"]["enum"],
            json!(["import", "cochange"])
        );
        assert_eq!(props["snapshot"]["type"], "string");
        assert!(props.get("budget_tokens").is_none(), "{props}");
        assert_eq!(tool["inputSchema"]["required"], json!(["paths"]));
    }

    #[test]
    fn get_graph_serves_facts_with_the_snapshot_and_validates_its_arguments() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(repo.0.join("b.rs"), "fn b() {}\n").unwrap();
        let mut engram = ready_engram(&repo);

        let r = engram
            .call_tool("get_graph", &json!({ "paths": ["a.rs"] }))
            .unwrap();
        let expanded = engram
            .call_tool("expand_connections", &json!({ "paths": ["a.rs"] }))
            .unwrap();
        assert_eq!(r["snapshot"], expanded["snapshot"]);
        assert_eq!(r["anchors"], json!(["a.rs"]));
        assert_eq!(r["nodes"][0]["path"], "a.rs");
        assert_eq!(r["nodes"][0]["role"], "anchor");
        assert_eq!(r["edges"], json!([]));
        assert_eq!(r["limits"]["truncated"], false);
        assert_eq!(r["limits"]["max_nodes"], 200);
        assert_eq!(r["limits"]["max_edges"], 500);
        assert!(r.get("snapshot_mismatch").is_none(), "{r}");
        assert!(r.get("index_state").is_none(), "healthy index: {r}");

        // Another snapshot is flagged, never silently served as requested.
        let current = r["snapshot"].as_str().unwrap();
        let stale = engram
            .call_tool(
                "get_graph",
                &json!({ "paths": ["a.rs"], "snapshot": "snap-from-yesterday" }),
            )
            .unwrap();
        assert_eq!(
            stale["snapshot_mismatch"],
            json!({ "requested": "snap-from-yesterday", "current": current })
        );
        let same = engram
            .call_tool(
                "get_graph",
                &json!({ "paths": ["a.rs"], "snapshot": current }),
            )
            .unwrap();
        assert!(same.get("snapshot_mismatch").is_none(), "{same}");

        // kinds narrows what is searched, and says so.
        let imports_only = engram
            .call_tool(
                "get_graph",
                &json!({ "paths": ["a.rs"], "kinds": ["import"], "max_hops": 2 }),
            )
            .unwrap();
        assert_eq!(imports_only["scope"]["kinds"], json!(["import"]));
        assert_eq!(imports_only["scope"]["import_max_hops"], 2);

        for (args, needle) in [
            (
                json!({ "paths": ["missing.rs"] }),
                "not an indexed text file",
            ),
            (json!({ "paths": ["a.rs", "missing.rs"] }), "missing.rs"),
            (json!({}), "paths"),
            (json!({ "paths": [] }), "paths"),
            (json!({ "paths": [7] }), "strings"),
            (json!({ "paths": ["a.rs"], "max_hops": 3 }), "max_hops"),
            (json!({ "paths": ["a.rs"], "max_hops": 0 }), "max_hops"),
            (json!({ "paths": ["a.rs"], "kinds": [] }), "kinds"),
            (json!({ "paths": ["a.rs"], "kinds": ["calls"] }), "kinds"),
            (json!({ "paths": ["a.rs"], "kinds": "import" }), "kinds"),
            (json!({ "paths": ["a.rs"], "snapshot": 7 }), "snapshot"),
        ] {
            let err = engram
                .call_tool("get_graph", &args)
                .expect_err(&format!("{args} must be rejected"));
            assert!(err.contains(needle), "{args}: {err}");
        }
    }

    fn edge_ids_of(graph: &Value) -> std::collections::BTreeSet<String> {
        graph["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["edge_id"].as_str().unwrap().to_owned())
            .collect()
    }

    #[test]
    fn get_graph_defaults_to_the_expansion_depth_so_what_it_cites_is_listed() {
        let (_repo, mut engram) = fan_repo();
        // chain_c -> chain_b -> chain_a: chain_c is a 2-hop dependent.
        let anchor = json!({ "paths": ["app/chain_a.py"] });
        let expanded = engram.call_tool("expand_connections", &anchor).unwrap();
        let mut cited = std::collections::BTreeSet::new();
        for key in ["cochange_expansions", "import_expansions"] {
            for entry in expanded[key].as_array().unwrap() {
                for id in entry["evidence_ids"].as_array().unwrap() {
                    cited.insert(id.as_str().unwrap().to_owned());
                }
            }
        }
        assert_eq!(cited.len(), 2, "chain_b's edge and chain_c's outer hop");

        // (a) At default arguments the graph has both hops, nothing cut.
        let default = engram.call_tool("get_graph", &anchor).unwrap();
        assert_eq!(default["scope"]["import_max_hops"], 2);
        assert_eq!(default["limits"]["truncated"], false);
        assert!(cited.is_subset(&edge_ids_of(&default)), "{default}");

        // (b) With max_hops 1 the outer-hop id is absent, without truncation.
        let one = engram
            .call_tool(
                "get_graph",
                &json!({ "paths": ["app/chain_a.py"], "max_hops": 1 }),
            )
            .unwrap();
        assert_eq!(one["limits"]["truncated"], false);
        let one_ids = edge_ids_of(&one);
        let missing: Vec<_> = cited.difference(&one_ids).collect();
        assert_eq!(missing.len(), 1, "{missing:?}");

        // (c) A cap that cuts a cited edge is reported as truncation.
        let cut = engram
            .call_tool(
                "get_graph",
                &json!({ "paths": ["app/chain_a.py"], "max_edges": 1 }),
            )
            .unwrap();
        assert!(!cited.is_subset(&edge_ids_of(&cut)));
        assert_eq!(cut["limits"]["truncated"], true);
        assert_eq!(cut["limits"]["omitted_edges"], 1);
    }

    #[test]
    fn get_graph_max_edges_and_max_nodes_only_lower_the_hard_caps() {
        let (_repo, mut engram) = fan_repo();
        let anchor = |extra: Value| {
            let mut args = json!({ "paths": ["app/anchor.py"] });
            for (k, v) in extra.as_object().unwrap() {
                args[k] = v.clone();
            }
            args
        };
        for (arg, bad) in [
            ("max_edges", json!(0)),
            ("max_edges", json!(501)),
            ("max_edges", json!(-1)),
            ("max_edges", json!(1.5)),
            ("max_edges", json!(10.0)),
            ("max_edges", json!("10")),
            ("max_edges", json!(true)),
            ("max_nodes", json!(0)),
            ("max_nodes", json!(201)),
            ("max_nodes", json!(2.5)),
            ("max_nodes", json!([5])),
        ] {
            let err = engram
                .call_tool("get_graph", &anchor(json!({ arg: bad.clone() })))
                .expect_err(&format!("{arg}={bad} must be rejected"));
            assert!(err.contains(arg), "{arg}={bad}: {err}");
        }

        let full = engram.call_tool("get_graph", &anchor(json!({}))).unwrap();
        let (edges, nodes) = (
            full["edges"].as_array().unwrap().len(),
            full["nodes"].as_array().unwrap().len(),
        );
        // 12 co-change rows, 12 direct importers, and top -> mid (2 hops).
        assert_eq!((edges, nodes), (25, 26));
        assert_eq!(full["limits"]["truncated"], false);
        // The ceilings themselves, and null, are the defaults.
        for same in [
            json!({ "max_edges": 500, "max_nodes": 200 }),
            json!({ "max_edges": null, "max_nodes": null }),
        ] {
            assert_eq!(engram.call_tool("get_graph", &anchor(same)).unwrap(), full);
        }

        // Lowering works, and the counts stay exact.
        let few = engram
            .call_tool("get_graph", &anchor(json!({ "max_edges": 10 })))
            .unwrap();
        assert_eq!(few["edges"].as_array().unwrap().len(), 10);
        assert_eq!(few["limits"]["max_edges"], 10);
        assert_eq!(few["limits"]["truncated"], true);
        assert_eq!(few["limits"]["omitted_edges"], edges - 10);
        let small = engram
            .call_tool("get_graph", &anchor(json!({ "max_nodes": 5 })))
            .unwrap();
        assert_eq!(small["nodes"].as_array().unwrap().len(), 5);
        assert_eq!(small["limits"]["max_nodes"], 5);
        assert_eq!(small["limits"]["omitted_nodes"], nodes - 5);
        let kept = small["edges"].as_array().unwrap().len();
        assert_eq!(small["limits"]["omitted_edges"], edges - kept);

        // More anchors than max_nodes is refused outright.
        let err = engram
            .call_tool(
                "get_graph",
                &json!({ "paths": ["app/anchor.py", "app/mid.py"], "max_nodes": 1 }),
            )
            .expect_err("two anchors do not fit in one node");
        assert!(err.contains("node cap"), "{err}");
    }

    #[test]
    fn a_budget_that_is_not_a_positive_integer_is_an_error() {
        let repo = billing_repo(&[0]);
        let mut engram = ready_engram(&repo);
        for bad in [json!(0), json!(-5), json!("2000"), json!(12.5), json!(true)] {
            let err = engram
                .call_tool(
                    "search_context",
                    &json!({ "query": BILLING_QUERY, "budget_tokens": bad }),
                )
                .expect_err("invalid budget_tokens must be an error");
            assert!(err.contains("budget_tokens"), "{err}");
        }
        // null means "no budget", like leaving it out.
        let r = engram
            .call_tool(
                "search_context",
                &json!({ "query": BILLING_QUERY, "budget_tokens": null }),
            )
            .unwrap();
        assert_eq!(r["pack"]["budget_tokens"], Value::Null);
    }

    #[test]
    fn the_schemas_advertise_budget_hops_and_snapshot() {
        let repo = TempRepo::new();
        let engram = Engram::start_with(repo.0.clone(), false, || Ok(()));
        let tools = engram.list_tools();
        let props = |name: &str| -> Value {
            tools["tools"]
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["name"] == name)
                .unwrap()["inputSchema"]["properties"]
                .clone()
        };
        for tool in ["search_context", "get_task_context"] {
            let p = props(tool);
            assert_eq!(p["budget_tokens"]["type"], "integer", "{tool}");
            assert_eq!(p["budget_tokens"]["minimum"], 1, "{tool}");
            assert_eq!(p["snapshot"]["type"], "string", "{tool}");
        }
        for tool in ["expand_connections", "find_connected_files"] {
            let p = props(tool);
            assert_eq!(p["budget_tokens"]["minimum"], 1, "{tool}");
            assert_eq!(p["max_hops"]["minimum"], 1, "{tool}");
            assert_eq!(p["max_hops"]["maximum"], 2, "{tool}");
            assert_eq!(p["max_hops"]["default"], 2, "{tool}");
            assert_eq!(p["snapshot"]["type"], "string", "{tool}");
        }
    }

    #[test]
    fn the_mcp_content_text_is_exactly_the_rendered_result_and_within_the_budget() {
        // Through the real protocol loop, not just `call_tool`: the budget is a
        // promise about the decoded `content[0].text` the client receives.
        let repo = billing_repo(&(0..30).collect::<Vec<_>>());
        let mut engram = ready_engram(&repo);
        for budget_tokens in [700usize, 2000] {
            let args = json!({ "query": BILLING_QUERY, "budget_tokens": budget_tokens });
            let direct = engram.call_tool("search_context", &args).unwrap();
            let request = json!({
                "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                "params": { "name": "search_context", "arguments": args },
            });
            let mut out = Vec::new();
            crate::mcp::serve_io(
                &mut engram,
                std::io::Cursor::new(format!("{request}\n")),
                &mut out,
            );
            let out = String::from_utf8(out).unwrap();
            let reply: Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
            assert_eq!(reply["id"], 7);
            assert_eq!(reply["result"]["isError"], false, "{reply}");
            let text = reply["result"]["content"][0]["text"].as_str().unwrap();
            assert_eq!(text, budget::render(&direct), "budget {budget_tokens}");
            assert!(
                budget::estimate_tokens(text) <= budget_tokens,
                "budget {budget_tokens}: {} tokens",
                budget::estimate_tokens(text)
            );
            let decoded: Value = serde_json::from_str(text).unwrap();
            assert_eq!(decoded["pack"]["within_budget"], true);
        }
    }

    #[test]
    fn a_snapshot_mismatch_is_still_disclosed_when_the_budget_leaves_only_the_pack() {
        let repo = billing_repo(&(0..6).collect::<Vec<_>>());
        let mut engram = ready_engram(&repo);
        let mut flagged_pack_only = 0;
        for budget_tokens in 1..=400 {
            let r = engram
                .call_tool(
                    "search_context",
                    &json!({
                        "query": BILLING_QUERY,
                        "budget_tokens": budget_tokens,
                        "snapshot": "not-the-current-snapshot",
                    }),
                )
                .unwrap();
            if r["pack"]["within_budget"] != true {
                continue;
            }
            assert_eq!(
                r["pack"]["snapshot_mismatch"], true,
                "budget {budget_tokens}: {r}"
            );
            if r.as_object().unwrap().len() == 1 {
                flagged_pack_only += 1;
            }
        }
        assert!(
            flagged_pack_only > 0,
            "the sweep never reached the pack-only layout"
        );
    }
}
