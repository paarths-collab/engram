//! engram binary: MCP server + automatic background indexing.
//! Usage: engram --repo /path/to/repo   (defaults to cwd)

use crate::mcp::{serve, ToolHandler};
use engram_repo_map::store::Store;
use engram_retrieval::Engine;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const EAGER_TIER1_LIMIT: usize = 3000;

pub struct Engram {
    repo_root: PathBuf,
    engine: Option<Engine>,
    store: Option<Store>,
    index_ready: Arc<AtomicBool>,
}

impl Engram {
    pub fn start(repo_root: PathBuf) -> Self {
        let index_ready = Arc::new(AtomicBool::new(false));
        let ready = index_ready.clone();
        let root = repo_root.clone();
        // Background indexing: server accepts connections immediately;
        // tools report progress until the index is ready.
        std::thread::spawn(
            move || match engram_repo_map::index_repo(&root, EAGER_TIER1_LIMIT) {
                Ok(stats) => {
                    eprintln!(
                        "[engram] indexed: {} files, {} cochange edges, {} tier1 files",
                        stats.files, stats.cochange_edges, stats.tier1_files
                    );
                    ready.store(true, Ordering::SeqCst);
                }
                Err(e) => eprintln!("[engram] indexing failed: {e}"),
            },
        );
        Engram {
            repo_root,
            engine: None,
            store: None,
            index_ready,
        }
    }

    /// Ensure the background index is ready and the store is open. Cheap tools
    /// (verification plan) need only this, not the embedding index.
    fn ensure_store(&mut self) -> Result<(), String> {
        if !self.index_ready.load(Ordering::SeqCst) {
            return Err(
                "Engram is still indexing this repository in the background. \
                 Retry this tool call in a few seconds."
                    .into(),
            );
        }
        if self.store.is_none() {
            self.store = Some(Store::open(&self.repo_root).map_err(|e| format!("store: {e}"))?);
        }
        Ok(())
    }

    fn ensure_engine(&mut self) -> Result<(), String> {
        self.ensure_store()?;
        if self.engine.is_none() {
            let store = self.store.as_mut().unwrap();
            self.engine =
                Some(Engine::build(&self.repo_root, store).map_err(|e| format!("engine: {e}"))?);
        }
        Ok(())
    }
}

impl ToolHandler for Engram {
    fn list_tools(&self) -> Value {
        json!({ "tools": [
            {
                "name": "get_task_context",
                "description": "ALWAYS call this FIRST, before planning or implementing any coding task. Returns evidence packets about this repository: existing code relevant to the task, symbols to reuse instead of reimplementing, related tests, and files matched by hybrid (BM25+vector+symbol) retrieval. Using this prevents duplicate implementations and wrong assumptions about the codebase.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": { "type": "string", "description": "The task description in natural language, e.g. 'add retry logic to billing webhooks'" }
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
                "name": "predict_impact",
                "description": "Call this BEFORE modifying files. Predicts which files a task will likely affect: direct matches plus files that historically change together (git co-change analysis) and tests likely to be affected. Touching files outside this set should be justified explicitly.",
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
            }
        ]})
    }

    fn call_tool(&mut self, name: &str, args: &Value) -> Result<Value, String> {
        match name {
            "get_task_context" => {
                self.ensure_engine()?;
                let engine = self.engine.as_mut().unwrap();
                let store = self.store.as_mut().unwrap();
                let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("");
                if task.is_empty() {
                    return Err("missing required argument: task".into());
                }
                let packets = engine.search(store, task, 8).map_err(|e| e.to_string())?;
                Ok(json!({ "task": task, "evidence": packets }))
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
            other => Err(format!("unknown tool: {other}")),
        }
    }
}

pub fn run() {
    let mut repo = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--repo") {
        if let Some(p) = args.get(pos + 1) {
            repo = PathBuf::from(p);
        }
    }
    eprintln!("[engram] starting MCP server for {}", repo.display());
    let mut engram = Engram::start(repo);
    serve(&mut engram);
}
