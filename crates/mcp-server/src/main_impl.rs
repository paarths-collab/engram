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

    fn ensure_engine(&mut self) -> Result<(), String> {
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
        if self.engine.is_none() {
            let store = self.store.as_ref().unwrap();
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
            }
        ]})
    }

    fn call_tool(&mut self, name: &str, args: &Value) -> Result<Value, String> {
        self.ensure_engine()?;
        let engine = self.engine.as_mut().unwrap();
        let store = self.store.as_mut().unwrap();
        match name {
            "get_task_context" => {
                let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("");
                if task.is_empty() {
                    return Err("missing required argument: task".into());
                }
                let packets = engine.search(store, task, 8).map_err(|e| e.to_string())?;
                Ok(json!({ "task": task, "evidence": packets }))
            }
            "find_existing_implementation" => {
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
                let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("");
                if task.is_empty() {
                    return Err("missing required argument: task".into());
                }
                let impact = engine
                    .predict_impact(store, task)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::to_value(impact).map_err(|e| e.to_string())?)
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
