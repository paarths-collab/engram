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
    /// Flipped by the watcher when files or HEAD change; triggers a lazy rebuild.
    dirty: Arc<AtomicBool>,
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
                        "[engram] indexed: {} files, {} cochange edges, {} tier1 files, {} pruned",
                        stats.files, stats.cochange_edges, stats.tier1_files, stats.pruned_files
                    );
                    ready.store(true, Ordering::SeqCst);
                }
                Err(e) => eprintln!("[engram] indexing failed: {e}"),
            },
        );
        // Incremental reindex: watch for file saves and HEAD moves.
        let dirty = Arc::new(AtomicBool::new(false));
        crate::watcher::spawn(repo_root.clone(), dirty.clone());
        Engram {
            repo_root,
            engine: None,
            store: None,
            index_ready,
            dirty,
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
        // A watcher change invalidates the in-memory index; rebuild lazily here
        // (cheap: persisted vectors mean only changed files re-embed).
        if self.dirty.swap(false, Ordering::SeqCst) {
            self.engine = None;
        }
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
                "name": "find_connected_files",
                "description": "Call this when you already know which file(s) you are changing (e.g. the current diff) and want everything CONNECTED to them, with zero guessing. Unlike predict_impact (which starts from a fuzzy task description), this takes exact file paths as anchors and returns only files linked by hard, deterministic facts already recorded in the store: the co-change graph (files that historically changed together in the same commits) and the import graph (files that statically import an anchor, up to 2 hops). Every result traces to a concrete recorded edge — this is fact-finding, not prediction. Prefer this over predict_impact whenever you have concrete anchor files.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Repo-relative paths of files you are changing or about to change, e.g. ['src/billing/cancel.rs']"
                        }
                    },
                    "required": ["files"]
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
            "find_connected_files" => {
                self.ensure_engine()?;
                let engine = self.engine.as_mut().unwrap();
                let store = self.store.as_mut().unwrap();
                let files: Vec<String> = args
                    .get("files")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                if files.is_empty() {
                    return Err("missing required argument: files (array of paths)".into());
                }
                let impact = engine
                    .impact_from_files(store, &files)
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

pub fn run() {
    let mut repo = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--repo") {
        if let Some(p) = args.get(pos + 1) {
            repo = PathBuf::from(p);
        }
    }
    // Subcommand: `engram ingest-github [--limit N]` ingests PR history, then exits.
    if args.iter().any(|a| a == "ingest-github") {
        let limit = args
            .iter()
            .position(|a| a == "--limit")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(50);
        std::process::exit(match ingest_github(&repo, limit) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[engram] ingest-github failed: {e}");
                1
            }
        });
    }
    eprintln!("[engram] starting MCP server for {}", repo.display());
    let mut engram = Engram::start(repo);
    serve(&mut engram);
}

/// Ingest pull-request history for the repo's `origin` remote into the store.
fn ingest_github(repo: &std::path::Path, limit: usize) -> Result<(), String> {
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
        "[engram] ingested: {} PRs, {} changed files, {} review comments",
        stats.pull_requests, stats.files, stats.review_comments
    );
    Ok(())
}
