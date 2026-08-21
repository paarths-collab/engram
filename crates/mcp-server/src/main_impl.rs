//! engram binary: MCP server + automatic background indexing.
//! Usage: engram --repo /path/to/repo   (defaults to cwd)

use crate::mcp::{serve, ToolHandler};
use engram_domain::{ReuseAssessment, ReuseCandidate, ReuseState};
use engram_repo_map::store::Store;
use engram_retrieval::Engine;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const EAGER_TIER1_LIMIT: usize = 3000;

#[derive(Debug, Clone)]
struct IndexBuild {
    head_sha: Option<String>,
    built_at_unix_seconds: u64,
}

pub struct Engram {
    repo_root: PathBuf,
    engine: Option<Engine>,
    store: Option<Store>,
    index_ready: Arc<AtomicBool>,
    /// Flipped by the watcher when files or HEAD change; triggers a lazy rebuild.
    dirty: Arc<AtomicBool>,
    index_build: Option<IndexBuild>,
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
                        "[engram] indexed: {} files, {} symbols, {}/{} tier1 files \
                         ({} extracted this run), {} cochange edges, {} PRs, {} reviews, {} pruned",
                        stats.files,
                        stats.symbols,
                        stats.tier1_files,
                        stats.parser_supported_files,
                        stats.tier1_files_extracted_this_run,
                        stats.cochange_edges,
                        stats.pull_requests,
                        stats.review_comments,
                        stats.pruned_files
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
            index_build: None,
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
            let engine =
                Engine::build(&self.repo_root, store).map_err(|e| format!("engine: {e}"))?;
            self.index_build = Some(IndexBuild {
                head_sha: git_text(&self.repo_root, &["rev-parse", "HEAD"]),
                built_at_unix_seconds: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|duration| duration.as_secs())
                    .unwrap_or(0),
            });
            self.engine = Some(engine);
        }
        Ok(())
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
                "description": "Call this BEFORE writing any new function, class, service, or utility. Searches the current-code index for observed evidence of a reusable implementation and reports whether reuse is likely, possible, unsupported by evidence, or unverifiable because the index is incomplete. A no-evidence result does not prove absence.",
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
                "description": "Call this to see WHY two files are connected, from recorded facts only. Returns the concrete edges linking them — import edges (with direction) and historical co-change — each with a weight, or an empty list if no recorded connection exists. Never a guess.",
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
                let concept = args
                    .get("concept")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .unwrap_or("");
                if concept.is_empty() {
                    return Err("missing required argument: concept".into());
                }
                let (assessment, coverage) = {
                    let engine = self.engine.as_mut().unwrap();
                    let store = self.store.as_mut().unwrap();
                    let assessment = engine
                        .assess_reuse(store, concept)
                        .map_err(|e| e.to_string())?;
                    let coverage = store.coverage().map_err(|e| e.to_string())?;
                    (assessment, coverage)
                };
                Ok(reuse_response(
                    concept,
                    assessment,
                    coverage,
                    &self.repo_root,
                    self.index_build.as_ref(),
                ))
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
                let reasons = engine
                    .explain_connection(store, source, target)
                    .map_err(|e| e.to_string())?;
                Ok(json!({
                    "source": source,
                    "target": target,
                    "connected": !reasons.is_empty(),
                    "reasons": reasons,
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

fn reuse_response(
    concept: &str,
    assessment: ReuseAssessment,
    coverage: engram_repo_map::store::StoreCoverage,
    repo_root: &std::path::Path,
    index_build: Option<&IndexBuild>,
) -> Value {
    let current_head_sha = git_text(repo_root, &["rev-parse", "HEAD"]);
    let branch = git_text(repo_root, &["symbolic-ref", "--short", "-q", "HEAD"]);
    let dirty = git_dirty(repo_root);
    let indexed_head_sha = index_build.and_then(|build| build.head_sha.clone());
    let mut missing_reasons = Vec::new();

    if index_build.is_none() {
        missing_reasons.push("index_build_timestamp_unavailable");
    }
    match (&indexed_head_sha, &current_head_sha) {
        (Some(indexed), Some(current)) if indexed != current => {
            missing_reasons.push("index_snapshot_differs_from_current_head")
        }
        (None, _) => missing_reasons.push("indexed_snapshot_sha_unavailable"),
        (_, None) => missing_reasons.push("current_snapshot_sha_unavailable"),
        _ => {}
    }
    match dirty {
        Some(true) => missing_reasons.push("working_tree_has_uncommitted_changes"),
        None => missing_reasons.push("working_tree_state_unknown"),
        Some(false) => {}
    }
    let supported_files = coverage.files.saturating_sub(coverage.ineligible_files);
    if assessment.indexed_files != supported_files {
        missing_reasons.push("engine_file_count_differs_from_persisted_inventory");
    }
    if coverage.ineligible_files > 0 {
        missing_reasons.push("inventory_contains_files_ineligible_for_indexing");
    }
    if coverage.unsupported_source_files > 0 {
        missing_reasons.push("source_files_without_symbol_parser");
    }
    if coverage.tier1_done_files != coverage.parser_supported_files {
        missing_reasons.push("parser_supported_files_not_fully_tier1_indexed");
    }
    if !assessment.index_complete {
        missing_reasons.push("retrieval_engine_reports_incomplete_reuse_index");
    }
    missing_reasons.sort_unstable();
    missing_reasons.dedup();

    let complete = missing_reasons.is_empty();
    let state = if !complete && assessment.state == ReuseState::NoEvidence {
        ReuseState::IndexIncomplete
    } else {
        assessment.state
    };
    let candidates: Vec<Value> = assessment
        .candidates
        .iter()
        .take(3)
        .map(reuse_candidate_json)
        .collect();
    let repository_path = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf())
        .to_string_lossy()
        .to_string();
    let pr_missing_reasons = [
        "no_ingestion_history_boundary_is_recorded",
        "ingestion_is_limited_to_recent_closed_pull_requests",
        "closed_pull_requests_beyond_the_first_100_are_not_imported",
        "pull_request_files_beyond_the_first_100_are_not_imported",
        "pull_request_commits_beyond_the_first_100_are_not_imported",
        "review_comments_beyond_the_first_100_are_not_imported",
        "pull_request_memory_is_not_used_by_reuse_retrieval",
    ];
    let mut coverage_missing = missing_reasons.clone();
    coverage_missing.extend(pr_missing_reasons);
    coverage_missing.sort_unstable();
    coverage_missing.dedup();
    let response_snapshot_sha = indexed_head_sha
        .clone()
        .or_else(|| current_head_sha.clone());

    json!({
        "concept": concept,
        "status": state,
        "existing_candidates": candidates,
        "recommendation": reuse_recommendation(state),
        "repository": {
            "origin": git_text(repo_root, &["remote", "get-url", "origin"]),
            "path": repository_path,
        },
        "snapshot": {
            "sha": current_head_sha,
            "branch": branch,
            "dirty": dirty,
        },
        "snapshot_sha": response_snapshot_sha,
        "coverage": {
            "discovered_files": coverage.files,
            "supported_files": supported_files,
            "indexed_files": assessment.indexed_files,
            "ineligible_files": coverage.ineligible_files,
            "unsupported_source_files": coverage.unsupported_source_files,
            "symbols": coverage.symbols,
            "symbol_parser_supported_files": coverage.parser_supported_files,
            "tier1_indexed_files": coverage.tier1_done_files,
            "prs_imported": coverage.pull_requests,
            "pr_import_complete": false,
            "index_complete": complete,
            "missing": coverage_missing,
        },
        "index": {
            "built_at_unix_seconds": index_build.map(|build| build.built_at_unix_seconds),
            "snapshot_sha": indexed_head_sha,
            "complete": complete,
            "missing_reasons": missing_reasons,
            "files": {
                "discovered": coverage.files,
                "supported": supported_files,
                "ineligible": coverage.ineligible_files,
                "unsupported_source": coverage.unsupported_source_files,
                "parser_supported": coverage.parser_supported_files,
                "tier1_indexed": coverage.tier1_done_files,
                "retrieval_indexed": assessment.indexed_files,
            },
            "symbols": coverage.symbols,
        },
        "pull_request_memory": {
            "complete": false,
            "pull_requests": coverage.pull_requests,
            "review_comments": coverage.review_comments,
            "missing_reasons": pr_missing_reasons,
        }
    })
}

fn reuse_candidate_json(candidate: &ReuseCandidate) -> Value {
    let evidence = &candidate.evidence;
    json!({
        "status": candidate.state,
        "memory_status": "OBSERVED",
        "source": "current-code",
        "id": evidence.id,
        "evidence_id": evidence.id,
        "type": evidence.kind,
        "title": evidence.title,
        "path": evidence.path,
        "symbol": evidence.symbol,
        "symbol_kind": evidence.symbol_kind,
        "start_line": evidence.start_line,
        "end_line": evidence.end_line,
        "snippet": evidence.snippet,
        "retrieval_score": evidence.score,
        "score": evidence.score,
        "signals": evidence.signals,
    })
}

fn reuse_recommendation(state: ReuseState) -> &'static str {
    match state {
        ReuseState::ReuseLikely => {
            "Current-code evidence makes reuse likely. Inspect the cited implementation before writing a new one."
        }
        ReuseState::PossibleReuse => {
            "Current-code evidence suggests possible reuse. Inspect the candidates before deciding whether to implement."
        }
        ReuseState::NoEvidence => {
            "No sufficiently similar current implementation was found in the indexed coverage. This does not prove that no implementation exists."
        }
        ReuseState::IndexIncomplete => {
            "The index is incomplete or stale, so Engram cannot make an honest negative reuse claim."
        }
    }
}

fn git_text(repo_root: &std::path::Path, arguments: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(arguments)
        .current_dir(repo_root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!value.is_empty()).then_some(value)
}

fn git_dirty(repo_root: &std::path::Path) -> Option<bool> {
    let output = std::process::Command::new("git")
        .args(["status", "--porcelain=v1", "--untracked-files=normal"])
        .current_dir(repo_root)
        .output()
        .ok()?;
    output.status.success().then(|| !output.stdout.is_empty())
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
    use engram_domain::{EvidenceKind, EvidencePacket, ReuseCandidate, SymbolKind};

    fn candidate(state: ReuseState) -> ReuseCandidate {
        ReuseCandidate {
            state,
            evidence: EvidencePacket {
                id: "ev_retry".to_owned(),
                kind: EvidenceKind::Symbol,
                title: "retry in src/retry.rs:12".to_owned(),
                path: "src/retry.rs".to_owned(),
                symbol: Some("retry".to_owned()),
                start_line: Some(12),
                end_line: Some(20),
                symbol_kind: Some(SymbolKind::Function),
                snippet: Some("fn retry() {}".to_owned()),
                score: 0.91,
                bm25_score: Some(2.4),
                vector_score: Some(0.72),
                signals: vec!["bm25".to_owned(), "symbol_exact".to_owned()],
            },
        }
    }

    #[test]
    fn reuse_candidate_labels_score_and_observed_current_code() {
        let value = reuse_candidate_json(&candidate(ReuseState::ReuseLikely));
        assert_eq!(value["status"], "reuse_likely");
        assert_eq!(value["memory_status"], "OBSERVED");
        assert_eq!(value["source"], "current-code");
        assert_eq!(value["id"], "ev_retry");
        assert_eq!(value["evidence_id"], "ev_retry");
        assert_eq!(value["start_line"], 12);
        assert!((value["retrieval_score"].as_f64().unwrap() - 0.91).abs() < 1e-6);
        assert!((value["score"].as_f64().unwrap() - 0.91).abs() < 1e-6);
        assert_eq!(value["signals"], json!(["bm25", "symbol_exact"]));
    }

    #[test]
    fn an_unverifiable_negative_becomes_index_incomplete() {
        let assessment = ReuseAssessment {
            state: ReuseState::NoEvidence,
            candidates: Vec::new(),
            indexed_files: 0,
            index_complete: true,
        };
        let value = reuse_response(
            "transactional outbox",
            assessment,
            engram_repo_map::store::StoreCoverage::default(),
            std::path::Path::new("path-that-is-not-a-git-repository"),
            None,
        );
        assert_eq!(value["status"], "index_incomplete");
        assert_eq!(value["index"]["complete"], false);
        let recommendation = value["recommendation"].as_str().unwrap();
        assert!(!recommendation.to_ascii_lowercase().contains("safe"));
    }

    #[test]
    fn positive_observation_survives_incomplete_repository_metadata() {
        let assessment = ReuseAssessment {
            state: ReuseState::ReuseLikely,
            candidates: vec![candidate(ReuseState::ReuseLikely)],
            indexed_files: 1,
            index_complete: true,
        };
        let coverage = engram_repo_map::store::StoreCoverage {
            files: 1,
            parser_supported_files: 1,
            tier1_done_files: 1,
            symbols: 1,
            ..Default::default()
        };
        let value = reuse_response(
            "retry",
            assessment,
            coverage,
            std::path::Path::new("path-that-is-not-a-git-repository"),
            None,
        );
        assert_eq!(value["status"], "reuse_likely");
        assert_eq!(value["existing_candidates"].as_array().unwrap().len(), 1);
        assert_eq!(value["index"]["complete"], false);
        assert_eq!(value["coverage"]["indexed_files"], 1);
        assert_eq!(value["coverage"]["supported_files"], 1);
        assert_eq!(value["coverage"]["symbols"], 1);
        assert_eq!(value["coverage"]["pr_import_complete"], false);
    }

    #[test]
    fn no_recommendation_claims_it_is_safe_to_implement_new() {
        for state in [
            ReuseState::ReuseLikely,
            ReuseState::PossibleReuse,
            ReuseState::NoEvidence,
            ReuseState::IndexIncomplete,
        ] {
            let text = reuse_recommendation(state).to_ascii_lowercase();
            assert!(!text.contains("safe to implement"), "{state:?}: {text}");
        }
    }
}
