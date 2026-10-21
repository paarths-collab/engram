//! Engram domain types — shared across indexer, retrieval, and MCP server.

use serde::{Deserialize, Serialize};

/// A file discovered during Tier-0 inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub path: String,
    pub language: Language,
    pub size_bytes: u64,
    pub is_test: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Other,
}

/// File extension (without the dot) to language. The single table behind
/// [`Language::from_path`]; the inventory's rules fingerprint reads it too, so
/// adding an extension here invalidates stale indexes by itself.
pub const LANGUAGE_EXTENSIONS: &[(&str, Language)] = &[
    ("rs", Language::Rust),
    ("py", Language::Python),
    ("ts", Language::TypeScript),
    ("tsx", Language::TypeScript),
    ("js", Language::JavaScript),
    ("jsx", Language::JavaScript),
    ("mjs", Language::JavaScript),
    ("cjs", Language::JavaScript),
    ("go", Language::Go),
];

impl Language {
    pub fn from_path(path: &str) -> Self {
        let ext = path.rsplit('.').next().unwrap_or("");
        LANGUAGE_EXTENSIONS
            .iter()
            .find(|(e, _)| *e == ext)
            .map_or(Language::Other, |(_, lang)| *lang)
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Python => "python",
            Language::TypeScript => "typescript",
            Language::JavaScript => "javascript",
            Language::Go => "go",
            Language::Other => "other",
        }
    }
}

/// A symbol extracted during Tier-1 structural parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolRecord {
    pub name: String,
    pub kind: SymbolKind,
    pub path: String,
    pub start_line: usize,
    /// Last line of the definition, inclusive. Together with `start_line` this
    /// gives the symbol's source span, which is what lets retrieval embed and
    /// quote the definition itself rather than the head of its file.
    pub end_line: usize,
    /// First line of the definition, used as a cheap signature/preview.
    pub signature: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Method,
    Struct,
    Enum,
    Class,
    Trait,
    Interface,
    Const,
    Module,
}

/// Historical co-change edge derived from git history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoChange {
    pub path_a: String,
    pub path_b: String,
    /// Number of commits in which both files changed.
    pub count: u32,
    /// count / commits touching path_a  (asymmetric strength a→b).
    pub strength: f32,
}

/// Which way an import chain runs between the two files of a pairwise lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportDirection {
    SourceImportsTarget,
    TargetImportsSource,
}

/// Which stored co-change row a fact came from. Rows are asymmetric: the
/// strength of `a -> b` is relative to commits touching `a`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoChangeDirection {
    SourceToTarget,
    TargetToSource,
}

/// One hop of an import route: `from` imports `to`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImportEdge {
    pub from: String,
    pub to: String,
}

/// The limits the co-change table was built under. A pair that fell outside
/// them (too few shared commits, only in bulk commits, older than the window)
/// is absent from the table, so absence is only meaningful relative to these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CoChangeWindow {
    /// Most recent commits walked.
    pub max_commits: usize,
    /// Fewest shared commits for a pair to be recorded.
    pub min_count: u32,
    /// Commits touching more files than this are ignored.
    pub max_files_per_commit: usize,
}

/// One recorded fact linking two files, as returned by `explain_connection`.
///
/// `type`, `detail` and `weight` are the original contract keys; the remaining
/// fields are typed additions per variant. Import and co-change facts are
/// independent: neither implies, requires or replaces the other.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ConnectionReason {
    #[serde(rename = "import_edge")]
    Import {
        detail: String,
        /// 0.6 for a direct import, 0.35 for a 2-hop chain.
        weight: f32,
        /// Always `"import"`.
        edge_kind: &'static str,
        direction: ImportDirection,
        hops: usize,
        /// Repo-relative paths from the importer to the imported file,
        /// endpoints included; `hops + 1` entries.
        chain: Vec<String>,
        /// One `{from, to}` per hop, in route order.
        edges: Vec<ImportEdge>,
        /// Always `"heuristic"`: edges come from module-name matching, not
        /// compiler resolution.
        resolution: &'static str,
        /// Always `"heuristic_module_segments"`.
        resolver: &'static str,
        /// Always `"candidate"`: a plausible edge, not a compiler-verified one.
        resolution_status: &'static str,
    },
    #[serde(rename = "historical_cochange")]
    CoChange {
        detail: String,
        /// Equal to `strength`.
        weight: f32,
        /// Always `"cochange"`.
        edge_kind: &'static str,
        direction: CoChangeDirection,
        /// Commits in which both files changed.
        count: u32,
        /// `count` / commits touching the row's first file.
        strength: f32,
        window: CoChangeWindow,
        /// Commits in the window that touched the queried source file (bulk
        /// commits excluded, as for `strength`). `null` when the index was
        /// built before these totals were recorded; never a silent zero.
        opportunities_source: Option<u32>,
        /// Same, for the queried target file.
        opportunities_target: Option<u32>,
    },
}

/// Overall answer of a pairwise lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionStatus {
    /// At least one fact was recorded.
    Found,
    /// Nothing recorded, and every search ran to completion within its bound.
    NotFoundWithinScope,
    /// Nothing recorded, but the import search stopped at its hop bound with
    /// unexplored import edges remaining, so a longer chain may exist.
    UnknownBeyondBound,
}

/// How one directed import search ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportSearchOutcome {
    Found,
    /// The whole reachable import neighbourhood was explored; no chain.
    NoneWithinBound,
    /// No chain within the bound, but imports continue past it.
    BeyondBound,
}

/// What a pairwise lookup searched, so "not connected" is never read as more
/// than it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConnectionScope {
    pub import_max_hops: usize,
    pub import_directions_searched: Vec<ImportDirection>,
    pub import_search_source_imports_target: ImportSearchOutcome,
    pub import_search_target_imports_source: ImportSearchOutcome,
    pub cochange_lookup: &'static str,
    pub cochange_directions_searched: Vec<CoChangeDirection>,
    pub cochange_window: CoChangeWindow,
    /// Commits the last history walk covered (at most
    /// `cochange_window.max_commits`). `null` when not recorded yet.
    pub commits_scanned: Option<u32>,
    /// Of those, commits touching more than `max_files_per_commit` files,
    /// which contribute to neither counts nor strength. `null` when not
    /// recorded yet.
    pub bulk_commits_excluded: Option<u32>,
    pub note: String,
}

/// Result of `explain_connection`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConnectionExplanation {
    pub status: ConnectionStatus,
    pub scope: ConnectionScope,
    pub reasons: Vec<ConnectionReason>,
}

/// One compact, evidence-backed recommendation returned to the coding agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePacket {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: EvidenceKind,
    pub title: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    /// One-based first line of the definition this packet quotes, when known.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub start_line: Option<usize>,
    /// Last line of that definition, inclusive, when known.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub end_line: Option<usize>,
    pub score: f32,
    /// Which signals contributed (bm25, vector, symbol, path, cochange).
    pub signals: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    ExistingCode,
    Symbol,
    Test,
    RelatedFile,
}

/// Output of predict_impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactPrediction {
    pub likely_files: Vec<ScoredPath>,
    pub likely_tests: Vec<String>,
    pub cochange_expansions: Vec<ScoredPath>,
    /// Files that statically import one of the direct hits (import-graph expansion).
    pub import_expansions: Vec<ScoredPath>,
    /// Expansion candidates that existed but were cut by a recommendation cap
    /// (per-anchor co-change top-N, or the per-kind result limit). Never
    /// serialized with the prediction itself; the pack layer discloses them.
    #[serde(skip)]
    pub dropped: Vec<DroppedCandidate>,
}

/// An expansion candidate withheld by a recommendation cap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedCandidate {
    pub path: String,
    /// `"cochange"` or `"import"`.
    pub kind: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredPath {
    pub path: String,
    pub confidence: f32,
    pub reason: String,
}

/// A raw human review comment ingested from a pull request. Stored and returned
/// verbatim — Engram never summarizes review comments; the agent reads them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewComment {
    pub pr_number: i64,
    pub pr_title: String,
    /// Whether the PR the comment was left on was eventually merged.
    pub pr_merged: bool,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<i64>,
    pub body: String,
    pub author: String,
    /// ISO-8601 timestamp the comment was posted. Empty for rows ingested
    /// before timestamps were captured. This is what orders a comment against
    /// the commits that followed it, which is the basis of correction mining.
    #[serde(skip_serializing_if = "String::is_empty")]
    #[serde(default)]
    pub created_at: String,
    /// The diff hunk the reviewer was looking at when they commented — the
    /// `code_before` half of a correction triple.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub diff_hunk: Option<String>,
}

/// A review comment as ingested from the API, before it is joined with its PR.
///
/// Carries the fields needed to reconstruct a *correction*: when the comment
/// was made, which commit it was anchored to, the hunk under discussion, and
/// whether it is a reply in an existing thread rather than a new request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestedComment {
    pub path: String,
    pub line: Option<i64>,
    pub body: String,
    pub author: String,
    /// ISO-8601 creation timestamp, verbatim from the API.
    pub created_at: String,
    /// Diff hunk the comment was anchored to.
    pub diff_hunk: String,
    /// SHA the comment was left against.
    pub commit_id: String,
    /// Set when this comment replies to another; thread roots are the actual
    /// change requests, replies are usually discussion.
    pub in_reply_to: Option<i64>,
}

/// One file changed by a pull request, including the diff itself.
///
/// The `patch` is the unified-diff hunks GitHub returns for the file. Without
/// it a PR is just a list of filenames and no before/after can be recovered.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrFileChange {
    pub path: String,
    /// `added`, `modified`, `removed`, `renamed`, …
    pub status: String,
    pub additions: i64,
    pub deletions: i64,
    /// Unified diff hunks. Empty for binary files and for very large diffs,
    /// which GitHub omits.
    pub patch: String,
}

/// One commit belonging to a pull request, in the order the API returned it.
///
/// Ordering plus `authored_at` is what lets a later commit be attributed to an
/// earlier review comment: the push that came *after* the reviewer spoke.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrCommit {
    pub sha: String,
    pub message: String,
    pub author: String,
    /// ISO-8601 author timestamp.
    pub authored_at: String,
    /// Zero-based position in the PR's commit list.
    pub ordinal: i64,
}

/// Output of get_verification_plan: the merged checklist for a set of changed
/// files, plus repo-detected test commands and tests that historically changed
/// together with the changed files (co-changed test paths).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationPlan {
    /// Domain profiles whose detection rules matched the changed files.
    pub matched_profiles: Vec<String>,
    /// Merged, de-duplicated checklist items from the matched profiles.
    pub checklist: Vec<String>,
    /// Test/verify commands detected from the repo's manifests.
    pub test_commands: Vec<String>,
    /// Test paths that historically changed together (co-changed) with the
    /// changed files. The key name is legacy: it says "co-failing", but the
    /// data is co-change history, not test-failure history. It is kept so
    /// existing clients keep working.
    pub historically_co_failing_tests: Vec<String>,
}
