//! Engram domain types — shared across indexer, retrieval, and MCP server.

use serde::{Deserialize, Serialize};

/// A file discovered during Tier-0 inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub path: String,
    pub language: Language,
    pub size_bytes: u64,
    /// Stable digest of the complete file contents. Empty only for records
    /// deserialized from indexes that predate content-aware invalidation.
    #[serde(default)]
    pub content_hash: String,
    /// Why this file is inventoried but intentionally excluded from indexing.
    /// `None` means the file is eligible for Tier-1 and retrieval indexing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexing_ineligibility: Option<String>,
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

impl Language {
    pub fn from_path(path: &str) -> Self {
        match path.rsplit('.').next().unwrap_or("") {
            "rs" => Language::Rust,
            "py" => Language::Python,
            "ts" | "tsx" => Language::TypeScript,
            "js" | "jsx" | "mjs" | "cjs" => Language::JavaScript,
            "go" => Language::Go,
            _ => Language::Other,
        }
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
    /// One-based source line for symbol evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_line: Option<usize>,
    /// Inclusive one-based end line for symbol evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_kind: Option<SymbolKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    pub score: f32,
    /// Absolute (pre-fusion-normalization) lexical score, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_score: Option<f32>,
    /// Absolute cosine score, before per-query normalization, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f32>,
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

/// Honest conclusion returned by reuse-specific retrieval.
///
/// `NoEvidence` means no strong evidence was found; it is deliberately not a
/// proof that no implementation exists. `IndexIncomplete` makes incomplete
/// symbol coverage explicit instead of returning a false negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReuseState {
    ReuseLikely,
    PossibleReuse,
    NoEvidence,
    IndexIncomplete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuseCandidate {
    pub state: ReuseState,
    pub evidence: EvidencePacket,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuseAssessment {
    pub state: ReuseState,
    pub candidates: Vec<ReuseCandidate>,
    pub indexed_files: usize,
    pub index_complete: bool,
}

/// Output of predict_impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactPrediction {
    pub likely_files: Vec<ScoredPath>,
    pub likely_tests: Vec<String>,
    pub cochange_expansions: Vec<ScoredPath>,
    /// Files that statically import one of the direct hits (import-graph expansion).
    pub import_expansions: Vec<ScoredPath>,
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
/// files, plus repo-detected test commands and historically co-failing tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationPlan {
    /// Domain profiles whose detection rules matched the changed files.
    pub matched_profiles: Vec<String>,
    /// Merged, de-duplicated checklist items from the matched profiles.
    pub checklist: Vec<String>,
    /// Test/verify commands detected from the repo's manifests.
    pub test_commands: Vec<String>,
    /// Tests that historically change together with the changed files.
    pub historically_co_failing_tests: Vec<String>,
}
