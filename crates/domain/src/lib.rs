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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
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

/// Deterministic relationships discovered from explicit repository anchors.
///
/// Engram does not guess which files a natural-language task will change. The
/// caller supplies paths already present in its selection or diff, then Engram
/// expands only relationships backed by source code or git history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionMap {
    /// Files explicitly supplied by the caller. They are never inferred.
    pub anchors: Vec<String>,
    /// Files that directly or transitively import an anchor, as proved by the
    /// import graph.
    pub import_dependents: Vec<ScoredPath>,
    /// Files historically changed with an anchor, including the observed
    /// co-change strength. This is historical evidence, not a prediction.
    pub historical_connections: Vec<ScoredPath>,
    /// Test files reached through either evidence source.
    pub related_tests: Vec<ScoredPath>,
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
