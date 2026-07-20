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
