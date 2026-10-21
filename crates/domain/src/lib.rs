//! Engram domain types — shared across indexer, retrieval, and MCP server.

use serde::{Deserialize, Serialize};

/// FNV-1a, 64-bit. A change fingerprint, not a security boundary: stable
/// across platforms and releases, with no dependency.
pub struct Fnv64(u64);

impl Default for Fnv64 {
    fn default() -> Self {
        Fnv64(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv64 {
    pub fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    /// Lowercase 16-digit hex of the digest.
    pub fn finish_hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// `kind` of an import edge in [`edge_id`].
pub const EDGE_KIND_IMPORT: &str = "import";
/// `kind` of a co-change edge in [`edge_id`].
pub const EDGE_KIND_COCHANGE: &str = "cochange";
/// The only `direction` a stored co-change row has: the row's first file is
/// its source, and its strength is relative to the commits touching it.
pub const COCHANGE_ROW_DIRECTION: &str = "source_to_target";

/// Content-derived id of one graph fact: `"e_"` + 16 lowercase hex digits.
///
/// A 64-bit FNV-1a over a canonical, length-prefixed encoding of, in order:
/// the format tag `engram-edge-v1`, `snapshot_id`, `kind`, `source`, `target`,
/// then the number of `chain_or_direction` items and each item. Every string
/// is written as its byte length (`u64`, little-endian) followed by its bytes,
/// so no two different inputs encode alike. For an import edge the items are
/// the full chain of paths in order; for a co-change edge, the direction.
///
/// Stable across runs and machines, and a function of the snapshot: any change
/// to the indexed content gives every edge a new id. It is a 64-bit hash, so
/// ids are unique within a snapshot with overwhelming but not absolute
/// probability, and are not globally unique across snapshots or repositories.
/// A fingerprint to cite a fact by, not a security boundary.
pub fn edge_id(
    snapshot_id: &str,
    kind: &str,
    source: &str,
    target: &str,
    chain_or_direction: &[&str],
) -> String {
    let mut h = Fnv64::default();
    let mut field = |s: &str| {
        h.write(&(s.len() as u64).to_le_bytes());
        h.write(s.as_bytes());
    };
    field("engram-edge-v1");
    field(snapshot_id);
    field(kind);
    field(source);
    field(target);
    h.write(&(chain_or_direction.len() as u64).to_le_bytes());
    for item in chain_or_direction {
        h.write(&(item.len() as u64).to_le_bytes());
        h.write(item.as_bytes());
    }
    format!("e_{}", h.finish_hex())
}

/// [`edge_id`] of an import edge or chain: `chain[0]` imports `chain[1]`, and
/// so on. A single hop is a two-element chain; its id is the same whether it
/// is cited alone or as one hop of a longer route.
pub fn import_edge_id<S: AsRef<str>>(snapshot_id: &str, chain: &[S]) -> String {
    let items: Vec<&str> = chain.iter().map(AsRef::as_ref).collect();
    let source = items.first().copied().unwrap_or("");
    let target = items.last().copied().unwrap_or("");
    edge_id(snapshot_id, EDGE_KIND_IMPORT, source, target, &items)
}

/// [`edge_id`] of the stored co-change row `source -> target` (its strength is
/// relative to commits touching `source`). The id belongs to the row, not to
/// the question that found it: `a -> b` and `b -> a` are different rows with
/// different ids whichever way a pairwise lookup was asked.
pub fn cochange_edge_id(snapshot_id: &str, source: &str, target: &str) -> String {
    edge_id(
        snapshot_id,
        EDGE_KIND_COCHANGE,
        source,
        target,
        &[COCHANGE_ROW_DIRECTION],
    )
}

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
    /// Id of this single hop (`import_edge_id(snapshot, [from, to])`): the same
    /// id `get_graph` emits for the edge, whatever route it sits on.
    pub edge_id: String,
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
        /// Id of the whole route (`import_edge_id(snapshot, chain)`). For a
        /// direct import it equals the id of its only hop; for a 2-hop route
        /// it is distinct, and the hops are cited through `edges[].edge_id`.
        edge_id: String,
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
        /// Id of the stored row this fact came from (`cochange_edge_id` of the
        /// row's own source and target), the id `get_graph` emits for it.
        edge_id: String,
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
    /// Which signals contributed. The server emits exactly these names:
    /// `bm25`, `vector`, `path`, `recent`, `symbol_exact`, `symbol`,
    /// `changelog_demoted`, `doc_demoted`.
    /// `vector` is a hashed n-gram embedding, not a learned semantic model.
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
    /// Ids of the recorded edges this recommendation rests on: the co-change
    /// row(s) from an anchor, or every hop of the import route to an anchor.
    /// Each is an id `get_graph` emits for the same anchors in the same
    /// snapshot. Absent for anchors themselves and for search-derived entries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_ids: Vec<String>,
    /// The numbers `confidence` was taken from. Absent where there are none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_components: Option<ScoreComponents>,
}

/// Where a recommendation's `confidence` came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "basis", rename_all = "snake_case")]
pub enum ScoreComponents {
    /// `confidence` is the strongest co-change strength among the anchors'
    /// kept rows for this file.
    CochangeStrength {
        strength: f32,
        /// Commits in which the anchor and this file changed together.
        count: u32,
        /// The anchor whose row supplied `strength`.
        anchor: String,
    },
    /// `confidence` is a fixed weight by import distance (0.6 for one hop,
    /// 0.35 for two).
    ImportHops { hops: usize, weight: f32 },
}

/// Whether a node of [`GraphFacts`] was asked for or found around one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeRole {
    Anchor,
    Neighbor,
}

/// One file in a `get_graph` result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GraphNode {
    pub path: String,
    pub role: GraphNodeRole,
    /// Import hops from the nearest anchor (0 for anchors; co-change partners
    /// are 1), the smallest distance at which the file was reached.
    pub distance: usize,
    /// Omitted when the extension is not a supported language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<&'static str>,
    pub is_test: bool,
}

/// One recorded fact in a `get_graph` result. Import and co-change facts are
/// independent: a file pair can have both, each with its own id.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum GraphEdge {
    /// `source` imports `target`: one hop, as the import graph resolved it.
    Import {
        edge_id: String,
        source: String,
        target: String,
        /// Always `source_imports_target`: `source` is the importer.
        direction: ImportDirection,
        /// Always 1: every import edge is a single hop.
        hops: usize,
        /// Always `"heuristic_module_segments"`.
        resolver: &'static str,
        /// Always `"candidate"`.
        resolution_status: &'static str,
    },
    /// The stored co-change row `source -> target`.
    Cochange {
        edge_id: String,
        source: String,
        target: String,
        /// Always `source_to_target`: `strength` is relative to `source`.
        direction: CoChangeDirection,
        count: u32,
        strength: f32,
        window: CoChangeWindow,
        /// Commits in the window that touched `source` (the denominator of
        /// `strength`). `null` when the index predates these totals.
        opportunities_source: Option<u32>,
        /// Same, for `target`.
        opportunities_target: Option<u32>,
    },
}

impl GraphEdge {
    pub fn edge_id(&self) -> &str {
        match self {
            GraphEdge::Import { edge_id, .. } | GraphEdge::Cochange { edge_id, .. } => edge_id,
        }
    }
}

/// The bounds a `get_graph` result was produced under, and what they cut.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GraphLimits {
    pub max_nodes: usize,
    pub max_edges: usize,
    /// Most co-change rows read per anchor.
    pub cochange_per_anchor: usize,
    pub truncated: bool,
    /// Exactly the number of edges the same call would have emitted with no
    /// cap, minus the number it did emit. The unit is one edge: one directed
    /// co-change row or one import edge. Counts co-change rows past the
    /// per-anchor read limit (with the reverse rows of the partners it hides),
    /// edges touching a file past the node cap, and edges past the edge cap.
    pub omitted_edges: usize,
    /// Neighbouring files past the node cap. A co-change partner hidden by
    /// the per-anchor read limit is never added as a node, so it is not
    /// counted here; its rows are counted in `omitted_edges`.
    pub omitted_nodes: usize,
    pub reason: String,
}

/// What a `get_graph` call searched, so a missing edge is never read as more
/// than it is.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GraphScope {
    pub kinds: Vec<&'static str>,
    pub import_max_hops: usize,
    pub cochange_window: CoChangeWindow,
    pub commits_scanned: Option<u32>,
    pub bulk_commits_excluded: Option<u32>,
    pub note: String,
}

/// Result of `get_graph`: raw recorded facts about the neighbourhood of some
/// anchor files, independent of recommendation ranking and caps.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GraphFacts {
    pub snapshot: String,
    pub anchors: Vec<String>,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub limits: GraphLimits,
    pub scope: GraphScope,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Values computed independently of this crate (a separate FNV-1a over the
    /// documented byte encoding), so a change to the encoding fails here
    /// rather than quietly re-numbering every id.
    #[test]
    fn edge_ids_match_the_documented_encoding() {
        assert_eq!(
            import_edge_id("snap1", &["a.py", "b.py"]),
            "e_5f08b1d2946f9754"
        );
        assert_eq!(
            import_edge_id("snap1", &["a.py", "b.py", "c.py"]),
            "e_13301a0e24b41fb2"
        );
        assert_eq!(
            cochange_edge_id("snap1", "a.py", "b.py"),
            "e_9a00c641bdb973d0"
        );
        assert_eq!(
            cochange_edge_id("snap1", "b.py", "a.py"),
            "e_941ed47f442d546c"
        );
        assert_eq!(
            import_edge_id("snap2", &["a.py", "b.py"]),
            "e_42d58902cf175f51"
        );
    }

    #[test]
    fn edge_ids_are_16_lowercase_hex_digits_after_a_prefix() {
        let id = import_edge_id("s", &["x", "y"]);
        let hex = id.strip_prefix("e_").expect("prefix");
        assert_eq!(hex.len(), 16);
        assert!(hex
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
    }

    #[test]
    fn length_prefixing_keeps_field_boundaries_distinct() {
        // Without length prefixes these would concatenate to the same bytes.
        assert_ne!(
            edge_id("ab", "c", "d", "e", &[]),
            edge_id("a", "bc", "d", "e", &[])
        );
        assert_ne!(
            edge_id("s", "import", "a", "bc", &["a", "bc"]),
            edge_id("s", "import", "a", "bc", &["ab", "c"])
        );
    }

    #[test]
    fn direction_chain_snapshot_and_kind_each_change_the_id() {
        let base = import_edge_id("s", &["a", "b", "c"]);
        assert_ne!(base, import_edge_id("s", &["a", "x", "c"]), "chain");
        assert_ne!(base, import_edge_id("t", &["a", "b", "c"]), "snapshot");
        assert_ne!(base, import_edge_id("s", &["c", "b", "a"]), "direction");
        assert_ne!(
            import_edge_id("s", &["a", "b"]),
            cochange_edge_id("s", "a", "b"),
            "kind"
        );
        assert_ne!(
            cochange_edge_id("s", "a", "b"),
            cochange_edge_id("s", "b", "a"),
            "cochange direction"
        );
    }
}
