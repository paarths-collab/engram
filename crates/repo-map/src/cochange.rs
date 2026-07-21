//! Historical signals built from a single `git log --name-only` walk:
//!   - co-change graph: which files change together (asymmetric edge strength)
//!   - file recency: each file's most-recent commit timestamp
//!
//! Both come from the same pass so we never shell out per file.

use engram_domain::CoChange;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

const MAX_COMMITS: usize = 2000;
const MAX_FILES_PER_COMMIT: usize = 30; // skip mega-commits (renames, vendoring)
const MIN_PAIR_COUNT: u32 = 2;

/// Output of the history walk.
pub struct History {
    pub edges: Vec<CoChange>,
    /// path -> most-recent commit unix timestamp.
    pub last_commit: HashMap<String, i64>,
}

/// One commit's contribution to history: `(committer_timestamp, changed_files)`.
pub type CommitRecord = (i64, Vec<String>);

/// Parse recent git history into per-commit `(unix_timestamp, changed_files)`.
/// Newest first, matching `git log` order.
pub fn commit_records(repo_root: &Path) -> Vec<CommitRecord> {
    let output = Command::new("git")
        .args([
            "log",
            &format!("--max-count={MAX_COMMITS}"),
            "--name-only",
            // %ct = committer date, unix timestamp — tacked onto the marker line.
            "--pretty=format:@@COMMIT@@%ct",
        ])
        .current_dir(repo_root)
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let mut records = Vec::new();
    let mut cur_ts: i64 = 0;
    let mut current: Vec<String> = Vec::new();
    for line in text.lines() {
        if let Some(ts) = line.strip_prefix("@@COMMIT@@") {
            if !current.is_empty() {
                records.push((cur_ts, std::mem::take(&mut current)));
            }
            cur_ts = ts.trim().parse().unwrap_or(0);
        } else if !line.trim().is_empty() {
            current.push(line.trim().to_string());
        }
    }
    if !current.is_empty() {
        records.push((cur_ts, current));
    }
    records
}

/// Build the co-change edge list and per-file recency from the full git history.
pub fn build(repo_root: &Path) -> History {
    build_from_records(&commit_records(repo_root), None)
}

/// Fold `(timestamp, changed_files)` records into co-change edges and recency.
///
/// `before` bounds the history to commits *strictly older* than the given
/// unix timestamp. `None` uses everything, which is what the server wants.
///
/// The cutoff exists for the benchmark harness. A co-change edge is a direct
/// function of the commits that produced it, so scoring a commit against a
/// graph built from history that includes that commit is scoring against the
/// answer key. Strict `<` (not `<=`) is deliberate: commits sharing a second
/// are excluded rather than risk leaking a sibling commit.
pub fn build_from_records(records: &[CommitRecord], before: Option<i64>) -> History {
    let mut file_commit_count: HashMap<String, u32> = HashMap::new();
    let mut pair_count: HashMap<(String, String), u32> = HashMap::new();
    let mut last_commit: HashMap<String, i64> = HashMap::new();

    for (ts, set) in records {
        if before.is_some_and(|cutoff| *ts >= cutoff) {
            continue;
        }
        // Recency covers every commit (git log is newest-first, so the first
        // sighting of a file is its most-recent commit). Done before the
        // mega-commit skip so a file only touched in a big commit still dates.
        for f in set {
            last_commit.entry(f.clone()).or_insert(*ts);
        }
        if set.len() > MAX_FILES_PER_COMMIT {
            continue;
        }
        for f in set {
            *file_commit_count.entry(f.clone()).or_insert(0) += 1;
        }
        for i in 0..set.len() {
            for j in 0..set.len() {
                if i == j {
                    continue;
                }
                *pair_count
                    .entry((set[i].clone(), set[j].clone()))
                    .or_insert(0) += 1;
            }
        }
    }

    let mut edges = Vec::new();
    for ((a, b), count) in pair_count {
        if count < MIN_PAIR_COUNT {
            continue;
        }
        let total_a = *file_commit_count.get(&a).unwrap_or(&1) as f32;
        edges.push(CoChange {
            strength: count as f32 / total_a,
            path_a: a,
            path_b: b,
            count,
        });
    }
    edges.sort_by(|x, y| y.strength.partial_cmp(&x.strength).unwrap());
    History { edges, last_commit }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Newest-first, the order `git log` produces. `future.rs` co-changes with
    /// `a.rs` only in the two most recent commits.
    fn records() -> Vec<CommitRecord> {
        vec![
            (300, vec!["a.rs".to_owned(), "future.rs".to_owned()]),
            (250, vec!["a.rs".to_owned(), "future.rs".to_owned()]),
            (200, vec!["a.rs".to_owned(), "b.rs".to_owned()]),
            (100, vec!["a.rs".to_owned(), "b.rs".to_owned()]),
        ]
    }

    fn has_edge(history: &History, a: &str, b: &str) -> bool {
        history
            .edges
            .iter()
            .any(|e| e.path_a == a && e.path_b == b)
    }

    #[test]
    fn full_history_includes_every_pair() {
        let history = build_from_records(&records(), None);
        assert!(has_edge(&history, "a.rs", "b.rs"));
        assert!(has_edge(&history, "a.rs", "future.rs"));
    }

    #[test]
    fn cutoff_excludes_edges_from_commits_at_or_after_it() {
        // Evaluating the commit at t=250 must not see that commit or any newer
        // one. Without the cutoff the a.rs<->future.rs edge is built from the
        // very commits under evaluation.
        let history = build_from_records(&records(), Some(250));
        assert!(has_edge(&history, "a.rs", "b.rs"), "past edge survives");
        assert!(
            !has_edge(&history, "a.rs", "future.rs"),
            "edge leaked from the commit under evaluation and newer history"
        );
    }

    #[test]
    fn cutoff_also_bounds_recency() {
        let full = build_from_records(&records(), None);
        assert_eq!(full.last_commit.get("a.rs"), Some(&300));
        assert_eq!(full.last_commit.get("future.rs"), Some(&300));

        let bounded = build_from_records(&records(), Some(250));
        assert_eq!(bounded.last_commit.get("a.rs"), Some(&200));
        assert_eq!(
            bounded.last_commit.get("future.rs"),
            None,
            "a file first seen after the cutoff must not be dated"
        );
    }

    #[test]
    fn cutoff_is_strict_so_same_second_commits_are_excluded() {
        let same_second = vec![
            (100, vec!["a.rs".to_owned(), "b.rs".to_owned()]),
            (100, vec!["a.rs".to_owned(), "b.rs".to_owned()]),
        ];
        let history = build_from_records(&same_second, Some(100));
        assert!(history.edges.is_empty());
    }
}
