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

/// Parse recent git history into per-commit `(unix_timestamp, changed_files)`.
fn commit_records(repo_root: &Path) -> Vec<(i64, Vec<String>)> {
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

/// Build the co-change edge list and per-file recency from git history.
pub fn build(repo_root: &Path) -> History {
    let records = commit_records(repo_root);
    let mut file_commit_count: HashMap<String, u32> = HashMap::new();
    let mut pair_count: HashMap<(String, String), u32> = HashMap::new();
    let mut last_commit: HashMap<String, i64> = HashMap::new();

    for (ts, set) in &records {
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
