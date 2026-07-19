//! Historical co-change graph built from `git log --name-only`.
//! Which files change together — the cheapest, highest-value historical signal.

use engram_domain::CoChange;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

const MAX_COMMITS: usize = 2000;
const MAX_FILES_PER_COMMIT: usize = 30; // skip mega-commits (renames, vendoring)
const MIN_PAIR_COUNT: u32 = 2;

/// Parse recent git history into per-commit changed-file sets.
fn commit_file_sets(repo_root: &Path) -> Vec<Vec<String>> {
    let output = Command::new("git")
        .args([
            "log",
            &format!("--max-count={MAX_COMMITS}"),
            "--name-only",
            "--pretty=format:@@COMMIT@@",
        ])
        .current_dir(repo_root)
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let mut sets = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for line in text.lines() {
        if line == "@@COMMIT@@" {
            if !current.is_empty() {
                sets.push(std::mem::take(&mut current));
            }
        } else if !line.trim().is_empty() {
            current.push(line.trim().to_string());
        }
    }
    if !current.is_empty() {
        sets.push(current);
    }
    sets
}

/// Build the co-change edge list.
pub fn build(repo_root: &Path) -> Vec<CoChange> {
    let sets = commit_file_sets(repo_root);
    let mut file_commit_count: HashMap<String, u32> = HashMap::new();
    let mut pair_count: HashMap<(String, String), u32> = HashMap::new();

    for set in &sets {
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
    edges
}
