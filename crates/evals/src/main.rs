//! engram-evals: deterministic retrieval benchmarks.
//!
//! Benchmark 1 — file prediction (text -> files): each git commit is a
//! (message -> changed files) sample. We hide the changed files, feed the
//! commit subject to each ranking strategy, and measure Recall@k and MRR.
//! Tests whether hybrid text retrieval beats BM25-only / vector-only.
//!
//! Benchmark 2 — connection recovery (file -> files, no text, no guessing):
//! for each commit, take one changed file as a known anchor (the file the
//! agent is already editing) and ask whether the deterministic graph — the
//! co-change table (git history fact) and the import graph (static-analysis
//! fact) — recovers the *other* files that changed in the same commit. There
//! is no prediction step here: `find_connected_files` only returns files
//! linked to the anchor by a concrete recorded edge. The co-change graph is
//! built with a temporal cutoff (`--history-commits` older commits, skipping
//! the evaluation window) so it cannot see the future it is being asked to
//! recover — no leakage.
//!
//! Usage: engram-evals [--repo PATH] [--max-commits N]
//!                      [--eval-commits N] [--history-commits N]

use anyhow::Result;
use engram_repo_map::cochange;
use engram_repo_map::store::Store;
use engram_retrieval::{Engine, RankMode};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

const MARKER: &str = "@@ENGRAM_COMMIT@@";

struct Sample {
    query: String,
    changed: HashSet<String>,
}

/// A commit reduced to (anchor file, the other files that changed with it).
struct ConnectionSample {
    anchor: String,
    targets: HashSet<String>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let repo = arg(&args, "--repo")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let max_commits: usize = arg(&args, "--max-commits")
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let eval_commits: usize = arg(&args, "--eval-commits")
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let history_commits: usize = arg(&args, "--history-commits")
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);

    eprintln!("[evals] indexing {}", repo.display());
    engram_repo_map::index_repo(&repo, 100_000)?;
    let mut store = Store::open(&repo)?;
    let indexed: HashSet<String> = store.all_files()?.into_iter().map(|f| f.path).collect();

    run_benchmark1_file_prediction(&repo, &mut store, max_commits, &indexed)?;
    println!();
    run_benchmark2_connection_recovery(&repo, &mut store, eval_commits, history_commits, &indexed)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Benchmark 1: file prediction (text query -> files)
// ---------------------------------------------------------------------------

fn run_benchmark1_file_prediction(
    repo: &Path,
    store: &mut Store,
    max_commits: usize,
    indexed: &HashSet<String>,
) -> Result<()> {
    let mut engine = Engine::build(repo, store)?;
    let samples = collect_samples(repo, max_commits, indexed);
    println!("# Benchmark 1: file prediction (text -> files)\n");
    if samples.is_empty() {
        println!("no usable commits found\n");
        return Ok(());
    }
    println!(
        "Repo: `{}` · {} files · {} commits\n",
        repo.display(),
        indexed.len(),
        samples.len()
    );
    println!("| strategy | Recall@5 | Recall@10 | Recall@20 | MRR |");
    println!("|---|---|---|---|---|");
    let modes = [
        ("hybrid", RankMode::Hybrid),
        ("bm25-only", RankMode::Bm25),
        ("vector-only", RankMode::Vector),
        ("random", RankMode::Random),
    ];
    for (name, mode) in modes {
        let (mut r5, mut r10, mut r20, mut mrr) = (0.0, 0.0, 0.0, 0.0);
        for s in &samples {
            let ranked = engine.rank(store, &s.query, mode, 20)?;
            r5 += recall_at_k(&ranked, &s.changed, 5);
            r10 += recall_at_k(&ranked, &s.changed, 10);
            r20 += recall_at_k(&ranked, &s.changed, 20);
            mrr += reciprocal_rank(&ranked, &s.changed);
        }
        let n = samples.len() as f64;
        println!(
            "| {name} | {:.3} | {:.3} | {:.3} | {:.3} |",
            r5 / n,
            r10 / n,
            r20 / n,
            mrr / n
        );
    }
    Ok(())
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Parse `git log` into (subject, changed-files) samples, keeping only commits
/// with 1..=8 changed files that still exist in the index.
fn collect_samples(repo: &Path, max_commits: usize, indexed: &HashSet<String>) -> Vec<Sample> {
    let out = Command::new("git")
        .args([
            "log",
            &format!("--max-count={max_commits}"),
            "--name-only",
            &format!("--pretty=format:{MARKER}%s"),
        ])
        .current_dir(repo)
        .output();
    let Ok(out) = out else { return Vec::new() };
    let text = String::from_utf8_lossy(&out.stdout);

    let mut samples = Vec::new();
    let mut subject = String::new();
    let mut files: HashSet<String> = HashSet::new();
    let flush = |subject: &str, files: &mut HashSet<String>, out: &mut Vec<Sample>| {
        let changed: HashSet<String> = files.drain().filter(|f| indexed.contains(f)).collect();
        if !subject.is_empty() && (1..=8).contains(&changed.len()) && !subject.starts_with("Merge ")
        {
            out.push(Sample {
                query: subject.to_string(),
                changed,
            });
        }
    };
    for line in text.lines() {
        if let Some(subj) = line.strip_prefix(MARKER) {
            flush(&subject, &mut files, &mut samples);
            subject = subj.trim().to_string();
        } else if !line.trim().is_empty() {
            files.insert(line.trim().to_string());
        }
    }
    flush(&subject, &mut files, &mut samples);
    samples
}

fn recall_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = ranked
        .iter()
        .take(k)
        .filter(|p| relevant.contains(*p))
        .count();
    hits as f64 / relevant.len() as f64
}

fn reciprocal_rank(ranked: &[String], relevant: &HashSet<String>) -> f64 {
    for (i, p) in ranked.iter().enumerate() {
        if relevant.contains(p) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

// ---------------------------------------------------------------------------
// Benchmark 2: connection recovery (anchor file -> connected files, no text)
// ---------------------------------------------------------------------------

fn run_benchmark2_connection_recovery(
    repo: &Path,
    store: &mut Store,
    eval_commits: usize,
    history_commits: usize,
    indexed: &HashSet<String>,
) -> Result<()> {
    println!("# Benchmark 2: connection recovery (anchor file -> connected files)\n");
    println!(
        "No text, no guessing: history is built ONLY from the {history_commits} commits older \
         than the {eval_commits}-commit evaluation window (temporal cutoff — the co-change \
         graph cannot see the future it's being asked to recover). `find_connected_files` is \
         called with one changed file as the anchor; we check whether the co-change graph and \
         import graph recover the other files that changed in the same commit.\n"
    );

    // Leakage-free history: skip the eval_commits newest commits, then take
    // history_commits older ones. Overwrite the store's cochange table with
    // this restricted history before building the graph.
    let history = cochange::build_from(repo, eval_commits, history_commits);
    store.replace_cochange(&history.edges)?;
    eprintln!(
        "[evals] leakage-free co-change graph: {} edges (from commits {}..{})",
        history.edges.len(),
        eval_commits,
        eval_commits + history_commits
    );

    let mut engine = Engine::build(repo, store)?;
    let samples = collect_connection_samples(repo, eval_commits, indexed);
    if samples.is_empty() {
        println!("no usable commits found (need commits with 2-8 changed files)\n");
        return Ok(());
    }
    eprintln!("[evals] {} connection-recovery samples", samples.len());

    let (mut hit1, mut r5_co, mut r5_imp, mut r5_both) = (0.0, 0.0, 0.0, 0.0);
    let (mut avg_targets, mut avg_candidates) = (0.0, 0.0);
    for s in &samples {
        let impact = engine.impact_from_files(store, std::slice::from_ref(&s.anchor))?;
        let co: Vec<String> = impact
            .cochange_expansions
            .iter()
            .map(|p| p.path.clone())
            .collect();
        let imp: Vec<String> = impact
            .import_expansions
            .iter()
            .map(|p| p.path.clone())
            .collect();
        let mut both = co.clone();
        both.extend(imp.iter().cloned());

        r5_co += recall_at_k(&co, &s.targets, 5);
        r5_imp += recall_at_k(&imp, &s.targets, 5);
        r5_both += recall_at_k(&both, &s.targets, 10); // combined list, wider cap
        if both.iter().any(|p| s.targets.contains(p)) {
            hit1 += 1.0;
        }
        avg_targets += s.targets.len() as f64;
        avg_candidates += both.len() as f64;
    }
    let n = samples.len() as f64;
    println!("| source | Recall@5 |");
    println!("|---|---|");
    println!("| co-change only | {:.3} |", r5_co / n);
    println!("| import graph only | {:.3} |", r5_imp / n);
    println!("| combined (Recall@10) | {:.3} |", r5_both / n);
    println!();
    println!(
        "Hit rate (found >=1 real connection): **{:.1}%** over {} samples. \
         Avg held-out targets/commit: {:.1}. Avg candidates returned: {:.1}.",
        100.0 * hit1 / n,
        samples.len(),
        avg_targets / n,
        avg_candidates / n
    );
    Ok(())
}

/// Newest `eval_commits` commits, reduced to (anchor, other-changed-files).
/// The anchor is the alphabetically-first changed file (deterministic, no
/// heuristic pick). Keeps only non-merge commits touching 2..=8 indexed files.
fn collect_connection_samples(
    repo: &Path,
    eval_commits: usize,
    indexed: &HashSet<String>,
) -> Vec<ConnectionSample> {
    let out = Command::new("git")
        .args([
            "log",
            &format!("--max-count={eval_commits}"),
            "--name-only",
            &format!("--pretty=format:{MARKER}%s"),
        ])
        .current_dir(repo)
        .output();
    let Ok(out) = out else { return Vec::new() };
    let text = String::from_utf8_lossy(&out.stdout);

    let mut samples = Vec::new();
    let mut is_merge = false;
    let mut have_commit = false;
    let mut files: Vec<String> = Vec::new();
    let flush = |is_merge: bool, files: &mut Vec<String>, out: &mut Vec<ConnectionSample>| {
        let mut changed: Vec<String> = files.drain(..).filter(|f| indexed.contains(f)).collect();
        changed.sort();
        changed.dedup();
        if !is_merge && (2..=8).contains(&changed.len()) {
            let anchor = changed.remove(0);
            out.push(ConnectionSample {
                anchor,
                targets: changed.into_iter().collect(),
            });
        }
    };
    for line in text.lines() {
        if let Some(subj) = line.strip_prefix(MARKER) {
            if have_commit {
                flush(is_merge, &mut files, &mut samples);
            }
            have_commit = true;
            is_merge = subj.trim().starts_with("Merge ");
        } else if !line.trim().is_empty() {
            files.push(line.trim().to_string());
        }
    }
    if have_commit {
        flush(is_merge, &mut files, &mut samples);
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn recall_counts_relevant_in_top_k() {
        let ranked: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let rel = set(&["b", "d", "z"]); // z never retrieved
        assert!((recall_at_k(&ranked, &rel, 2) - 1.0 / 3.0).abs() < 1e-9); // only b in top2
        assert!((recall_at_k(&ranked, &rel, 4) - 2.0 / 3.0).abs() < 1e-9); // b and d
    }

    #[test]
    fn reciprocal_rank_uses_first_hit() {
        let ranked: Vec<String> = ["x", "y", "a"].iter().map(|s| s.to_string()).collect();
        assert!((reciprocal_rank(&ranked, &set(&["a"])) - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(reciprocal_rank(&ranked, &set(&["none"])), 0.0);
    }
}
