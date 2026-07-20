//! engram-evals: deterministic retrieval benchmarks.
//!
//! Benchmark 1 — file prediction: each git commit is a (message -> changed files)
//! sample. We hide the changed files, feed the commit subject to each ranking
//! strategy, and measure Recall@k and MRR. This tests the core thesis: does
//! hybrid retrieval beat BM25-only / vector-only?
//!
//! Usage: engram-evals [--repo PATH] [--max-commits N]

use anyhow::Result;
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let repo = arg(&args, "--repo")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let max_commits: usize = arg(&args, "--max-commits")
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    eprintln!("[evals] indexing {}", repo.display());
    engram_repo_map::index_repo(&repo, 100_000)?;
    let mut store = Store::open(&repo)?;
    let mut engine = Engine::build(&repo, &mut store)?;
    let indexed: HashSet<String> = store.all_files()?.into_iter().map(|f| f.path).collect();

    let samples = collect_samples(&repo, max_commits, &indexed);
    if samples.is_empty() {
        eprintln!("[evals] no usable commits found");
        return Ok(());
    }
    eprintln!(
        "[evals] {} files indexed, {} eval samples",
        indexed.len(),
        samples.len()
    );

    let modes = [
        ("hybrid", RankMode::Hybrid),
        ("bm25-only", RankMode::Bm25),
        ("vector-only", RankMode::Vector),
        ("random", RankMode::Random),
    ];

    println!("# Engram file-prediction benchmark\n");
    println!(
        "Repo: `{}` · {} files · {} commits\n",
        repo.display(),
        indexed.len(),
        samples.len()
    );
    println!("| strategy | Recall@5 | Recall@10 | Recall@20 | MRR |");
    println!("|---|---|---|---|---|");
    for (name, mode) in modes {
        let (mut r5, mut r10, mut r20, mut mrr) = (0.0, 0.0, 0.0, 0.0);
        for s in &samples {
            let ranked = engine.rank(&mut store, &s.query, mode, 20)?;
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
