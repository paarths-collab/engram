//! engram-evals: deterministic connection-recovery benchmarks.
//!
//! One file from each historical commit is supplied as an explicit anchor. The
//! remaining changed files are hidden, then recovered through static imports and
//! historical co-change evidence.
//!
//! # Temporal isolation
//!
//! A co-change edge is a direct function of the commits that produced it, so a
//! graph built from the full log already contains the commit being predicted.
//! Scoring against it measures leakage, not retrieval. Every sample therefore
//! rebuilds the co-change table from commits *strictly older* than the commit
//! under evaluation.
//!
//! One leak remains and is deliberately not hidden: the static import graph is
//! built from working-tree file contents at HEAD, not reconstructed per commit.
//! It is a weaker leak (current imports are not derived from any single commit's
//! file set) but it is real, and the `static import` row should be read as an
//! upper bound until per-commit checkout is added.

use anyhow::Result;
use engram_domain::ConnectionMap;
use engram_repo_map::cochange::{self, CommitRecord};
use engram_repo_map::store::Store;
use engram_retrieval::Engine;
use std::collections::HashSet;
use std::path::PathBuf;

/// Evidence rows reported, in output order.
const KINDS: [&str; 3] = ["static import", "historical co-change", "combined"];

struct Sample {
    anchor: String,
    hidden: HashSet<String>,
    /// Committer timestamp of the commit under evaluation. History at or after
    /// this instant is withheld from the graph when scoring this sample.
    timestamp: i64,
}

#[derive(Default, Clone, Copy)]
struct Totals {
    recall_at_5: f64,
    recall_at_10: f64,
    mrr: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let repo = arg(&args, "--repo")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let max_commits = arg(&args, "--max-commits")
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    engram_repo_map::index_repo(&repo, 100_000)?;
    let mut store = Store::open(&repo)?;
    let engine = Engine::build(&repo, &mut store)?;
    let indexed: HashSet<String> = store.all_files()?.into_iter().map(|f| f.path).collect();

    let records = cochange::commit_records(&repo);
    let samples = collect_samples(&records, max_commits, &indexed);
    if samples.is_empty() {
        eprintln!("[evals] no commits with an anchor and hidden files");
        return Ok(());
    }

    // Scoring rewrites the co-change table once per sample, so the full graph
    // must be restored before returning — including on failure. Otherwise a
    // benchmark run silently leaves the repository's real index holding
    // nothing but the last sample's anchor edges.
    let result = score(&engine, &mut store, &records, &samples);
    let full = cochange::build_from_records(&records, None);
    store.replace_cochange(&full.edges)?;
    store.update_recency(&full.last_commit)?;
    let totals = result?;

    println!("# Engram connection-recovery benchmark\n");
    println!(
        "Repo: `{}` · {} files · {} samples\n",
        repo.display(),
        indexed.len(),
        samples.len()
    );
    println!(
        "Co-change evidence is rebuilt per sample from commits strictly older \
         than the commit under evaluation. The static import graph is still \
         built at HEAD, so that row is an upper bound.\n"
    );
    println!("| evidence | Recall@5 | Recall@10 | MRR |");
    println!("|---|---|---|---|");
    let n = samples.len() as f64;
    for (kind, total) in KINDS.iter().zip(totals) {
        println!(
            "| {kind} | {:.3} | {:.3} | {:.3} |",
            total.recall_at_5 / n,
            total.recall_at_10 / n,
            total.mrr / n
        );
    }
    Ok(())
}

/// Score every sample against a temporally isolated co-change graph.
/// Returns one `Totals` per entry in [`KINDS`], summed over samples.
fn score(
    engine: &Engine,
    store: &mut Store,
    records: &[CommitRecord],
    samples: &[Sample],
) -> Result<Vec<Totals>> {
    let mut totals = vec![Totals::default(); KINDS.len()];
    for (i, sample) in samples.iter().enumerate() {
        if i % 25 == 0 {
            eprintln!("[evals] sample {}/{}", i + 1, samples.len());
        }
        // Hide the commit under evaluation, and everything after it, from the
        // co-change graph before asking for its connections. Edge *strength*
        // still comes from the production builder, so this measures the real
        // scoring logic rather than a benchmark-local reimplementation.
        //
        // Only the anchor's own edges are written back: `expand_connections`
        // reads co-change exclusively through `cochange_for(anchor)`, so
        // persisting the whole graph 300 times would cost minutes of SQLite
        // writes to produce identical results.
        let history = cochange::build_from_records(records, Some(sample.timestamp));
        let anchor_edges: Vec<_> = history
            .edges
            .into_iter()
            .filter(|edge| edge.path_a == sample.anchor)
            .collect();
        store.replace_cochange(&anchor_edges)?;

        let map = engine.expand_connections(store, std::slice::from_ref(&sample.anchor), 2)?;
        for (kind, total) in KINDS.iter().zip(totals.iter_mut()) {
            let ranked = paths_for(kind, &map);
            total.recall_at_5 += recall_at_k(&ranked, &sample.hidden, 5);
            total.recall_at_10 += recall_at_k(&ranked, &sample.hidden, 10);
            total.mrr += reciprocal_rank(&ranked, &sample.hidden);
        }
    }
    Ok(totals)
}

fn paths_for(kind: &str, map: &ConnectionMap) -> Vec<String> {
    let mut paths: Vec<String> = match kind {
        "static import" => map.import_dependents.iter().map(|p| p.path.clone()).collect(),
        "historical co-change" => map
            .historical_connections
            .iter()
            .map(|p| p.path.clone())
            .collect(),
        _ => map
            .import_dependents
            .iter()
            .chain(&map.historical_connections)
            .chain(&map.related_tests)
            .map(|p| p.path.clone())
            .collect(),
    };
    let mut seen = HashSet::new();
    paths.retain(|path| seen.insert(path.clone()));
    paths
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Turn commit records into evaluation samples: one anchor plus the hidden
/// files it changed alongside. Commits touching only one indexed file, or a
/// sprawling set, carry no recoverable signal and are skipped.
fn collect_samples(
    records: &[CommitRecord],
    max_commits: usize,
    indexed: &HashSet<String>,
) -> Vec<Sample> {
    let mut samples = Vec::new();
    for (timestamp, files) in records.iter().take(max_commits) {
        let mut paths: Vec<String> = files
            .iter()
            .filter(|path| indexed.contains(*path))
            .cloned()
            .collect();
        paths.sort();
        paths.dedup();
        if paths.len() < 2 || paths.len() > 8 {
            continue;
        }
        let anchor = paths.remove(0);
        samples.push(Sample {
            anchor,
            hidden: paths.into_iter().collect(),
            timestamp: *timestamp,
        });
    }
    samples
}

fn recall_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    ranked
        .iter()
        .take(k)
        .filter(|path| relevant.contains(*path))
        .count() as f64
        / relevant.len() as f64
}

fn reciprocal_rank(ranked: &[String], relevant: &HashSet<String>) -> f64 {
    ranked
        .iter()
        .position(|path| relevant.contains(path))
        .map(|i| 1.0 / (i as f64 + 1.0))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_counts_only_hidden_connections() {
        let relevant = HashSet::from(["b".to_owned(), "d".to_owned()]);
        let ranked = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        assert!((recall_at_k(&ranked, &relevant, 2) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn samples_carry_the_commit_timestamp_as_their_cutoff() {
        let indexed = HashSet::from(["a.rs".to_owned(), "b.rs".to_owned()]);
        let records = vec![(500, vec!["a.rs".to_owned(), "b.rs".to_owned()])];
        let samples = collect_samples(&records, 10, &indexed);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].timestamp, 500);
        assert_eq!(samples[0].anchor, "a.rs");
        assert_eq!(samples[0].hidden, HashSet::from(["b.rs".to_owned()]));
    }

    #[test]
    fn skips_commits_without_a_recoverable_pair() {
        let indexed = HashSet::from(["a.rs".to_owned()]);
        // Only one indexed file changed: nothing to recover.
        let records = vec![(1, vec!["a.rs".to_owned(), "untracked.bin".to_owned()])];
        assert!(collect_samples(&records, 10, &indexed).is_empty());
    }

    #[test]
    fn skips_sprawling_commits() {
        let files: Vec<String> = (0..9).map(|i| format!("f{i}.rs")).collect();
        let indexed: HashSet<String> = files.iter().cloned().collect();
        let records = vec![(1, files)];
        assert!(collect_samples(&records, 10, &indexed).is_empty());
    }
}
