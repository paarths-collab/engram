//! engram-repo-map: Tier-0 inventory, Tier-1 symbols, co-change graph, SQLite store.

pub mod cochange;
pub mod graph;
pub mod imports;
pub mod inventory;
pub mod store;
pub mod symbols;

use anyhow::Result;
use std::path::Path;
use store::Store;

/// Run the full background indexing pass: Tier-0 inventory + co-change.
/// Tier-1 symbol extraction is done lazily by retrieval, or eagerly here for small repos.
pub fn index_repo(repo_root: &Path, eager_tier1_limit: usize) -> Result<IndexStats> {
    let mut store = Store::open(repo_root)?;

    let files = inventory::scan(repo_root);
    store.upsert_files(&files)?;
    // `upsert_files` never deletes, so files removed from the repo since the
    // last pass would otherwise keep their row, symbols, and import edges and
    // go on being cited as evidence.
    let present: std::collections::HashSet<String> = files.iter().map(|f| f.path.clone()).collect();
    let pruned_files = store.prune_missing_files(&present)?;

    let history = cochange::build(repo_root);
    store.replace_cochange(&history.edges)?;
    store.update_recency(&history.last_commit)?;

    // Eagerly extract symbols for up to N source files (small repos = instant full coverage).
    // Languages without a parser are skipped rather than "extracted" to nothing:
    // see symbols::supports.
    let mut extracted = 0usize;
    for f in files
        .iter()
        .filter(|f| f.indexing_ineligibility.is_none() && symbols::supports(f.language))
    {
        if extracted >= eager_tier1_limit {
            break;
        }
        if store.is_tier1_done(&f.path) {
            continue;
        }
        if let Ok(src) = std::fs::read_to_string(repo_root.join(&f.path)) {
            let syms = symbols::extract_file(&f.path, &src, f.language);
            store.replace_symbols_for_file(&f.path, &syms)?;
            let imports = symbols::extract_imports(&src, f.language);
            store.replace_imports_for_file(&f.path, &imports)?;
            extracted += 1;
        }
    }

    let coverage = store.coverage()?;
    Ok(IndexStats {
        files: coverage.files,
        ineligible_files: coverage.ineligible_files,
        cochange_edges: history.edges.len(),
        parser_supported_files: coverage.parser_supported_files,
        unsupported_source_files: coverage.unsupported_source_files,
        tier1_files: coverage.tier1_done_files,
        tier1_files_extracted_this_run: extracted,
        symbols: coverage.symbols,
        pull_requests: coverage.pull_requests,
        review_comments: coverage.review_comments,
        pruned_files,
    })
}

/// Lazily ensure a file is Tier-1 extracted (called by retrieval on cache miss).
pub fn ensure_tier1(store: &mut Store, repo_root: &Path, path: &str) -> Result<()> {
    if store.is_tier1_done(path) {
        return Ok(());
    }
    let lang = engram_domain::Language::from_path(path);
    // Checked before touching the disk: this runs per retrieval hit, and an
    // unsupported file would otherwise be re-read on every single search.
    if !symbols::supports(lang) {
        return Ok(());
    }
    if let Ok(src) = std::fs::read_to_string(repo_root.join(path)) {
        let syms = symbols::extract_file(path, &src, lang);
        store.replace_symbols_for_file(path, &syms)?;
        let imports = symbols::extract_imports(&src, lang);
        store.replace_imports_for_file(path, &imports)?;
    }
    Ok(())
}

/// Incrementally re-extract a single changed file: refresh its `files` row,
/// symbols, and imports, and invalidate its cached vector so it re-embeds.
/// If the file no longer exists (or is unsupported), its extracted data is
/// cleared. Used by the file-watcher for reindex-on-save.
pub fn reindex_file(store: &mut Store, repo_root: &Path, path: &str) -> Result<()> {
    match inventory::scan_path(repo_root, path) {
        Some(file) => {
            let lang = file.language;
            let index_eligible = file.indexing_ineligibility.is_none();
            store.upsert_files(&[file])?;
            if index_eligible && symbols::supports(lang) {
                if let Ok(src) = std::fs::read_to_string(repo_root.join(path)) {
                    store
                        .replace_symbols_for_file(path, &symbols::extract_file(path, &src, lang))?;
                    store.replace_imports_for_file(path, &symbols::extract_imports(&src, lang))?;
                }
            }
        }
        _ => {
            // Removed or unreadable: clear extracted data.
            store.replace_symbols_for_file(path, &[])?;
            store.replace_imports_for_file(path, &[])?;
        }
    }
    store.invalidate_vector(path)?;
    Ok(())
}

/// Rebuild the co-change graph and file recency from git history (called when
/// HEAD moves — a new commit landed).
pub fn refresh_history(store: &mut Store, repo_root: &Path) -> Result<()> {
    let history = cochange::build(repo_root);
    store.replace_cochange(&history.edges)?;
    store.update_recency(&history.last_commit)?;
    Ok(())
}

#[derive(Debug)]
pub struct IndexStats {
    /// Cumulative files in the persisted Tier-0 inventory.
    pub files: usize,
    /// Inventoried files omitted from content and Tier-1 indexing.
    pub ineligible_files: usize,
    pub cochange_edges: usize,
    /// Cumulative inventory files whose language has a Tier-1 parser.
    pub parser_supported_files: usize,
    /// Eligible source files in languages without a Tier-1 parser.
    pub unsupported_source_files: usize,
    /// Cumulative parser-supported files whose Tier-1 extraction completed.
    pub tier1_files: usize,
    /// Work performed during this pass; unlike `tier1_files`, this can be zero
    /// for a fully covered warm index.
    pub tier1_files_extracted_this_run: usize,
    /// Cumulative extracted symbols in the persisted store.
    pub symbols: usize,
    /// Cumulative GitHub ingestion counts. These are observed rows, not a claim
    /// that pull-request history is complete.
    pub pull_requests: usize,
    pub review_comments: usize,
    /// Paths dropped because they no longer exist in the repository.
    pub pruned_files: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    #[test]
    fn index_stats_keep_cumulative_tier1_coverage_on_a_warm_run() {
        let root = std::env::temp_dir().join(format!(
            "engram-index-stats-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("lib.rs"), "pub fn reusable() {}\n").unwrap();
        std::fs::write(root.join("worker.go"), "package worker\n").unwrap();

        let first = index_repo(&root, 100).unwrap();
        assert_eq!(first.files, 2);
        assert_eq!(first.ineligible_files, 0);
        assert_eq!(first.parser_supported_files, 1);
        assert_eq!(first.unsupported_source_files, 1);
        assert_eq!(first.tier1_files, 1);
        assert_eq!(first.tier1_files_extracted_this_run, 1);
        assert_eq!(first.symbols, 1);

        let second = index_repo(&root, 100).unwrap();
        assert_eq!(second.files, first.files);
        assert_eq!(second.ineligible_files, first.ineligible_files);
        assert_eq!(second.parser_supported_files, first.parser_supported_files);
        assert_eq!(
            second.unsupported_source_files,
            first.unsupported_source_files
        );
        assert_eq!(second.tier1_files, first.tier1_files);
        assert_eq!(second.symbols, first.symbols);
        assert_eq!(second.tier1_files_extracted_this_run, 0);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cold_reindex_replaces_a_same_size_files_stale_symbol() {
        let root = std::env::temp_dir().join(format!(
            "engram-cold-reindex-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("lib.rs");
        let old_source = "pub fn old_name() {}\n";
        let new_source = "pub fn new_name() {}\n";
        assert_eq!(old_source.len(), new_source.len());

        std::fs::write(&path, old_source).unwrap();
        let first = index_repo(&root, 100).unwrap();
        assert_eq!(first.tier1_files_extracted_this_run, 1);
        {
            let mut store = Store::open(&root).unwrap();
            assert_eq!(store.symbols_exact("old_name", 5).unwrap().len(), 1);
            store
                .upsert_vectors(&[("lib.rs".to_owned(), "old".to_owned(), vec![1])])
                .unwrap();
            store
                .upsert_chunk_vectors(&[("lib.rs".to_owned(), 1, "old".to_owned(), vec![1])])
                .unwrap();
        }

        // Simulate a restart: every Store used above was dropped before the
        // same-size edit and second full indexing pass.
        std::fs::write(&path, new_source).unwrap();
        let second = index_repo(&root, 100).unwrap();
        assert_eq!(second.tier1_files_extracted_this_run, 1);

        let store = Store::open(&root).unwrap();
        assert!(store.symbols_exact("old_name", 5).unwrap().is_empty());
        assert_eq!(store.symbols_exact("new_name", 5).unwrap().len(), 1);
        assert!(store.load_vectors().unwrap().is_empty());
        assert!(store.load_chunk_vectors().unwrap().is_empty());

        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn oversized_source_is_reported_but_not_returned_for_indexing() {
        let root = std::env::temp_dir().join(format!(
            "engram-oversized-index-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("generated.rs"), vec![b'x'; 1_500_001]).unwrap();

        let stats = index_repo(&root, 100).unwrap();
        assert_eq!(stats.files, 1);
        assert_eq!(stats.ineligible_files, 1);
        assert_eq!(stats.parser_supported_files, 0);
        assert!(Store::open(&root).unwrap().all_files().unwrap().is_empty());

        let _ = std::fs::remove_dir_all(root);
    }
}
