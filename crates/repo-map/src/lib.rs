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
    for f in files.iter().filter(|f| symbols::supports(f.language)) {
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

    Ok(IndexStats {
        files: files.len(),
        cochange_edges: history.edges.len(),
        tier1_files: extracted,
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
    use engram_domain::{FileRecord, Language};
    let lang = Language::from_path(path);
    let full = repo_root.join(path);
    match std::fs::metadata(&full) {
        Ok(meta) if meta.is_file() => {
            store.upsert_files(&[FileRecord {
                is_test: inventory::is_test_path(path),
                path: path.to_string(),
                language: lang,
                size_bytes: meta.len(),
            }])?;
            if symbols::supports(lang) {
                if let Ok(src) = std::fs::read_to_string(&full) {
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
    pub files: usize,
    pub cochange_edges: usize,
    pub tier1_files: usize,
    /// Paths dropped because they no longer exist in the repository.
    pub pruned_files: usize,
}
