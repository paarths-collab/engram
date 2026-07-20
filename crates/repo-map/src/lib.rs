//! engram-repo-map: Tier-0 inventory, Tier-1 symbols, co-change graph, SQLite store.

pub mod cochange;
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

    let history = cochange::build(repo_root);
    store.replace_cochange(&history.edges)?;
    store.update_recency(&history.last_commit)?;

    // Eagerly extract symbols for up to N source files (small repos = instant full coverage).
    let mut extracted = 0usize;
    for f in files
        .iter()
        .filter(|f| f.language != engram_domain::Language::Other)
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

    Ok(IndexStats {
        files: files.len(),
        cochange_edges: history.edges.len(),
        tier1_files: extracted,
    })
}

/// Lazily ensure a file is Tier-1 extracted (called by retrieval on cache miss).
pub fn ensure_tier1(store: &mut Store, repo_root: &Path, path: &str) -> Result<()> {
    if store.is_tier1_done(path) {
        return Ok(());
    }
    let lang = engram_domain::Language::from_path(path);
    if lang == engram_domain::Language::Other {
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

#[derive(Debug)]
pub struct IndexStats {
    pub files: usize,
    pub cochange_edges: usize,
    pub tier1_files: usize,
}
