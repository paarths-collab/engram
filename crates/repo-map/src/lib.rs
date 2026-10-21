//! engram-repo-map: Tier-0 inventory, Tier-1 symbols, co-change graph, SQLite store.
//!
//! # Staged, atomically published index generations
//!
//! Every production path that changes the index works in two steps. First it
//! STAGES: it reads and parses files and walks git history into an in-memory
//! [`store::IndexBatch`], holding no write transaction while it does the slow
//! work. Then it PUBLISHES with one [`Store::publish_batch`] call, which applies
//! the whole batch in a single transaction and bumps the store's `generation`
//! inside it. A reader therefore sees either all of a batch or none of it, and
//! a failure before the commit leaves the previous generation untouched.

pub mod cochange;
pub mod graph;
pub mod imports;
pub mod inventory;
pub mod store;
pub mod symbols;

use anyhow::Result;
use std::path::Path;
use store::{Extraction, ExtractionUpdate, FileUpdate, HistoryUpdate, IndexBatch, Store};

/// Most extractions one published generation carries. A full reconcile whose
/// eager extraction fits in this many files is ONE generation, including every
/// removal, modification and addition. Beyond it the extractions are published
/// in further generations of this size (the inventory, removals and history are
/// always in the first), so memory stays bounded; each such generation is
/// itself atomic, but the reconcile as a whole is not. Extraction results are
/// small (symbols and import targets, never file contents), so the limit is
/// generous: the server's own eager limit is 3000.
pub const MAX_EXTRACTIONS_PER_GENERATION: usize = 5000;

/// Run the full background indexing pass: Tier-0 inventory + co-change.
/// Tier-1 symbol extraction is done lazily by retrieval, or eagerly here for small repos.
///
/// The whole pass is staged first (scan, git history, parsing up to
/// `eager_tier1_limit` files) and then published as one generation containing
/// every change: files edited or added while the server was stopped, files
/// removed, the refreshed history, and the new symbols and imports. When the
/// call returns, the stale facts are gone and the new ones are in, together;
/// a call that finds nothing to change does not move the generation.
pub fn index_repo(repo_root: &Path, eager_tier1_limit: usize) -> Result<IndexStats> {
    index_repo_in_generations(repo_root, eager_tier1_limit, MAX_EXTRACTIONS_PER_GENERATION)
}

/// [`index_repo`] with the per-generation extraction limit spelled out, so the
/// chunked case can be tested without thousands of files.
fn index_repo_in_generations(
    repo_root: &Path,
    eager_tier1_limit: usize,
    per_generation: usize,
) -> Result<IndexStats> {
    let mut store = Store::open(repo_root)?;

    let files = inventory::scan(repo_root);
    // Every file is upserted with its (size, mtime, hash) fingerprint, which
    // resets `tier1_done` and drops the derived rows of any file whose
    // fingerprint changed, so a file edited while the server was stopped is
    // re-extracted below instead of keeping the symbols and import edges of its
    // previous contents. `retain_only` removes the files that are gone since
    // the last pass, which would otherwise keep their row, symbols, and import
    // edges and go on being cited as evidence.
    let present: std::collections::HashSet<String> =
        files.iter().map(|f| f.record.path.clone()).collect();

    let history = cochange::build(repo_root);
    let cochange_edges = history.edges.len();
    let file_count = files.len();

    // Eagerly extract symbols for up to N source files (small repos = instant full coverage).
    // Languages without a parser are skipped rather than "extracted" to nothing:
    // see symbols::supports. A file already extracted at its current fingerprint
    // is skipped, so an unchanged file is never parsed again.
    let pending = store.pending_extraction(&files)?;
    let mut extractions = Vec::new();
    let mut extracted = 0usize;
    for f in files
        .iter()
        .map(|scanned| &scanned.record)
        .filter(|f| symbols::supports(f.language))
    {
        if extracted >= eager_tier1_limit {
            break;
        }
        if !pending.contains(&f.path) {
            continue;
        }
        let update = stage_extraction(repo_root, &f.path, f.language);
        if matches!(update.extraction, Extraction::Extracted { .. }) {
            extracted += 1;
        }
        extractions.push(update);
    }

    // Beyond the per-generation limit the surplus extractions go in later
    // generations; the first always carries the inventory, removals, history.
    let mut later = if extractions.len() > per_generation {
        extractions.split_off(per_generation)
    } else {
        Vec::new()
    };
    let first = IndexBatch {
        retain_only: Some(present),
        removals: Vec::new(),
        files: files
            .into_iter()
            .map(|scanned| FileUpdate {
                scanned,
                invalidate_vectors: false,
            })
            .collect(),
        extractions,
        history: Some(HistoryUpdate::from(history)),
    };
    let published = store.publish_batch_detailed(first)?;
    while !later.is_empty() {
        let next = if later.len() > per_generation {
            later.split_off(per_generation)
        } else {
            Vec::new()
        };
        store.publish_batch(IndexBatch {
            extractions: std::mem::replace(&mut later, next),
            ..IndexBatch::default()
        })?;
    }

    Ok(IndexStats {
        files: file_count,
        cochange_edges,
        tier1_files: extracted,
        pruned_files: published.pruned_files,
    })
}

impl From<cochange::History> for HistoryUpdate {
    fn from(history: cochange::History) -> Self {
        HistoryUpdate {
            edges: history.edges,
            facts: history.facts,
            recency: history.last_commit,
        }
    }
}

/// Read a file and extract BOTH its symbols and its imports as it is now, or
/// note that it could not be read (the store then clears whatever it holds for
/// it, and leaves it un-extracted so a later pass retries it). Touches no
/// store: the result is applied by a publish.
fn stage_extraction(
    repo_root: &Path,
    path: &str,
    lang: engram_domain::Language,
) -> ExtractionUpdate {
    let extraction = match std::fs::read_to_string(repo_root.join(path)) {
        Ok(src) => Extraction::Extracted {
            symbols: symbols::extract_file(path, &src, lang),
            imports: symbols::extract_imports(&src, lang),
        },
        Err(_) => Extraction::Unreadable,
    };
    ExtractionUpdate {
        path: path.to_owned(),
        extraction,
    }
}

/// Lazily ensure a file is Tier-1 extracted (called by retrieval on cache miss).
///
/// Returns whether this call changed the stored symbols or imports, which it
/// publishes as a generation of its own (see [`Store::publish_batch`]). A
/// caller holding an in-memory index built from the store (chunk boundaries,
/// the import graph) must treat `true` as "my copy is now stale".
pub fn ensure_tier1(store: &mut Store, repo_root: &Path, path: &str) -> Result<bool> {
    if store.is_tier1_done(path) {
        return Ok(false);
    }
    let lang = engram_domain::Language::from_path(path);
    // Checked before touching the disk: this runs per retrieval hit, and an
    // unsupported file would otherwise be re-read on every single search.
    if !symbols::supports(lang) {
        return Ok(false);
    }
    let batch = IndexBatch {
        extractions: vec![stage_extraction(repo_root, path, lang)],
        ..IndexBatch::default()
    };
    Ok(store.publish_batch_detailed(batch)?.changed)
}

/// Stage one file that is part of the inventory: its `files` row (with its
/// cached vectors invalidated, so they re-embed) and, for a language with a
/// parser, its symbols and imports as they are now.
fn stage_scanned(batch: &mut IndexBatch, repo_root: &Path, scanned: inventory::ScannedFile) {
    let path = scanned.record.path.clone();
    let lang = scanned.record.language;
    batch.files.push(FileUpdate {
        scanned,
        invalidate_vectors: true,
    });
    if symbols::supports(lang) {
        batch
            .extractions
            .push(stage_extraction(repo_root, &path, lang));
    }
}

/// Stage the refresh of a single changed path into `batch`, from the disk as it
/// is right now: a file that is still part of the inventory is re-recorded and
/// re-extracted; anything else (removed, replaced by something that is not a
/// file, or no longer indexable) is forgotten entirely, and a directory path
/// takes everything beneath it. The caller publishes the batch.
///
/// Staging a path reads the disk's current state, not "what the event said", so
/// several events for one path, in any order, stage the same thing.
pub fn stage_reindex_file(batch: &mut IndexBatch, repo_root: &Path, path: &str) {
    use engram_domain::{FileRecord, Language};
    let full = repo_root.join(path);
    match std::fs::metadata(&full) {
        // Same inclusion rules as the full scan, so a live-updated index and a
        // freshly built one cover the same files.
        Ok(meta) if meta.is_file() && inventory::is_indexable(path, meta.len()) => {
            stage_scanned(
                batch,
                repo_root,
                inventory::ScannedFile {
                    record: FileRecord {
                        is_test: inventory::is_test_path(path),
                        path: path.to_string(),
                        language: Language::from_path(path),
                        size_bytes: meta.len(),
                    },
                    mtime_ns: inventory::mtime_ns(&meta),
                    content_hash: inventory::hash_file(&full),
                },
            );
        }
        _ => batch.removals.push(path.to_string()),
    }
}

/// Stage every file under a directory that appeared or moved into the repo.
/// A directory rename reaches the watcher as an event for the directory only,
/// so its files would otherwise never be seen until the next restart.
/// Returns the number of files staged.
///
/// The scan already read and hashed each file, so its result is staged as is
/// rather than going back through [`stage_reindex_file`] (which would hash it
/// again).
pub fn stage_reindex_dir(batch: &mut IndexBatch, repo_root: &Path, rel_dir: &str) -> usize {
    let prefix = rel_dir.trim_end_matches('/');
    let mut staged = 0;
    for mut scanned in inventory::scan(&repo_root.join(prefix)) {
        // The scan is rooted at the directory; the store is rooted at the repo.
        scanned.record.path = format!("{prefix}/{}", scanned.record.path);
        scanned.record.is_test = inventory::is_test_path(&scanned.record.path);
        // The scan could not see the ignored directories above `prefix`.
        if !inventory::is_indexable(&scanned.record.path, scanned.record.size_bytes) {
            continue;
        }
        stage_scanned(batch, repo_root, scanned);
        staged += 1;
    }
    staged
}

/// Stage a rebuild of the co-change graph and file recency from git history.
pub fn stage_history(batch: &mut IndexBatch, repo_root: &Path) {
    batch.history = Some(HistoryUpdate::from(cochange::build(repo_root)));
}

/// Incrementally re-extract a single changed file and publish it as one
/// generation: refresh its `files` row, symbols, and imports, and invalidate
/// its cached vector so it re-embeds. If the file no longer exists, every trace
/// of it is removed (row, symbols, imports, vectors). A rename arrives as a
/// remove of the old path plus a create of the new one, and each half lands
/// correctly on its own. The file-watcher stages many events into one batch
/// instead (see [`stage_reindex_file`]); this is the single-path form.
///
/// Returns whether the index changed: `false` when nothing was stored for the
/// path and nothing is now, or when it was re-recorded exactly as it was, so
/// the caller can skip rebuilding.
pub fn reindex_file(store: &mut Store, repo_root: &Path, path: &str) -> Result<bool> {
    let mut batch = IndexBatch::default();
    stage_reindex_file(&mut batch, repo_root, path);
    Ok(store.publish_batch_detailed(batch)?.changed)
}

/// Index every file under a directory that appeared or moved into the repo, as
/// one generation. Returns the number of files reindexed.
pub fn reindex_dir(store: &mut Store, repo_root: &Path, rel_dir: &str) -> Result<usize> {
    let mut batch = IndexBatch::default();
    let staged = stage_reindex_dir(&mut batch, repo_root, rel_dir);
    store.publish_batch(batch)?;
    Ok(staged)
}

/// The commit the repository has checked out, or `"no-git"` when there is none
/// (not a git repository, or no commits yet).
pub fn head_commit(repo_root: &Path) -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|sha| !sha.is_empty())
        .unwrap_or_else(|| "no-git".to_owned())
}

/// Identity of what is currently indexed; see [`Store::snapshot_id`].
pub fn snapshot_id(store: &Store, repo_root: &Path) -> Result<String> {
    store.snapshot_id(&head_commit(repo_root))
}

/// Rebuild the co-change graph and file recency from git history (called when
/// HEAD moves — a new commit landed), published as one generation: the edges,
/// their completeness facts and the recency land together or not at all.
pub fn refresh_history(store: &mut Store, repo_root: &Path) -> Result<()> {
    let mut batch = IndexBatch::default();
    stage_history(&mut batch, repo_root);
    store.publish_batch(batch)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, UNIX_EPOCH};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Throwaway repo root, removed on drop. Not `tempfile`: CI builds
    /// `--locked`, so a dev-dependency would need a Cargo.lock regeneration.
    /// It is not a git repository; `index_repo` must cope with that.
    struct TempRepo(PathBuf);

    impl TempRepo {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "engram-repomap-test-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create temp repo");
            TempRepo(path)
        }

        /// Write a file and pin its mtime, so the test controls the
        /// fingerprint rather than racing the filesystem's clock granularity.
        fn write_at(&self, rel: &str, contents: &str, mtime_secs: u64) {
            let full = self.0.join(rel);
            std::fs::create_dir_all(full.parent().unwrap()).expect("mkdir");
            std::fs::write(&full, contents).expect("write");
            let file = std::fs::File::options()
                .write(true)
                .open(&full)
                .expect("reopen");
            file.set_modified(UNIX_EPOCH + Duration::from_secs(mtime_secs))
                .expect("set mtime");
        }

        fn store(&self) -> Store {
            Store::open(&self.0).expect("open store")
        }
    }

    impl Drop for TempRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn symbol_names(store: &Store, path: &str) -> Vec<String> {
        store
            .symbols_for_path(path)
            .unwrap()
            .into_iter()
            .map(|s| s.name)
            .collect()
    }

    fn imports_of(store: &Store, path: &str) -> Vec<String> {
        store
            .all_imports()
            .unwrap()
            .into_iter()
            .filter(|(p, _)| p == path)
            .map(|(_, t)| t)
            .collect()
    }

    fn count(store: &Store, table: &str, path: &str) -> i64 {
        store
            .conn
            .query_row(
                &format!("SELECT COUNT(*) FROM {table} WHERE path = ?1"),
                [path],
                |r| r.get(0),
            )
            .unwrap()
    }

    const OLD: &str = "use helpers::thing;\nfn old_name() {}\n";
    const NEW: &str = "use other_mod::stuff;\nfn new_name_here() {}\n";

    #[test]
    fn a_failed_history_walk_leaves_the_cochange_facts_unknown_not_zero() {
        // Not a repository: the log command exits nonzero.
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        let stats = index_repo(&repo.0, 100).unwrap();
        assert_eq!(stats.cochange_edges, 0);
        let store = repo.store();
        assert_eq!(store.cochange_totals().unwrap(), None);
        assert_eq!(store.cochange_commits_touching("a.rs").unwrap(), None);
        // Again, over rows that did carry facts: a failed refresh clears them
        // to unknown instead of recording a walk of zero commits.
        let mut store = repo.store();
        let facts = cochange::CoChangeFacts {
            commits_scanned: 9,
            ..Default::default()
        };
        store.replace_cochange_with_facts(&[], &facts).unwrap();
        assert_eq!(store.cochange_totals().unwrap(), Some((9, 0)));
        refresh_history(&mut store, &repo.0).unwrap();
        assert_eq!(store.cochange_totals().unwrap(), None);
    }

    #[test]
    fn an_offline_edit_replaces_both_symbols_and_imports() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        index_repo(&repo.0, 100).unwrap();
        {
            let store = repo.store();
            assert_eq!(symbol_names(&store, "a.rs"), vec!["old_name"]);
            assert_eq!(imports_of(&store, "a.rs"), vec!["helpers/thing"]);
        }

        // The server is "stopped" here: edit the file, then start again.
        repo.write_at("a.rs", NEW, 2_000_000);
        index_repo(&repo.0, 100).unwrap();

        let store = repo.store();
        assert_eq!(symbol_names(&store, "a.rs"), vec!["new_name_here"]);
        assert_eq!(imports_of(&store, "a.rs"), vec!["other_mod/stuff"]);
    }

    #[test]
    fn a_same_size_edit_is_caught_by_its_mtime() {
        let repo = TempRepo::new();
        let before = "fn alpha_one() {}\n";
        let after = "fn betaa_two() {}\n";
        assert_eq!(before.len(), after.len(), "the test needs equal sizes");
        repo.write_at("a.rs", before, 1_000_000);
        index_repo(&repo.0, 100).unwrap();

        repo.write_at("a.rs", after, 1_000_050);
        index_repo(&repo.0, 100).unwrap();

        let store = repo.store();
        assert_eq!(symbol_names(&store, "a.rs"), vec!["betaa_two"]);
    }

    #[test]
    fn an_untouched_file_is_not_re_extracted() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        assert_eq!(index_repo(&repo.0, 100).unwrap().tier1_files, 1);
        assert_eq!(
            index_repo(&repo.0, 100).unwrap().tier1_files,
            0,
            "an unchanged fingerprint must keep the cached extraction"
        );
    }

    #[test]
    fn an_offline_delete_leaves_no_trace() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        repo.write_at("keep.rs", "fn keeper() {}\n", 1_000_000);
        index_repo(&repo.0, 100).unwrap();
        {
            // Give it embeddings too, as a built Engine would have.
            let mut store = repo.store();
            store
                .upsert_chunk_vectors(&[("a.rs".into(), 0, "h".into(), vec![1])])
                .unwrap();
            store
                .upsert_vectors(&[("a.rs".into(), "h".into(), vec![1])])
                .unwrap();
        }

        std::fs::remove_file(repo.0.join("a.rs")).unwrap();
        let stats = index_repo(&repo.0, 100).unwrap();
        assert_eq!(stats.pruned_files, 1);

        let store = repo.store();
        for table in [
            "files",
            "symbols",
            "file_imports",
            "vectors",
            "chunk_vectors",
        ] {
            assert_eq!(count(&store, table, "a.rs"), 0, "{table} kept a row");
        }
        assert_eq!(symbol_names(&store, "keep.rs"), vec!["keeper"]);
    }

    #[test]
    fn a_database_from_before_fingerprints_re_extracts_once() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        index_repo(&repo.0, 100).unwrap();
        {
            // Roll the DB back to its pre-fingerprint shape...
            let store = repo.store();
            store
                .conn
                .execute_batch(
                    "ALTER TABLE files DROP COLUMN mtime_ns;
                     ALTER TABLE files DROP COLUMN content_hash;",
                )
                .unwrap();
        }
        // ...and edit the file so that size AND mtime match what is cached:
        // only the missing fingerprint can reveal that the rows are stale.
        let same_size = OLD.replace("old_name", "ald_name");
        repo.write_at("a.rs", &same_size, 1_000_000);

        index_repo(&repo.0, 100).unwrap();
        let store = repo.store();
        assert_eq!(symbol_names(&store, "a.rs"), vec!["ald_name"]);
    }

    #[test]
    fn reindex_of_a_removed_file_deletes_its_row_and_traces() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        index_repo(&repo.0, 100).unwrap();
        let mut store = repo.store();
        store
            .upsert_chunk_vectors(&[("a.rs".into(), 0, "h".into(), vec![1])])
            .unwrap();
        store
            .upsert_vectors(&[("a.rs".into(), "h".into(), vec![1])])
            .unwrap();

        std::fs::remove_file(repo.0.join("a.rs")).unwrap();
        reindex_file(&mut store, &repo.0, "a.rs").unwrap();

        for table in [
            "files",
            "symbols",
            "file_imports",
            "vectors",
            "chunk_vectors",
        ] {
            assert_eq!(count(&store, table, "a.rs"), 0, "{table} kept a row");
        }
    }

    #[test]
    fn a_rename_seen_as_remove_plus_create_ends_up_correct() {
        let repo = TempRepo::new();
        repo.write_at("old.rs", OLD, 1_000_000);
        index_repo(&repo.0, 100).unwrap();
        let mut store = repo.store();

        std::fs::rename(repo.0.join("old.rs"), repo.0.join("moved.rs")).unwrap();
        // Either order of arrival must converge on the same state.
        for order in [["old.rs", "moved.rs"], ["moved.rs", "old.rs"]] {
            for path in order {
                reindex_file(&mut store, &repo.0, path).unwrap();
            }
            assert_eq!(count(&store, "files", "old.rs"), 0);
            assert_eq!(count(&store, "symbols", "old.rs"), 0);
            assert_eq!(count(&store, "file_imports", "old.rs"), 0);
            assert_eq!(symbol_names(&store, "moved.rs"), vec!["old_name"]);
            assert_eq!(imports_of(&store, "moved.rs"), vec!["helpers/thing"]);
        }
    }

    #[test]
    fn a_renamed_directory_is_removed_and_re_added_as_a_tree() {
        let repo = TempRepo::new();
        repo.write_at("pkg/a.rs", OLD, 1_000_000);
        index_repo(&repo.0, 100).unwrap();
        let mut store = repo.store();

        std::fs::rename(repo.0.join("pkg"), repo.0.join("lib2")).unwrap();
        reindex_file(&mut store, &repo.0, "pkg").unwrap();
        assert_eq!(count(&store, "files", "pkg/a.rs"), 0);
        assert_eq!(count(&store, "symbols", "pkg/a.rs"), 0);

        assert_eq!(reindex_dir(&mut store, &repo.0, "lib2").unwrap(), 1);
        assert_eq!(symbol_names(&store, "lib2/a.rs"), vec!["old_name"]);
    }

    #[test]
    fn reindex_dir_stores_repo_relative_paths_with_the_right_test_flag() {
        // The scan is rooted at the directory, where `a.rs` is not a test
        // path; rooted at the repo, `tests/a.rs` is.
        let repo = TempRepo::new();
        repo.write_at("tests/a.rs", OLD, 1_000_000);
        repo.write_at("tests/target/gen.rs", NEW, 1_000_000);
        let mut store = repo.store();

        assert_eq!(reindex_dir(&mut store, &repo.0, "tests/").unwrap(), 1);
        let files = store.all_files().unwrap();
        assert_eq!(files.len(), 1, "{files:?}");
        assert_eq!(files[0].path, "tests/a.rs");
        assert!(files[0].is_test);
        assert_eq!(symbol_names(&store, "tests/a.rs"), vec!["old_name"]);
    }

    #[test]
    fn reindex_dir_does_not_index_beneath_an_ignored_ancestor() {
        let repo = TempRepo::new();
        repo.write_at("vendor/lib/a.rs", OLD, 1_000_000);
        let mut store = repo.store();
        assert_eq!(reindex_dir(&mut store, &repo.0, "vendor/lib").unwrap(), 0);
        assert!(store.all_files().unwrap().is_empty());
    }

    #[test]
    fn reindex_file_reports_whether_the_index_changed() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        let mut store = repo.store();
        assert!(reindex_file(&mut store, &repo.0, "a.rs").unwrap());
        // Not indexed and never was: nothing to forget, nothing changed.
        assert!(!reindex_file(&mut store, &repo.0, "ghost.rs").unwrap());
        std::fs::remove_file(repo.0.join("a.rs")).unwrap();
        assert!(reindex_file(&mut store, &repo.0, "a.rs").unwrap());
        assert!(!reindex_file(&mut store, &repo.0, "a.rs").unwrap());
    }

    #[test]
    fn ensure_tier1_reports_whether_it_wrote_anything() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        repo.write_at("empty.rs", "// nothing here\n", 1_000_000);
        let mut store = repo.store();
        let files = inventory::scan(&repo.0);
        store.upsert_fingerprinted(&files).unwrap();

        assert!(ensure_tier1(&mut store, &repo.0, "a.rs").unwrap());
        assert_eq!(symbol_names(&store, "a.rs"), vec!["old_name"]);
        // Already done: nothing new, so a caller's in-memory copy stays valid.
        assert!(!ensure_tier1(&mut store, &repo.0, "a.rs").unwrap());
        // Extracted, but there was nothing to record.
        assert!(!ensure_tier1(&mut store, &repo.0, "empty.rs").unwrap());
        assert!(store.is_tier1_done("empty.rs"));
    }

    #[test]
    fn a_same_size_same_mtime_edit_is_caught_by_the_content_hash() {
        // The hardest case for a size+mtime fingerprint: the edit keeps the
        // length and the timestamp is put back. Only the bytes differ.
        let repo = TempRepo::new();
        let before = "fn alpha_one() {}\n";
        let after = "fn betaa_two() {}\n";
        assert_eq!(before.len(), after.len());
        repo.write_at("a.rs", before, 1_000_000);
        index_repo(&repo.0, 100).unwrap();

        repo.write_at("a.rs", after, 1_000_000);
        index_repo(&repo.0, 100).unwrap();

        let store = repo.store();
        assert_eq!(symbol_names(&store, "a.rs"), vec!["betaa_two"]);
    }

    #[test]
    fn a_changed_file_loses_its_old_derived_rows_at_once() {
        // Even if extraction is deferred (past the eager limit), the previous
        // contents' symbols, imports, and vectors must not survive the edit.
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        index_repo(&repo.0, 100).unwrap();
        {
            let mut store = repo.store();
            store
                .upsert_chunk_vectors(&[("a.rs".into(), 1, "h".into(), vec![1])])
                .unwrap();
        }
        repo.write_at("a.rs", NEW, 2_000_000);
        // Eager limit 0: inventory is reconciled but nothing is re-extracted.
        index_repo(&repo.0, 0).unwrap();

        let store = repo.store();
        assert!(!store.is_tier1_done("a.rs"));
        assert!(symbol_names(&store, "a.rs").is_empty());
        assert!(imports_of(&store, "a.rs").is_empty());
        assert_eq!(count(&store, "chunk_vectors", "a.rs"), 0);
    }

    #[test]
    fn offline_edit_rename_and_delete_are_all_reconciled() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", "use helpers::thing;\nfn user() {}\n", 1_000_000);
        repo.write_at("helpers.rs", "pub fn thing() {}\n", 1_000_000);
        repo.write_at("gone.rs", "fn ghost() {}\n", 1_000_000);
        index_repo(&repo.0, 100).unwrap();

        // Offline: change the import, rename the module, delete a file.
        repo.write_at("a.rs", "use renamed::thing;\nfn user() {}\n", 2_000_000);
        std::fs::rename(repo.0.join("helpers.rs"), repo.0.join("renamed.rs")).unwrap();
        std::fs::remove_file(repo.0.join("gone.rs")).unwrap();
        index_repo(&repo.0, 100).unwrap();

        let store = repo.store();
        assert_eq!(imports_of(&store, "a.rs"), vec!["renamed/thing"]);
        assert_eq!(symbol_names(&store, "renamed.rs"), vec!["thing"]);
        for table in ["files", "symbols", "file_imports"] {
            assert_eq!(count(&store, table, "helpers.rs"), 0, "{table}");
            assert_eq!(count(&store, table, "gone.rs"), 0, "{table}");
        }
    }

    fn snapshot_of(repo: &TempRepo) -> String {
        index_repo(&repo.0, 100).unwrap();
        snapshot_id(&repo.store(), &repo.0).unwrap()
    }

    #[test]
    fn identical_content_gives_the_same_snapshot_id_whatever_the_mtimes() {
        let (a, b) = (TempRepo::new(), TempRepo::new());
        a.write_at("src/x.rs", OLD, 1_000_000);
        b.write_at("src/x.rs", OLD, 7_000_000);
        assert_eq!(snapshot_of(&a), snapshot_of(&b));
        // And re-indexing the same checkout does not move it.
        assert_eq!(snapshot_of(&a), snapshot_of(&a));
    }

    #[test]
    fn same_paths_with_different_content_give_different_snapshot_ids() {
        let (a, b) = (TempRepo::new(), TempRepo::new());
        a.write_at("src/x.rs", OLD, 1_000_000);
        b.write_at("src/x.rs", NEW, 1_000_000);
        assert_ne!(snapshot_of(&a), snapshot_of(&b));
    }

    #[test]
    fn a_different_set_of_paths_gives_a_different_snapshot_id() {
        let (a, b) = (TempRepo::new(), TempRepo::new());
        a.write_at("x.rs", OLD, 1_000_000);
        b.write_at("y.rs", OLD, 1_000_000);
        assert_ne!(snapshot_of(&a), snapshot_of(&b));
    }

    #[test]
    fn an_edit_moves_the_snapshot_id() {
        let repo = TempRepo::new();
        repo.write_at("x.rs", OLD, 1_000_000);
        let before = snapshot_of(&repo);
        repo.write_at("x.rs", NEW, 2_000_000);
        assert_ne!(before, snapshot_of(&repo));
    }

    #[test]
    fn bumping_the_parser_config_version_invalidates_all_derived_data() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        let v1 = snapshot_of(&repo);
        assert_eq!(symbol_names(&repo.store(), "a.rs"), vec!["old_name"]);
        {
            let mut store = repo.store();
            store
                .upsert_vectors(&[("a.rs".into(), "h".into(), vec![1])])
                .unwrap();
        }

        let next = inventory::rules_fingerprint(&["other"], &[], 0, store::PARSER_CONFIG_VERSION);
        let store = Store::open_with_rules_fingerprint(&repo.0, &next).unwrap();
        assert!(!store.is_tier1_done("a.rs"));
        assert!(symbol_names(&store, "a.rs").is_empty());
        assert!(imports_of(&store, "a.rs").is_empty());
        assert_eq!(count(&store, "vectors", "a.rs"), 0);
        assert_ne!(
            snapshot_id(&store, &repo.0).unwrap(),
            v1,
            "the snapshot id must change with the parser version"
        );
        // The file row itself survives, ready to be re-extracted.
        assert_eq!(count(&store, "files", "a.rs"), 1);
    }

    #[test]
    fn reindexing_a_file_the_inventory_would_skip_keeps_it_out() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        repo.write_at("notes.txt", "hello", 1_000_000);
        repo.write_at("pkg/target/gen.rs", "fn generated() {}\n", 1_000_000);
        index_repo(&repo.0, 100).unwrap();
        let before = snapshot_id(&repo.store(), &repo.0).unwrap();

        let mut store = repo.store();
        reindex_file(&mut store, &repo.0, "notes.txt").unwrap();
        reindex_file(&mut store, &repo.0, "pkg/target/gen.rs").unwrap();
        assert_eq!(count(&store, "files", "notes.txt"), 0);
        assert_eq!(count(&store, "files", "pkg/target/gen.rs"), 0);
        assert_eq!(snapshot_id(&store, &repo.0).unwrap(), before);
    }

    // ----- generations -----

    fn generation_of(repo: &TempRepo) -> u64 {
        repo.store().generation().unwrap()
    }

    #[test]
    fn a_reconcile_of_edits_removals_and_additions_is_exactly_one_generation() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", "use helpers::thing;\nfn user() {}\n", 1_000_000);
        repo.write_at("helpers.rs", "pub fn thing() {}\n", 1_000_000);
        repo.write_at("gone.rs", "fn ghost() {}\n", 1_000_000);
        index_repo(&repo.0, 100).unwrap();
        let first = generation_of(&repo);
        assert_eq!(first, 1, "indexing a repository for the first time");

        // Nothing changed: reconciling again is not a new generation.
        index_repo(&repo.0, 100).unwrap();
        assert_eq!(generation_of(&repo), first);

        // Offline: edit one file, delete one, add one.
        repo.write_at("a.rs", "use renamed::thing;\nfn user_two() {}\n", 2_000_000);
        std::fs::remove_file(repo.0.join("gone.rs")).unwrap();
        repo.write_at("fresh.rs", "fn brand_new() {}\n", 2_000_000);
        let stats = index_repo(&repo.0, 100).unwrap();

        assert_eq!(stats.pruned_files, 1);
        assert_eq!(
            generation_of(&repo),
            first + 1,
            "the whole reconcile is one generation"
        );
        let store = repo.store();
        assert_eq!(symbol_names(&store, "a.rs"), vec!["user_two"]);
        assert_eq!(imports_of(&store, "a.rs"), vec!["renamed/thing"]);
        assert_eq!(symbol_names(&store, "fresh.rs"), vec!["brand_new"]);
        assert!(symbol_names(&store, "gone.rs").is_empty());
        assert_eq!(count(&store, "files", "gone.rs"), 0);
    }

    #[test]
    fn a_reconcile_beyond_the_per_generation_limit_is_published_in_atomic_chunks() {
        let repo = TempRepo::new();
        for i in 0..5 {
            repo.write_at(
                &format!("f{i}.rs"),
                &format!("fn func_{i}() {{}}\n"),
                1_000_000,
            );
        }
        let stats = index_repo_in_generations(&repo.0, 100, 2).unwrap();
        assert_eq!(stats.tier1_files, 5);
        // 5 extractions at 2 per generation: 2 + 2 + 1.
        assert_eq!(generation_of(&repo), 3);
        let store = repo.store();
        for i in 0..5 {
            assert!(store.is_tier1_done(&format!("f{i}.rs")), "f{i}.rs");
        }

        // The same repository fits one generation under the real limit.
        let other = TempRepo::new();
        for i in 0..5 {
            other.write_at(
                &format!("f{i}.rs"),
                &format!("fn func_{i}() {{}}\n"),
                1_000_000,
            );
        }
        index_repo(&other.0, 100).unwrap();
        assert_eq!(generation_of(&other), 1);
    }

    #[test]
    fn lazy_extraction_publishes_one_generation_per_file_that_gained_facts() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        repo.write_at("empty.rs", "// nothing here\n", 1_000_000);
        let mut store = repo.store();
        store
            .upsert_fingerprinted(&inventory::scan(&repo.0))
            .unwrap();
        let before = store.generation().unwrap();

        assert!(ensure_tier1(&mut store, &repo.0, "a.rs").unwrap());
        assert_eq!(store.generation().unwrap(), before + 1);
        // Already done, then extracted-but-empty: neither adds facts.
        assert!(!ensure_tier1(&mut store, &repo.0, "a.rs").unwrap());
        assert!(!ensure_tier1(&mut store, &repo.0, "empty.rs").unwrap());
        assert_eq!(store.generation().unwrap(), before + 1);
    }

    #[test]
    fn a_save_that_changes_nothing_is_not_a_new_generation_and_a_real_one_is() {
        let repo = TempRepo::new();
        repo.write_at("a.rs", OLD, 1_000_000);
        let mut store = repo.store();
        assert!(reindex_file(&mut store, &repo.0, "a.rs").unwrap());
        let first = store.generation().unwrap();

        // The editor re-saved identical bytes (mtime pinned): nothing to publish.
        assert!(!reindex_file(&mut store, &repo.0, "a.rs").unwrap());
        assert_eq!(store.generation().unwrap(), first);

        repo.write_at("a.rs", NEW, 2_000_000);
        assert!(reindex_file(&mut store, &repo.0, "a.rs").unwrap());
        assert_eq!(store.generation().unwrap(), first + 1);
        assert_eq!(symbol_names(&store, "a.rs"), vec!["new_name_here"]);
    }

    #[test]
    fn a_directory_of_files_is_one_generation_and_so_is_a_burst_of_staged_paths() {
        let repo = TempRepo::new();
        for name in ["x", "y", "z"] {
            repo.write_at(&format!("pkg/{name}.rs"), OLD, 1_000_000);
        }
        let mut store = repo.store();
        assert_eq!(reindex_dir(&mut store, &repo.0, "pkg").unwrap(), 3);
        assert_eq!(store.generation().unwrap(), 1);

        // What the watcher does with a burst: stage many paths, publish once.
        repo.write_at("pkg/x.rs", NEW, 2_000_000);
        std::fs::remove_file(repo.0.join("pkg/y.rs")).unwrap();
        repo.write_at("pkg/w.rs", NEW, 2_000_000);
        let mut batch = store::IndexBatch::default();
        for path in ["pkg/x.rs", "pkg/y.rs", "pkg/w.rs", "pkg/x.rs"] {
            stage_reindex_file(&mut batch, &repo.0, path);
        }
        let published = store.publish_batch_detailed(batch).unwrap();
        assert!(published.changed);
        assert_eq!(published.generation, 2);
        assert_eq!(symbol_names(&store, "pkg/x.rs"), vec!["new_name_here"]);
        assert_eq!(symbol_names(&store, "pkg/w.rs"), vec!["new_name_here"]);
        assert_eq!(count(&store, "files", "pkg/y.rs"), 0);
    }
}
