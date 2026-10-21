//! Incremental reindex watcher. Runs on its own thread so tool calls never
//! block: on a source-file save it re-extracts just that file and invalidates
//! its cached vector; on a HEAD move (new commit) it refreshes the co-change
//! graph. Either way it flips a `dirty` flag; the server rebuilds the in-memory
//! index lazily on the next tool call (cheap — only changed files re-embed).
//!
//! Events are coalesced: a burst is staged in memory and published as ONE
//! generation (one transaction), so readers see a burst entirely or not at all.
//! If a publish fails the previous generation stays intact and the failure is
//! recorded in the health as stale; the next start reconciles the whole
//! inventory in a single generation.
//!
//! The watcher is best-effort, but never silently so: every failure is logged
//! with the path involved and recorded in a [`WatcherHealth`] the server shares,
//! so a client can be told its results may not reflect recent edits.

use engram_repo_map::inventory::is_ignored_component;
use engram_repo_map::store::{IndexBatch, Store};
use notify::{event::ModifyKind, EventKind, RecursiveMode, Watcher};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// How many individual refresh failures to spell out before summarizing.
const MAX_LISTED_FAILURES: usize = 5;

/// What a failed git-history refresh is recorded under.
const HISTORY_KEY: &str = "git history";

/// A burst of events is published once it has been quiet this long.
const DEBOUNCE: Duration = Duration::from_millis(75);
/// A burst is published at the latest after this many events, or this long
/// after its first one, however busy the stream stays.
const MAX_BATCH_EVENTS: usize = 2000;
const MAX_BATCH_WAIT: Duration = Duration::from_secs(1);

/// What is currently wrong with live reindexing, if anything.
#[derive(Debug, Default)]
pub struct WatcherHealth {
    /// The watcher never started, or died: nothing is being tracked.
    not_running: Option<String>,
    /// The OS watcher reported an error, which can mean dropped events.
    event_error: Option<String>,
    /// Paths (or `"git history"`) whose last refresh failed, with the error.
    /// Cleared per key by the next successful refresh of the same thing.
    failures: BTreeMap<String, String>,
}

pub type SharedHealth = Arc<Mutex<WatcherHealth>>;

fn lock(health: &SharedHealth) -> MutexGuard<'_, WatcherHealth> {
    health
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl WatcherHealth {
    /// Human-readable warnings for the client; empty when all is well.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(err) = &self.not_running {
            out.push(format!(
                "file watcher is not running ({err}); results may not reflect edits made since startup"
            ));
        }
        if let Some(err) = &self.event_error {
            out.push(format!(
                "file watcher reported an error ({err}); some file changes may have been missed"
            ));
        }
        for (what, err) in self.failures.iter().take(MAX_LISTED_FAILURES) {
            out.push(format!(
                "file watcher could not refresh {what} ({err}); results for it may be stale"
            ));
        }
        if self.failures.len() > MAX_LISTED_FAILURES {
            out.push(format!(
                "file watcher could not refresh {} more paths",
                self.failures.len() - MAX_LISTED_FAILURES
            ));
        }
        out
    }

    /// Record that the watcher is not (or no longer) running, and why.
    pub fn mark_not_running(&mut self, reason: &str) {
        self.not_running = Some(reason.to_owned());
    }

    fn record_failure(&mut self, what: &str, err: &str) {
        self.failures.insert(what.to_owned(), err.to_owned());
    }

    fn clear_failure(&mut self, what: &str) {
        self.failures.remove(what);
    }

    fn clear_event_error(&mut self) {
        self.event_error = None;
    }
}

/// Convenience for the server: the current warnings of a shared health.
pub fn warnings(health: &SharedHealth) -> Vec<String> {
    lock(health).warnings()
}

/// Spawn the watcher thread. The server stays fully functional if it fails
/// (just without live reindex), and the failure is recorded in `health`.
pub fn spawn(repo_root: PathBuf, dirty: Arc<AtomicBool>, health: SharedHealth) {
    let thread_health = health.clone();
    let spawned = std::thread::Builder::new()
        .name("engram-watcher".into())
        .spawn(move || run_and_record(&repo_root, &dirty, &thread_health));
    if let Err(e) = spawned {
        eprintln!("[engram] could not start the file watcher thread: {e}");
        lock(&health).mark_not_running(&format!("could not start its thread: {e}"));
    }
}

/// The watcher thread body: run to completion, and make sure that however it
/// ends (an error, a clean stop, a panic) the health says it is not running.
fn run_and_record(repo_root: &Path, dirty: &Arc<AtomicBool>, health: &SharedHealth) {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(repo_root, dirty, health)
    }));
    let reason = match outcome {
        Ok(Ok(())) => "its event stream ended".to_owned(),
        Ok(Err(e)) => e.to_string(),
        Err(_) => "it panicked".to_owned(),
    };
    eprintln!(
        "[engram] file watcher stopped for {}: {reason}",
        repo_root.display()
    );
    lock(health).mark_not_running(&reason);
}

fn run(repo_root: &Path, dirty: &Arc<AtomicBool>, health: &SharedHealth) -> notify::Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })?;
    watcher.watch(repo_root, RecursiveMode::Recursive)?;
    eprintln!("[engram] watching {} for changes", repo_root.display());

    // The watcher handle must stay alive for the lifetime of the loop.
    let mut store = Store::open(repo_root).map_err(|e| notify::Error::generic(&e.to_string()))?;
    // One batch per burst of events: a save, a `git checkout` or a build step
    // produces dozens of events within milliseconds, and each used to be its own
    // transaction (and its own generation). Wait for the burst to go quiet, then
    // stage everything it touched and publish it as ONE generation.
    while let Ok(first) = rx.recv() {
        let started = Instant::now();
        let mut burst = vec![first];
        while burst.len() < MAX_BATCH_EVENTS && started.elapsed() < MAX_BATCH_WAIT {
            match rx.recv_timeout(DEBOUNCE) {
                Ok(next) => burst.push(next),
                // Quiet for a whole debounce window, or the channel closed
                // (the next `recv` then ends the loop).
                Err(_) => break,
            }
        }
        if process_burst(burst, repo_root, &mut store, health) {
            dirty.store(true, Ordering::SeqCst);
        }
    }
    Ok(())
}

/// Handle one item from the watcher's channel. Returns whether the index
/// changed (see [`handle_events`]).
#[cfg(test)]
fn process(
    res: notify::Result<notify::Event>,
    repo_root: &Path,
    store: &mut Store,
    health: &SharedHealth,
) -> bool {
    process_burst(vec![res], repo_root, store, health)
}

/// Handle a burst of items from the watcher's channel as one batch. Returns
/// whether the index changed (see [`handle_events`]).
fn process_burst(
    items: Vec<notify::Result<notify::Event>>,
    repo_root: &Path,
    store: &mut Store,
    health: &SharedHealth,
) -> bool {
    let mut events = Vec::with_capacity(items.len());
    for res in items {
        match res {
            Ok(event) => {
                // Events are flowing again, so the error that may have dropped
                // some is no longer news worth warning about on every response.
                lock(health).clear_event_error();
                events.push(event);
            }
            Err(e) => {
                // An OS-level watcher error (queue overflow, a watched path
                // vanishing) can mean events were lost; say so rather than
                // carry on looking healthy.
                eprintln!("[engram] file watcher error: {e}");
                lock(health).event_error = Some(e.to_string());
            }
        }
    }
    handle_events(&events.iter().collect::<Vec<_>>(), repo_root, store, health)
}

/// Apply one filesystem event to the store (see [`handle_events`]).
#[cfg(test)]
fn handle_event(
    event: &notify::Event,
    repo_root: &Path,
    store: &mut Store,
    health: &SharedHealth,
) -> bool {
    handle_events(&[event], repo_root, store, health)
}

/// Apply a burst of filesystem events to the store as ONE published generation.
/// Returns whether anything changed (so the in-memory index should be rebuilt).
///
/// The events only say which paths to look at. Each distinct path is staged
/// from the disk as it is now (so several events for one path, in any order,
/// stage the same thing), the git history is staged once if HEAD moved, and
/// the whole batch is published in a single transaction: a reader sees the
/// burst entirely or not at all. A failed publish leaves the previous
/// generation intact; it is logged and recorded in `health` against every path
/// in the batch, rather than discarded.
fn handle_events(
    events: &[&notify::Event],
    repo_root: &Path,
    store: &mut Store,
    health: &SharedHealth,
) -> bool {
    let mut head_moved = false;
    // rel path -> whether any event for it said it appeared or moved in.
    let mut paths: BTreeMap<String, bool> = BTreeMap::new();
    for event in events {
        if !matches!(
            event.kind,
            EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
        ) {
            continue;
        }
        // A directory that appeared or moved in brings files nobody will report
        // individually. Plain directory modifications (a child was added) are
        // not worth a rescan; the child's own event covers it.
        let dir_arrived = matches!(
            event.kind,
            EventKind::Create(_) | EventKind::Modify(ModifyKind::Name(_))
        );
        for path in &event.paths {
            let Ok(rel) = path.strip_prefix(repo_root) else {
                continue;
            };
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            // The repository root itself (its own mtime changing, say) is not a
            // path inside the repository; treating it as one would rescan it all.
            if rel_str.is_empty() {
                continue;
            }
            if is_head_ref(&rel_str) {
                head_moved = true;
                continue;
            }
            if is_ignored(&rel_str) {
                continue;
            }
            *paths.entry(rel_str).or_insert(false) |= dir_arrived;
        }
    }

    let mut batch = IndexBatch::default();
    let mut involved: Vec<String> = Vec::new();
    if head_moved {
        engram_repo_map::stage_history(&mut batch, repo_root);
        involved.push(HISTORY_KEY.to_owned());
    }
    for (rel_str, dir_arrived) in &paths {
        if repo_root.join(rel_str).is_dir() {
            if !dir_arrived {
                continue;
            }
            engram_repo_map::stage_reindex_dir(&mut batch, repo_root, rel_str);
        } else {
            engram_repo_map::stage_reindex_file(&mut batch, repo_root, rel_str);
        }
        involved.push(rel_str.clone());
    }
    if batch.is_empty() {
        return false;
    }

    match store.publish_batch_detailed(batch) {
        Ok(published) => {
            let mut health = lock(health);
            for what in &involved {
                health.clear_failure(what);
            }
            // A moved HEAD is reported as a change even when the history came
            // out identical: the owner then rechecks the store's generation,
            // which is cheap, instead of the watcher judging it.
            head_moved || published.changed
        }
        Err(e) => {
            let mut health = lock(health);
            for what in &involved {
                eprintln!("[engram] could not publish the refresh of {what}: {e}");
                health.record_failure(what, &e.to_string());
            }
            // Nothing of the batch was applied, but let the owner look again.
            true
        }
    }
}

/// A git HEAD/ref move (new commit or checkout) — refresh co-change history.
fn is_head_ref(rel: &str) -> bool {
    rel == ".git/logs/HEAD" || rel == ".git/HEAD"
}

/// Paths the indexer never covers: build output and tool state, at any depth.
/// Uses the inventory's own rule ([`is_ignored_component`]), so the watcher can
/// never spend work on, or add rows for, anything a full scan would skip.
fn is_ignored(rel: &str) -> bool {
    let parts: Vec<&str> = rel.split('/').collect();
    let last = parts.len() - 1;
    // The final component is the file itself, not a directory; only a
    // top-level entry (the directory event for `target`) is judged by name.
    parts
        .iter()
        .enumerate()
        .any(|(i, part)| (i < last || last == 0) && is_ignored_component(part))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    struct TempRepo(PathBuf);

    impl TempRepo {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "engram-watcher-test-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create temp repo");
            TempRepo(path)
        }
    }

    impl Drop for TempRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn modify_event(path: PathBuf) -> notify::Event {
        notify::Event::new(EventKind::Modify(ModifyKind::Any)).add_path(path)
    }

    #[test]
    fn ignores_build_and_git_internals() {
        assert!(is_ignored("target/debug/foo"));
        assert!(is_ignored(".engram/engram.db"));
        assert!(is_ignored(".git/objects/ab/cd"));
        assert!(!is_ignored("src/main.rs"));
        assert!(is_head_ref(".git/logs/HEAD"));
        assert!(!is_head_ref("src/HEAD.rs"));
    }

    #[test]
    fn ignores_build_output_at_any_depth() {
        assert!(is_ignored("crates/foo/target/debug/foo"));
        assert!(is_ignored("web/node_modules/left-pad/index.js"));
        assert!(is_ignored("a/b/__pycache__/x.pyc"));
        // A file merely named like an ignored directory is still a file.
        assert!(!is_ignored("src/build"));
        assert!(!is_ignored("src/target.rs"));
        assert!(!is_ignored("src/builder/mod.rs"));
    }

    #[test]
    fn the_watcher_ignores_everything_the_inventory_ignores() {
        for dir in [
            "vendor", ".next", ".cache", "venv", ".idea", ".vscode", ".git",
        ] {
            assert!(is_ignored(&format!("{dir}/x.rs")), "{dir}");
            assert!(is_ignored(&format!("pkg/{dir}/deep/x.rs")), "{dir}");
            assert!(is_ignored(dir), "{dir} (the directory event itself)");
        }
    }

    #[test]
    fn events_in_ignored_directories_do_no_work() {
        let repo = TempRepo::new();
        std::fs::create_dir_all(repo.0.join("vendor/lib")).unwrap();
        std::fs::write(repo.0.join("vendor/lib/a.rs"), "fn a() {}\n").unwrap();
        let mut store = Store::open(&repo.0).unwrap();
        let health = SharedHealth::default();

        for event in [
            modify_event(repo.0.join("vendor/lib/a.rs")),
            notify::Event::new(EventKind::Create(notify::event::CreateKind::Folder))
                .add_path(repo.0.join("vendor")),
        ] {
            assert!(!handle_event(&event, &repo.0, &mut store, &health));
        }
        assert!(store.all_files().unwrap().is_empty());
        assert!(warnings(&health).is_empty());
    }

    #[test]
    fn an_event_for_the_repository_root_itself_is_skipped() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        let mut store = Store::open(&repo.0).unwrap();
        let health = SharedHealth::default();

        // Were the empty relative path treated as a directory that arrived,
        // this would index the whole repository.
        let created = notify::Event::new(EventKind::Create(notify::event::CreateKind::Folder))
            .add_path(repo.0.clone());
        assert!(!handle_event(&created, &repo.0, &mut store, &health));
        assert!(!handle_event(
            &modify_event(repo.0.clone()),
            &repo.0,
            &mut store,
            &health
        ));
        assert!(store.all_files().unwrap().is_empty());
        assert!(warnings(&health).is_empty());
    }

    #[test]
    fn removing_something_never_indexed_does_not_dirty_the_index() {
        let repo = TempRepo::new();
        let mut store = Store::open(&repo.0).unwrap();
        let health = SharedHealth::default();
        let removed = notify::Event::new(EventKind::Remove(notify::event::RemoveKind::File))
            .add_path(repo.0.join("never-indexed.rs"));
        assert!(!handle_event(&removed, &repo.0, &mut store, &health));
    }

    #[test]
    fn a_later_successful_event_clears_the_watcher_error() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        let mut store = Store::open(&repo.0).unwrap();
        let health = SharedHealth::default();

        assert!(!process(
            Err(notify::Error::generic("queue overflow")),
            &repo.0,
            &mut store,
            &health
        ));
        let now = warnings(&health);
        assert_eq!(now.len(), 1, "{now:?}");
        assert!(now[0].contains("queue overflow"), "{now:?}");

        assert!(process(
            Ok(modify_event(repo.0.join("a.rs"))),
            &repo.0,
            &mut store,
            &health
        ));
        assert!(warnings(&health).is_empty(), "{:?}", warnings(&health));
    }

    #[test]
    fn a_watcher_that_cannot_start_reports_it() {
        let health = SharedHealth::default();
        let dirty = Arc::new(AtomicBool::new(false));
        // A path that does not exist cannot be watched.
        let missing = std::env::temp_dir().join(format!(
            "engram-watcher-missing-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        run_and_record(&missing, &dirty, &health);

        let warnings = warnings(&health);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].starts_with("file watcher is not running ("),
            "{warnings:?}"
        );
        assert!(warnings[0].ends_with("results may not reflect edits made since startup"));
    }

    #[test]
    fn a_watcher_whose_store_cannot_open_reports_it() {
        // `.engram` as a regular file makes Store::open fail, and unlike a
        // permissions trick it fails the same way when the tests run as root.
        let repo = TempRepo::new();
        std::fs::write(repo.0.join(".engram"), "not a directory").unwrap();
        let health = SharedHealth::default();
        run_and_record(&repo.0, &Arc::new(AtomicBool::new(false)), &health);
        assert_eq!(warnings(&health).len(), 1);
    }

    #[test]
    fn a_failed_reindex_is_recorded_then_cleared_by_a_success() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        let mut store = Store::open(&repo.0).unwrap();
        let health = SharedHealth::default();
        let event = modify_event(repo.0.join("a.rs"));

        // Break the store so the reindex cannot succeed.
        store
            .conn
            .execute_batch("ALTER TABLE files RENAME TO files_x;")
            .unwrap();
        assert!(handle_event(&event, &repo.0, &mut store, &health));
        let warnings_now = warnings(&health);
        assert_eq!(warnings_now.len(), 1, "{warnings_now:?}");
        assert!(warnings_now[0].contains("a.rs"), "{warnings_now:?}");
        assert!(warnings_now[0].contains("may be stale"), "{warnings_now:?}");

        // Fix it; the next successful reindex of the same path clears the note.
        store
            .conn
            .execute_batch("ALTER TABLE files_x RENAME TO files;")
            .unwrap();
        assert!(handle_event(&event, &repo.0, &mut store, &health));
        assert!(warnings(&health).is_empty(), "{:?}", warnings(&health));
    }

    #[test]
    fn a_healthy_watcher_has_no_warnings() {
        assert!(WatcherHealth::default().warnings().is_empty());
    }

    #[test]
    fn many_failures_are_summarized() {
        let mut health = WatcherHealth::default();
        for i in 0..8 {
            health.record_failure(&format!("f{i}.rs"), "boom");
        }
        let w = health.warnings();
        assert_eq!(w.len(), MAX_LISTED_FAILURES + 1, "{w:?}");
        assert!(w.last().unwrap().contains("3 more"), "{w:?}");
    }

    #[test]
    fn a_removal_event_deletes_the_file_from_the_store() {
        let repo = TempRepo::new();
        std::fs::write(repo.0.join("a.rs"), "fn a() {}\n").unwrap();
        let mut store = Store::open(&repo.0).unwrap();
        let health = SharedHealth::default();
        let path = repo.0.join("a.rs");
        assert!(handle_event(
            &modify_event(path.clone()),
            &repo.0,
            &mut store,
            &health
        ));
        assert_eq!(store.all_files().unwrap().len(), 1);

        std::fs::remove_file(&path).unwrap();
        let removed =
            notify::Event::new(EventKind::Remove(notify::event::RemoveKind::File)).add_path(path);
        assert!(handle_event(&removed, &repo.0, &mut store, &health));
        assert!(store.all_files().unwrap().is_empty());
        assert!(store.symbols_matching("a", 5).unwrap().is_empty());
    }
}
