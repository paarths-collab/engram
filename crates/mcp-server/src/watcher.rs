//! Incremental reindex watcher. Runs on its own thread so tool calls never
//! block: on a source-file save it re-extracts just that file and invalidates
//! its cached vector; on a HEAD move (new commit) it refreshes the co-change
//! graph. Either way it flips a `dirty` flag; the server rebuilds the in-memory
//! index lazily on the next tool call (cheap — only changed files re-embed).

use engram_repo_map::store::Store;
use notify::{EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const IGNORED: &[&str] = &[
    ".engram",
    "target",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "__pycache__",
];

/// Spawn the watcher thread. Best-effort: on error it logs and exits, leaving
/// the server fully functional (just without live reindex).
pub fn spawn(repo_root: PathBuf, dirty: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        if let Err(e) = run(&repo_root, &dirty) {
            eprintln!("[engram] file watcher disabled: {e}");
        }
    });
}

fn run(repo_root: &Path, dirty: &Arc<AtomicBool>) -> notify::Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })?;
    watcher.watch(repo_root, RecursiveMode::Recursive)?;
    eprintln!("[engram] watching {} for changes", repo_root.display());

    // The watcher handle must stay alive for the lifetime of the loop.
    let mut store = Store::open(repo_root).map_err(|e| notify::Error::generic(&e.to_string()))?;
    for res in rx {
        let Ok(event) = res else { continue };
        if !matches!(
            event.kind,
            EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
        ) {
            continue;
        }
        let mut changed = false;
        for path in &event.paths {
            let Ok(rel) = path.strip_prefix(repo_root) else {
                continue;
            };
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            if is_head_ref(&rel_str) {
                let _ = engram_repo_map::refresh_history(&mut store, repo_root);
                changed = true;
                continue;
            }
            if is_ignored(&rel_str) {
                continue;
            }
            if path.is_dir() {
                continue;
            }
            let _ = engram_repo_map::reindex_file(&mut store, repo_root, &rel_str);
            changed = true;
        }
        if changed {
            dirty.store(true, Ordering::SeqCst);
        }
    }
    Ok(())
}

/// A git HEAD/ref move (new commit or checkout) — refresh co-change history.
fn is_head_ref(rel: &str) -> bool {
    rel == ".git/logs/HEAD" || rel == ".git/HEAD"
}

fn is_ignored(rel: &str) -> bool {
    rel.split('/')
        .next()
        .is_some_and(|top| IGNORED.contains(&top))
        || rel.starts_with(".git/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_build_and_git_internals() {
        assert!(is_ignored("target/debug/foo"));
        assert!(is_ignored(".engram/engram.db"));
        assert!(is_ignored(".git/objects/ab/cd"));
        assert!(!is_ignored("src/main.rs"));
        assert!(is_head_ref(".git/logs/HEAD"));
        assert!(!is_head_ref("src/HEAD.rs"));
    }
}
