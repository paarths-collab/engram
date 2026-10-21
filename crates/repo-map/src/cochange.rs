//! Historical signals built from a single `git log --name-only` walk:
//!   - co-change graph: which files change together (asymmetric edge strength)
//!   - file recency: each file's most-recent commit timestamp
//!
//! Both come from the same pass so we never shell out per file.

use engram_domain::CoChange;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

pub const MAX_COMMITS: usize = 2000;
pub const MAX_FILES_PER_COMMIT: usize = 30; // skip mega-commits (renames, vendoring)
pub const MIN_PAIR_COUNT: u32 = 2;

/// Output of the history walk.
pub struct History {
    pub edges: Vec<CoChange>,
    /// path -> most-recent commit unix timestamp.
    pub last_commit: HashMap<String, i64>,
    /// What the walk covered, so a co-change count can be read against how
    /// many commits had a chance to produce it. `None` when `git log` could
    /// not be run or failed (not a repository, no commits yet, git missing):
    /// the walk then covered nothing, and "nothing" must not be recorded as a
    /// count of zero. The edges and recency are empty in that case.
    pub facts: Option<CoChangeFacts>,
}

impl History {
    /// Replace the store's co-change rows with this walk. A walk that failed
    /// (`facts` is `None`) stores its empty edge list through the plain
    /// replace, so the completeness facts read back as unknown rather than as
    /// "0 commits scanned".
    pub fn record_into(&self, store: &mut crate::store::Store) -> anyhow::Result<()> {
        match &self.facts {
            Some(facts) => store.replace_cochange_with_facts(&self.edges, facts),
            None => store.replace_cochange(&self.edges),
        }
    }
}

/// Completeness facts of one history walk.
///
/// A pair's `strength` is `count / file_commits[path_a]`, so `file_commits` is
/// the exact denominator, and the two totals say how much of the window the
/// table could not see.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoChangeFacts {
    /// Commits the walk covered, including ones listing no files (merges).
    pub commits_scanned: u32,
    /// Commits touching more than [`MAX_FILES_PER_COMMIT`] files; they feed
    /// neither `count` nor `file_commits`.
    pub bulk_commits_excluded: u32,
    /// path -> non-bulk commits in the window that touched it.
    pub file_commits: HashMap<String, u32>,
}

/// Parsed log: per-commit `(unix_timestamp, changed_files)` for commits that
/// list files, plus the number of commits walked (merges list none).
struct Records {
    commits: Vec<(i64, Vec<String>)>,
    walked: u32,
    /// The log ran and exited successfully. An empty window in a healthy
    /// repository is still `ok`.
    ok: bool,
}

/// Parse recent git history into per-commit `(unix_timestamp, changed_files)`,
/// skipping the `skip` newest commits first (used to build a leakage-free
/// history that cannot see a held-out evaluation window — see
/// [`build_from`]).
fn commit_records_from(repo_root: &Path, skip: usize, max_commits: usize) -> Records {
    let output = Command::new("git")
        // Settings from the user's own configuration must not change what is
        // parsed below: a signature check would add lines to the stream, and
        // path quoting would turn non-ASCII paths into escaped strings that
        // match no inventory path. (The config files themselves stay in
        // effect: ignoring the global one would also drop `safe.directory`,
        // and with it any repository owned by another user.)
        .args([
            "-c",
            "log.showSignature=false",
            "-c",
            "core.quotePath=false",
            "log",
            "--no-color",
            "--encoding=UTF-8",
            &format!("--skip={skip}"),
            &format!("--max-count={max_commits}"),
            "--name-only",
            // %ct = committer date, unix timestamp — tacked onto the marker line.
            "--pretty=format:@@COMMIT@@%ct",
        ])
        .current_dir(repo_root)
        .output();
    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => {
            return Records {
                commits: Vec::new(),
                walked: 0,
                ok: false,
            }
        }
    };
    parse_log(&String::from_utf8_lossy(&output.stdout))
}

/// Parse the `--name-only` / `@@COMMIT@@%ct` text of a successful log.
fn parse_log(text: &str) -> Records {
    let mut records = Vec::new();
    let mut walked: u32 = 0;
    let mut cur_ts: i64 = 0;
    let mut current: Vec<String> = Vec::new();
    for line in text.lines() {
        if let Some(ts) = line.strip_prefix("@@COMMIT@@") {
            walked = walked.saturating_add(1);
            if !current.is_empty() {
                records.push((cur_ts, std::mem::take(&mut current)));
            }
            cur_ts = ts.trim().parse().unwrap_or(0);
        } else if !line.trim().is_empty() {
            current.push(line.trim().to_string());
        }
    }
    if !current.is_empty() {
        records.push((cur_ts, current));
    }
    Records {
        commits: records,
        walked,
        ok: true,
    }
}

/// Build the co-change edge list and per-file recency from git history.
pub fn build(repo_root: &Path) -> History {
    build_from(repo_root, 0, MAX_COMMITS)
}

/// Build history from commits `skip..skip+max_commits` back from HEAD (newest
/// first). Used directly by benchmarks that need a leakage-free graph: skip
/// the held-out evaluation window so the graph cannot see the future.
pub fn build_from(repo_root: &Path, skip: usize, max_commits: usize) -> History {
    build_from_records(commit_records_from(repo_root, skip, max_commits))
}

fn build_from_records(records: Records) -> History {
    let Records {
        commits: records,
        walked,
        ok,
    } = records;
    let mut bulk_commits_excluded: u32 = 0;
    let mut file_commit_count: HashMap<String, u32> = HashMap::new();
    let mut pair_count: HashMap<(String, String), u32> = HashMap::new();
    let mut last_commit: HashMap<String, i64> = HashMap::new();

    for (ts, set) in &records {
        // Recency covers every commit (git log is newest-first, so the first
        // sighting of a file is its most-recent commit). Done before the
        // mega-commit skip so a file only touched in a big commit still dates.
        for f in set {
            last_commit.entry(f.clone()).or_insert(*ts);
        }
        if set.len() > MAX_FILES_PER_COMMIT {
            bulk_commits_excluded += 1;
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
    History {
        edges,
        last_commit,
        facts: ok.then_some(CoChangeFacts {
            commits_scanned: walked,
            bulk_commits_excluded,
            file_commits: file_commit_count,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// 2020-01-01T00:00:00Z; commit `n` is dated `n` days later.
    const DAY0: i64 = 1_577_836_800;
    const DAY: i64 = 86_400;

    /// Throwaway directory, removed on drop. Not `tempfile`: CI builds
    /// `--locked`, so a dev-dependency would need a Cargo.lock regeneration.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "engram-cochange-test-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create temp dir");
            TempDir(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Run the VCS against the fixture with nothing ambient: a fixed identity
    /// and date, and no user or system configuration to change its behavior.
    fn run(dir: &Path, day: i64, args: &[&str]) {
        let date = format!("{} +0000", DAY0 + day * DAY);
        let out = Command::new("git")
            .args([
                "-c",
                "commit.gpgsign=false",
                "-c",
                "log.showSignature=false",
                "-c",
                "init.defaultBranch=main",
            ])
            .args(args)
            .current_dir(dir)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .env("GIT_AUTHOR_NAME", "Anonymous")
            .env("GIT_AUTHOR_EMAIL", "anonymous@example.invalid")
            .env("GIT_COMMITTER_NAME", "Anonymous")
            .env("GIT_COMMITTER_EMAIL", "anonymous@example.invalid")
            .env("GIT_AUTHOR_DATE", &date)
            .env("GIT_COMMITTER_DATE", &date)
            .output()
            .expect("git must be installed to run these tests");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// Commit number `day` (0-based), rewriting every file in `names`.
    fn commit_files(dir: &Path, day: i64, names: &[String]) {
        for name in names {
            std::fs::write(dir.join(name), format!("day {day}\n")).unwrap();
        }
        run(dir, day, &["add", "--all"]);
        run(dir, day, &["commit", "-q", "-m", &format!("day {day}")]);
    }

    fn names(prefix: &str, n: usize) -> Vec<String> {
        (0..n).map(|i| format!("{prefix}{i:02}.rs")).collect()
    }

    /// Four commits, oldest first: two touching exactly MAX_FILES_PER_COMMIT
    /// files (`a00..`), then two touching one more than that (`b00..`).
    fn boundary_repo() -> TempDir {
        let repo = TempDir::new();
        run(&repo.0, 0, &["init", "-q"]);
        let at_cap = names("a", MAX_FILES_PER_COMMIT);
        let over_cap = names("b", MAX_FILES_PER_COMMIT + 1);
        commit_files(&repo.0, 0, &at_cap);
        commit_files(&repo.0, 1, &at_cap);
        commit_files(&repo.0, 2, &over_cap);
        commit_files(&repo.0, 3, &over_cap);
        repo
    }

    #[test]
    fn a_commit_of_exactly_the_cap_is_included_and_one_over_is_excluded() {
        let repo = boundary_repo();
        let history = build(&repo.0);
        let facts = history.facts.expect("the walk succeeded");

        assert_eq!(facts.commits_scanned, 4);
        assert_eq!(facts.bulk_commits_excluded, 2, "the two 31-file commits");
        // At the cap: counted, and every pair among its files is an edge.
        assert_eq!(facts.file_commits.get("a00.rs"), Some(&2));
        assert_eq!(facts.file_commits.len(), MAX_FILES_PER_COMMIT);
        let n = MAX_FILES_PER_COMMIT;
        assert_eq!(history.edges.len(), n * (n - 1));
        assert!(history
            .edges
            .iter()
            .all(|e| e.count == 2 && e.strength == 1.0 && e.path_a.starts_with('a')));
        // Over the cap: neither a denominator nor a pair...
        assert!(!facts.file_commits.contains_key("b00.rs"));
        // ...but still dated, newest commit first.
        assert_eq!(history.last_commit["b00.rs"], DAY0 + 3 * DAY);
        assert_eq!(history.last_commit["a00.rs"], DAY0 + DAY);
    }

    #[test]
    fn the_history_window_bounds_which_commits_are_walked() {
        let repo = boundary_repo();

        // Newest two only: both bulk, so nothing is counted and no edge exists.
        let newest = build_from(&repo.0, 0, 2);
        let facts = newest.facts.expect("ok");
        assert_eq!((facts.commits_scanned, facts.bulk_commits_excluded), (2, 2));
        assert!(facts.file_commits.is_empty());
        assert!(newest.edges.is_empty());

        // Skip them: the two at-cap commits remain.
        let older = build_from(&repo.0, 2, 10);
        let facts = older.facts.expect("ok");
        assert_eq!((facts.commits_scanned, facts.bulk_commits_excluded), (2, 0));
        assert_eq!(facts.file_commits.get("a00.rs"), Some(&2));
        assert!(!older.last_commit.contains_key("b00.rs"));

        // A window straddling the boundary: one bulk, one at the cap. A single
        // shared commit is below MIN_PAIR_COUNT, so there is no edge to show.
        let straddle = build_from(&repo.0, 1, 2);
        let facts = straddle.facts.expect("ok");
        assert_eq!((facts.commits_scanned, facts.bulk_commits_excluded), (2, 1));
        assert_eq!(facts.file_commits.get("a00.rs"), Some(&1));
        assert!(straddle.edges.is_empty());

        // A window past the end of history is a successful, empty walk.
        let past = build_from(&repo.0, 50, 5);
        assert_eq!(past.facts.expect("ok").commits_scanned, 0);
    }

    #[test]
    fn a_non_ascii_path_comes_back_unquoted_so_it_can_match_the_inventory() {
        let repo = TempDir::new();
        run(&repo.0, 0, &["init", "-q"]);
        let unicode = vec!["caf\u{e9}.rs".to_string(), "plain.rs".to_string()];
        commit_files(&repo.0, 0, &unicode);
        commit_files(&repo.0, 1, &unicode);
        let history = build(&repo.0);
        let facts = history.facts.expect("ok");
        assert_eq!(facts.file_commits.get("caf\u{e9}.rs"), Some(&2));
        assert!(history.edges.iter().any(|e| e.path_a == "caf\u{e9}.rs"));
    }

    #[test]
    fn the_boundary_holds_on_synthetic_log_text_too() {
        let commit = |ts: i64, n: usize| {
            let files: Vec<String> = (0..n).map(|i| format!("f{i}.rs")).collect();
            format!("@@COMMIT@@{ts}\n{}\n\n", files.join("\n"))
        };
        let text = format!(
            "{}{}{}@@COMMIT@@200\n\n",
            commit(400, MAX_FILES_PER_COMMIT + 1),
            commit(300, MAX_FILES_PER_COMMIT),
            commit(100, MAX_FILES_PER_COMMIT),
        );
        let history = build_from_records(parse_log(&text));
        let facts = history.facts.expect("ok");
        // The empty commit (a merge) is walked but lists no files.
        assert_eq!(facts.commits_scanned, 4);
        assert_eq!(facts.bulk_commits_excluded, 1);
        assert_eq!(facts.file_commits["f0.rs"], 2);
        let only_in_bulk = format!("f{MAX_FILES_PER_COMMIT}.rs");
        assert!(!facts.file_commits.contains_key(&only_in_bulk));
        // The file only the bulk commit lists is still dated by it.
        assert_eq!(history.last_commit[&only_in_bulk], 400);
        assert_eq!(history.last_commit["f0.rs"], 400);
    }

    #[test]
    fn a_failed_log_has_unknown_facts_and_no_rows() {
        let failed = build_from_records(Records {
            commits: Vec::new(),
            walked: 0,
            ok: false,
        });
        assert!(failed.facts.is_none());
        assert!(failed.edges.is_empty() && failed.last_commit.is_empty());

        // A directory that is not a repository fails for real.
        let dir = TempDir::new();
        let history = build(&dir.0);
        assert!(history.facts.is_none(), "not a repo is unknown, not zero");
        assert!(history.edges.is_empty());

        // A repository with no commits yet also fails to log.
        run(&dir.0, 0, &["init", "-q"]);
        assert!(build(&dir.0).facts.is_none());
    }
}
