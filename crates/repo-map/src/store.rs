//! SQLite persistence for the repo map. Single-file DB in .engram/engram.db.
//! Tracks which files are extracted to Tier-1 so retrieval can lazily extract on miss.

use anyhow::Result;
use engram_domain::{
    CoChange, FileRecord, IngestedComment, Language, PrCommit, PrFileChange, ReviewComment,
    SymbolKind, SymbolRecord,
};
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::Path;

/// Identity of one embedded chunk: the file, and the one-based line its span
/// starts at. `0` is reserved for the whole-file fallback used when a file has
/// no extracted symbols.
pub type ChunkKey = (String, usize);

/// Cumulative facts about what the persisted repository index contains.
///
/// These are totals in the database, not counts from only the most recent
/// indexing pass. Callers can therefore report coverage honestly after a warm
/// restart where no files needed to be re-extracted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoreCoverage {
    pub files: usize,
    /// Inventoried files intentionally excluded from indexing (currently
    /// oversized source or source-adjacent files).
    pub ineligible_files: usize,
    pub parser_supported_files: usize,
    /// Eligible source files whose language currently has no Tier-1 parser.
    pub unsupported_source_files: usize,
    pub tier1_done_files: usize,
    pub symbols: usize,
    pub pull_requests: usize,
    pub review_comments: usize,
}

pub struct Store {
    pub conn: Connection,
}

impl Store {
    pub fn open(repo_root: &Path) -> Result<Self> {
        let dir = repo_root.join(".engram");
        std::fs::create_dir_all(&dir)?;
        let conn = Connection::open(dir.join("engram.db"))?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA busy_timeout = 5000;
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                content_hash TEXT NOT NULL DEFAULT '',
                indexing_ineligibility TEXT,
                is_test INTEGER NOT NULL,
                tier1_done INTEGER NOT NULL DEFAULT 0,
                last_commit_ts INTEGER
            );
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL DEFAULT 0,
                signature TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);
            CREATE TABLE IF NOT EXISTS cochange (
                path_a TEXT NOT NULL,
                path_b TEXT NOT NULL,
                count INTEGER NOT NULL,
                strength REAL NOT NULL,
                PRIMARY KEY (path_a, path_b)
            );
            CREATE TABLE IF NOT EXISTS file_imports (
                path TEXT NOT NULL,
                target TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_file_imports_target ON file_imports(target);
            CREATE INDEX IF NOT EXISTS idx_file_imports_path ON file_imports(path);
            CREATE TABLE IF NOT EXISTS vectors (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                data BLOB NOT NULL
            );
            -- One embedding per symbol span, so retrieval can match a
            -- definition rather than the head of the file containing it.
            -- start_line 0 is the whole-file fallback for files with no
            -- extracted symbols; real symbols are one-based.
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                hash TEXT NOT NULL,
                data BLOB NOT NULL,
                PRIMARY KEY (path, start_line)
            );
            CREATE INDEX IF NOT EXISTS idx_chunk_vectors_path ON chunk_vectors(path);
            CREATE TABLE IF NOT EXISTS pull_requests (
                number INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                merged INTEGER NOT NULL,
                author TEXT NOT NULL,
                base_sha TEXT NOT NULL DEFAULT '',
                head_sha TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                merged_at TEXT NOT NULL DEFAULT ''
            );
            -- `patch` holds the unified-diff hunks. A PR without it is just a
            -- list of filenames: no before/after, so no correction can be
            -- reconstructed from it.
            CREATE TABLE IF NOT EXISTS pr_files (
                pr_number INTEGER NOT NULL,
                path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT '',
                additions INTEGER NOT NULL DEFAULT 0,
                deletions INTEGER NOT NULL DEFAULT 0,
                patch TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (pr_number, path)
            );
            CREATE INDEX IF NOT EXISTS idx_pr_files_path ON pr_files(path);
            -- Commits in PR order. Ordering against review_comments.created_at
            -- is what attributes a push to the review comment that prompted it.
            CREATE TABLE IF NOT EXISTS pr_commits (
                pr_number INTEGER NOT NULL,
                sha TEXT NOT NULL,
                message TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT '',
                authored_at TEXT NOT NULL DEFAULT '',
                ordinal INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (pr_number, sha)
            );
            CREATE INDEX IF NOT EXISTS idx_pr_commits_pr ON pr_commits(pr_number);
            CREATE TABLE IF NOT EXISTS review_comments (
                id INTEGER PRIMARY KEY,
                pr_number INTEGER NOT NULL,
                path TEXT NOT NULL,
                line INTEGER,
                body TEXT NOT NULL,
                author TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT '',
                diff_hunk TEXT NOT NULL DEFAULT '',
                commit_id TEXT NOT NULL DEFAULT '',
                in_reply_to INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_review_comments_path ON review_comments(path);
            CREATE INDEX IF NOT EXISTS idx_review_comments_pr ON review_comments(pr_number);
            "#,
        )?;
        Self::migrate(&conn);
        Ok(Store { conn })
    }

    /// Bring a database created by an older Engram up to the current schema.
    ///
    /// Every statement here is additive and idempotent: `ALTER TABLE ... ADD
    /// COLUMN` fails with "duplicate column" on a DB that already has it, which
    /// is why the results are deliberately discarded. `CREATE TABLE IF NOT
    /// EXISTS` in [`Store::open`] covers new tables, so only columns need this.
    fn migrate(conn: &Connection) {
        // Predates recency tracking.
        let _ = conn.execute("ALTER TABLE files ADD COLUMN last_commit_ts INTEGER", []);

        // A path and byte count cannot detect same-size edits. Databases from
        // before content hashing have no trustworthy relationship between the
        // inventory rows and any derived symbols/imports/vectors, so clear all
        // derived caches once and let the next index pass rebuild them.
        let _ = conn.execute(
            "ALTER TABLE files ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''",
            [],
        );
        let has_untrusted_derived_data = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM files WHERE content_hash = '')",
                [],
                |row| row.get::<_, bool>(0),
            )
            .unwrap_or(false);
        if has_untrusted_derived_data {
            let _ = conn.execute("DELETE FROM symbols", []);
            let _ = conn.execute("DELETE FROM file_imports", []);
            let _ = conn.execute("DELETE FROM vectors", []);
            let _ = conn.execute("DELETE FROM chunk_vectors", []);
            let _ = conn.execute("UPDATE files SET tier1_done = 0", []);
        }
        let _ = conn.execute(
            "ALTER TABLE files ADD COLUMN indexing_ineligibility TEXT",
            [],
        );

        // Symbols predating span tracking have no end_line, and a stored zero
        // is indistinguishable from a real span. The ALTER succeeds exactly
        // once — on a DB that predates the column — so use that as the signal
        // to drop the spanless rows and let every file re-extract.
        if conn
            .execute(
                "ALTER TABLE symbols ADD COLUMN end_line INTEGER NOT NULL DEFAULT 0",
                [],
            )
            .is_ok()
        {
            let _ = conn.execute("DELETE FROM symbols", []);
            let _ = conn.execute("UPDATE files SET tier1_done = 0", []);
        }

        // PR ingestion depth: diffs, SHAs, and timestamps. Rows ingested before
        // these existed keep their filenames but carry empty patches, so a
        // re-ingest is needed to mine corrections from an older database.
        for stmt in [
            "ALTER TABLE pull_requests ADD COLUMN base_sha TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE pull_requests ADD COLUMN head_sha TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE pull_requests ADD COLUMN created_at TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE pull_requests ADD COLUMN merged_at TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE pr_files ADD COLUMN status TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE pr_files ADD COLUMN additions INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE pr_files ADD COLUMN deletions INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE pr_files ADD COLUMN patch TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE review_comments ADD COLUMN created_at TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE review_comments ADD COLUMN diff_hunk TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE review_comments ADD COLUMN commit_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE review_comments ADD COLUMN in_reply_to INTEGER",
        ] {
            let _ = conn.execute(stmt, []);
        }
    }

    /// Store the last-commit unix timestamp for files touched in git history.
    /// Paths not present in the files table are silently skipped.
    pub fn update_recency(&mut self, recency: &HashMap<String, i64>) -> Result<()> {
        let tx = self.conn.transaction()?;
        for (path, ts) in recency {
            tx.execute(
                "UPDATE files SET last_commit_ts = ?2 WHERE path = ?1",
                params![path, ts],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Map of path -> last-commit unix timestamp for files that have one.
    pub fn recency_map(&self) -> Result<HashMap<String, i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, last_commit_ts FROM files WHERE last_commit_ts IS NOT NULL")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Load every cached embedding as `path -> (content_hash, raw_bytes)`.
    /// Retrieval decodes the bytes back into `f32` vectors.
    pub fn load_vectors(&self) -> Result<HashMap<String, (String, Vec<u8>)>> {
        let mut stmt = self.conn.prepare("SELECT path, hash, data FROM vectors")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                (r.get::<_, String>(1)?, r.get::<_, Vec<u8>>(2)?),
            ))
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Upsert embeddings: `(path, content_hash, raw_bytes)`.
    pub fn upsert_vectors(&mut self, rows: &[(String, String, Vec<u8>)]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for (path, hash, data) in rows {
            tx.execute(
                "INSERT INTO vectors (path, hash, data) VALUES (?1, ?2, ?3)
                 ON CONFLICT(path) DO UPDATE SET hash=?2, data=?3",
                params![path, hash, data],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Invalidate one file's cached embeddings so they re-embed on the next
    /// build. Clears the chunk vectors too: the file's symbols have moved, so
    /// every span keyed to an old line number is stale.
    pub fn invalidate_vector(&mut self, path: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM vectors WHERE path = ?1", params![path])?;
        self.conn
            .execute("DELETE FROM chunk_vectors WHERE path = ?1", params![path])?;
        Ok(())
    }

    /// Load every cached chunk embedding as `(path, start_line) -> (hash, bytes)`.
    pub fn load_chunk_vectors(&self) -> Result<HashMap<ChunkKey, (String, Vec<u8>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, start_line, hash, data FROM chunk_vectors")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                (r.get::<_, String>(0)?, r.get::<_, i64>(1)? as usize),
                (r.get::<_, String>(2)?, r.get::<_, Vec<u8>>(3)?),
            ))
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Upsert chunk embeddings: `(path, start_line, content_hash, raw_bytes)`.
    pub fn upsert_chunk_vectors(
        &mut self,
        rows: &[(String, usize, String, Vec<u8>)],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        for (path, start_line, hash, data) in rows {
            tx.execute(
                "INSERT INTO chunk_vectors (path, start_line, hash, data) VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(path, start_line) DO UPDATE SET hash=?3, data=?4",
                params![path, *start_line as i64, hash, data],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Drop cached chunk embeddings whose span no longer exists. Editing a file
    /// moves its symbols, so stale spans accumulate faster than stale files do.
    pub fn prune_chunk_vectors(
        &mut self,
        keep: &std::collections::HashSet<ChunkKey>,
    ) -> Result<()> {
        let existing: Vec<ChunkKey> = {
            let mut stmt = self
                .conn
                .prepare("SELECT path, start_line FROM chunk_vectors")?;
            let rows = stmt.query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as usize))
            })?;
            rows.filter_map(Result::ok).collect()
        };
        let tx = self.conn.transaction()?;
        for key in existing {
            if !keep.contains(&key) {
                tx.execute(
                    "DELETE FROM chunk_vectors WHERE path = ?1 AND start_line = ?2",
                    params![key.0, key.1 as i64],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Drop cached embeddings for files no longer present in the repo.
    pub fn prune_vectors(&mut self, keep: &std::collections::HashSet<String>) -> Result<()> {
        let existing: Vec<String> = {
            let mut stmt = self.conn.prepare("SELECT path FROM vectors")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.filter_map(Result::ok).collect()
        };
        let tx = self.conn.transaction()?;
        for path in existing {
            if !keep.contains(&path) {
                tx.execute("DELETE FROM vectors WHERE path = ?1", params![path])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Drop every trace of files no longer present in the repository.
    ///
    /// `upsert_files` only inserts and updates, so without this a file deleted
    /// from the repo keeps its `files` row, its extracted symbols, and its
    /// import edges forever. Those ghosts still match `symbols_matching`, still
    /// resolve as import targets in the code graph, and still rank in
    /// retrieval — evidence citing a path that is not there any more.
    ///
    /// Returns the number of paths removed.
    pub fn prune_missing_files(
        &mut self,
        keep: &std::collections::HashSet<String>,
    ) -> Result<usize> {
        let stale: Vec<String> = {
            let mut stmt = self.conn.prepare("SELECT path FROM files")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.filter_map(Result::ok)
                .filter(|path| !keep.contains(path))
                .collect()
        };
        if stale.is_empty() {
            return Ok(0);
        }
        let tx = self.conn.transaction()?;
        for path in &stale {
            tx.execute("DELETE FROM files WHERE path = ?1", params![path])?;
            tx.execute("DELETE FROM symbols WHERE path = ?1", params![path])?;
            tx.execute("DELETE FROM file_imports WHERE path = ?1", params![path])?;
            tx.execute("DELETE FROM vectors WHERE path = ?1", params![path])?;
            tx.execute("DELETE FROM chunk_vectors WHERE path = ?1", params![path])?;
        }
        tx.commit()?;
        Ok(stale.len())
    }

    /// Upsert a pull request row.
    ///
    /// `base_sha`/`head_sha` bound the PR's diff; `created_at`/`merged_at`
    /// bound it in time. Both are needed to reconstruct the state a reviewer
    /// was looking at.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_pull_request(
        &mut self,
        number: i64,
        title: &str,
        body: &str,
        merged: bool,
        author: &str,
        base_sha: &str,
        head_sha: &str,
        created_at: &str,
        merged_at: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO pull_requests
                 (number, title, body, merged, author, base_sha, head_sha, created_at, merged_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(number) DO UPDATE SET title=?2, body=?3, merged=?4, author=?5,
                 base_sha=?6, head_sha=?7, created_at=?8, merged_at=?9",
            params![
                number,
                title,
                body,
                merged as i64,
                author,
                base_sha,
                head_sha,
                created_at,
                merged_at
            ],
        )?;
        Ok(())
    }

    /// Replace the changed files recorded for a PR, including their diffs.
    pub fn replace_pr_files(&mut self, pr_number: i64, files: &[PrFileChange]) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM pr_files WHERE pr_number = ?1",
            params![pr_number],
        )?;
        for f in files {
            tx.execute(
                "INSERT OR REPLACE INTO pr_files
                     (pr_number, path, status, additions, deletions, patch)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    pr_number,
                    f.path,
                    f.status,
                    f.additions,
                    f.deletions,
                    f.patch
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Replace the commit list recorded for a PR.
    pub fn replace_pr_commits(&mut self, pr_number: i64, commits: &[PrCommit]) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM pr_commits WHERE pr_number = ?1",
            params![pr_number],
        )?;
        for c in commits {
            tx.execute(
                "INSERT OR REPLACE INTO pr_commits
                     (pr_number, sha, message, author, authored_at, ordinal)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    pr_number,
                    c.sha,
                    c.message,
                    c.author,
                    c.authored_at,
                    c.ordinal
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Replace the review comments recorded for a PR.
    pub fn replace_review_comments_for_pr(
        &mut self,
        pr_number: i64,
        comments: &[IngestedComment],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM review_comments WHERE pr_number = ?1",
            params![pr_number],
        )?;
        for c in comments {
            tx.execute(
                "INSERT INTO review_comments
                     (pr_number, path, line, body, author, created_at, diff_hunk,
                      commit_id, in_reply_to)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    pr_number,
                    c.path,
                    c.line,
                    c.body,
                    c.author,
                    c.created_at,
                    c.diff_hunk,
                    c.commit_id,
                    c.in_reply_to
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Commits recorded for a PR, in the order the API returned them.
    pub fn pr_commits(&self, pr_number: i64) -> Result<Vec<PrCommit>> {
        let mut stmt = self.conn.prepare(
            "SELECT sha, message, author, authored_at, ordinal
             FROM pr_commits WHERE pr_number = ?1 ORDER BY ordinal",
        )?;
        let rows = stmt.query_map(params![pr_number], |r| {
            Ok(PrCommit {
                sha: r.get(0)?,
                message: r.get(1)?,
                author: r.get(2)?,
                authored_at: r.get(3)?,
                ordinal: r.get(4)?,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Changed files recorded for a PR, with their diffs.
    pub fn pr_files(&self, pr_number: i64) -> Result<Vec<PrFileChange>> {
        let mut stmt = self.conn.prepare(
            "SELECT path, status, additions, deletions, patch
             FROM pr_files WHERE pr_number = ?1 ORDER BY path",
        )?;
        let rows = stmt.query_map(params![pr_number], |r| {
            Ok(PrFileChange {
                path: r.get(0)?,
                status: r.get(1)?,
                additions: r.get(2)?,
                deletions: r.get(3)?,
                patch: r.get(4)?,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn count_pull_requests(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM pull_requests", [], |r| r.get(0))?)
    }

    pub fn count_review_comments(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM review_comments", [], |r| r.get(0))?)
    }

    /// All ingested review comments, joined with their PR's title/merged status.
    pub fn all_review_comments(&self) -> Result<Vec<ReviewComment>> {
        self.query_review_comments("1=1", params![])
    }

    /// Review comments left on a specific file path.
    pub fn review_comments_for_path(&self, path: &str) -> Result<Vec<ReviewComment>> {
        self.query_review_comments("rc.path = ?1", params![path])
    }

    fn query_review_comments(
        &self,
        where_clause: &str,
        p: &[&dyn rusqlite::ToSql],
    ) -> Result<Vec<ReviewComment>> {
        let sql = format!(
            "SELECT rc.pr_number, pr.title, pr.merged, rc.path, rc.line, rc.body, rc.author,
                    rc.created_at, rc.diff_hunk
             FROM review_comments rc JOIN pull_requests pr ON pr.number = rc.pr_number
             WHERE {where_clause}"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(p, |r| {
            let hunk: String = r.get(8)?;
            Ok(ReviewComment {
                pr_number: r.get(0)?,
                pr_title: r.get(1)?,
                pr_merged: r.get::<_, i64>(2)? == 1,
                path: r.get(3)?,
                line: r.get(4)?,
                body: r.get(5)?,
                author: r.get(6)?,
                created_at: r.get(7)?,
                diff_hunk: (!hunk.is_empty()).then_some(hunk),
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Replace the import targets recorded for a file. Targets are normalized
    /// (see [`crate::imports::normalize_target`]) before storage; empties dropped.
    pub fn replace_imports_for_file(&mut self, path: &str, targets: &[String]) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM file_imports WHERE path = ?1", params![path])?;
        for raw in targets {
            let target = crate::imports::normalize_target(raw);
            if target.is_empty() {
                continue;
            }
            tx.execute(
                "INSERT INTO file_imports (path, target) VALUES (?1, ?2)",
                params![path, target],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// All `(importer_path, normalized_target)` import rows. Used to build the
    /// in-memory code graph (see `crate::graph`).
    pub fn all_imports(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare("SELECT path, target FROM file_imports")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// All directed co-change edges `(path_a, path_b, strength)`.
    pub fn all_cochange_edges(&self) -> Result<Vec<(String, String, f32)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path_a, path_b, strength FROM cochange")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, f32>(2)?,
            ))
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Files whose import targets contain `needle` (a module key from
    /// [`crate::imports::module_needles`]), excluding `exclude`.
    pub fn importers_of(&self, needle: &str, exclude: &str, limit: usize) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT path FROM file_imports
             WHERE target LIKE ?1 AND path <> ?2 LIMIT ?3",
        )?;
        let pattern = format!("%{needle}%");
        let rows = stmt.query_map(params![pattern, exclude, limit as i64], |r| {
            r.get::<_, String>(0)
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn upsert_files(&mut self, files: &[FileRecord]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for f in files {
            // Invalidate every derived artifact before replacing an inventory
            // fingerprint. This transaction is the cold-restart equivalent of
            // the file watcher's explicit invalidation path.
            let changed = tx.execute(
                "UPDATE files SET tier1_done = 0
                 WHERE path = ?1 AND (
                   content_hash <> ?2 OR
                   COALESCE(indexing_ineligibility, '') <> COALESCE(?3, '')
                 )",
                params![f.path, f.content_hash, f.indexing_ineligibility],
            )?;
            if changed > 0 {
                tx.execute("DELETE FROM symbols WHERE path = ?1", params![f.path])?;
                tx.execute("DELETE FROM file_imports WHERE path = ?1", params![f.path])?;
                tx.execute("DELETE FROM vectors WHERE path = ?1", params![f.path])?;
                tx.execute("DELETE FROM chunk_vectors WHERE path = ?1", params![f.path])?;
            }
            tx.execute(
                "INSERT INTO files (
                    path, language, size_bytes, content_hash,
                    indexing_ineligibility, is_test
                 )
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(path) DO UPDATE SET
                   language=?2, size_bytes=?3, content_hash=?4,
                   indexing_ineligibility=?5, is_test=?6",
                params![
                    f.path,
                    f.language.as_str(),
                    f.size_bytes as i64,
                    f.content_hash,
                    f.indexing_ineligibility,
                    f.is_test as i64
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn replace_symbols_for_file(&mut self, path: &str, symbols: &[SymbolRecord]) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM symbols WHERE path = ?1", params![path])?;
        for s in symbols {
            tx.execute(
                "INSERT INTO symbols (name, kind, path, start_line, end_line, signature)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    s.name,
                    format!("{:?}", s.kind).to_lowercase(),
                    s.path,
                    s.start_line as i64,
                    s.end_line as i64,
                    s.signature
                ],
            )?;
        }
        tx.execute(
            "UPDATE files SET tier1_done = 1 WHERE path = ?1",
            params![path],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn replace_cochange(&mut self, edges: &[CoChange]) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM cochange", [])?;
        for e in edges {
            tx.execute(
                "INSERT OR REPLACE INTO cochange (path_a, path_b, count, strength)
                 VALUES (?1, ?2, ?3, ?4)",
                params![e.path_a, e.path_b, e.count as i64, e.strength],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn is_tier1_done(&self, path: &str) -> bool {
        self.conn
            .query_row(
                "SELECT tier1_done FROM files WHERE path = ?1",
                params![path],
                |r| r.get::<_, i64>(0),
            )
            .map(|v| v == 1)
            .unwrap_or(false)
    }

    /// Return cumulative coverage and ingestion counts from the persisted
    /// store. Parser support is deliberately scoped to languages for which the
    /// current Tier-1 extractor exists; unsupported source is still available
    /// to body retrieval but is not misreported as symbol-indexed.
    pub fn coverage(&self) -> Result<StoreCoverage> {
        let (
            files,
            ineligible_files,
            parser_supported_files,
            unsupported_source_files,
            tier1_done_files,
            symbols,
            pull_requests,
            reviews,
        ) = self.conn.query_row(
            "SELECT
                    (SELECT COUNT(*) FROM files),
                    (SELECT COUNT(*) FROM files
                     WHERE indexing_ineligibility IS NOT NULL),
                    (SELECT COUNT(*) FROM files
                     WHERE indexing_ineligibility IS NULL
                       AND language IN ('rust', 'python', 'typescript', 'javascript')),
                    (SELECT COUNT(*) FROM files
                     WHERE indexing_ineligibility IS NULL AND language = 'go'),
                    (SELECT COUNT(*) FROM files
                     WHERE indexing_ineligibility IS NULL
                       AND tier1_done = 1
                       AND language IN ('rust', 'python', 'typescript', 'javascript')),
                    (SELECT COUNT(*) FROM symbols),
                    (SELECT COUNT(*) FROM pull_requests),
                    (SELECT COUNT(*) FROM review_comments)",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                ))
            },
        )?;
        Ok(StoreCoverage {
            files: files as usize,
            ineligible_files: ineligible_files as usize,
            parser_supported_files: parser_supported_files as usize,
            unsupported_source_files: unsupported_source_files as usize,
            tier1_done_files: tier1_done_files as usize,
            symbols: symbols as usize,
            pull_requests: pull_requests as usize,
            review_comments: reviews as usize,
        })
    }

    pub fn all_files(&self) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT path, language, size_bytes, content_hash, is_test FROM files
             WHERE indexing_ineligibility IS NULL ORDER BY path",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(FileRecord {
                path: r.get(0)?,
                language: lang_from_str(&r.get::<_, String>(1)?),
                size_bytes: r.get::<_, i64>(2)? as u64,
                content_hash: r.get(3)?,
                indexing_ineligibility: None,
                is_test: r.get::<_, i64>(4)? == 1,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn symbols_matching(&self, needle: &str, limit: usize) -> Result<Vec<SymbolRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, kind, path, start_line, end_line, signature FROM symbols
             WHERE name LIKE ?1 COLLATE NOCASE
             ORDER BY name COLLATE NOCASE, path, start_line, end_line
             LIMIT ?2",
        )?;
        let pattern = format!("%{}%", needle);
        let rows = stmt.query_map(params![pattern, limit as i64], row_to_symbol)?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Symbols whose name equals `needle` exactly (case-insensitive).
    ///
    /// `symbols_matching` is a substring `LIKE` search, so a query naming
    /// `merge_dicts` competes with `merge_content`, `merge_configs`, and every
    /// other name containing "merge", and the one the caller actually named can
    /// lose. When a query contains a whole identifier that IS a symbol name,
    /// that is the strongest signal retrieval has, and it must not be diluted.
    pub fn symbols_exact(&self, needle: &str, limit: usize) -> Result<Vec<SymbolRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, kind, path, start_line, end_line, signature FROM symbols
             WHERE name = ?1 COLLATE NOCASE
             ORDER BY path, start_line, end_line, name COLLATE NOCASE
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![needle, limit as i64], row_to_symbol)?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Every extracted symbol for a file, in source order. Retrieval uses the
    /// spans to embed and quote definitions rather than the head of the file.
    pub fn symbols_for_path(&self, path: &str) -> Result<Vec<SymbolRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, kind, path, start_line, end_line, signature FROM symbols
             WHERE path = ?1 ORDER BY start_line, end_line, name COLLATE NOCASE",
        )?;
        let rows = stmt.query_map(params![path], row_to_symbol)?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn cochange_for(&self, path: &str, limit: usize) -> Result<Vec<CoChange>> {
        let mut stmt = self.conn.prepare(
            "SELECT path_a, path_b, count, strength FROM cochange
             WHERE path_a = ?1 ORDER BY strength DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![path, limit as i64], |r| {
            Ok(CoChange {
                path_a: r.get(0)?,
                path_b: r.get(1)?,
                count: r.get::<_, i64>(2)? as u32,
                strength: r.get(3)?,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }
}

fn lang_from_str(s: &str) -> Language {
    match s {
        "rust" => Language::Rust,
        "python" => Language::Python,
        "typescript" => Language::TypeScript,
        "javascript" => Language::JavaScript,
        "go" => Language::Go,
        _ => Language::Other,
    }
}

/// Shared row mapper for the symbol `SELECT`s above; both must project the
/// same columns in the same order.
fn row_to_symbol(r: &rusqlite::Row) -> rusqlite::Result<SymbolRecord> {
    Ok(SymbolRecord {
        name: r.get(0)?,
        kind: kind_from_str(&r.get::<_, String>(1)?),
        path: r.get(2)?,
        start_line: r.get::<_, i64>(3)? as usize,
        end_line: r.get::<_, i64>(4)? as usize,
        signature: r.get(5)?,
    })
}

fn kind_from_str(s: &str) -> SymbolKind {
    match s {
        "function" => SymbolKind::Function,
        "method" => SymbolKind::Method,
        "struct" => SymbolKind::Struct,
        "enum" => SymbolKind::Enum,
        "class" => SymbolKind::Class,
        "trait" => SymbolKind::Trait,
        "interface" => SymbolKind::Interface,
        "const" => SymbolKind::Const,
        _ => SymbolKind::Module,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// A throwaway repo root under the OS temp dir, removed on drop.
    ///
    /// Deliberately not `tempfile`: CI builds `--locked`, so adding a
    /// dev-dependency means regenerating `Cargo.lock` in the same change.
    struct TempRepo(PathBuf);

    impl TempRepo {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "engram-store-test-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            std::fs::create_dir_all(&path).expect("create temp repo");
            TempRepo(path)
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

    fn file(path: &str) -> FileRecord {
        FileRecord {
            path: path.to_owned(),
            language: Language::Rust,
            size_bytes: 10,
            content_hash: format!("hash:{path}"),
            indexing_ineligibility: None,
            is_test: false,
        }
    }

    fn symbol(path: &str, name: &str) -> SymbolRecord {
        SymbolRecord {
            name: name.to_owned(),
            kind: SymbolKind::Function,
            path: path.to_owned(),
            start_line: 1,
            end_line: 4,
            signature: format!("fn {name}()"),
        }
    }

    #[test]
    fn prune_removes_every_trace_of_a_deleted_file() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store
            .upsert_files(&[file("kept.rs"), file("gone.rs")])
            .unwrap();
        store
            .replace_symbols_for_file("kept.rs", &[symbol("kept.rs", "keeper")])
            .unwrap();
        store
            .replace_symbols_for_file("gone.rs", &[symbol("gone.rs", "ghost")])
            .unwrap();
        store
            .replace_imports_for_file("gone.rs", &["utils/retry".to_owned()])
            .unwrap();

        let keep = HashSet::from(["kept.rs".to_owned()]);
        assert_eq!(store.prune_missing_files(&keep).unwrap(), 1);

        let paths: Vec<String> = store
            .all_files()
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(paths, vec!["kept.rs".to_owned()]);
        // The deleted file must stop being citable as evidence, which means
        // its symbols and import edges have to go too, not just its row.
        assert!(store.symbols_matching("ghost", 5).unwrap().is_empty());
        assert_eq!(store.symbols_matching("keeper", 5).unwrap().len(), 1);
        assert!(store.all_imports().unwrap().is_empty());
    }

    #[test]
    fn chunk_vectors_round_trip_and_upsert_in_place() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store
            .upsert_chunk_vectors(&[
                ("a.rs".to_owned(), 1, "h1".to_owned(), vec![1u8, 2]),
                ("a.rs".to_owned(), 40, "h2".to_owned(), vec![3u8]),
            ])
            .unwrap();

        let loaded = store.load_chunk_vectors().unwrap();
        assert_eq!(loaded.len(), 2, "same path, two spans, two rows");
        assert_eq!(
            loaded.get(&("a.rs".to_owned(), 40)),
            Some(&("h2".to_owned(), vec![3u8]))
        );

        // Re-embedding the same span replaces it rather than duplicating.
        store
            .upsert_chunk_vectors(&[("a.rs".to_owned(), 1, "h1b".to_owned(), vec![9u8])])
            .unwrap();
        let loaded = store.load_chunk_vectors().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(
            loaded.get(&("a.rs".to_owned(), 1)),
            Some(&("h1b".to_owned(), vec![9u8]))
        );
    }

    #[test]
    fn prune_chunk_vectors_drops_spans_that_moved() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store
            .upsert_chunk_vectors(&[
                ("a.rs".to_owned(), 1, "h".to_owned(), vec![1u8]),
                ("a.rs".to_owned(), 40, "h".to_owned(), vec![2u8]),
                ("gone.rs".to_owned(), 1, "h".to_owned(), vec![3u8]),
            ])
            .unwrap();

        // The symbol at line 40 moved, and gone.rs was deleted.
        let keep = HashSet::from([("a.rs".to_owned(), 1)]);
        store.prune_chunk_vectors(&keep).unwrap();

        let loaded = store.load_chunk_vectors().unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key(&("a.rs".to_owned(), 1)));
    }

    #[test]
    fn invalidating_a_file_clears_its_chunk_vectors_too() {
        // A file's symbols move when it is edited, so every span keyed to an
        // old line number is stale, not just the file-level vector.
        let repo = TempRepo::new();
        let mut store = repo.store();
        store
            .upsert_vectors(&[("a.rs".to_owned(), "h".to_owned(), vec![1u8])])
            .unwrap();
        store
            .upsert_chunk_vectors(&[
                ("a.rs".to_owned(), 1, "h".to_owned(), vec![1u8]),
                ("b.rs".to_owned(), 1, "h".to_owned(), vec![2u8]),
            ])
            .unwrap();

        store.invalidate_vector("a.rs").unwrap();

        assert!(store.load_vectors().unwrap().is_empty());
        let chunks = store.load_chunk_vectors().unwrap();
        assert_eq!(chunks.len(), 1, "only a.rs chunks are cleared");
        assert!(chunks.contains_key(&("b.rs".to_owned(), 1)));
    }

    #[test]
    fn changed_inventory_hash_invalidates_all_derived_file_data() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        let original = file("a.rs");
        store.upsert_files(&[original.clone()]).unwrap();
        store
            .replace_symbols_for_file("a.rs", &[symbol("a.rs", "stale")])
            .unwrap();
        store
            .replace_imports_for_file("a.rs", &["crate/old".to_owned()])
            .unwrap();
        store
            .upsert_vectors(&[("a.rs".to_owned(), "old".to_owned(), vec![1])])
            .unwrap();
        store
            .upsert_chunk_vectors(&[("a.rs".to_owned(), 1, "old".to_owned(), vec![1])])
            .unwrap();

        let mut changed = original;
        changed.content_hash = "different-content-same-size".to_owned();
        store.upsert_files(&[changed]).unwrap();

        assert!(!store.is_tier1_done("a.rs"));
        assert!(store.symbols_for_path("a.rs").unwrap().is_empty());
        assert!(store.all_imports().unwrap().is_empty());
        assert!(store.load_vectors().unwrap().is_empty());
        assert!(store.load_chunk_vectors().unwrap().is_empty());
    }

    #[test]
    fn pruning_a_deleted_file_takes_its_chunk_vectors() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store.upsert_files(&[file("gone.rs")]).unwrap();
        store
            .upsert_chunk_vectors(&[("gone.rs".to_owned(), 1, "h".to_owned(), vec![1u8])])
            .unwrap();

        assert_eq!(store.prune_missing_files(&HashSet::new()).unwrap(), 1);
        assert!(store.load_chunk_vectors().unwrap().is_empty());
    }

    #[test]
    fn symbol_spans_survive_a_round_trip() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store.upsert_files(&[file("a.rs")]).unwrap();
        store
            .replace_symbols_for_file("a.rs", &[symbol("a.rs", "spanned")])
            .unwrap();

        let found = store.symbols_matching("spanned", 5).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].start_line, 1);
        assert_eq!(found[0].end_line, 4, "span must survive storage");
    }

    #[test]
    fn symbols_for_path_returns_them_in_source_order() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store.upsert_files(&[file("a.rs")]).unwrap();
        let mut second = symbol("a.rs", "later");
        second.start_line = 40;
        second.end_line = 60;
        // Inserted out of order on purpose; the query must sort.
        store
            .replace_symbols_for_file("a.rs", &[second, symbol("a.rs", "earlier")])
            .unwrap();

        let found = store.symbols_for_path("a.rs").unwrap();
        let names: Vec<&str> = found.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["earlier", "later"]);
        assert_eq!(found[1].start_line, 40);
        assert_eq!(found[1].end_line, 60);
    }

    #[test]
    fn symbol_lookups_have_a_deterministic_tie_order() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store.upsert_files(&[file("z.rs"), file("a.rs")]).unwrap();
        store
            .replace_symbols_for_file("z.rs", &[symbol("z.rs", "retry")])
            .unwrap();
        store
            .replace_symbols_for_file("a.rs", &[symbol("a.rs", "retry")])
            .unwrap();

        let matching: Vec<String> = store
            .symbols_matching("retry", 10)
            .unwrap()
            .into_iter()
            .map(|item| item.path)
            .collect();
        let exact: Vec<String> = store
            .symbols_exact("retry", 10)
            .unwrap()
            .into_iter()
            .map(|item| item.path)
            .collect();
        assert_eq!(matching, vec!["a.rs", "z.rs"]);
        assert_eq!(exact, vec!["a.rs", "z.rs"]);
    }

    #[test]
    fn coverage_reports_cumulative_index_and_ingestion_totals() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        let mut go_file = file("worker.go");
        go_file.language = Language::Go;
        store
            .upsert_files(&[file("a.rs"), file("b.rs"), go_file])
            .unwrap();
        store
            .replace_symbols_for_file("a.rs", &[symbol("a.rs", "alpha")])
            .unwrap();
        // A supported file with no declarations is still a completed Tier-1
        // extraction and must remain counted on later warm runs.
        store.replace_symbols_for_file("b.rs", &[]).unwrap();
        store
            .upsert_pull_request(7, "reuse", "", true, "alice", "", "", "", "")
            .unwrap();
        store
            .replace_review_comments_for_pr(
                7,
                &[IngestedComment {
                    path: "a.rs".to_owned(),
                    body: "reuse alpha".to_owned(),
                    author: "bob".to_owned(),
                    ..Default::default()
                }],
            )
            .unwrap();

        let coverage = store.coverage().unwrap();
        assert_eq!(coverage.files, 3);
        assert_eq!(coverage.ineligible_files, 0);
        assert_eq!(coverage.parser_supported_files, 2);
        assert_eq!(coverage.unsupported_source_files, 1);
        assert_eq!(coverage.tier1_done_files, 2);
        assert_eq!(coverage.symbols, 1);
        assert_eq!(coverage.pull_requests, 1);
        assert_eq!(coverage.review_comments, 1);

        drop(store);
        assert_eq!(repo.store().coverage().unwrap(), coverage);
    }

    #[test]
    fn coverage_counts_ineligible_files_but_all_files_excludes_them() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        let mut oversized = file("generated.rs");
        oversized.indexing_ineligibility = Some("file_exceeds_limit".to_owned());
        store.upsert_files(&[file("lib.rs"), oversized]).unwrap();

        let coverage = store.coverage().unwrap();
        assert_eq!(coverage.files, 2);
        assert_eq!(coverage.ineligible_files, 1);
        assert_eq!(coverage.parser_supported_files, 1);
        assert_eq!(coverage.unsupported_source_files, 0);
        assert_eq!(
            store
                .all_files()
                .unwrap()
                .into_iter()
                .map(|file| file.path)
                .collect::<Vec<_>>(),
            vec!["lib.rs"]
        );
    }

    #[test]
    fn prune_is_a_noop_when_every_file_is_still_present() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store.upsert_files(&[file("a.rs"), file("b.rs")]).unwrap();

        let keep = HashSet::from(["a.rs".to_owned(), "b.rs".to_owned()]);
        assert_eq!(store.prune_missing_files(&keep).unwrap(), 0);
        assert_eq!(store.all_files().unwrap().len(), 2);
    }

    #[test]
    fn prune_leaves_review_history_alone() {
        // Review comments are historical evidence about a path. A file being
        // deleted does not make what reviewers said about it untrue, and
        // get_review_history takes a raw path, not an indexed file.
        let repo = TempRepo::new();
        let mut store = repo.store();
        store.upsert_files(&[file("gone.rs")]).unwrap();
        store
            .upsert_pull_request(1, "old change", "", true, "alice", "", "", "", "")
            .unwrap();
        store
            .replace_review_comments_for_pr(
                1,
                &[IngestedComment {
                    path: "gone.rs".to_owned(),
                    line: Some(4),
                    body: "this needs a guard".to_owned(),
                    author: "bob".to_owned(),
                    ..Default::default()
                }],
            )
            .unwrap();

        assert_eq!(store.prune_missing_files(&HashSet::new()).unwrap(), 1);
        assert_eq!(store.count_review_comments().unwrap(), 1);
        assert_eq!(store.review_comments_for_path("gone.rs").unwrap().len(), 1);
    }

    #[test]
    fn round_trips_pr_diffs_commits_and_comment_timestamps() {
        // The correction triple needs all three to survive storage: the diff
        // (what changed), the commit order (when), and the comment timestamp
        // (what prompted it).
        let repo = TempRepo::new();
        let mut store = repo.store();
        store
            .upsert_pull_request(
                7,
                "fix billing",
                "body",
                true,
                "alice",
                "base111",
                "head222",
                "2026-05-01T08:00:00Z",
                "2026-05-02T08:00:00Z",
            )
            .unwrap();
        store
            .replace_pr_files(
                7,
                &[PrFileChange {
                    path: "src/api.rs".to_owned(),
                    status: "modified".to_owned(),
                    additions: 3,
                    deletions: 1,
                    patch: "@@ -1 +1 @@\n-a.unwrap()\n+a.context(\"a\")?".to_owned(),
                }],
            )
            .unwrap();
        store
            .replace_pr_commits(
                7,
                &[
                    PrCommit {
                        sha: "aaa".to_owned(),
                        message: "first pass".to_owned(),
                        author: "alice".to_owned(),
                        authored_at: "2026-05-01T09:00:00Z".to_owned(),
                        ordinal: 0,
                    },
                    PrCommit {
                        sha: "bbb".to_owned(),
                        message: "address review".to_owned(),
                        author: "alice".to_owned(),
                        authored_at: "2026-05-01T11:00:00Z".to_owned(),
                        ordinal: 1,
                    },
                ],
            )
            .unwrap();
        store
            .replace_review_comments_for_pr(
                7,
                &[IngestedComment {
                    path: "src/api.rs".to_owned(),
                    line: Some(1),
                    body: "wrap this in Context".to_owned(),
                    author: "bob".to_owned(),
                    created_at: "2026-05-01T10:00:00Z".to_owned(),
                    diff_hunk: "@@ -1 +1 @@\n+a.unwrap()".to_owned(),
                    commit_id: "aaa".to_owned(),
                    in_reply_to: None,
                }],
            )
            .unwrap();

        let files = store.pr_files(7).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].patch.contains("context"));
        assert_eq!(files[0].additions, 3);

        let commits = store.pr_commits(7).unwrap();
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[0].sha, "aaa", "commits come back in PR order");
        assert_eq!(commits[1].message, "address review");

        let comments = store.review_comments_for_path("src/api.rs").unwrap();
        assert_eq!(comments[0].created_at, "2026-05-01T10:00:00Z");
        assert!(comments[0].diff_hunk.as_deref().unwrap().contains("unwrap"));

        // The whole point: the comment lands between the two commits, so the
        // second commit is attributable to it.
        assert!(comments[0].created_at > commits[0].authored_at);
        assert!(comments[0].created_at < commits[1].authored_at);
    }

    #[test]
    fn content_hash_migration_invalidates_legacy_derived_data() {
        let repo = TempRepo::new();
        let db_dir = repo.0.join(".engram");
        std::fs::create_dir_all(&db_dir).unwrap();
        {
            let conn = Connection::open(db_dir.join("engram.db")).unwrap();
            conn.execute_batch(
                "CREATE TABLE files (
                    path TEXT PRIMARY KEY, language TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, is_test INTEGER NOT NULL,
                    tier1_done INTEGER NOT NULL DEFAULT 0, last_commit_ts INTEGER
                 );
                 CREATE TABLE symbols (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
                    path TEXT NOT NULL, start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL DEFAULT 0, signature TEXT NOT NULL
                 );
                 CREATE TABLE file_imports (path TEXT NOT NULL, target TEXT NOT NULL);
                 CREATE TABLE vectors (path TEXT PRIMARY KEY, hash TEXT NOT NULL, data BLOB NOT NULL);
                 CREATE TABLE chunk_vectors (
                    path TEXT NOT NULL, start_line INTEGER NOT NULL,
                    hash TEXT NOT NULL, data BLOB NOT NULL,
                    PRIMARY KEY (path, start_line)
                 );
                 INSERT INTO files VALUES ('legacy.rs', 'rust', 10, 0, 1, NULL);
                 INSERT INTO symbols VALUES (1, 'stale', 'function', 'legacy.rs', 1, 1, 'fn stale()');
                 INSERT INTO file_imports VALUES ('legacy.rs', 'crate/old');
                 INSERT INTO vectors VALUES ('legacy.rs', 'old', X'01');
                 INSERT INTO chunk_vectors VALUES ('legacy.rs', 1, 'old', X'01');",
            )
            .unwrap();
        }

        let store = Store::open(&repo.0).unwrap();
        assert!(!store.is_tier1_done("legacy.rs"));
        assert!(store.symbols_for_path("legacy.rs").unwrap().is_empty());
        assert!(store.all_imports().unwrap().is_empty());
        assert!(store.load_vectors().unwrap().is_empty());
        assert!(store.load_chunk_vectors().unwrap().is_empty());
        let files = store.all_files().unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].content_hash, "");
    }

    #[test]
    fn migrates_a_pre_ingestion_depth_database() {
        // A DB written by an older Engram must keep working: open it with the
        // old PR schema, then reopen and confirm the new columns are usable.
        let repo = TempRepo::new();
        {
            // Create the DB, then rewrite the PR tables in their old shape.
            drop(repo.store());
            let conn = Connection::open(repo.0.join(".engram").join("engram.db")).unwrap();
            conn.execute_batch(
                "DROP TABLE IF EXISTS pull_requests;
                 DROP TABLE IF EXISTS pr_files;
                 DROP TABLE IF EXISTS review_comments;
                 CREATE TABLE pull_requests (number INTEGER PRIMARY KEY, title TEXT NOT NULL,
                     body TEXT NOT NULL, merged INTEGER NOT NULL, author TEXT NOT NULL);
                 CREATE TABLE pr_files (pr_number INTEGER NOT NULL, path TEXT NOT NULL,
                     PRIMARY KEY (pr_number, path));
                 CREATE TABLE review_comments (id INTEGER PRIMARY KEY, pr_number INTEGER NOT NULL,
                     path TEXT NOT NULL, line INTEGER, body TEXT NOT NULL, author TEXT NOT NULL);
                 INSERT INTO pull_requests VALUES (1, 'old', '', 1, 'alice');
                 INSERT INTO pr_files VALUES (1, 'legacy.rs');
                 INSERT INTO review_comments VALUES (1, 1, 'legacy.rs', 2, 'old note', 'bob');",
            )
            .unwrap();
        }

        let mut store = Store::open(&repo.0).unwrap();
        // Pre-existing rows survive, with empty defaults for the new columns.
        let files = store.pr_files(1).unwrap();
        assert_eq!(files[0].path, "legacy.rs");
        assert_eq!(
            files[0].patch, "",
            "legacy rows have no diff until re-ingest"
        );
        let comments = store.review_comments_for_path("legacy.rs").unwrap();
        assert_eq!(comments[0].body, "old note");
        assert_eq!(comments[0].created_at, "");
        // And the new columns actually work on the migrated tables.
        store
            .replace_pr_commits(
                1,
                &[PrCommit {
                    sha: "aaa".to_owned(),
                    ordinal: 0,
                    ..Default::default()
                }],
            )
            .unwrap();
        assert_eq!(store.pr_commits(1).unwrap().len(), 1);
    }
}
