//! SQLite persistence for the repo map. Single-file DB in .engram/engram.db.
//! Tracks which files are extracted to Tier-1 so retrieval can lazily extract on miss.

use anyhow::Result;
use engram_domain::{CoChange, FileRecord, Language, ReviewComment, SymbolKind, SymbolRecord};
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::Path;

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
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
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
            CREATE TABLE IF NOT EXISTS pull_requests (
                number INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                merged INTEGER NOT NULL,
                author TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pr_files (
                pr_number INTEGER NOT NULL,
                path TEXT NOT NULL,
                PRIMARY KEY (pr_number, path)
            );
            CREATE INDEX IF NOT EXISTS idx_pr_files_path ON pr_files(path);
            CREATE TABLE IF NOT EXISTS review_comments (
                id INTEGER PRIMARY KEY,
                pr_number INTEGER NOT NULL,
                path TEXT NOT NULL,
                line INTEGER,
                body TEXT NOT NULL,
                author TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_review_comments_path ON review_comments(path);
            "#,
        )?;
        // Migration for DBs created before recency tracking: add the column if
        // it is missing. Fails harmlessly ("duplicate column") on fresh DBs.
        let _ = conn.execute("ALTER TABLE files ADD COLUMN last_commit_ts INTEGER", []);
        Ok(Store { conn })
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

    /// Upsert a pull request row.
    pub fn upsert_pull_request(
        &mut self,
        number: i64,
        title: &str,
        body: &str,
        merged: bool,
        author: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO pull_requests (number, title, body, merged, author)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(number) DO UPDATE SET title=?2, body=?3, merged=?4, author=?5",
            params![number, title, body, merged as i64, author],
        )?;
        Ok(())
    }

    /// Replace the changed-file list recorded for a PR.
    pub fn replace_pr_files(&mut self, pr_number: i64, paths: &[String]) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM pr_files WHERE pr_number = ?1",
            params![pr_number],
        )?;
        for path in paths {
            tx.execute(
                "INSERT OR IGNORE INTO pr_files (pr_number, path) VALUES (?1, ?2)",
                params![pr_number, path],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Replace the review comments recorded for a PR. `(path, line, body, author)`.
    pub fn replace_review_comments_for_pr(
        &mut self,
        pr_number: i64,
        comments: &[(String, Option<i64>, String, String)],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM review_comments WHERE pr_number = ?1",
            params![pr_number],
        )?;
        for (path, line, body, author) in comments {
            tx.execute(
                "INSERT INTO review_comments (pr_number, path, line, body, author)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![pr_number, path, line, body, author],
            )?;
        }
        tx.commit()?;
        Ok(())
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
            "SELECT rc.pr_number, pr.title, pr.merged, rc.path, rc.line, rc.body, rc.author
             FROM review_comments rc JOIN pull_requests pr ON pr.number = rc.pr_number
             WHERE {where_clause}"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(p, |r| {
            Ok(ReviewComment {
                pr_number: r.get(0)?,
                pr_title: r.get(1)?,
                pr_merged: r.get::<_, i64>(2)? == 1,
                path: r.get(3)?,
                line: r.get(4)?,
                body: r.get(5)?,
                author: r.get(6)?,
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
    /// [`crate::imports::module_needle`]), excluding `exclude`.
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
            tx.execute(
                "INSERT INTO files (path, language, size_bytes, is_test)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(path) DO UPDATE SET
                   language=?2, size_bytes=?3, is_test=?4",
                params![
                    f.path,
                    f.language.as_str(),
                    f.size_bytes as i64,
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
                "INSERT INTO symbols (name, kind, path, start_line, signature)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    s.name,
                    format!("{:?}", s.kind).to_lowercase(),
                    s.path,
                    s.start_line as i64,
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

    pub fn all_files(&self) -> Result<Vec<FileRecord>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, language, size_bytes, is_test FROM files")?;
        let rows = stmt.query_map([], |r| {
            Ok(FileRecord {
                path: r.get(0)?,
                language: lang_from_str(&r.get::<_, String>(1)?),
                size_bytes: r.get::<_, i64>(2)? as u64,
                is_test: r.get::<_, i64>(3)? == 1,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn symbols_matching(&self, needle: &str, limit: usize) -> Result<Vec<SymbolRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, kind, path, start_line, signature FROM symbols
             WHERE name LIKE ?1 COLLATE NOCASE LIMIT ?2",
        )?;
        let pattern = format!("%{}%", needle);
        let rows = stmt.query_map(params![pattern, limit as i64], |r| {
            Ok(SymbolRecord {
                name: r.get(0)?,
                kind: kind_from_str(&r.get::<_, String>(1)?),
                path: r.get(2)?,
                start_line: r.get::<_, i64>(3)? as usize,
                signature: r.get(4)?,
            })
        })?;
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
