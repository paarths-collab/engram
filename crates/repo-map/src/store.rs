//! SQLite persistence for the repo map. Single-file DB in .engram/engram.db.
//! Tracks which files are extracted to Tier-1 so retrieval can lazily extract on miss.

use anyhow::Result;
use engram_domain::{CoChange, FileRecord, Language, SymbolKind, SymbolRecord};
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
