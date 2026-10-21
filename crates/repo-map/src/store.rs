//! SQLite persistence for the repo map. Single-file DB in .engram/engram.db.
//! Tracks which files are extracted to Tier-1 so retrieval can lazily extract on miss.

use crate::cochange::CoChangeFacts;
use crate::inventory::ScannedFile;
use anyhow::Result;
use engram_domain::{
    CoChange, FileRecord, IngestedComment, Language, PrCommit, PrFileChange, ReviewComment,
    SymbolKind, SymbolRecord,
};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Duration;

/// How long a connection waits for another writer's lock before giving up with
/// "database is locked". Indexing, the file watcher and request handlers each
/// open their own connection to the same file, so brief contention is normal.
const BUSY_TIMEOUT: Duration = Duration::from_secs(10);

/// Identity of one embedded chunk: the file, and the one-based line its span
/// starts at. `0` is reserved for the whole-file fallback used when a file has
/// no extracted symbols.
pub type ChunkKey = (String, usize);

/// Result of [`Store::cochange_neighbors`].
#[derive(Debug, Clone)]
pub struct CoChangeNeighbors {
    /// The strongest rows, at most the requested limit, in a total order.
    pub rows: Vec<CoChange>,
    /// Every row from the path to an indexed file, including those past the
    /// limit.
    pub total: usize,
}

/// One file's new place in the index: its fingerprint, and whether its cached
/// embeddings must be dropped even when the fingerprint did not change.
#[derive(Debug, Clone)]
pub struct FileUpdate {
    pub scanned: ScannedFile,
    /// Drop the file's cached vectors regardless of whether it changed. The
    /// watcher sets this on every save; a full reconcile leaves it off because
    /// a changed fingerprint already drops them.
    pub invalidate_vectors: bool,
}

/// What reading a file for Tier-1 produced.
#[derive(Debug, Clone)]
pub enum Extraction {
    /// Symbols and (raw, not yet normalized) import targets of the file as it
    /// was read. Either may be empty: a file can legitimately hold neither.
    Extracted {
        symbols: Vec<SymbolRecord>,
        imports: Vec<String>,
    },
    /// The file could not be read. Whatever is stored for it describes text
    /// that is gone, so it is cleared; the file stays un-extracted and a later
    /// pass retries it.
    Unreadable,
}

/// One file's extraction, applied together with everything else in a batch.
#[derive(Debug, Clone)]
pub struct ExtractionUpdate {
    pub path: String,
    pub extraction: Extraction,
}

/// A replacement of the git-history facts (co-change rows, their completeness
/// facts, per-file recency), applied as one unit.
#[derive(Debug, Clone, Default)]
pub struct HistoryUpdate {
    pub edges: Vec<CoChange>,
    /// `None` for a walk that failed: the facts then read back as unknown.
    pub facts: Option<CoChangeFacts>,
    pub recency: HashMap<String, i64>,
}

/// A staged set of index changes, built entirely in memory (reading and parsing
/// files, walking git history) with NO write transaction open, and then made
/// visible at once by [`Store::publish_batch`].
///
/// Application order inside the publish transaction is fixed, because later
/// steps rely on earlier ones: (1) the `retain_only` prune, (2) `removals`,
/// (3) `files` (fingerprint upserts, which also drop the derived rows of a
/// changed file), (4) `extractions` (symbols + imports + `tier1_done`, written
/// only for a path that has a `files` row by then), (5) `history`.
#[derive(Debug, Clone, Default)]
pub struct IndexBatch {
    /// Full reconcile: every `files` row whose path is not in this set is
    /// removed with everything derived from it.
    pub retain_only: Option<HashSet<String>>,
    /// Paths (or directories, which take everything beneath them) to forget.
    pub removals: Vec<String>,
    pub files: Vec<FileUpdate>,
    pub extractions: Vec<ExtractionUpdate>,
    pub history: Option<HistoryUpdate>,
}

impl IndexBatch {
    pub fn is_empty(&self) -> bool {
        self.retain_only.is_none()
            && self.removals.is_empty()
            && self.files.is_empty()
            && self.extractions.is_empty()
            && self.history.is_none()
    }
}

/// The outcome of one [`Store::publish_batch_detailed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Published {
    /// The generation after this publish (unchanged when nothing changed).
    pub generation: u64,
    /// Whether the batch changed any fact and therefore bumped the generation.
    pub changed: bool,
    /// Files dropped by the batch's `retain_only` prune.
    pub pruned_files: usize,
}

/// A consistent view of the database: one read transaction, which in WAL mode
/// is a snapshot. Every read made through the same [`Store`] while it is alive
/// sees the facts of exactly one generation, however many batches other
/// connections publish meanwhile. Ends (read-only, so rollback) on drop.
///
/// It borrows the store immutably, so a write through the same handle cannot
/// happen while a view is open: the compiler enforces "read, then write".
pub struct ReadView<'a> {
    conn: &'a Connection,
    generation: u64,
    owns_tx: bool,
}

impl ReadView<'_> {
    /// The generation every read in this view sees.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl Drop for ReadView<'_> {
    fn drop(&mut self) {
        if self.owns_tx {
            let _ = self.conn.execute_batch("ROLLBACK");
        }
    }
}

/// Stored fingerprint of one file, as far as change detection is concerned.
struct StoredFingerprint {
    size: i64,
    mtime_ns: Option<i64>,
    content_hash: Option<String>,
}

impl StoredFingerprint {
    /// A file is *changed* when its size differs from what is stored, or the
    /// new mtime differs, or the new content hash differs. Unknown values on
    /// the new side (`None`) never count as a change on their own, but a
    /// stored NULL against a known value does: a row from before a
    /// fingerprint existed has never been verified.
    fn differs_from(&self, scanned: &ScannedFile) -> bool {
        self.size != scanned.record.size_bytes as i64
            || (scanned.mtime_ns.is_some() && self.mtime_ns != scanned.mtime_ns)
            || (scanned.content_hash.is_some() && self.content_hash != scanned.content_hash)
    }
}

pub struct Store {
    pub conn: Connection,
}

/// Version of what the parsers and their queries produce from a file. Bump it
/// by hand when extraction logic changes in a way the inventory rules cannot
/// reveal. The rules themselves (ignored directories, extension list, size
/// cap) need no bump: `inventory::inventory_rules_fingerprint` folds them in
/// together with this number, and that fingerprint is what the database
/// records. On open, a database written under a different fingerprint has all
/// its derived data (symbols, imports, vectors) discarded and every file
/// marked for re-extraction, and the snapshot id changes with it.
pub const PARSER_CONFIG_VERSION: u32 = 1;

/// Meta key holding the rules fingerprint. The name predates the fingerprint
/// (it used to hold the bare version number); an old value never matches, so
/// such a database is simply re-extracted once.
const PARSER_VERSION_KEY: &str = "parser_config_version";
const COCHANGE_SCANNED_KEY: &str = "cochange_commits_scanned";
const COCHANGE_BULK_KEY: &str = "cochange_bulk_commits_excluded";
/// Meta key of the publish counter; see [`Store::generation`].
const GENERATION_KEY: &str = "generation";

impl Store {
    pub fn open(repo_root: &Path) -> Result<Self> {
        Self::open_with_rules_fingerprint(
            repo_root,
            &crate::inventory::inventory_rules_fingerprint(),
        )
    }

    /// [`Store::open`] under an explicit rules fingerprint. Exists so the
    /// invalidation on a fingerprint change can be tested without editing the
    /// rules themselves.
    pub fn open_with_rules_fingerprint(repo_root: &Path, fingerprint: &str) -> Result<Self> {
        let dir = repo_root.join(".engram");
        std::fs::create_dir_all(&dir)?;
        let conn = Connection::open(dir.join("engram.db"))?;
        // First, before any statement: switching to WAL and creating the schema
        // both need a lock another connection may hold at this very moment.
        conn.busy_timeout(BUSY_TIMEOUT)?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                is_test INTEGER NOT NULL,
                tier1_done INTEGER NOT NULL DEFAULT 0,
                last_commit_ts INTEGER,
                mtime_ns INTEGER,
                content_hash TEXT
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
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
            -- Per-file denominator of cochange.strength: non-bulk commits in
            -- the window that touched the file. Absent (with the meta totals)
            -- until the next history walk on a database that predates it.
            CREATE TABLE IF NOT EXISTS cochange_file_commits (
                path TEXT PRIMARY KEY,
                commits INTEGER NOT NULL
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
        Self::reconcile_parser_version(&conn, fingerprint)?;
        Ok(Store { conn })
    }

    /// Discard derived data if the database was written under a different
    /// rules fingerprint (see [`PARSER_CONFIG_VERSION`]). A brand-new database
    /// has nothing to discard and just records the fingerprint.
    ///
    /// The check and the wipe happen under one `BEGIN IMMEDIATE` transaction,
    /// and the stored value is read again inside it: of several connections
    /// that opened at the same moment and all saw the mismatch, only the first
    /// to take the write lock wipes; the rest find the fingerprint already
    /// updated and must not discard what the first has since written.
    fn reconcile_parser_version(conn: &Connection, version: &str) -> Result<()> {
        // Cheap path with no write lock: the common case is a match.
        if Self::stored_parser_version(conn)?.as_deref() == Some(version) {
            return Ok(());
        }
        Self::reset_if_still_mismatched(conn, version)
    }

    fn stored_parser_version(conn: &Connection) -> Result<Option<String>> {
        Ok(conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![PARSER_VERSION_KEY],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// The locked half of [`Store::reconcile_parser_version`]: the caller has
    /// seen a mismatch, but that may be stale by the time the write lock is
    /// held, so the stored value is read again inside the transaction.
    fn reset_if_still_mismatched(conn: &Connection, version: &str) -> Result<()> {
        // One transaction: a crash midway must not leave the derived data
        // wiped but the old version recorded, which would trust the empty tables.
        conn.execute_batch("BEGIN IMMEDIATE")?;
        let result = (|| -> Result<()> {
            let stored = Self::stored_parser_version(conn)?;
            if stored.as_deref() == Some(version) {
                return Ok(());
            }
            // Discarding derived data changes what readers see, so it is a
            // generation of its own. A brand-new database (nothing recorded
            // yet) has nothing to discard and stays at generation 0.
            if stored.is_some() {
                Self::bump_generation(conn)?;
            }
            conn.execute_batch(
                "DELETE FROM symbols;
                 DELETE FROM file_imports;
                 DELETE FROM vectors;
                 DELETE FROM chunk_vectors;
                 UPDATE files SET tier1_done = 0;",
            )?;
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = ?2",
                params![PARSER_VERSION_KEY, version],
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                conn.execute_batch("COMMIT")?;
                Ok(())
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    /// A write transaction that takes the write lock up front. A deferred one
    /// that reads first and writes later fails at the upgrade with an immediate
    /// "database is locked" (the busy timeout does not apply) whenever another
    /// connection committed in between; taking the lock first waits instead.
    fn write_tx(&mut self) -> Result<rusqlite::Transaction<'_>> {
        Ok(self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?)
    }

    /// The publish counter: how many batches of changes to the indexed facts
    /// (files, symbols, imports, co-change, recency) have been committed to
    /// this database, `0` for one that has never published.
    ///
    /// It is a counter of *publishes*, not an identity of content (that is
    /// [`Store::snapshot_id`]): it only goes up, it is bumped inside the same
    /// transaction that changes the facts (so a reader sees the new number
    /// exactly when it sees the new facts), and a batch that changes nothing
    /// does not move it. A cheap single-row read, so callers can poll it.
    pub fn generation(&self) -> Result<u64> {
        Self::read_generation(&self.conn)
    }

    fn read_generation(conn: &Connection) -> Result<u64> {
        let value: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![GENERATION_KEY],
                |r| r.get(0),
            )
            .optional()?;
        Ok(value.and_then(|v| v.parse().ok()).unwrap_or(0))
    }

    /// Add one to the counter and return the new value. Only ever called
    /// inside the transaction that wrote the facts it accounts for.
    fn bump_generation(conn: &Connection) -> Result<u64> {
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, '1')
             ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)",
            params![GENERATION_KEY],
        )?;
        Self::read_generation(conn)
    }

    /// Start a consistent read: every read made through this store until the
    /// returned view is dropped sees one generation, whatever other connections
    /// publish meanwhile (SQLite WAL snapshot isolation). The generation is
    /// read first, which is what pins the snapshot, so [`ReadView::generation`]
    /// is exactly the generation of all the facts read inside it.
    ///
    /// If the connection is already inside a transaction, that transaction is
    /// the snapshot: nothing new is started, and nothing is ended on drop.
    pub fn begin_read(&self) -> Result<ReadView<'_>> {
        let owns_tx = self.conn.is_autocommit();
        if owns_tx {
            self.conn.execute_batch("BEGIN")?;
        }
        match Self::read_generation(&self.conn) {
            Ok(generation) => Ok(ReadView {
                conn: &self.conn,
                generation,
                owns_tx,
            }),
            Err(e) => {
                if owns_tx {
                    let _ = self.conn.execute_batch("ROLLBACK");
                }
                Err(e)
            }
        }
    }

    /// Apply a staged [`IndexBatch`] as ONE generation: a single `IMMEDIATE`
    /// transaction holds every fingerprint upsert, derived-row invalidation,
    /// symbol and import replacement, removal and co-change replacement, and
    /// bumps the generation by one inside that same transaction when anything
    /// changed. Returns the generation after the publish.
    ///
    /// All-or-nothing: an error (or a panic, or a crash) before the commit
    /// leaves the previous generation fully intact, with none of the batch
    /// applied. A batch that changes nothing does not bump. Concurrent readers
    /// using [`Store::begin_read`] see either all of the batch or none of it.
    pub fn publish_batch(&mut self, batch: IndexBatch) -> Result<u64> {
        Ok(self.publish_batch_detailed(batch)?.generation)
    }

    /// [`Store::publish_batch`], also saying whether anything changed and how
    /// many files the batch's prune dropped.
    pub fn publish_batch_detailed(&mut self, batch: IndexBatch) -> Result<Published> {
        let tx = self.write_tx()?;
        let mut changed = false;
        let mut pruned_files = 0;

        if let Some(keep) = &batch.retain_only {
            pruned_files = Self::prune_missing_tx(&tx, keep)?;
            changed |= pruned_files > 0;
        }
        for path in &batch.removals {
            changed |= Self::remove_path_tx(&tx, path)? > 0;
        }
        for update in &batch.files {
            changed |= Self::upsert_one_tx(&tx, &update.scanned)?;
            if update.invalidate_vectors {
                // Cache rows: dropping them is not a change to the facts.
                Self::delete_vectors_tx(&tx, &update.scanned.record.path)?;
            }
        }
        for update in &batch.extractions {
            changed |= Self::apply_extraction_tx(&tx, update)?;
        }
        if let Some(history) = &batch.history {
            changed |= Self::apply_history_tx(&tx, history)?;
        }

        let generation = if changed {
            Self::bump_generation(&tx)?
        } else {
            Self::read_generation(&tx)?
        };
        tx.commit()?;
        Ok(Published {
            generation,
            changed,
            pruned_files,
        })
    }

    /// Of `files` (about to be upserted), the paths that will have no usable
    /// extraction afterwards: not stored yet, changed since, or never
    /// extracted. Read-only, so staging can decide what to parse without
    /// holding a write transaction. The answer is a snapshot; if another
    /// connection extracts one meanwhile, re-extracting it is harmless.
    pub fn pending_extraction(&self, files: &[ScannedFile]) -> Result<HashSet<String>> {
        let mut stored: HashMap<String, (StoredFingerprint, bool)> = HashMap::new();
        {
            let mut stmt = self.conn.prepare(
                "SELECT path, size_bytes, mtime_ns, content_hash, tier1_done FROM files",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    StoredFingerprint {
                        size: r.get(1)?,
                        mtime_ns: r.get(2)?,
                        content_hash: r.get(3)?,
                    },
                    r.get::<_, i64>(4)? == 1,
                ))
            })?;
            for row in rows {
                let (path, fingerprint, done) = row?;
                stored.insert(path, (fingerprint, done));
            }
        }
        Ok(files
            .iter()
            .filter(|f| match stored.get(&f.record.path) {
                None => true,
                Some((fingerprint, done)) => !done || fingerprint.differs_from(f),
            })
            .map(|f| f.record.path.clone())
            .collect())
    }

    /// Content hash recorded for every indexed file (`None` where a row never
    /// had one).
    pub fn file_hashes(&self) -> Result<HashMap<String, Option<String>>> {
        let mut stmt = self.conn.prepare("SELECT path, content_hash FROM files")?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Every extracted symbol, grouped by file and in source order within one.
    pub fn all_symbols_by_path(&self) -> Result<HashMap<String, Vec<SymbolRecord>>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, kind, path, start_line, end_line, signature FROM symbols
             ORDER BY path, start_line, id",
        )?;
        let rows = stmt.query_map([], row_to_symbol)?;
        let mut out: HashMap<String, Vec<SymbolRecord>> = HashMap::new();
        for symbol in rows {
            let symbol = symbol?;
            out.entry(symbol.path.clone()).or_default().push(symbol);
        }
        Ok(out)
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

        // Predates the per-file fingerprint. The ALTER succeeds exactly once, on
        // a DB whose cached symbols and imports were never checked against the
        // file on disk; distrust them all so each file re-extracts once rather
        // than serving whatever an old run (or an offline edit) left behind.
        if conn
            .execute("ALTER TABLE files ADD COLUMN mtime_ns INTEGER", [])
            .is_ok()
        {
            let _ = conn.execute("UPDATE files SET tier1_done = 0", []);
        }

        // Predates content hashing. Same reasoning as above: with no stored
        // hash there is nothing to compare an on-disk file against.
        if conn
            .execute("ALTER TABLE files ADD COLUMN content_hash TEXT", [])
            .is_ok()
        {
            let _ = conn.execute("UPDATE files SET tier1_done = 0", []);
        }

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
        let tx = self.write_tx()?;
        if Self::apply_recency_tx(&tx, recency)? {
            Self::bump_generation(&tx)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Set the recency of the listed paths that are indexed; true when any
    /// value actually changed.
    fn apply_recency_tx(
        tx: &rusqlite::Transaction<'_>,
        recency: &HashMap<String, i64>,
    ) -> Result<bool> {
        let mut changed = 0;
        for (path, ts) in recency {
            changed += tx.execute(
                "UPDATE files SET last_commit_ts = ?2
                 WHERE path = ?1 AND (last_commit_ts IS NULL OR last_commit_ts <> ?2)",
                params![path, ts],
            )?;
        }
        Ok(changed > 0)
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
        let tx = self.write_tx()?;
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
        let tx = self.write_tx()?;
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
        let tx = self.write_tx()?;
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
        let tx = self.write_tx()?;
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
        let tx = self.write_tx()?;
        let pruned = Self::prune_missing_tx(&tx, keep)?;
        if pruned > 0 {
            Self::bump_generation(&tx)?;
        }
        tx.commit()?;
        Ok(pruned)
    }

    fn prune_missing_tx(
        tx: &rusqlite::Transaction<'_>,
        keep: &std::collections::HashSet<String>,
    ) -> Result<usize> {
        let stale: Vec<String> = {
            let mut stmt = tx.prepare("SELECT path FROM files")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.collect::<rusqlite::Result<Vec<String>>>()?
                .into_iter()
                .filter(|path| !keep.contains(path))
                .collect()
        };
        for path in &stale {
            Self::delete_file_rows(tx, path)?;
        }
        Ok(stale.len())
    }

    /// Delete everything derived from a file's contents, keeping its `files`
    /// row: symbols, import edges, and embeddings.
    fn delete_derived_rows(tx: &rusqlite::Transaction<'_>, path: &str) -> Result<()> {
        tx.execute("DELETE FROM symbols WHERE path = ?1", params![path])?;
        tx.execute("DELETE FROM file_imports WHERE path = ?1", params![path])?;
        Self::delete_vectors_tx(tx, path)
    }

    /// Delete a file's cached embeddings (a cache, not part of the facts).
    fn delete_vectors_tx(tx: &rusqlite::Transaction<'_>, path: &str) -> Result<()> {
        tx.execute("DELETE FROM vectors WHERE path = ?1", params![path])?;
        tx.execute("DELETE FROM chunk_vectors WHERE path = ?1", params![path])?;
        Ok(())
    }

    /// Delete every row that belongs to one file.
    fn delete_file_rows(tx: &rusqlite::Transaction<'_>, path: &str) -> Result<()> {
        tx.execute("DELETE FROM files WHERE path = ?1", params![path])?;
        Self::delete_derived_rows(tx, path)
    }

    /// Remove every trace of one file, and of anything beneath it should
    /// `path` have been a directory (a renamed or deleted directory reaches the
    /// watcher as one event for the directory, not one per file).
    ///
    /// Everything beneath a directory is matched with an index range
    /// (`dir/` <= path < `dir0`, since `0` is the character after `/`), so a
    /// removal costs a seek rather than a scan of every indexed file. Rows for
    /// the exact path are cleaned even when its `files` row is already gone, so
    /// rows orphaned by an older Engram do not survive a removal.
    ///
    /// Returns the number of rows deleted across all tables; zero means the
    /// index did not change.
    pub fn remove_path(&mut self, path: &str) -> Result<usize> {
        let tx = self.write_tx()?;
        let removed = Self::remove_path_tx(&tx, path)?;
        if removed > 0 {
            Self::bump_generation(&tx)?;
        }
        tx.commit()?;
        Ok(removed)
    }

    fn remove_path_tx(tx: &rusqlite::Transaction<'_>, path: &str) -> Result<usize> {
        let dir = path.trim_end_matches('/');
        let range_start = format!("{dir}/");
        let range_end = format!("{dir}0");
        let mut removed = 0;
        for table in [
            "files",
            "symbols",
            "file_imports",
            "vectors",
            "chunk_vectors",
        ] {
            removed += tx.execute(
                &format!("DELETE FROM {table} WHERE path = ?1 OR (path >= ?2 AND path < ?3)"),
                params![dir, range_start, range_end],
            )?;
        }
        Ok(removed)
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
        let tx = self.write_tx()?;
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
        let tx = self.write_tx()?;
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
        let tx = self.write_tx()?;
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
    /// Ordered by `pr_number, path, line, created_at` (then body, author, id as
    /// final tie-breaks), never by insertion order.
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
             WHERE {where_clause}
             ORDER BY rc.pr_number, rc.path, rc.line, rc.created_at, rc.body, rc.author, rc.id"
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
        let tx = self.write_tx()?;
        if Self::replace_imports_tx(&tx, path, targets)? {
            Self::bump_generation(&tx)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Replace a file's import rows; true when they differ from what was
    /// stored (an identical re-extraction writes nothing).
    fn replace_imports_tx(
        tx: &rusqlite::Transaction<'_>,
        path: &str,
        targets: &[String],
    ) -> Result<bool> {
        let new: Vec<String> = targets
            .iter()
            .map(|raw| crate::imports::normalize_target(raw))
            .filter(|target| !target.is_empty())
            .collect();
        let old: Vec<String> = {
            let mut stmt =
                tx.prepare("SELECT target FROM file_imports WHERE path = ?1 ORDER BY rowid")?;
            let rows = stmt.query_map(params![path], |r| r.get::<_, String>(0))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        if old == new {
            return Ok(false);
        }
        tx.execute("DELETE FROM file_imports WHERE path = ?1", params![path])?;
        let mut insert =
            tx.prepare_cached("INSERT INTO file_imports (path, target) VALUES (?1, ?2)")?;
        for target in &new {
            insert.execute(params![path, target])?;
        }
        Ok(true)
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

    /// Upsert files whose modification time is unknown. A size change still
    /// invalidates extraction; see [`Store::upsert_scanned`].
    pub fn upsert_files(&mut self, files: &[FileRecord]) -> Result<()> {
        let scanned: Vec<(FileRecord, Option<i64>)> =
            files.iter().cloned().map(|f| (f, None)).collect();
        self.upsert_scanned(&scanned)
    }

    /// Upsert files with their modification times but no content hash.
    pub fn upsert_scanned(&mut self, files: &[(FileRecord, Option<i64>)]) -> Result<()> {
        let scanned: Vec<ScannedFile> = files
            .iter()
            .map(|(record, mtime_ns)| ScannedFile {
                record: record.clone(),
                mtime_ns: *mtime_ns,
                content_hash: None,
            })
            .collect();
        self.upsert_fingerprinted(&scanned)
    }

    /// Upsert inventoried files with their full fingerprint.
    ///
    /// A file is *changed* when its size differs from what is stored, or the
    /// new mtime differs, or the new content hash differs. Unknown values on
    /// the new side (`None`) never count as a change on their own, but a
    /// stored NULL against a known value does: a row from before a
    /// fingerprint existed has never been verified, so it is re-extracted once.
    ///
    /// For a changed file every derived row (symbols, imports, vectors, chunk
    /// vectors) is deleted and `tier1_done` reset, so the old contents can
    /// neither be served nor mistaken for current.
    pub fn upsert_fingerprinted(&mut self, files: &[ScannedFile]) -> Result<()> {
        let tx = self.write_tx()?;
        let mut changed = false;
        for scanned in files {
            changed |= Self::upsert_one_tx(&tx, scanned)?;
        }
        if changed {
            Self::bump_generation(&tx)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Upsert one inventoried file inside a transaction. True when the file is
    /// new or anything the index shows about it changed (fingerprint, language,
    /// test flag).
    fn upsert_one_tx(tx: &rusqlite::Transaction<'_>, scanned: &ScannedFile) -> Result<bool> {
        let f = &scanned.record;
        let old: Option<(StoredFingerprint, String, bool)> = tx
            .query_row(
                "SELECT size_bytes, mtime_ns, content_hash, language, is_test
                 FROM files WHERE path = ?1",
                params![f.path],
                |r| {
                    Ok((
                        StoredFingerprint {
                            size: r.get(0)?,
                            mtime_ns: r.get(1)?,
                            content_hash: r.get(2)?,
                        },
                        r.get::<_, String>(3)?,
                        r.get::<_, i64>(4)? == 1,
                    ))
                },
            )
            .optional()?;
        let changed = old
            .as_ref()
            .is_some_and(|(fingerprint, _, _)| fingerprint.differs_from(scanned));
        let visible_change = match &old {
            None => true,
            Some((_, language, is_test)) => {
                changed || language != f.language.as_str() || *is_test != f.is_test
            }
        };
        if changed {
            Self::delete_derived_rows(tx, &f.path)?;
        }
        tx.execute(
            "INSERT INTO files (path, language, size_bytes, is_test, mtime_ns, content_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(path) DO UPDATE SET
               language=?2,
               is_test=?4,
               tier1_done = CASE WHEN ?7 THEN 0 ELSE tier1_done END,
               size_bytes=?3,
               mtime_ns=COALESCE(?5, mtime_ns),
               content_hash=COALESCE(?6, content_hash)",
            params![
                f.path,
                f.language.as_str(),
                f.size_bytes as i64,
                f.is_test as i64,
                scanned.mtime_ns,
                scanned.content_hash,
                changed
            ],
        )?;
        Ok(visible_change)
    }

    /// A stable id for exactly what is indexed: the checked-out commit (`head`),
    /// the inventory rules fingerprint, and every indexed file's path and content
    /// hash. Identical content at the same commit gives the same id regardless
    /// of mtimes or checkout location; any content, path, or version
    /// difference gives a different one.
    pub fn snapshot_id(&self, head: &str) -> Result<String> {
        use crate::inventory::Fnv64;
        let version: String = self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![PARSER_VERSION_KEY],
                |r| r.get(0),
            )
            .unwrap_or_else(|_| crate::inventory::inventory_rules_fingerprint());
        let mut hasher = Fnv64::default();
        // 0xff cannot occur in UTF-8, so fields cannot run into each other.
        let mut field = |s: &str| {
            hasher.write(s.as_bytes());
            hasher.write(&[0xff]);
        };
        field("engram-snapshot-v1");
        field(head);
        field(&version);
        let mut stmt = self
            .conn
            .prepare("SELECT path, content_hash FROM files ORDER BY path")?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
        })?;
        for row in rows {
            let (path, hash) = row?;
            field(&path);
            field(hash.as_deref().unwrap_or(""));
        }
        Ok(hasher.finish_hex())
    }

    /// Drop a file's stored symbols and imports without marking it extracted,
    /// for when its contents can no longer be read: the old rows describe text
    /// that is gone, and the file should be retried rather than trusted empty.
    pub fn clear_extracted(&mut self, path: &str) -> Result<()> {
        let tx = self.write_tx()?;
        if Self::clear_extracted_tx(&tx, path)? {
            Self::bump_generation(&tx)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// True when a symbol or import row was removed.
    fn clear_extracted_tx(tx: &rusqlite::Transaction<'_>, path: &str) -> Result<bool> {
        let removed = tx.execute("DELETE FROM symbols WHERE path = ?1", params![path])?
            + tx.execute("DELETE FROM file_imports WHERE path = ?1", params![path])?;
        tx.execute(
            "UPDATE files SET tier1_done = 0 WHERE path = ?1",
            params![path],
        )?;
        Ok(removed > 0)
    }

    pub fn replace_symbols_for_file(&mut self, path: &str, symbols: &[SymbolRecord]) -> Result<()> {
        let tx = self.write_tx()?;
        if Self::replace_symbols_tx(&tx, path, symbols)? {
            Self::bump_generation(&tx)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Replace a file's symbols and mark it extracted. True when the symbols
    /// differ from what was stored (an identical re-extraction writes nothing,
    /// and flipping `tier1_done` alone is bookkeeping, not a change of facts).
    fn replace_symbols_tx(
        tx: &rusqlite::Transaction<'_>,
        path: &str,
        symbols: &[SymbolRecord],
    ) -> Result<bool> {
        type Row = (String, String, String, i64, i64, String);
        let new: Vec<Row> = symbols
            .iter()
            .map(|s| {
                (
                    s.name.clone(),
                    format!("{:?}", s.kind).to_lowercase(),
                    s.path.clone(),
                    s.start_line as i64,
                    s.end_line as i64,
                    s.signature.clone(),
                )
            })
            .collect();
        let old: Vec<Row> = {
            let mut stmt = tx.prepare(
                "SELECT name, kind, path, start_line, end_line, signature
                 FROM symbols WHERE path = ?1 ORDER BY id",
            )?;
            let rows = stmt.query_map(params![path], |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                ))
            })?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        let changed = old != new;
        if changed {
            tx.execute("DELETE FROM symbols WHERE path = ?1", params![path])?;
            let mut insert = tx.prepare_cached(
                "INSERT INTO symbols (name, kind, path, start_line, end_line, signature)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for row in &new {
                insert.execute(params![row.0, row.1, row.2, row.3, row.4, row.5])?;
            }
        }
        tx.execute(
            "UPDATE files SET tier1_done = 1 WHERE path = ?1",
            params![path],
        )?;
        Ok(changed)
    }

    /// Apply one staged extraction. Skipped (false) for a path with no `files`
    /// row, which was removed between staging and publishing: writing symbols
    /// for it would resurrect a ghost.
    fn apply_extraction_tx(
        tx: &rusqlite::Transaction<'_>,
        update: &ExtractionUpdate,
    ) -> Result<bool> {
        let known: bool = tx
            .query_row(
                "SELECT 1 FROM files WHERE path = ?1",
                params![update.path],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        if !known {
            return Ok(false);
        }
        match &update.extraction {
            Extraction::Extracted { symbols, imports } => {
                // Imports first: replacing symbols is what marks the file
                // extracted, so the order is the same as it always was.
                let imports_changed = Self::replace_imports_tx(tx, &update.path, imports)?;
                let symbols_changed = Self::replace_symbols_tx(tx, &update.path, symbols)?;
                Ok(imports_changed || symbols_changed)
            }
            Extraction::Unreadable => Self::clear_extracted_tx(tx, &update.path),
        }
    }

    /// Replace the co-change rows only. The completeness facts that described
    /// the previous rows no longer apply to these, so they are cleared and read
    /// back as unknown; use [`Store::replace_cochange_with_facts`] to record
    /// them.
    pub fn replace_cochange(&mut self, edges: &[CoChange]) -> Result<()> {
        self.replace_cochange_inner(edges, None)
    }

    /// Replace the co-change rows together with the facts of the walk that
    /// produced them, in one transaction.
    pub fn replace_cochange_with_facts(
        &mut self,
        edges: &[CoChange],
        facts: &CoChangeFacts,
    ) -> Result<()> {
        self.replace_cochange_inner(edges, Some(facts))
    }

    fn replace_cochange_inner(
        &mut self,
        edges: &[CoChange],
        facts: Option<&CoChangeFacts>,
    ) -> Result<()> {
        let tx = self.write_tx()?;
        if Self::replace_cochange_tx(&tx, edges, facts)? {
            Self::bump_generation(&tx)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Replace the co-change rows and their completeness facts. True when the
    /// stored result differs from what was there (compared by digest, so a
    /// history walk that found the same history changes nothing).
    fn replace_cochange_tx(
        tx: &rusqlite::Transaction<'_>,
        edges: &[CoChange],
        facts: Option<&CoChangeFacts>,
    ) -> Result<bool> {
        let before = Self::cochange_digest(tx)?;
        tx.execute("DELETE FROM cochange", [])?;
        {
            let mut insert = tx.prepare_cached(
                "INSERT OR REPLACE INTO cochange (path_a, path_b, count, strength)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            for e in edges {
                insert.execute(params![e.path_a, e.path_b, e.count as i64, e.strength])?;
            }
        }
        tx.execute("DELETE FROM cochange_file_commits", [])?;
        tx.execute(
            "DELETE FROM meta WHERE key IN (?1, ?2)",
            params![COCHANGE_SCANNED_KEY, COCHANGE_BULK_KEY],
        )?;
        if let Some(facts) = facts {
            for (path, commits) in &facts.file_commits {
                tx.execute(
                    "INSERT INTO cochange_file_commits (path, commits) VALUES (?1, ?2)",
                    params![path, i64::from(*commits)],
                )?;
            }
            for (key, value) in [
                (COCHANGE_SCANNED_KEY, facts.commits_scanned),
                (COCHANGE_BULK_KEY, facts.bulk_commits_excluded),
            ] {
                tx.execute(
                    "INSERT INTO meta (key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = ?2",
                    params![key, value.to_string()],
                )?;
            }
        }
        Ok(before != Self::cochange_digest(tx)?)
    }

    /// Replace the git-history facts of a staged batch: co-change rows (with
    /// their completeness facts) and per-file recency. True when anything
    /// differs afterwards.
    fn apply_history_tx(tx: &rusqlite::Transaction<'_>, history: &HistoryUpdate) -> Result<bool> {
        let cochange = Self::replace_cochange_tx(tx, &history.edges, history.facts.as_ref())?;
        let recency = Self::apply_recency_tx(tx, &history.recency)?;
        Ok(cochange || recency)
    }

    /// Digest of everything [`Store::replace_cochange_tx`] rewrites, in a
    /// canonical order, so before and after can be compared without keeping a
    /// copy of the rows.
    fn cochange_digest(conn: &Connection) -> Result<String> {
        use crate::inventory::Fnv64;
        let mut hasher = Fnv64::default();
        let mut field = |bytes: &[u8]| {
            hasher.write(bytes);
            hasher.write(&[0xff]);
        };
        let mut stmt = conn.prepare(
            "SELECT path_a, path_b, count, strength FROM cochange ORDER BY path_a, path_b",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            field(r.get::<_, String>(0)?.as_bytes());
            field(r.get::<_, String>(1)?.as_bytes());
            field(&r.get::<_, i64>(2)?.to_le_bytes());
            field(&r.get::<_, f64>(3)?.to_bits().to_le_bytes());
        }
        field(b"file-commits");
        let mut stmt =
            conn.prepare("SELECT path, commits FROM cochange_file_commits ORDER BY path")?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            field(r.get::<_, String>(0)?.as_bytes());
            field(&r.get::<_, i64>(1)?.to_le_bytes());
        }
        field(b"totals");
        for key in [COCHANGE_SCANNED_KEY, COCHANGE_BULK_KEY] {
            let value: Option<String> = conn
                .query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| {
                    r.get(0)
                })
                .optional()?;
            field(value.as_deref().unwrap_or("-").as_bytes());
        }
        Ok(hasher.finish_hex())
    }

    /// `(commits_scanned, bulk_commits_excluded)` of the walk that built the
    /// co-change table, or `None` when this database has not recorded one yet
    /// (it predates the totals, or the rows were written without them).
    pub fn cochange_totals(&self) -> Result<Option<(u32, u32)>> {
        let read = |key: &str| -> Result<Option<u32>> {
            let value: Option<String> = self
                .conn
                .query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| {
                    r.get(0)
                })
                .optional()?;
            Ok(value.and_then(|v| v.parse().ok()))
        };
        Ok(
            match (read(COCHANGE_SCANNED_KEY)?, read(COCHANGE_BULK_KEY)?) {
                (Some(scanned), Some(bulk)) => Some((scanned, bulk)),
                _ => None,
            },
        )
    }

    /// Non-bulk commits in the history window that touched `path` (the
    /// denominator of its co-change strengths). `None` when the totals were
    /// never recorded; `Some(0)` when they were and the file was not touched.
    pub fn cochange_commits_touching(&self, path: &str) -> Result<Option<u32>> {
        if self.cochange_totals()?.is_none() {
            return Ok(None);
        }
        let n: Option<i64> = self
            .conn
            .query_row(
                "SELECT commits FROM cochange_file_commits WHERE path = ?1",
                params![path],
                |r| r.get(0),
            )
            .optional()?;
        Ok(Some(n.map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX))))
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

    /// Every indexed file, ordered by path so callers never depend on insertion order.
    pub fn all_files(&self) -> Result<Vec<FileRecord>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, language, size_bytes, is_test FROM files ORDER BY path")?;
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
            "SELECT name, kind, path, start_line, end_line, signature FROM symbols
             WHERE name LIKE ?1 COLLATE NOCASE LIMIT ?2",
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
             WHERE name = ?1 COLLATE NOCASE LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![needle, limit as i64], row_to_symbol)?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Every extracted symbol for a file, in source order. Retrieval uses the
    /// spans to embed and quote definitions rather than the head of the file.
    pub fn symbols_for_path(&self, path: &str) -> Result<Vec<SymbolRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, kind, path, start_line, end_line, signature FROM symbols
             WHERE path = ?1 ORDER BY start_line",
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

    /// The strongest co-change rows `path -> partner` whose partner is an
    /// indexed file, at most `limit`, ordered by strength descending, then
    /// shared-commit count descending, then partner path ascending (a total
    /// order), plus how many such rows exist in all. Unlike
    /// [`Store::cochange_for`] the order is fully determined by the query, so
    /// the cut at `limit` is the same on every run, and the caller can say
    /// exactly how many rows the limit hid (`total - rows.len()`).
    pub fn cochange_neighbors(&self, path: &str, limit: usize) -> Result<CoChangeNeighbors> {
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM cochange c JOIN files f ON f.path = c.path_b
             WHERE c.path_a = ?1",
            params![path],
            |r| r.get(0),
        )?;
        let mut stmt = self.conn.prepare(
            "SELECT c.path_a, c.path_b, c.count, c.strength
             FROM cochange c JOIN files f ON f.path = c.path_b
             WHERE c.path_a = ?1
             ORDER BY c.strength DESC, c.count DESC, c.path_b ASC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![path, limit as i64], |r| {
            Ok(CoChange {
                path_a: r.get(0)?,
                path_b: r.get(1)?,
                count: r.get::<_, i64>(2)? as u32,
                strength: r.get(3)?,
            })
        })?;
        let rows: Vec<CoChange> = rows.collect::<rusqlite::Result<_>>()?;
        Ok(CoChangeNeighbors {
            rows,
            total: usize::try_from(total).unwrap_or(0),
        })
    }

    /// How many directed co-change rows the neighbourhood of `anchors` holds in
    /// all, read or not: every row `anchor -> partner` whose partner is an
    /// indexed file other than the anchor, plus every row `partner -> anchor`
    /// whose forward row `anchor -> partner` exists (the reverse row a graph
    /// lists with it). A row reachable from two anchors counts once. This is the
    /// number of co-change facts a graph of `anchors` holds with no limit
    /// applied, so `total - listed` is exactly what a limit hid.
    pub fn cochange_fact_count(&self, anchors: &[String]) -> Result<usize> {
        if anchors.is_empty() {
            return Ok(0);
        }
        // One bound variable per anchor, used twice below.
        anyhow::ensure!(
            anchors.len() <= 400,
            "at most 400 anchors per co-change count, got {}",
            anchors.len()
        );
        let marks = vec!["?"; anchors.len()].join(",");
        let sql = format!(
            "SELECT COUNT(*) FROM (
                 SELECT c.path_a AS a, c.path_b AS b
                 FROM cochange c JOIN files f ON f.path = c.path_b
                 WHERE c.path_a IN ({marks}) AND c.path_b <> c.path_a
                 UNION
                 SELECT r.path_a, r.path_b
                 FROM cochange r
                 JOIN cochange c ON c.path_a = r.path_b AND c.path_b = r.path_a
                 JOIN files f ON f.path = r.path_a
                 WHERE r.path_b IN ({marks}) AND r.path_a <> r.path_b
             )"
        );
        let bound = anchors.iter().chain(anchors.iter());
        let n: i64 = self
            .conn
            .query_row(&sql, rusqlite::params_from_iter(bound), |r| r.get(0))?;
        Ok(usize::try_from(n).unwrap_or(0))
    }

    /// The single directed co-change row `a -> b`, looked up by primary key.
    /// Unlike [`Store::cochange_for`] this is not ranked or capped, so a weak
    /// pair is found as reliably as a strong one. `b -> a` is a separate row.
    pub fn cochange_pair(&self, a: &str, b: &str) -> Result<Option<CoChange>> {
        Ok(self
            .conn
            .query_row(
                "SELECT path_a, path_b, count, strength FROM cochange
                 WHERE path_a = ?1 AND path_b = ?2",
                params![a, b],
                |r| {
                    Ok(CoChange {
                        path_a: r.get(0)?,
                        path_b: r.get(1)?,
                        count: r.get::<_, i64>(2)? as u32,
                        strength: r.get(3)?,
                    })
                },
            )
            .optional()?)
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
    fn all_files_and_review_comments_ignore_insertion_order() {
        let comment = |path: &str, line: Option<i64>, created_at: &str| IngestedComment {
            path: path.to_owned(),
            line,
            body: format!("{path}:{line:?}:{created_at}"),
            author: "bob".to_owned(),
            created_at: created_at.to_owned(),
            ..Default::default()
        };
        let build = |files: &[&str], prs: &[i64]| {
            let repo = TempRepo::new();
            let mut store = repo.store();
            let records: Vec<_> = files.iter().map(|f| file(f)).collect();
            store.upsert_files(&records).unwrap();
            for &pr in prs {
                store
                    .upsert_pull_request(pr, "t", "", true, "alice", "", "", "", "")
                    .unwrap();
                store
                    .replace_review_comments_for_pr(
                        pr,
                        &[
                            comment("b.rs", Some(9), "2026-01-02"),
                            comment("a.rs", Some(30), "2026-01-01"),
                            comment("a.rs", Some(4), "2026-01-03"),
                            comment("a.rs", Some(4), "2026-01-01"),
                        ],
                    )
                    .unwrap();
            }
            let paths: Vec<String> = store
                .all_files()
                .unwrap()
                .into_iter()
                .map(|f| f.path)
                .collect();
            let comments: Vec<(i64, String, Option<i64>, String)> = store
                .all_review_comments()
                .unwrap()
                .into_iter()
                .map(|c| (c.pr_number, c.path, c.line, c.created_at))
                .collect();
            let for_path: Vec<String> = store
                .review_comments_for_path("a.rs")
                .unwrap()
                .into_iter()
                .map(|c| c.body)
                .collect();
            (paths, comments, for_path)
        };
        let forward = build(&["a.rs", "b.rs", "c.rs"], &[1, 2, 3]);
        let reverse = build(&["c.rs", "b.rs", "a.rs"], &[3, 2, 1]);
        assert_eq!(forward, reverse);
        assert_eq!(forward.0, ["a.rs", "b.rs", "c.rs"]);
        // pr_number, then path, then line, then created_at.
        let first_pr: Vec<_> = forward.1.iter().take(4).collect();
        assert!(first_pr.iter().all(|c| c.0 == 1));
        assert_eq!(first_pr[0].1, "a.rs");
        assert_eq!(
            (first_pr[0].2, first_pr[0].3.as_str()),
            (Some(4), "2026-01-01")
        );
        assert_eq!(
            (first_pr[1].2, first_pr[1].3.as_str()),
            (Some(4), "2026-01-03")
        );
        assert_eq!(first_pr[2].2, Some(30));
        assert_eq!(first_pr[3].1, "b.rs");
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

    #[test]
    fn cochange_pair_reads_one_directed_row_regardless_of_rank() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        let mut edges: Vec<CoChange> = (0..10)
            .map(|i| CoChange {
                path_a: "a.rs".to_owned(),
                path_b: format!("strong{i}.rs"),
                count: 50,
                strength: 0.9,
            })
            .collect();
        edges.push(CoChange {
            path_a: "a.rs".to_owned(),
            path_b: "weak.rs".to_owned(),
            count: 2,
            strength: 0.05,
        });
        store.replace_cochange(&edges).unwrap();

        // Ranked far outside any top-N, still found, with its own count.
        let pair = store.cochange_pair("a.rs", "weak.rs").unwrap().unwrap();
        assert_eq!((pair.count, pair.path_b.as_str()), (2, "weak.rs"));
        // The reverse row was never recorded; direction matters.
        assert!(store.cochange_pair("weak.rs", "a.rs").unwrap().is_none());
        assert!(store.cochange_pair("a.rs", "nope.rs").unwrap().is_none());
    }

    #[test]
    fn cochange_fact_count_counts_each_listed_row_once_and_reverse_rows_with_their_forward() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store
            .upsert_files(&["a.rs", "b.rs", "c.rs", "d.rs"].map(file))
            .expect("files are indexed");
        let row = |a: &str, b: &str| CoChange {
            path_a: a.to_owned(),
            path_b: b.to_owned(),
            count: 3,
            strength: 0.5,
        };
        store
            .replace_cochange(&[
                // a <-> b both ways, a -> c forward only, d -> a reverse only,
                // a -> deleted.rs (not indexed), and c <-> d unrelated to a.
                row("a.rs", "b.rs"),
                row("b.rs", "a.rs"),
                row("a.rs", "c.rs"),
                row("d.rs", "a.rs"),
                row("a.rs", "deleted.rs"),
                row("c.rs", "d.rs"),
                row("d.rs", "c.rs"),
            ])
            .unwrap();
        let anchors =
            |names: &[&str]| -> Vec<String> { names.iter().map(|n| (*n).to_owned()).collect() };
        // a: a->b, a->c, and the reverse b->a (its forward exists); d->a has
        // no forward a->d, and deleted.rs is not a file.
        assert_eq!(store.cochange_fact_count(&anchors(&["a.rs"])).unwrap(), 3);
        // a and b together: b->a is both b's forward row and a's reverse row.
        assert_eq!(
            store
                .cochange_fact_count(&anchors(&["a.rs", "b.rs"]))
                .unwrap(),
            3
        );
        // d: d->a, d->c forward; reverse c->d exists with forward d->c.
        assert_eq!(store.cochange_fact_count(&anchors(&["d.rs"])).unwrap(), 3);
        assert_eq!(store.cochange_fact_count(&[]).unwrap(), 0);
    }

    #[test]
    fn cochange_neighbors_are_totally_ordered_limited_and_counted() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        let names = ["a.rs", "m1.rs", "m2.rs", "m3.rs", "m4.rs", "weak.rs"];
        store
            .upsert_files(&names.map(file))
            .expect("files are indexed");
        let row = |b: &str, count: u32, strength: f32| CoChange {
            path_a: "a.rs".to_owned(),
            path_b: b.to_owned(),
            count,
            strength,
        };
        store
            .replace_cochange(&[
                row("m3.rs", 5, 0.5),
                row("m1.rs", 5, 0.5),
                row("m2.rs", 9, 0.5),
                row("weak.rs", 2, 0.1),
                row("m4.rs", 7, 0.7),
                // Not an indexed file: neither returned nor counted.
                row("deleted.rs", 9, 0.99),
            ])
            .unwrap();

        let all = store.cochange_neighbors("a.rs", 100).unwrap();
        let order: Vec<&str> = all.rows.iter().map(|r| r.path_b.as_str()).collect();
        // strength desc, then count desc, then path asc.
        assert_eq!(order, ["m4.rs", "m2.rs", "m1.rs", "m3.rs", "weak.rs"]);
        assert_eq!(all.total, 5);

        let cut = store.cochange_neighbors("a.rs", 2).unwrap();
        let order: Vec<&str> = cut.rows.iter().map(|r| r.path_b.as_str()).collect();
        assert_eq!(order, ["m4.rs", "m2.rs"]);
        assert_eq!(
            cut.total, 5,
            "the limit hides rows but does not hide that they exist"
        );

        let none = store.cochange_neighbors("weak.rs", 10).unwrap();
        assert!(none.rows.is_empty() && none.total == 0);
    }

    fn sized(path: &str, size_bytes: u64) -> FileRecord {
        FileRecord {
            size_bytes,
            ..file(path)
        }
    }

    /// A store with `a.rs` upserted at (size 10, mtime 1000) and extracted.
    fn extracted_store(repo: &TempRepo) -> Store {
        let mut store = repo.store();
        store.upsert_scanned(&[(file("a.rs"), Some(1000))]).unwrap();
        store
            .replace_symbols_for_file("a.rs", &[symbol("a.rs", "old_name")])
            .unwrap();
        assert!(store.is_tier1_done("a.rs"));
        store
    }

    #[test]
    fn upsert_resets_tier1_when_the_size_changes() {
        let repo = TempRepo::new();
        let mut store = extracted_store(&repo);
        store
            .upsert_scanned(&[(sized("a.rs", 11), Some(1000))])
            .unwrap();
        assert!(!store.is_tier1_done("a.rs"));
    }

    #[test]
    fn upsert_resets_tier1_when_only_the_mtime_changes() {
        // A same-size edit (rename a function to a name of equal length) keeps
        // the size identical; only the mtime gives it away.
        let repo = TempRepo::new();
        let mut store = extracted_store(&repo);
        store.upsert_scanned(&[(file("a.rs"), Some(2000))]).unwrap();
        assert!(!store.is_tier1_done("a.rs"));
    }

    #[test]
    fn upsert_keeps_tier1_when_the_fingerprint_is_unchanged() {
        let repo = TempRepo::new();
        let mut store = extracted_store(&repo);
        store.upsert_scanned(&[(file("a.rs"), Some(1000))]).unwrap();
        assert!(store.is_tier1_done("a.rs"));
        // Unknown mtime and unchanged size is not evidence of a change either.
        store.upsert_files(&[file("a.rs")]).unwrap();
        assert!(store.is_tier1_done("a.rs"));
        assert_eq!(store.symbols_matching("old_name", 5).unwrap().len(), 1);
    }

    #[test]
    fn a_pre_fingerprint_database_re_extracts_once() {
        let repo = TempRepo::new();
        drop(extracted_store(&repo));
        {
            // Rewrite the files table the way an older Engram left it: no
            // mtime_ns column, symbols already extracted.
            let conn = Connection::open(repo.0.join(".engram").join("engram.db")).unwrap();
            conn.execute_batch("ALTER TABLE files DROP COLUMN mtime_ns;")
                .unwrap();
        }
        let mut store = repo.store();
        assert!(
            !store.is_tier1_done("a.rs"),
            "rows from before fingerprints existed must not be trusted"
        );
        // After it re-extracts, the migration does not fire again.
        store.upsert_scanned(&[(file("a.rs"), Some(1000))]).unwrap();
        store.replace_symbols_for_file("a.rs", &[]).unwrap();
        drop(store);
        let mut store = repo.store();
        store.upsert_scanned(&[(file("a.rs"), Some(1000))]).unwrap();
        assert!(store.is_tier1_done("a.rs"));
    }

    #[test]
    fn remove_path_deletes_every_trace_including_vectors() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store
            .upsert_files(&[file("gone.rs"), file("kept.rs")])
            .unwrap();
        for p in ["gone.rs", "kept.rs"] {
            store
                .replace_symbols_for_file(p, &[symbol(p, "thing")])
                .unwrap();
            store
                .replace_imports_for_file(p, &["utils/retry".to_owned()])
                .unwrap();
            store
                .upsert_vectors(&[(p.to_owned(), "h".to_owned(), vec![1, 2])])
                .unwrap();
            store
                .upsert_chunk_vectors(&[(p.to_owned(), 1, "h".to_owned(), vec![1, 2])])
                .unwrap();
        }
        // One row in each of the five tables.
        assert_eq!(store.remove_path("gone.rs").unwrap(), 5);

        let count = |table: &str, path: &str| -> i64 {
            store
                .conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM {table} WHERE path = ?1"),
                    params![path],
                    |r| r.get(0),
                )
                .unwrap()
        };
        for table in [
            "files",
            "symbols",
            "file_imports",
            "vectors",
            "chunk_vectors",
        ] {
            assert_eq!(count(table, "gone.rs"), 0, "{table} kept a row");
            assert_eq!(count(table, "kept.rs"), 1, "{table} lost the other file");
        }
    }

    #[test]
    fn remove_path_on_a_directory_takes_the_files_beneath_it() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store
            .upsert_files(&[file("pkg/a.rs"), file("pkg/sub/b.rs"), file("pkgx/c.rs")])
            .unwrap();
        assert_eq!(store.remove_path("pkg").unwrap(), 2);
        let left: Vec<String> = store
            .all_files()
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(left, vec!["pkgx/c.rs".to_owned()]);
    }

    #[test]
    fn remove_path_changes_nothing_for_an_unknown_path_and_spares_lookalikes() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        // Neighbours of "pkg" in sort order: the range bounds must not catch them.
        let lookalikes = ["pkg.rs", "pkg-x/a.rs", "pkg0/a.rs", "pkga/b.rs", "pkg"];
        let files: Vec<FileRecord> = lookalikes.iter().map(|p| file(p)).collect();
        store.upsert_files(&files).unwrap();
        assert_eq!(
            store.remove_path("pkg/").unwrap(),
            1,
            "only the exact `pkg`"
        );
        assert_eq!(store.all_files().unwrap().len(), lookalikes.len() - 1);
        // Nothing left to remove: reports that the index did not change.
        assert_eq!(store.remove_path("pkg").unwrap(), 0);
        assert_eq!(store.remove_path("nope/at/all.rs").unwrap(), 0);
    }

    #[test]
    fn a_failed_parser_version_reset_leaves_the_old_version_recorded() {
        // The wipe and the version bump are one transaction: if the wipe
        // cannot complete, the database must not claim the new version.
        let repo = TempRepo::new();
        {
            let mut store = repo.store();
            store.upsert_files(&[file("a.rs")]).unwrap();
            store
                .replace_symbols_for_file("a.rs", &[symbol("a.rs", "thing")])
                .unwrap();
        }
        let conn = Connection::open(repo.0.join(".engram").join("engram.db")).unwrap();
        // The symbols wipe runs before the statement that fails.
        conn.execute_batch("DROP TABLE vectors;").unwrap();
        assert!(Store::reconcile_parser_version(&conn, "another-fingerprint").is_err());
        let symbols: i64 = conn
            .query_row("SELECT COUNT(*) FROM symbols", [], |r| r.get(0))
            .unwrap();
        assert_eq!(symbols, 1, "the partial wipe must roll back");
        let stored: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![PARSER_VERSION_KEY],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stored, crate::inventory::inventory_rules_fingerprint());
    }

    fn stored_fingerprint(repo: &TempRepo) -> String {
        let conn = Connection::open(repo.0.join(".engram").join("engram.db")).unwrap();
        conn.query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![PARSER_VERSION_KEY],
            |r| r.get(0),
        )
        .unwrap()
    }

    #[test]
    fn a_new_database_records_the_rules_fingerprint_and_reopening_keeps_derived_data() {
        let repo = TempRepo::new();
        {
            let mut store = repo.store();
            store.upsert_files(&[file("a.rs")]).unwrap();
            store
                .replace_symbols_for_file("a.rs", &[symbol("a.rs", "thing")])
                .unwrap();
        }
        assert_eq!(
            stored_fingerprint(&repo),
            crate::inventory::inventory_rules_fingerprint()
        );
        // Same fingerprint on every later open: nothing is wiped (no loop).
        for _ in 0..3 {
            let store = repo.store();
            assert_eq!(store.symbols_for_path("a.rs").unwrap().len(), 1);
            assert!(store.is_tier1_done("a.rs"));
        }
    }

    #[test]
    fn a_database_stored_under_another_fingerprint_is_wiped_once() {
        let repo = TempRepo::new();
        {
            let mut store = repo.store();
            store
                .upsert_fingerprinted(&[hashed("a.rs", 1, "h1")])
                .unwrap();
            store
                .replace_symbols_for_file("a.rs", &[symbol("a.rs", "thing")])
                .unwrap();
            store
                .replace_imports_for_file("a.rs", &["utils/retry".to_owned()])
                .unwrap();
            store
                .upsert_vectors(&[("a.rs".to_owned(), "h".to_owned(), vec![1])])
                .unwrap();
        }
        let before = repo.store().snapshot_id("c1").unwrap();

        // The ignore/extension rules changed: the stored value no longer matches.
        let changed = crate::inventory::rules_fingerprint(&["only-this"], &[], 1, 1);
        assert_ne!(changed, crate::inventory::inventory_rules_fingerprint());
        {
            let store = Store::open_with_rules_fingerprint(&repo.0, &changed).unwrap();
            assert!(!store.is_tier1_done("a.rs"));
            assert!(store.symbols_for_path("a.rs").unwrap().is_empty());
            assert!(store.all_imports().unwrap().is_empty());
            assert!(store.load_vectors().unwrap().is_empty());
            assert_eq!(store.all_files().unwrap().len(), 1, "file rows survive");
            assert_ne!(store.snapshot_id("c1").unwrap(), before, "snapshot id");
        }
        assert_eq!(stored_fingerprint(&repo), changed);

        // Re-extract, then reopen under the same fingerprint: kept, not wiped again.
        {
            let mut store = Store::open_with_rules_fingerprint(&repo.0, &changed).unwrap();
            store
                .replace_symbols_for_file("a.rs", &[symbol("a.rs", "thing")])
                .unwrap();
        }
        let store = Store::open_with_rules_fingerprint(&repo.0, &changed).unwrap();
        assert_eq!(store.symbols_for_path("a.rs").unwrap().len(), 1);
    }

    #[test]
    fn a_second_handle_that_saw_a_stale_mismatch_does_not_wipe_what_the_first_wrote() {
        let repo = TempRepo::new();
        drop(repo.store());
        let changed = crate::inventory::rules_fingerprint(&["race"], &[], 1, 1);
        // Handle B opened first and is about to act on a mismatch it already saw...
        let b = Connection::open(repo.0.join(".engram").join("engram.db")).unwrap();
        assert_ne!(
            Store::stored_parser_version(&b).unwrap().as_deref(),
            Some(changed.as_str())
        );
        // ...but handle A got there first: it wiped, recorded the new
        // fingerprint, and re-extracted a file under it.
        {
            let mut a = Store::open_with_rules_fingerprint(&repo.0, &changed).unwrap();
            a.upsert_files(&[file("a.rs")]).unwrap();
            a.replace_symbols_for_file("a.rs", &[symbol("a.rs", "thing")])
                .unwrap();
        }
        // B takes the lock, re-reads, sees the updated key and leaves the data.
        Store::reset_if_still_mismatched(&b, &changed).unwrap();
        drop(b);
        let store = Store::open_with_rules_fingerprint(&repo.0, &changed).unwrap();
        assert_eq!(store.symbols_for_path("a.rs").unwrap().len(), 1);
        assert!(store.is_tier1_done("a.rs"));
        assert_eq!(stored_fingerprint(&repo), changed);
    }

    #[test]
    fn concurrent_opens_and_writes_from_several_threads_all_succeed() {
        use std::sync::{Arc, Barrier};
        const THREADS: usize = 6;
        let repo = TempRepo::new();
        let root = Arc::new(repo.0.clone());
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let root = Arc::clone(&root);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || -> Result<()> {
                    // All threads race to create and configure the same new
                    // database file, then to write to it.
                    barrier.wait();
                    let mut store = Store::open(&root)?;
                    for i in 0..40 {
                        let path = format!("t{t}/f{i}.rs");
                        store.upsert_files(&[file(&path)])?;
                        store.replace_symbols_for_file(&path, &[symbol(&path, "thing")])?;
                    }
                    Ok(())
                })
            })
            .collect();
        for h in handles {
            h.join()
                .expect("thread panicked")
                .expect("no `database is locked`");
        }
        let store = repo.store();
        assert_eq!(store.all_files().unwrap().len(), THREADS * 40);
    }

    #[test]
    fn a_database_holding_the_old_numeric_version_is_wiped_once() {
        let repo = TempRepo::new();
        {
            let mut store = repo.store();
            store.upsert_files(&[file("a.rs")]).unwrap();
            store
                .replace_symbols_for_file("a.rs", &[symbol("a.rs", "thing")])
                .unwrap();
        }
        let conn = Connection::open(repo.0.join(".engram").join("engram.db")).unwrap();
        conn.execute(
            "UPDATE meta SET value = '1' WHERE key = ?1",
            params![PARSER_VERSION_KEY],
        )
        .unwrap();
        drop(conn);
        let store = repo.store();
        assert!(store.symbols_for_path("a.rs").unwrap().is_empty());
        assert_eq!(
            stored_fingerprint(&repo),
            crate::inventory::inventory_rules_fingerprint()
        );
    }

    fn facts(scanned: u32, bulk: u32, files: &[(&str, u32)]) -> CoChangeFacts {
        CoChangeFacts {
            commits_scanned: scanned,
            bulk_commits_excluded: bulk,
            file_commits: files.iter().map(|(p, n)| ((*p).to_owned(), *n)).collect(),
        }
    }

    #[test]
    fn cochange_facts_round_trip_and_unrecorded_reads_as_unknown_not_zero() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        assert_eq!(store.cochange_totals().unwrap(), None);
        assert_eq!(store.cochange_commits_touching("a.rs").unwrap(), None);

        store
            .replace_cochange_with_facts(&[], &facts(7, 2, &[("a.rs", 5)]))
            .unwrap();
        assert_eq!(store.cochange_totals().unwrap(), Some((7, 2)));
        assert_eq!(store.cochange_commits_touching("a.rs").unwrap(), Some(5));
        assert_eq!(
            store.cochange_commits_touching("untouched.rs").unwrap(),
            Some(0)
        );

        // Rows replaced without facts: the old facts described other rows.
        store.replace_cochange(&[]).unwrap();
        assert_eq!(store.cochange_totals().unwrap(), None);
        assert_eq!(store.cochange_commits_touching("a.rs").unwrap(), None);
    }

    #[test]
    fn a_database_predating_the_cochange_facts_migrates_and_reads_as_unknown() {
        let repo = TempRepo::new();
        {
            let mut store = repo.store();
            store
                .replace_cochange(&[CoChange {
                    path_a: "a.rs".into(),
                    path_b: "b.rs".into(),
                    count: 3,
                    strength: 0.5,
                }])
                .unwrap();
        }
        // Make it look like an older database: no facts table, no totals.
        let conn = Connection::open(repo.0.join(".engram").join("engram.db")).unwrap();
        conn.execute_batch(
            "DROP TABLE cochange_file_commits;
             DELETE FROM meta WHERE key IN ('cochange_commits_scanned', 'cochange_bulk_commits_excluded');",
        )
        .unwrap();
        drop(conn);

        // Opening migrates idempotently, repeatedly, and keeps the old rows.
        for _ in 0..2 {
            let store = repo.store();
            assert_eq!(store.cochange_totals().unwrap(), None);
            assert_eq!(store.cochange_commits_touching("a.rs").unwrap(), None);
            let pair = store.cochange_pair("a.rs", "b.rs").unwrap().unwrap();
            assert_eq!(pair.count, 3);
        }
    }

    fn hashed(path: &str, mtime: i64, hash: &str) -> ScannedFile {
        ScannedFile {
            record: file(path),
            mtime_ns: Some(mtime),
            content_hash: Some(hash.to_owned()),
        }
    }

    #[test]
    fn a_changed_content_hash_alone_invalidates_the_file() {
        // Same size, same mtime: only the hash differs.
        let repo = TempRepo::new();
        let mut store = repo.store();
        store
            .upsert_fingerprinted(&[hashed("a.rs", 5, "h1")])
            .unwrap();
        store
            .replace_symbols_for_file("a.rs", &[symbol("a.rs", "old_name")])
            .unwrap();
        store
            .replace_imports_for_file("a.rs", &["utils/retry".to_owned()])
            .unwrap();
        store
            .upsert_vectors(&[("a.rs".to_owned(), "h".to_owned(), vec![1])])
            .unwrap();
        store
            .upsert_chunk_vectors(&[("a.rs".to_owned(), 1, "h".to_owned(), vec![1])])
            .unwrap();

        // Unchanged hash: everything is kept.
        store
            .upsert_fingerprinted(&[hashed("a.rs", 5, "h1")])
            .unwrap();
        assert!(store.is_tier1_done("a.rs"));
        assert_eq!(store.symbols_for_path("a.rs").unwrap().len(), 1);

        store
            .upsert_fingerprinted(&[hashed("a.rs", 5, "h2")])
            .unwrap();
        assert!(!store.is_tier1_done("a.rs"));
        assert!(store.symbols_for_path("a.rs").unwrap().is_empty());
        assert!(store.all_imports().unwrap().is_empty());
        assert!(store.load_vectors().unwrap().is_empty());
        assert!(store.load_chunk_vectors().unwrap().is_empty());
    }

    #[test]
    fn a_row_with_no_stored_hash_is_re_extracted_once_a_hash_is_known() {
        let repo = TempRepo::new();
        let mut store = extracted_store(&repo);
        store
            .upsert_fingerprinted(&[hashed("a.rs", 1000, "h1")])
            .unwrap();
        assert!(
            !store.is_tier1_done("a.rs"),
            "NULL stored hash vs a known hash means unverified"
        );
    }

    #[test]
    fn snapshot_ids_follow_content_commit_and_version() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store
            .upsert_fingerprinted(&[hashed("a.rs", 1, "h1"), hashed("b.rs", 2, "h2")])
            .unwrap();
        let base = store.snapshot_id("c1").unwrap();
        assert_eq!(base, store.snapshot_id("c1").unwrap());
        assert_ne!(base, store.snapshot_id("c2").unwrap(), "commit");
        // mtime is not part of the identity.
        store
            .upsert_fingerprinted(&[hashed("a.rs", 99, "h1")])
            .unwrap();
        assert_eq!(base, store.snapshot_id("c1").unwrap());
        store
            .upsert_fingerprinted(&[hashed("a.rs", 99, "other")])
            .unwrap();
        assert_ne!(base, store.snapshot_id("c1").unwrap(), "content");
    }

    #[test]
    fn pruning_a_deleted_file_takes_its_legacy_vector_too() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store.upsert_files(&[file("gone.rs")]).unwrap();
        store
            .upsert_vectors(&[("gone.rs".to_owned(), "h".to_owned(), vec![1])])
            .unwrap();
        store.prune_missing_files(&HashSet::new()).unwrap();
        assert!(store.load_vectors().unwrap().is_empty());
    }

    // ----- generations: staged batches published atomically -----

    /// Every row of every table that publish touches, in a canonical order, as
    /// text. Two dumps are equal exactly when the tables are identical.
    fn dump(store: &Store) -> Vec<String> {
        let mut out = Vec::new();
        for table in [
            "files",
            "symbols",
            "file_imports",
            "cochange",
            "cochange_file_commits",
            "meta",
            "vectors",
            "chunk_vectors",
        ] {
            let mut stmt = store
                .conn
                .prepare(&format!("SELECT * FROM {table} ORDER BY rowid"))
                .unwrap();
            let columns = stmt.column_count();
            let rows = stmt
                .query_map([], |r| {
                    let mut cells = Vec::new();
                    for i in 0..columns {
                        let value: rusqlite::types::Value = r.get(i)?;
                        cells.push(format!("{value:?}"));
                    }
                    Ok(cells.join("|"))
                })
                .unwrap();
            for row in rows {
                out.push(format!("{table}: {}", row.unwrap()));
            }
        }
        out
    }

    fn extraction(path: &str, symbol_name: &str, import: &str) -> ExtractionUpdate {
        ExtractionUpdate {
            path: path.to_owned(),
            extraction: Extraction::Extracted {
                symbols: vec![symbol(path, symbol_name)],
                imports: vec![import.to_owned()],
            },
        }
    }

    fn history(partner_edge: bool) -> HistoryUpdate {
        let mut facts = CoChangeFacts {
            commits_scanned: 12,
            bulk_commits_excluded: 1,
            ..Default::default()
        };
        facts.file_commits.insert("a.rs".to_owned(), 4);
        HistoryUpdate {
            edges: if partner_edge {
                vec![CoChange {
                    path_a: "a.rs".to_owned(),
                    path_b: "b.rs".to_owned(),
                    count: 3,
                    strength: 0.75,
                }]
            } else {
                Vec::new()
            },
            facts: Some(facts),
            recency: HashMap::from([("a.rs".to_owned(), 1_700_000_000)]),
        }
    }

    /// Generation 1 of the tests below: a.rs and old.rs, both extracted.
    fn seeded(repo: &TempRepo) -> Store {
        let mut store = repo.store();
        let published = store
            .publish_batch_detailed(IndexBatch {
                files: vec![
                    FileUpdate {
                        scanned: hashed("a.rs", 1, "ha1"),
                        invalidate_vectors: false,
                    },
                    FileUpdate {
                        scanned: hashed("old.rs", 1, "ho1"),
                        invalidate_vectors: false,
                    },
                ],
                extractions: vec![
                    extraction("a.rs", "alpha_old", "dep_old"),
                    extraction("old.rs", "legacy", "dep_legacy"),
                ],
                ..IndexBatch::default()
            })
            .unwrap();
        assert!(published.changed);
        assert_eq!(published.generation, 1);
        store
    }

    #[test]
    fn a_fresh_database_is_generation_zero() {
        let repo = TempRepo::new();
        assert_eq!(repo.store().generation().unwrap(), 0);
        // An empty batch is not a change.
        let mut store = repo.store();
        assert_eq!(store.publish_batch(IndexBatch::default()).unwrap(), 0);
    }

    #[test]
    fn one_batch_applies_every_kind_of_change_and_bumps_the_generation_once() {
        let repo = TempRepo::new();
        let mut store = seeded(&repo);
        store
            .upsert_chunk_vectors(&[("a.rs".to_owned(), 1, "h".to_owned(), vec![1])])
            .unwrap();

        let published = store
            .publish_batch_detailed(IndexBatch {
                // old.rs is gone from the repository, a.rs was edited, b.rs is new.
                retain_only: Some(HashSet::from(["a.rs".to_owned(), "b.rs".to_owned()])),
                removals: vec!["ghost/".to_owned()],
                files: vec![
                    FileUpdate {
                        scanned: hashed("a.rs", 2, "ha2"),
                        invalidate_vectors: false,
                    },
                    FileUpdate {
                        scanned: hashed("b.rs", 2, "hb2"),
                        invalidate_vectors: false,
                    },
                ],
                extractions: vec![
                    extraction("a.rs", "alpha_new", "dep_new"),
                    extraction("b.rs", "beta", "dep_beta"),
                ],
                history: Some(history(true)),
            })
            .unwrap();

        assert_eq!(published.generation, 2, "one batch is one generation");
        assert!(published.changed);
        assert_eq!(published.pruned_files, 1);
        assert_eq!(store.generation().unwrap(), 2);

        let paths: Vec<String> = store
            .all_files()
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(paths, ["a.rs", "b.rs"]);
        let names = |path: &str| -> Vec<String> {
            store
                .symbols_for_path(path)
                .unwrap()
                .into_iter()
                .map(|s| s.name)
                .collect()
        };
        assert_eq!(
            names("a.rs"),
            ["alpha_new"],
            "the edit replaced the symbols"
        );
        assert_eq!(names("b.rs"), ["beta"]);
        assert!(
            names("old.rs").is_empty(),
            "the removed file left no symbols"
        );
        let mut imports = store.all_imports().unwrap();
        imports.sort();
        assert_eq!(
            imports,
            [
                ("a.rs".to_owned(), "dep_new".to_owned()),
                ("b.rs".to_owned(), "dep_beta".to_owned())
            ],
            "stale import edges are gone with the batch, new ones are in"
        );
        assert!(store.is_tier1_done("a.rs") && store.is_tier1_done("b.rs"));
        assert!(
            store.load_chunk_vectors().unwrap().is_empty(),
            "the edit dropped the old spans' cached vectors"
        );
        assert_eq!(store.cochange_totals().unwrap(), Some((12, 1)));
        assert_eq!(
            store.cochange_pair("a.rs", "b.rs").unwrap().unwrap().count,
            3
        );
        assert_eq!(
            store.recency_map().unwrap().get("a.rs"),
            Some(&1_700_000_000)
        );
    }

    #[test]
    fn a_batch_that_changes_nothing_does_not_move_the_generation() {
        let repo = TempRepo::new();
        let mut store = seeded(&repo);
        store
            .publish_batch(IndexBatch {
                history: Some(history(true)),
                ..IndexBatch::default()
            })
            .unwrap();
        let generation = store.generation().unwrap();
        let before = dump(&store);

        let again = IndexBatch {
            retain_only: Some(HashSet::from(["a.rs".to_owned(), "old.rs".to_owned()])),
            removals: vec!["never/there.rs".to_owned()],
            files: vec![
                FileUpdate {
                    scanned: hashed("a.rs", 1, "ha1"),
                    invalidate_vectors: false,
                },
                FileUpdate {
                    scanned: hashed("old.rs", 1, "ho1"),
                    invalidate_vectors: false,
                },
            ],
            // The very same facts again: an identical re-extraction is no change.
            extractions: vec![
                extraction("a.rs", "alpha_old", "dep_old"),
                extraction("old.rs", "legacy", "dep_legacy"),
            ],
            history: Some(history(true)),
        };
        let published = store.publish_batch_detailed(again).unwrap();
        assert!(!published.changed);
        assert_eq!(published.generation, generation);
        assert_eq!(store.generation().unwrap(), generation);
        assert_eq!(dump(&store), before, "nothing was rewritten either");
    }

    #[test]
    fn extracting_a_file_with_nothing_in_it_is_bookkeeping_not_a_new_generation() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        store.upsert_files(&[file("empty.rs")]).unwrap();
        let generation = store.generation().unwrap();
        let published = store
            .publish_batch_detailed(IndexBatch {
                extractions: vec![ExtractionUpdate {
                    path: "empty.rs".to_owned(),
                    extraction: Extraction::Extracted {
                        symbols: Vec::new(),
                        imports: Vec::new(),
                    },
                }],
                ..IndexBatch::default()
            })
            .unwrap();
        assert!(!published.changed);
        assert_eq!(published.generation, generation);
        assert!(
            store.is_tier1_done("empty.rs"),
            "but it is marked extracted"
        );
    }

    #[test]
    fn an_extraction_for_a_file_removed_meanwhile_does_not_resurrect_it() {
        let repo = TempRepo::new();
        let mut store = repo.store();
        // Staged against a file that no longer has a row when published.
        let published = store
            .publish_batch_detailed(IndexBatch {
                extractions: vec![extraction("gone.rs", "ghost", "dep_ghost")],
                ..IndexBatch::default()
            })
            .unwrap();
        assert!(!published.changed);
        assert!(store.symbols_matching("ghost", 5).unwrap().is_empty());
        assert!(store.all_imports().unwrap().is_empty());
    }

    #[test]
    fn a_batch_that_fails_midway_leaves_the_previous_generation_exactly_as_it_was() {
        let repo = TempRepo::new();
        let mut store = seeded(&repo);
        store
            .publish_batch(IndexBatch {
                history: Some(history(true)),
                ..IndexBatch::default()
            })
            .unwrap();
        store
            .upsert_chunk_vectors(&[("a.rs".to_owned(), 1, "h".to_owned(), vec![1])])
            .unwrap();
        let generation = store.generation().unwrap();
        let before = dump(&store);

        // The LAST thing a batch writes is the co-change completeness facts;
        // make that insert fail, after every other part has been applied.
        store
            .conn
            .execute_batch(
                "CREATE TRIGGER fail_the_last_step BEFORE INSERT ON cochange_file_commits
                 BEGIN SELECT RAISE(ABORT, 'injected failure'); END;",
            )
            .unwrap();
        let batch = IndexBatch {
            retain_only: Some(HashSet::from(["a.rs".to_owned(), "b.rs".to_owned()])),
            removals: vec!["x/".to_owned()],
            files: vec![
                FileUpdate {
                    scanned: hashed("a.rs", 2, "ha2"),
                    invalidate_vectors: true,
                },
                FileUpdate {
                    scanned: hashed("b.rs", 2, "hb2"),
                    invalidate_vectors: false,
                },
            ],
            extractions: vec![
                extraction("a.rs", "alpha_new", "dep_new"),
                extraction("b.rs", "beta", "dep_beta"),
            ],
            history: Some(history(false)),
        };
        let err = store.publish_batch(batch.clone()).unwrap_err();
        assert!(err.to_string().contains("injected failure"), "{err:#}");

        assert_eq!(store.generation().unwrap(), generation);
        assert_eq!(
            dump(&store),
            before,
            "files, symbols, imports, co-change and meta must all be untouched"
        );

        // The same batch is fine once the fault is gone: it was the fault, not
        // the batch, and the connection is usable afterwards.
        store
            .conn
            .execute_batch("DROP TRIGGER fail_the_last_step;")
            .unwrap();
        assert_eq!(store.publish_batch(batch).unwrap(), generation + 1);
    }

    #[test]
    fn a_reader_in_a_read_view_sees_one_generation_until_it_ends() {
        let repo = TempRepo::new();
        let mut publisher = seeded(&repo);
        let reader = repo.store();

        let view = reader.begin_read().unwrap();
        assert_eq!(view.generation(), 1);

        publisher
            .publish_batch(IndexBatch {
                retain_only: Some(HashSet::from(["a.rs".to_owned()])),
                files: vec![FileUpdate {
                    scanned: hashed("a.rs", 2, "ha2"),
                    invalidate_vectors: false,
                }],
                extractions: vec![extraction("a.rs", "alpha_new", "dep_new")],
                ..IndexBatch::default()
            })
            .unwrap();
        assert_eq!(publisher.generation().unwrap(), 2);

        // Inside the view: all old, none of the new, files included.
        assert_eq!(reader.generation().unwrap(), 1);
        let names: Vec<String> = reader
            .symbols_for_path("a.rs")
            .unwrap()
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(names, ["alpha_old"]);
        let mut imports = reader.all_imports().unwrap();
        imports.sort();
        assert_eq!(
            imports,
            [
                ("a.rs".to_owned(), "dep_old".to_owned()),
                ("old.rs".to_owned(), "dep_legacy".to_owned())
            ]
        );
        assert_eq!(reader.all_files().unwrap().len(), 2, "old.rs still exists");

        // Ending the view ends the isolation: now it is all new.
        drop(view);
        assert_eq!(reader.generation().unwrap(), 2);
        let names: Vec<String> = reader
            .symbols_for_path("a.rs")
            .unwrap()
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(names, ["alpha_new"]);
        assert_eq!(
            reader.all_imports().unwrap(),
            [("a.rs".to_owned(), "dep_new".to_owned())]
        );
        assert_eq!(reader.all_files().unwrap().len(), 1);
    }

    #[test]
    fn a_reader_never_observes_new_symbols_with_old_imports() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let repo = TempRepo::new();
        drop(seeded(&repo));
        let root = repo.0.clone();
        let done = Arc::new(AtomicBool::new(false));

        let writer = {
            let (root, done) = (root.clone(), done.clone());
            std::thread::spawn(move || {
                let mut store = Store::open(&root).unwrap();
                for k in 2..=40_i64 {
                    store
                        .publish_batch(IndexBatch {
                            files: vec![FileUpdate {
                                scanned: hashed("a.rs", k, &format!("hash{k}")),
                                invalidate_vectors: false,
                            }],
                            extractions: vec![extraction(
                                "a.rs",
                                &format!("sym_{k}"),
                                &format!("dep_{k}"),
                            )],
                            ..IndexBatch::default()
                        })
                        .unwrap();
                }
                done.store(true, Ordering::SeqCst);
            })
        };

        let reader = Store::open(&root).unwrap();
        let mut observed = 0;
        loop {
            let finished = done.load(Ordering::SeqCst);
            let view = reader.begin_read().unwrap();
            let symbols = reader.symbols_for_path("a.rs").unwrap();
            let imports: Vec<_> = reader
                .all_imports()
                .unwrap()
                .into_iter()
                .filter(|(p, _)| p == "a.rs")
                .collect();
            drop(view);
            // Both halves of one extraction (`sym_7` with `dep_7`, or the
            // seeded `alpha_old` with `dep_old`): never one of each.
            assert_eq!(symbols.len(), 1, "{symbols:?}");
            assert_eq!(imports.len(), 1, "{imports:?}");
            let sym_tag = symbols[0].name.rsplit('_').next().unwrap();
            let dep_tag = imports[0].1.rsplit('_').next().unwrap();
            assert_eq!(
                sym_tag, dep_tag,
                "symbols {symbols:?} with imports {imports:?}"
            );
            observed += 1;
            if finished {
                break;
            }
        }
        writer.join().unwrap();
        assert!(observed > 0);
        assert_eq!(reader.generation().unwrap(), 40);
    }

    #[test]
    fn the_public_writers_keep_the_generation_honest() {
        // The thin wrappers older callers (and tests) use bump when, and only
        // when, they change something.
        let repo = TempRepo::new();
        let mut store = repo.store();
        let mut expect = 0;
        let mut step = |store: &Store, moved: bool, what: &str| {
            if moved {
                expect += 1;
            }
            assert_eq!(store.generation().unwrap(), expect, "{what}");
        };
        store.upsert_files(&[file("a.rs"), file("b.rs")]).unwrap();
        step(&store, true, "new files");
        store.upsert_files(&[file("a.rs"), file("b.rs")]).unwrap();
        step(&store, false, "the same files again");
        store
            .replace_symbols_for_file("a.rs", &[symbol("a.rs", "x")])
            .unwrap();
        step(&store, true, "new symbols");
        store
            .replace_symbols_for_file("a.rs", &[symbol("a.rs", "x")])
            .unwrap();
        step(&store, false, "the same symbols again");
        store
            .replace_imports_for_file("a.rs", &["dep".to_owned()])
            .unwrap();
        step(&store, true, "new imports");
        store
            .replace_imports_for_file("a.rs", &["dep".to_owned()])
            .unwrap();
        step(&store, false, "the same imports again");
        let edge = CoChange {
            path_a: "a.rs".to_owned(),
            path_b: "b.rs".to_owned(),
            count: 2,
            strength: 0.5,
        };
        store.replace_cochange(std::slice::from_ref(&edge)).unwrap();
        step(&store, true, "new co-change rows");
        store.replace_cochange(std::slice::from_ref(&edge)).unwrap();
        step(&store, false, "the same co-change rows again");
        store
            .update_recency(&HashMap::from([("a.rs".to_owned(), 5)]))
            .unwrap();
        step(&store, true, "new recency");
        store
            .update_recency(&HashMap::from([("a.rs".to_owned(), 5)]))
            .unwrap();
        step(&store, false, "the same recency again");
        store.remove_path("b.rs").unwrap();
        step(&store, true, "a removal");
        store.remove_path("b.rs").unwrap();
        step(&store, false, "removing nothing");
        store.invalidate_vector("a.rs").unwrap();
        step(
            &store,
            false,
            "dropping cache rows is not a change of facts",
        );
    }

    #[test]
    fn the_generation_is_durable_and_a_rules_change_is_a_generation_of_its_own() {
        let repo = TempRepo::new();
        drop(seeded(&repo));
        assert_eq!(repo.store().generation().unwrap(), 1, "survives a reopen");

        let next = crate::inventory::rules_fingerprint(&["other"], &[], 0, PARSER_CONFIG_VERSION);
        let store = Store::open_with_rules_fingerprint(&repo.0, &next).unwrap();
        assert_eq!(
            store.generation().unwrap(),
            2,
            "discarding every symbol and import changes what readers see"
        );
        assert!(store.all_imports().unwrap().is_empty());
        // Reopening under the same rules discards nothing and moves nothing.
        drop(store);
        let again = Store::open_with_rules_fingerprint(&repo.0, &next).unwrap();
        assert_eq!(again.generation().unwrap(), 2);
    }

    #[test]
    fn pending_extraction_names_exactly_what_a_publish_would_leave_unextracted() {
        let repo = TempRepo::new();
        let mut store = seeded(&repo);
        let unchanged = hashed("a.rs", 1, "ha1");
        let edited = hashed("old.rs", 9, "ho9");
        let added = hashed("new.rs", 1, "hn1");
        let pending = store
            .pending_extraction(&[unchanged.clone(), edited, added])
            .unwrap();
        assert_eq!(
            pending,
            HashSet::from(["old.rs".to_owned(), "new.rs".to_owned()])
        );
        // A row that was never extracted is pending even when unchanged.
        store.clear_extracted("a.rs").unwrap();
        assert!(store
            .pending_extraction(&[unchanged])
            .unwrap()
            .contains("a.rs"));
    }
}
