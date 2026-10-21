//! engram-retrieval: hybrid BM25 (Tantivy) + vector (cosine) retrieval with
//! score fusion, lazy Tier-1 extraction on miss, and co-change impact expansion.

pub mod budget;
pub mod embed;
pub mod stopwords;
pub mod weights;

use anyhow::Result;
use embed::{cosine, Embedder, HashedNgramEmbedder};
use engram_domain::{
    cochange_edge_id, import_edge_id, CoChangeDirection, CoChangeWindow, ConnectionExplanation,
    ConnectionReason, ConnectionScope, ConnectionStatus, DroppedCandidate, EvidenceKind,
    EvidencePacket, GraphEdge, GraphFacts, GraphLimits, GraphNode, GraphNodeRole, GraphScope,
    ImpactPrediction, ImportDirection, ImportEdge, ImportSearchOutcome, Language, ScoreComponents,
    ScoredPath, SymbolRecord,
};
use engram_repo_map::graph::{CodeGraph, EdgeKind, ImportSearch};
use engram_repo_map::store::Store;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, STORED, TEXT};
use tantivy::{doc, Index, IndexReader};
use weights::{classify_path, recency_boost, DocClass, WeightsConfig};

pub use weights::Weights;

/// Ranking strategy, used by the benchmark harness to compare hybrid retrieval
/// against isolated-signal baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankMode {
    /// Full fusion (bm25 + vector + symbol + path + recency + demotions).
    Hybrid,
    /// Tantivy BM25 lexical only.
    Bm25,
    /// Hashed-ngram cosine only.
    Vector,
    /// Deterministic pseudo-random order (a floor baseline).
    Random,
}

struct DocMeta {
    path: String,
    is_test: bool,
    preview: String,
    last_commit_ts: Option<i64>,
}

/// One embedded slice of a file.
///
/// Embedding whole files means embedding a head slice, because files are far
/// longer than anything a single vector can represent. That leaves a function
/// at line 900 with no vector at all. A chunk is instead one symbol's source
/// span, so the vector side can match a definition rather than the file that
/// happens to contain it.
struct Chunk {
    /// Index into `Engine::docs` of the file this span came from.
    doc: usize,
    /// `None` for the whole-file head chunk, which carries imports and module
    /// docs that no symbol span covers.
    symbol: Option<String>,
    /// One-based; `0` for the head chunk. Doubles as the cache key.
    start_line: usize,
    /// Last line of the symbol's span, inclusive; `0` for the head chunk.
    end_line: usize,
    /// The span's source text, capped at [`CHUNK_BYTES`]. Returned as the
    /// snippet, so evidence quotes the matching definition itself.
    text: String,
    vector: Vec<f32>,
}

/// Everything [`Engine::build`] takes from the store, read in one transaction
/// so all of it belongs to `generation`.
struct StoreFacts {
    generation: u64,
    files: Vec<engram_domain::FileRecord>,
    symbols_by_path: HashMap<String, Vec<SymbolRecord>>,
    hashes: HashMap<String, Option<String>>,
    recency: HashMap<String, i64>,
    cached: HashMap<engram_repo_map::store::ChunkKey, (String, Vec<u8>)>,
    snapshot: String,
    graph: CodeGraph,
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub struct Engine {
    repo_root: PathBuf,
    index: Index,
    reader: IndexReader,
    f_path: Field,
    f_body: Field,
    docs: Vec<DocMeta>,
    chunks: Vec<Chunk>,
    by_path: HashMap<String, usize>,
    embedder: HashedNgramEmbedder,
    config: WeightsConfig,
    graph: CodeGraph,
    stopwords: std::sync::Arc<HashSet<String>>,
    /// Set when lazy Tier-1 extraction wrote symbols or imports that this
    /// engine's chunks and code graph, built before the write, do not reflect.
    stale: std::sync::atomic::AtomicBool,
    /// Identity of the indexed content this engine was built from; see
    /// `Store::snapshot_id`. Constant for the engine's life.
    snapshot: String,
    /// The store generation (`Store::generation`) every fact this engine holds
    /// was read at: all of them inside ONE read transaction, so none comes from
    /// a different generation. Constant for the engine's life. Unlike the
    /// snapshot id this is a publish counter, not a content identity.
    built_generation: u64,
    /// The newest generation this engine accounts for: `built_generation`, plus
    /// each publish made by its own lazy extraction (exactly one generation
    /// each, noticed as such). A store generation beyond it was published by
    /// somebody else, and live store reads are then not trusted.
    accepted_generation: std::sync::atomic::AtomicU64,
    /// How many indexed files had changed on disk since the generation they
    /// were indexed at, when this engine was built. Their symbols are left out
    /// (they describe other text, so their spans would point at the wrong
    /// lines); a later publish of the watcher's refresh replaces the engine.
    files_ahead_of_index: usize,
}

/// Longest import chain the import graph is searched for, shared by the
/// expansion path and `explain_connection`.
const MAX_IMPORT_HOPS: usize = 2;
/// Co-change rows kept per anchor file.
const COCHANGE_PER_SEED: usize = 5;
/// Entries kept per expansion kind (co-change, imports).
const EXPANSION_CAP: usize = 8;
/// Rows read per anchor so the ones past the per-anchor cap can be counted.
/// Far above any real fan-out; SQLite's LIMIT is applied in the query.
/// A hub with more rows than this is undercounted: the rows past the limit are
/// neither ranked nor counted in `pack.omitted_count`. The rows that are read
/// are the strongest ones (`cochange_for` orders by strength), but ties at the
/// cutoff are not tie-broken by the query.
const COCHANGE_SCAN_LIMIT: usize = 100_000;
/// Import edges come from matching module names against file paths, not from a
/// compiler, so every import fact is labelled with how it was resolved.
const IMPORT_RESOLUTION: &str = "heuristic";
const IMPORT_RESOLVER: &str = "heuristic_module_segments";

/// Hard bounds of [`Engine::graph_facts`] (`get_graph`); the `max_nodes` and
/// `max_edges` arguments can only lower them.
pub const GRAPH_MAX_NODES: usize = 200;
pub const GRAPH_MAX_EDGES: usize = 500;
/// Co-change partners read per anchor. Well above the 5 per anchor the
/// recommendations keep, and ordered the same way, so every co-change edge a
/// recommendation cites is inside it.
pub const GRAPH_COCHANGE_PER_ANCHOR: usize = 25;

/// The bounds of one [`Engine::graph_facts`] call. The defaults are the hard
/// ones `get_graph` serves; `get_graph` and the tests can lower them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphCaps {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub cochange_per_anchor: usize,
}

impl Default for GraphCaps {
    fn default() -> Self {
        GraphCaps {
            max_nodes: GRAPH_MAX_NODES,
            max_edges: GRAPH_MAX_EDGES,
            cochange_per_anchor: GRAPH_COCHANGE_PER_ANCHOR,
        }
    }
}

/// Which kinds of edge [`Engine::graph_facts`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphKinds {
    pub import: bool,
    pub cochange: bool,
}

impl Default for GraphKinds {
    fn default() -> Self {
        GraphKinds {
            import: true,
            cochange: true,
        }
    }
}

/// Whole-number percentage of a co-change row, for prose. With the
/// denominator known it is computed in integers from the two counts, rounded to
/// nearest, so it can never disagree with "N of M" through float truncation;
/// otherwise it is the stored strength, rounded.
fn cochange_percent(count: u32, opportunities: Option<u32>, strength: f32) -> u32 {
    match opportunities {
        Some(m) if m > 0 => {
            let (count, m) = (u64::from(count), u64::from(m));
            ((count * 100 + m / 2) / m) as u32
        }
        _ => (strength * 100.0).round() as u32,
    }
}

/// The co-change table's build limits, as every co-change fact reports them.
fn cochange_window() -> CoChangeWindow {
    CoChangeWindow {
        max_commits: engram_repo_map::cochange::MAX_COMMITS,
        min_count: engram_repo_map::cochange::MIN_PAIR_COUNT,
        max_files_per_commit: engram_repo_map::cochange::MAX_FILES_PER_COMMIT,
    }
}

const PREVIEW_BYTES: usize = 400;
const INDEX_BODY_BYTES: usize = 60_000;
/// Cap on the source text embedded and quoted for one chunk. Long enough for a
/// normal definition, short enough that one sprawling function cannot dominate.
const CHUNK_BYTES: usize = 4_000;

/// Reuse a cached embedding when its content hash still matches, otherwise
/// embed fresh. Returns the vector and whether it came from cache.
fn embed_or_reuse(
    embedder: &HashedNgramEmbedder,
    cached: Option<&(String, Vec<u8>)>,
    text: &str,
    hash: &str,
) -> (Vec<f32>, bool) {
    if let Some((cached_hash, bytes)) = cached {
        if cached_hash == hash {
            if let Some(vector) = embed::bytes_to_vector(bytes) {
                return (vector, true);
            }
        }
    }
    (embedder.embed(text), false)
}

/// Whole identifiers in a query, underscores and dots preserved, lowercased.
///
/// The fusion tokenizer splits on every non-alphanumeric, so `merge_dicts`
/// and `model.name` fragment into pieces and the identifier a bug report names
/// is never searched as a name. This keeps them whole for exact symbol lookup.
/// A dotted path also yields its last segment, since `utils.merge_dicts` names
/// the symbol `merge_dicts`.
fn query_identifiers(query: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for raw in query.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')) {
        for candidate in [raw, raw.rsplit('.').next().unwrap_or(raw)] {
            let ident = candidate.trim_matches('.').to_lowercase();
            // Require an underscore or mixed case to look identifier-shaped and
            // be at least 3 chars; a bare English word is not a symbol lookup.
            let identifier_shaped =
                ident.contains('_') || candidate.chars().any(|c| c.is_uppercase());
            if ident.len() >= 3 && identifier_shaped && seen.insert(ident.clone()) {
                out.push(ident);
            }
        }
    }
    out
}

/// Plan a file's chunks: the head, then one span per extracted symbol.
///
/// The head chunk always exists. A file with no chunks could never be returned
/// by vector search, and symbol extraction is lazy, so plenty of files have no
/// symbols yet on any given build.
fn plan_chunks(
    source: &str,
    body: &str,
    symbols: &[SymbolRecord],
) -> Vec<(usize, Option<String>, String)> {
    let mut planned = vec![(0usize, None, body.chars().take(CHUNK_BYTES).collect())];
    let lines: Vec<&str> = source.lines().collect();
    for symbol in symbols {
        // Spans from a pre-span database are zero, and a file can shrink
        // between extraction and this build.
        if symbol.start_line == 0
            || symbol.end_line < symbol.start_line
            || symbol.start_line > lines.len()
        {
            continue;
        }
        let end = symbol.end_line.min(lines.len());
        let text: String = lines[symbol.start_line - 1..end]
            .join("\n")
            .chars()
            .take(CHUNK_BYTES)
            .collect();
        planned.push((symbol.start_line, Some(symbol.name.clone()), text));
    }
    planned
}

impl Engine {
    /// Build the in-memory hybrid index from the current repo + store.
    ///
    /// Embeddings are cached in SQLite keyed by a content hash: unchanged files
    /// reuse their stored vector instead of re-embedding, so large repos don't
    /// pay the embedding cost on every server start.
    ///
    /// Every fact is read from the store inside ONE read transaction (a WAL
    /// snapshot) before any of the slow work starts: the file list, symbols,
    /// import edges, co-change edges, recency, hashes, snapshot id and cached
    /// vectors all come from the same generation, recorded as
    /// [`Engine::generation`], however many batches are published meanwhile.
    /// Embeddings computed here are cache written back afterwards; they are not
    /// part of the generation.
    pub fn build(repo_root: &Path, store: &mut Store) -> Result<Engine> {
        let StoreFacts {
            generation: built_generation,
            files,
            mut symbols_by_path,
            hashes,
            recency,
            cached,
            snapshot,
            graph,
        } = {
            // The commit is asked of the repository (a subprocess) BEFORE the
            // read view opens: a view is a read snapshot that pins the write-
            // ahead log, and should be held for database reads only.
            let head = engram_repo_map::head_commit(repo_root);
            let view = store.begin_read()?;
            let facts = StoreFacts {
                generation: view.generation(),
                files: store.all_files()?,
                symbols_by_path: store.all_symbols_by_path()?,
                hashes: store.file_hashes()?,
                recency: store.recency_map()?,
                cached: store.load_chunk_vectors()?,
                snapshot: store.snapshot_id(&head)?,
                graph: CodeGraph::build(store, repo_root)?,
            };
            drop(view);
            facts
        };
        let mut files_ahead_of_index = 0usize;

        let mut schema_builder = Schema::builder();
        let f_path = schema_builder.add_text_field("path", TEXT | STORED);
        let f_body = schema_builder.add_text_field("body", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        // One indexing thread, so the index is a single segment whose documents
        // are in the order added (path order, see `Store::all_files`). Tantivy
        // breaks score ties by document address, so with several writer threads
        // the address of a document, and with it the BM25 top-k cut among equal
        // scores, would depend on thread scheduling.
        let mut writer = index.writer_with_num_threads(1, 64_000_000)?;

        let stopwords = std::sync::Arc::new(stopwords::load(repo_root));
        let embedder = HashedNgramEmbedder::new(stopwords.clone());
        let mut docs = Vec::new();
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut by_path = HashMap::new();
        let mut fresh: Vec<(String, usize, String, Vec<u8>)> = Vec::new();
        let mut present: HashSet<(String, usize)> = HashSet::new();
        let (mut reused, mut embedded) = (0usize, 0usize);

        for f in files {
            let full = repo_root.join(&f.path);
            let Ok(src) = std::fs::read_to_string(&full) else {
                continue;
            };
            let body: String = src.chars().take(INDEX_BODY_BYTES).collect();
            let doc_index = docs.len();

            // Symbols are extracted lazily, so a file may legitimately have
            // none yet; plan_chunks always yields at least the head chunk.
            let mut symbols = symbols_by_path.remove(&f.path).unwrap_or_default();
            // The symbols' line spans describe the text they were extracted
            // from. If the file on disk is no longer that text (an edit the
            // watcher has not published yet), cutting spans out of the new text
            // with the old line numbers would quote the wrong lines; leave the
            // symbols out until the refresh is published and this engine is
            // replaced. A row with no recorded hash cannot be checked.
            if let Some(Some(stored)) = hashes.get(&f.path) {
                if *stored != engram_repo_map::inventory::hash_bytes(src.as_bytes()) {
                    // Only a file whose derived facts the pending refresh will
                    // replace is "ahead": it has symbols extracted from the
                    // older bytes, or a language that is extracted at all. A
                    // Markdown or JSON file has no symbols or imports, so its
                    // edit leaves nothing stale and says nothing.
                    if !symbols.is_empty() || engram_repo_map::symbols::supports(f.language) {
                        files_ahead_of_index += 1;
                    }
                    symbols.clear();
                }
            }
            for (start_line, symbol, text) in plan_chunks(&src, &body, &symbols) {
                let embed_text = match &symbol {
                    Some(name) => format!("{} {} {}", f.path, name, text),
                    None => format!("{} {}", f.path, text),
                };
                let hash = embed::content_hash(&embed_text);
                let key = (f.path.clone(), start_line);
                let (vector, from_cache) =
                    embed_or_reuse(&embedder, cached.get(&key), &embed_text, &hash);
                if from_cache {
                    reused += 1;
                } else {
                    embedded += 1;
                    fresh.push((
                        f.path.clone(),
                        start_line,
                        hash,
                        embed::vector_to_bytes(&vector),
                    ));
                }
                present.insert(key);
                let end_line = symbol
                    .as_deref()
                    .and_then(|name| {
                        symbols
                            .iter()
                            .find(|s| s.start_line == start_line && s.name == name)
                    })
                    .map_or(0, |s| s.end_line);
                chunks.push(Chunk {
                    doc: doc_index,
                    symbol,
                    start_line,
                    end_line,
                    text,
                    vector,
                });
            }

            writer.add_document(doc!(f_path => f.path.clone(), f_body => body.clone()))?;
            by_path.insert(f.path.clone(), doc_index);
            docs.push(DocMeta {
                preview: body.chars().take(PREVIEW_BYTES).collect(),
                last_commit_ts: recency.get(&f.path).copied(),
                path: f.path,
                is_test: f.is_test,
            });
        }
        writer.commit()?;
        if !fresh.is_empty() {
            store.upsert_chunk_vectors(&fresh)?;
        }
        store.prune_chunk_vectors(&present)?;
        // The head chunk superseded the per-file vector; drop the legacy rows
        // rather than leave a table that nothing reads slowly going stale.
        store.prune_vectors(&HashSet::new())?;
        eprintln!(
            "[engram] chunks: {} across {} files ({reused} reused, {embedded} embedded)",
            chunks.len(),
            docs.len()
        );
        if files_ahead_of_index > 0 {
            eprintln!(
                "[engram] {files_ahead_of_index} file(s) changed on disk since they were indexed; \
                 their symbols are left out until the refresh is published"
            );
        }
        eprintln!(
            "[engram] code graph: {} nodes, {} edges",
            graph.node_count(),
            graph.edge_count()
        );
        let reader = index.reader()?;

        Ok(Engine {
            repo_root: repo_root.to_path_buf(),
            index,
            reader,
            f_path,
            f_body,
            docs,
            chunks,
            by_path,
            embedder,
            config: WeightsConfig::load(repo_root),
            graph,
            stopwords,
            stale: std::sync::atomic::AtomicBool::new(false),
            snapshot,
            built_generation,
            accepted_generation: std::sync::atomic::AtomicU64::new(built_generation),
            files_ahead_of_index,
        })
    }

    /// Identity of the indexed content (commit, parser version, and every
    /// file's path and content hash) this engine was built from.
    pub fn snapshot_id(&self) -> &str {
        &self.snapshot
    }

    /// The store generation this engine's facts were all read at. A publish
    /// counter, not a content identity: see [`Engine::snapshot_id`] for that.
    pub fn generation(&self) -> u64 {
        self.built_generation
    }

    /// Indexed files that, when this engine was built, no longer matched the
    /// text their symbols were extracted from (edited, refresh not published
    /// yet). Their symbols are left out of this engine.
    pub fn files_ahead_of_index(&self) -> usize {
        self.files_ahead_of_index
    }

    /// Whether this engine is known to be out of date. Latched: it turns true
    /// when lazy Tier-1 extraction wrote symbols or imports this engine never
    /// saw, or when [`Engine::check_generation`] (also called by every query
    /// path that reads the store live) saw the store publish a newer
    /// generation, and never turns false again. The owner should discard the
    /// engine and rebuild it before the next query. For a check against the
    /// store as it is right now use [`Engine::is_stale_in`].
    pub fn is_stale(&self) -> bool {
        self.stale.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// [`Engine::is_stale`], also true when the store's generation is not the
    /// one this engine was built at (one cheap single-row read).
    pub fn is_stale_in(&self, store: &Store) -> bool {
        self.is_stale() || self.check_generation(store)
    }

    /// Compare the store's generation with the one this engine was built at,
    /// latching the stale flag on a difference. Returns whether they differ.
    /// A failed read changes nothing and reports "not moved".
    pub fn check_generation(&self, store: &Store) -> bool {
        match store.generation() {
            Ok(current) if current != self.built_generation => {
                self.mark_stale();
                true
            }
            _ => false,
        }
    }

    fn mark_stale(&self) {
        self.stale.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn accepted(&self) -> u64 {
        self.accepted_generation
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Whether the store holds nothing this engine does not account for: its
    /// generation is the one the engine was built at, plus only the publishes
    /// the engine made itself. A failed read counts as accounted for (every
    /// other read would fail the same way).
    fn accounts_for_store(&self, store: &Store) -> bool {
        store
            .generation()
            .map_or(true, |current| current == self.accepted())
    }

    /// Called right after this engine's own lazy extraction published. If the
    /// store is exactly one generation past what was accounted for, that
    /// generation is ours; if it is further (somebody else published around
    /// it), it stays unaccounted and live reads stay off until a rebuild.
    fn accept_own_publish(&self, store: &Store) {
        let before = self.accepted();
        if let Ok(now) = store.generation() {
            if now == before + 1 {
                self.accepted_generation
                    .store(now, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }

    /// Run live store reads against one consistent generation. The reads see a
    /// single snapshot, so they cannot mix two generations among themselves; if
    /// that snapshot is not the generation this engine was built at, the engine
    /// is latched stale, so the response built from them says so instead of
    /// silently combining this engine's facts with newer store facts.
    fn read_live<T>(&self, store: &Store, read: impl FnOnce(&Store) -> Result<T>) -> Result<T> {
        let view = store.begin_read()?;
        if view.generation() != self.built_generation {
            self.mark_stale();
        }
        let out = read(store);
        drop(view);
        out
    }

    /// Rank file paths for a query under a given strategy (top `k`). Used by the
    /// benchmark harness to compare hybrid retrieval against baselines.
    pub fn rank(
        &mut self,
        store: &mut Store,
        query: &str,
        mode: RankMode,
        k: usize,
    ) -> Result<Vec<String>> {
        let paths = match mode {
            RankMode::Hybrid => {
                let mut seen = HashSet::new();
                self.search(store, query, k)?
                    .into_iter()
                    .map(|p| p.path)
                    .filter(|p| seen.insert(p.clone()))
                    .take(k)
                    .collect()
            }
            RankMode::Bm25 => self
                .bm25_candidates(query, k)
                .into_iter()
                .map(|(i, _)| self.docs[i].path.clone())
                .collect(),
            RankMode::Vector => self
                .vector_candidates(query, k)
                .into_iter()
                .map(|(i, _)| self.docs[i].path.clone())
                .collect(),
            RankMode::Random => {
                let mut idx: Vec<usize> = (0..self.docs.len()).collect();
                // Deterministic pseudo-random: hash of (query, path).
                idx.sort_by_key(|&i| embed::content_hash(&format!("{query}{}", self.docs[i].path)));
                idx.into_iter()
                    .take(k)
                    .map(|i| self.docs[i].path.clone())
                    .collect()
            }
        };
        Ok(paths)
    }

    fn bm25_candidates(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let parser = QueryParser::for_index(&self.index, vec![self.f_path, self.f_body]);
        // lenient: drops tokens that fail to parse instead of erroring
        let (q, _errors) = parser.parse_query_lenient(query);
        let searcher = self.reader.searcher();
        let Ok(top) = searcher.search(&q, &TopDocs::with_limit(k)) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (score, addr) in top {
            if let Ok(doc) = searcher.doc::<tantivy::TantivyDocument>(addr) {
                use tantivy::schema::Value;
                if let Some(path) = doc.get_first(self.f_path).and_then(|v| v.as_str()) {
                    if let Some(&i) = self.by_path.get(path) {
                        out.push((i, score));
                    }
                }
            }
        }
        out
    }

    /// Score every chunk, then max-pool to the file that owns it: a file is as
    /// relevant as its single best-matching definition, not as its average.
    /// Returns `(doc, score, best chunk)` so callers can quote the span that
    /// actually matched.
    fn vector_candidates_with_chunk(&self, query: &str, k: usize) -> Vec<(usize, f32, usize)> {
        let qv = self.embedder.embed(query);
        let mut best: HashMap<usize, (f32, usize)> = HashMap::new();
        for (index, chunk) in self.chunks.iter().enumerate() {
            let score = cosine(&qv, &chunk.vector);
            match best.get_mut(&chunk.doc) {
                Some(slot) if slot.0 >= score => {}
                Some(slot) => *slot = (score, index),
                None => {
                    best.insert(chunk.doc, (score, index));
                }
            }
        }
        let mut scored: Vec<(usize, f32, usize)> = best
            .into_iter()
            .map(|(doc, (score, chunk))| (doc, score, chunk))
            .collect();
        // Total order for the top-k cut: score desc, then path, then the best
        // chunk's start line and index. Never the doc index or HashMap order,
        // so the cut does not depend on the order files were indexed in.
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.docs[a.0].path.cmp(&self.docs[b.0].path))
                .then_with(|| {
                    self.chunks[a.2]
                        .start_line
                        .cmp(&self.chunks[b.2].start_line)
                })
                .then_with(|| a.2.cmp(&b.2))
        });
        scored.truncate(k);
        scored
    }

    fn vector_candidates(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        self.vector_candidates_with_chunk(query, k)
            .into_iter()
            .map(|(doc, score, _)| (doc, score))
            .collect()
    }

    /// Hybrid search: BM25 + vector fused with weighted normalized scores,
    /// plus symbol exact-match boost. Lazily Tier-1 extracts top hits on miss.
    pub fn search(
        &mut self,
        store: &mut Store,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<EvidencePacket>> {
        Ok(self.search_with_outcome(store, query, top_k)?.packets)
    }

    /// [`Engine::search`], also reporting what the result caps cut off, so a
    /// caller can disclose it instead of presenting the capped list as the
    /// whole answer. The returned `packets` are exactly what `search` returns.
    pub fn search_with_outcome(
        &mut self,
        store: &mut Store,
        query: &str,
        top_k: usize,
    ) -> Result<SearchOutcome> {
        // Dev hot-reload: pick up config/scoring.toml edits without a restart.
        self.config.reload_if_changed();
        // Has another connection published a generation this engine does not
        // account for (it was built at one, and its own lazy extractions added
        // the ones it made itself)? Then the store's symbols no longer describe
        // this engine's chunks, so the live store reads below (lazy extraction,
        // the symbol fallbacks) are skipped and the answer comes from the
        // engine's own facts only. The engine is latched stale, which the
        // response reports and the owner repairs by rebuilding.
        self.check_generation(store);
        let in_sync = self.accounts_for_store(store);
        let bm25 = self.bm25_candidates(query, 50);
        let vector_hits = self.vector_candidates_with_chunk(query, 50);
        let best_chunk: HashMap<usize, usize> = vector_hits
            .iter()
            .map(|(doc, _, chunk)| (*doc, *chunk))
            .collect();
        let vecs: Vec<(usize, f32)> = vector_hits
            .iter()
            .map(|(doc, score, _)| (*doc, *score))
            .collect();
        let now = now_unix();
        let w = &self.config.weights;

        let norm = |list: &[(usize, f32)]| -> HashMap<usize, f32> {
            let max = list.iter().map(|x| x.1).fold(0f32, f32::max).max(1e-6);
            list.iter().map(|(i, s)| (*i, s / max)).collect()
        };
        let bm25_n = norm(&bm25);
        let vec_n = norm(&vecs);

        let query_wants_tests = query.to_lowercase().contains("test");

        let mut fused: HashMap<usize, (f32, Vec<String>)> = HashMap::new();
        for (i, s) in &bm25_n {
            let e = fused.entry(*i).or_insert((0.0, Vec::new()));
            e.0 += w.bm25 * s;
            e.1.push("bm25".into());
        }
        for (i, s) in &vec_n {
            let e = fused.entry(*i).or_insert((0.0, Vec::new()));
            e.0 += w.vector * s;
            e.1.push("vector".into());
        }

        // path token match, recency, test handling, and doc/changelog demotion
        let q_tokens: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 2)
            .map(|t| t.to_lowercase())
            .filter(|t| !self.stopwords.contains(t.as_str()))
            .collect();
        for (i, (score, signals)) in fused.iter_mut() {
            let d = &self.docs[*i];
            let pl = d.path.to_lowercase();
            if q_tokens.iter().any(|t| pl.contains(t)) {
                *score += w.path_match;
                signals.push("path".into());
            }
            // Additive recency: recently-committed files rank higher.
            let boost = recency_boost(w, d.last_commit_ts, now);
            if boost > 1e-4 {
                *score += boost;
                signals.push("recent".into());
            }
            if d.is_test {
                if query_wants_tests {
                    *score += w.test_bonus_for_tests_query;
                } else {
                    *score *= 0.8; // slight demotion, tests still surfaced via impact
                }
            }
            // Demote low-signal docs; changelogs mention every feature so hit hardest.
            match classify_path(&d.path) {
                DocClass::Changelog => {
                    *score *= w.changelog_penalty;
                    signals.push("changelog_demoted".into());
                }
                DocClass::Doc => {
                    *score *= w.doc_file_penalty;
                    signals.push("doc_demoted".into());
                }
                DocClass::Code => {}
            }
        }

        let mut ranked: Vec<(usize, f32, Vec<String>)> =
            fused.into_iter().map(|(i, (s, sig))| (i, s, sig)).collect();
        // Total order, so the top_k cut and the packet ids never depend on
        // HashMap iteration order: score desc, then path asc, then doc index.
        // (Paths are unique, so the doc index never actually decides.)
        ranked.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.docs[a.0].path.cmp(&self.docs[b.0].path))
                .then_with(|| a.0.cmp(&b.0))
        });
        let beyond_first_cap = ranked.len().saturating_sub(top_k);
        // The best of the cut candidates, by path, so the pack can name them.
        let beyond_paths: Vec<String> = ranked
            .iter()
            .skip(top_k)
            .take(budget::MAX_OMITTED_LISTED)
            .map(|(i, _, _)| self.docs[*i].path.clone())
            .collect();
        ranked.truncate(top_k);

        // lazy Tier-1: ensure symbols exist for top hits, then attach best symbols
        let mut packets = Vec::new();
        for (rank, (i, score, signals)) in ranked.iter().enumerate() {
            let d = &self.docs[*i];
            // Each lazy extraction is a generation of its own (it publishes),
            // which this engine's copy cannot see: it is stale from then on.
            if in_sync && engram_repo_map::ensure_tier1(store, &self.repo_root, &d.path)? {
                self.mark_stale();
                self.accept_own_publish(store);
            }
            // Prefer the span the vector side actually matched: its text is the
            // definition itself, where the symbol table only holds a signature
            // line and the file preview is just the first few hundred bytes.
            let matched_span = best_chunk
                .get(i)
                .map(|&index| &self.chunks[index])
                .filter(|chunk| chunk.symbol.is_some());
            let (title, symbol, snippet, span) = match matched_span {
                Some(chunk) => {
                    let name = chunk.symbol.clone().unwrap_or_default();
                    // path:line, so the agent can open the definition directly
                    // instead of searching the file for the name again.
                    (
                        format!("{} in {}:{}", name, d.path, chunk.start_line),
                        Some(name),
                        Some(chunk.text.clone()),
                        line_span(chunk.start_line, chunk.end_line),
                    )
                }
                None => match if in_sync {
                    best_symbols_for(store, &d.path, &q_tokens)
                } else {
                    Vec::new()
                }
                .first()
                {
                    Some(s) => (
                        format!("{} in {}", s.name, d.path),
                        Some(s.name.clone()),
                        Some(s.signature.clone()),
                        line_span(s.start_line, s.end_line),
                    ),
                    None => (d.path.clone(), None, Some(d.preview.clone()), None),
                },
            };
            packets.push(EvidencePacket {
                id: format!("ev_{rank:03}"),
                kind: if d.is_test {
                    EvidenceKind::Test
                } else if symbol.is_some() {
                    EvidenceKind::Symbol
                } else {
                    EvidenceKind::ExistingCode
                },
                title,
                path: d.path.clone(),
                symbol,
                snippet,
                start_line: span.map(|(start, _)| start),
                end_line: span.map(|(_, end)| end),
                score: *score,
                signals: signals.clone(),
            });
        }

        // Whole-identifier hits. Splitting on non-alphanumerics turns
        // `merge_dicts` into `merge` + `dicts`, so a query that names a symbol
        // never searches for it by name. Recover the identifiers, underscores
        // and dots intact, and look each up as an exact symbol name.
        //
        // An exact identifier match is the strongest signal retrieval has: the
        // caller typed the symbol's name. It has to outrank fusion, so it is
        // scored above the current top rather than at the flat symbol_exact
        // weight (which the langchain miss showed sits *below* fused noise).
        let top_fused = packets.iter().map(|p| p.score).fold(0.0_f32, f32::max);
        let exact_score = top_fused + self.config.weights.symbol_exact;
        // The two symbol lookups below read the store live; held in one read
        // transaction they see a single generation between them, and a
        // generation other than this engine's latches it stale.
        let live = if in_sync {
            let view = store.begin_read()?;
            if view.generation() != self.built_generation {
                self.mark_stale();
            }
            // Somebody else may have published since the check above.
            (view.generation() == self.accepted()).then_some(view)
        } else {
            None
        };
        let query_idents = if live.is_some() {
            query_identifiers(query)
        } else {
            Vec::new()
        };
        for ident in query_idents {
            for s in store.symbols_exact(&ident, 5)? {
                if packets.iter().any(|p| p.symbol.as_deref() == Some(&s.name)) {
                    continue;
                }
                packets.push(EvidencePacket {
                    id: format!("ev_exact_{}", s.name.to_lowercase()),
                    kind: EvidenceKind::Symbol,
                    title: format!("{} in {}:{}", s.name, s.path, s.start_line),
                    path: s.path.clone(),
                    symbol: Some(s.name.clone()),
                    snippet: Some(s.signature.clone()),
                    start_line: line_span(s.start_line, s.end_line).map(|(start, _)| start),
                    end_line: line_span(s.start_line, s.end_line).map(|(_, end)| end),
                    score: exact_score,
                    signals: vec!["symbol_exact".into()],
                });
            }
        }

        // Weaker substring hits that BM25/vector may have missed entirely.
        let substring_tokens: &[String] = if live.is_some() { &q_tokens } else { &[] };
        for t in substring_tokens {
            for s in store.symbols_matching(t, 3)? {
                if packets.iter().any(|p| p.symbol.as_deref() == Some(&s.name)) {
                    continue;
                }
                packets.push(EvidencePacket {
                    id: format!("ev_sym_{}", s.name.to_lowercase()),
                    kind: EvidenceKind::Symbol,
                    title: format!("{} in {}", s.name, s.path),
                    path: s.path.clone(),
                    symbol: Some(s.name.clone()),
                    snippet: Some(s.signature.clone()),
                    start_line: line_span(s.start_line, s.end_line).map(|(start, _)| start),
                    end_line: line_span(s.start_line, s.end_line).map(|(_, end)| end),
                    score: self.config.weights.symbol_exact,
                    signals: vec!["symbol".into()],
                });
            }
        }

        // Re-rank before cutting. The symbol hits above are appended, not
        // inserted in score order, so without this an exact match sits below
        // whatever fusion put in the first top_k slots and gets truncated away
        // — which is exactly how the langchain `merge_dicts` miss happened.
        packets.sort_by(budget::compare_packets);
        let cap = (top_k + 3).min(packets.len());
        let cap_dropped = packets.split_off(cap);
        // A path that came back as evidence (a symbol hit can re-find a file
        // the first cap cut) is not an omission.
        let beyond_first_cap_paths = beyond_paths
            .into_iter()
            .filter(|path| !packets.iter().chain(&cap_dropped).any(|p| &p.path == path))
            .collect();
        Ok(SearchOutcome {
            packets,
            cap_dropped,
            beyond_first_cap,
            beyond_first_cap_paths,
        })
    }

    /// predict_impact: hybrid hits = direct; co-change graph = historical expansion.
    pub fn predict_impact(&mut self, store: &mut Store, query: &str) -> Result<ImpactPrediction> {
        let direct = self.search(store, query, 8)?;
        let mut likely_files = Vec::new();
        let mut likely_tests = Vec::new();

        let max_score = direct.iter().map(|p| p.score).fold(1e-6, f32::max);
        let mut seeds = Vec::new();
        for p in &direct {
            if p.kind == EvidenceKind::Test {
                likely_tests.push(p.path.clone());
            } else {
                likely_files.push(ScoredPath {
                    path: p.path.clone(),
                    confidence: (p.score / max_score).min(1.0),
                    reason: format!("matched task via {}", p.signals.join("+")),
                    evidence_ids: Vec::new(),
                    score_components: None,
                });
                seeds.push(p.path.clone());
            }
        }

        let (cochange_expansions, import_expansions, mut expansion_tests, dropped) =
            self.expand_from_seeds(store, &seeds, MAX_IMPORT_HOPS)?;
        likely_tests.append(&mut expansion_tests);
        likely_tests.sort();
        likely_tests.dedup();

        Ok(ImpactPrediction {
            likely_files,
            likely_tests,
            cochange_expansions,
            import_expansions,
            dropped,
        })
    }

    /// Deterministic impact analysis from known anchor files — no text search,
    /// no ranking heuristics, no guessing. Given files the agent already knows
    /// it is touching (e.g. the current diff's changed files, passed as `paths` to
    /// the `expand_connections` tool),
    /// return everything *connected* to them by hard facts already recorded in
    /// the store: the co-change graph (git history) and the import graph
    /// (static analysis). Every result traces to a concrete edge, so this is
    /// pure fact-finding, not prediction.
    pub fn impact_from_files(
        &mut self,
        store: &mut Store,
        anchors: &[String],
    ) -> Result<ImpactPrediction> {
        self.impact_from_files_with(store, anchors, MAX_IMPORT_HOPS)
    }

    /// [`Engine::impact_from_files`] with an explicit import-traversal depth.
    /// `max_hops` must be in `1..=2`; anything else is an error rather than a
    /// silent clamp. The result also lists, in `dropped`, every candidate the
    /// recommendation caps cut, so callers can disclose them.
    ///
    /// Anchors are de-duplicated and sorted, so the result does not depend on
    /// the order the caller listed them in.
    pub fn impact_from_files_with(
        &mut self,
        store: &mut Store,
        anchors: &[String],
        max_hops: usize,
    ) -> Result<ImpactPrediction> {
        if !(1..=MAX_IMPORT_HOPS).contains(&max_hops) {
            anyhow::bail!("max_hops must be between 1 and {MAX_IMPORT_HOPS}, got {max_hops}");
        }
        let mut known: Vec<String> = anchors
            .iter()
            .filter(|p| self.by_path.contains_key(p.as_str()))
            .cloned()
            .collect();
        known.sort();
        known.dedup();
        let likely_files: Vec<ScoredPath> = known
            .iter()
            .map(|path| ScoredPath {
                path: path.clone(),
                confidence: 1.0,
                reason: "given as an anchor file".to_string(),
                evidence_ids: Vec::new(),
                score_components: None,
            })
            .collect();
        let mut likely_tests: Vec<String> = known
            .iter()
            .filter(|p| engram_repo_map::inventory::is_test_path(p))
            .cloned()
            .collect();

        let (cochange_expansions, import_expansions, mut expansion_tests, dropped) =
            self.expand_from_seeds(store, &known, max_hops)?;
        likely_tests.append(&mut expansion_tests);
        likely_tests.sort();
        likely_tests.dedup();

        Ok(ImpactPrediction {
            likely_files,
            likely_tests,
            cochange_expansions,
            import_expansions,
            dropped,
        })
    }

    /// Explain, from recorded facts only, why two files are connected.
    ///
    /// A direct pairwise lookup, deliberately independent of the expansion path
    /// (`expand_from_seeds`) and its caps, ranking and dedup:
    /// - co-change: the exact `source -> target` and `target -> source` rows,
    ///   each reported with its own `count` and `strength`;
    /// - imports: the shortest import chain of up to [`MAX_IMPORT_HOPS`] hops in
    ///   each direction, with every file and every edge on the chain named.
    ///
    /// Every fact found is returned (an import and a co-change for the same pair
    /// both appear). With no facts, `status` says whether the search was
    /// exhaustive within its scope or stopped at the hop bound, and `scope`
    /// states exactly what was searched, so absence is never overstated.
    /// Errors when either path is not an indexed file or both are the same file,
    /// so a typo cannot masquerade as "not connected".
    pub fn explain_connection(
        &mut self,
        store: &mut Store,
        source: &str,
        target: &str,
    ) -> Result<ConnectionExplanation> {
        for path in [source, target] {
            self.require_indexed(path)?;
        }
        if source == target {
            anyhow::bail!(
                "source and target are the same file ({source}); \
                 pass two different files to explain how they are connected"
            );
        }

        let window = cochange_window();
        let mut reasons = Vec::new();

        // Completeness facts of the last history walk. `None` (never zero)
        // for an index built before they were recorded.
        //
        // All the co-change rows are read in one snapshot, so the totals, the
        // denominators and the two pair rows are from one generation; if that
        // is not the one this engine was built at, the engine is latched stale
        // and the response says so.
        let (totals, opportunities_source, opportunities_target, pair_rows) =
            self.read_live(store, |s| {
                Ok((
                    s.cochange_totals()?,
                    s.cochange_commits_touching(source)?,
                    s.cochange_commits_touching(target)?,
                    [
                        s.cochange_pair(source, target)?,
                        s.cochange_pair(target, source)?,
                    ],
                ))
            })?;

        for ((direction, from, to), row) in [
            (CoChangeDirection::SourceToTarget, source, target),
            (CoChangeDirection::TargetToSource, target, source),
        ]
        .into_iter()
        .zip(pair_rows)
        {
            if let Some(row) = row {
                let from_opportunities = match direction {
                    CoChangeDirection::SourceToTarget => opportunities_source,
                    CoChangeDirection::TargetToSource => opportunities_target,
                };
                let percent = cochange_percent(row.count, from_opportunities, row.strength);
                // The denominator counts only non-bulk commits (those touching
                // at most `max_files_per_commit` files); bulk ones are excluded
                // from it, so the wording says so rather than "all commits".
                let detail = match from_opportunities {
                    Some(m) => format!(
                        "{from} and {to} changed together in {} of the {m} non-bulk commit{} \
                         that touched {from} ({percent}%)",
                        row.count,
                        if m == 1 { "" } else { "s" },
                    ),
                    None => format!(
                        "{from} and {to} changed together in {} commit{}, {percent}% of the \
                         non-bulk commits that touched {from} (that commit count is not \
                         recorded in this index)",
                        row.count,
                        if row.count == 1 { "" } else { "s" },
                    ),
                };
                reasons.push(ConnectionReason::CoChange {
                    detail,
                    weight: row.strength,
                    edge_kind: "cochange",
                    direction,
                    count: row.count,
                    strength: row.strength,
                    window,
                    opportunities_source,
                    opportunities_target,
                    // The id of the stored row `from -> to`, not of the
                    // question: it matches what `get_graph` emits for it.
                    edge_id: cochange_edge_id(&self.snapshot, from, to),
                });
            }
        }

        let mut outcomes = Vec::new();
        for (direction, from, to) in [
            (ImportDirection::SourceImportsTarget, source, target),
            (ImportDirection::TargetImportsSource, target, source),
        ] {
            match self.graph.import_search(from, to, MAX_IMPORT_HOPS) {
                ImportSearch::Found(chain) => {
                    outcomes.push(ImportSearchOutcome::Found);
                    let hops = chain.len() - 1;
                    let route = if hops == 1 {
                        format!("{from} imports {to} directly")
                    } else {
                        format!(
                            "{from} reaches {to} in {hops} hops through imports: {}",
                            chain.join(" -> ")
                        )
                    };
                    let edges = chain
                        .windows(2)
                        .map(|w| ImportEdge {
                            from: w[0].clone(),
                            to: w[1].clone(),
                            edge_id: import_edge_id(&self.snapshot, w),
                        })
                        .collect();
                    let route_id = import_edge_id(&self.snapshot, &chain);
                    reasons.push(ConnectionReason::Import {
                        detail: format!(
                            "{route} (heuristic module-name resolution, not compiler resolution)"
                        ),
                        weight: if hops <= 1 { 0.6 } else { 0.35 },
                        edge_kind: "import",
                        direction,
                        hops,
                        chain,
                        edges,
                        resolution: IMPORT_RESOLUTION,
                        resolver: IMPORT_RESOLVER,
                        resolution_status: "candidate",
                        edge_id: route_id,
                    });
                }
                ImportSearch::None => outcomes.push(ImportSearchOutcome::NoneWithinBound),
                ImportSearch::BeyondBound => outcomes.push(ImportSearchOutcome::BeyondBound),
            }
        }

        let status = if !reasons.is_empty() {
            ConnectionStatus::Found
        } else if outcomes.contains(&ImportSearchOutcome::BeyondBound) {
            ConnectionStatus::UnknownBeyondBound
        } else {
            ConnectionStatus::NotFoundWithinScope
        };
        let scope = ConnectionScope {
            import_max_hops: MAX_IMPORT_HOPS,
            import_directions_searched: vec![
                ImportDirection::SourceImportsTarget,
                ImportDirection::TargetImportsSource,
            ],
            import_search_source_imports_target: outcomes[0],
            import_search_target_imports_source: outcomes[1],
            cochange_lookup: "exact_pair",
            cochange_directions_searched: vec![
                CoChangeDirection::SourceToTarget,
                CoChangeDirection::TargetToSource,
            ],
            cochange_window: window,
            commits_scanned: totals.map(|(scanned, _)| scanned),
            bulk_commits_excluded: totals.map(|(_, bulk)| bulk),
            note: format!(
                "Import chains up to {MAX_IMPORT_HOPS} hops were searched in both directions \
                 over heuristically resolved import edges. Co-change covers only the last {} \
                 commits, pairs sharing at least {} commits, ignoring commits touching more \
                 than {} files (counted in bulk_commits_excluded). Absence of a fact means \
                 none was recorded within this scope, not that the files are unrelated.",
                window.max_commits, window.min_count, window.max_files_per_commit
            ),
        };
        Ok(ConnectionExplanation {
            status,
            scope,
            reasons,
        })
    }

    /// Err unless `path` is a file in this engine's index.
    fn require_indexed(&self, path: &str) -> Result<()> {
        if !self.by_path.contains_key(path) {
            anyhow::bail!(
                "{path} is not an indexed text file in the Engram index; use a \
                 repo-relative path returned by search_context/expand_connections"
            );
        }
        Ok(())
    }

    /// Raw recorded facts about the neighbourhood of `anchors` (`get_graph`).
    ///
    /// Independent of recommendation ranking, dedup and per-kind caps: it lists
    /// edges, it does not rank files.
    /// - imports: every single-hop import edge, in either direction, incident
    ///   to an anchor or to a file reached within `max_hops - 1` import hops of
    ///   one (so `max_hops = 1` is "everything touching an anchor", `2` adds
    ///   the edges touching those neighbours). Each edge is emitted once as
    ///   `importer -> imported`.
    /// - co-change: each anchor's stored rows to its strongest partners (at
    ///   most `caps.cochange_per_anchor`), plus the reverse row of each partner
    ///   (its own strength, relative to the partner's commits) when one exists.
    ///
    /// Every edge carries the content-derived id the recommendations and
    /// `explain_connection` cite it by. Output order is deterministic: nodes
    /// by path, edges by (kind, source, target, direction).
    ///
    /// Accounting. The unit is one emitted edge, i.e. one directed co-change
    /// row or one import edge. `omitted_edges` is exactly the number of edges
    /// the same call would have emitted with every cap removed, minus the
    /// number it did emit: co-change rows past the per-anchor limit (and the
    /// reverse rows of those partners), edges touching a file past the node
    /// cap, and edges past the edge cap. `omitted_nodes` counts neighbouring
    /// files past the node cap; a partner hidden by the per-anchor co-change
    /// limit is never added as a node, so it is not counted there, only
    /// through its edges. Nothing is cut silently.
    ///
    /// The work is bounded as well as the answer: the anchor count is checked
    /// before any anchor is looked up, and import edges are counted as they
    /// are walked but only the `max_edges` that survive the cut are ever built.
    ///
    /// Errors for a path that is not an indexed file, an empty anchor list,
    /// more anchors than `caps.max_nodes`, or `max_hops` outside `1..=2`.
    pub fn graph_facts(
        &self,
        store: &Store,
        anchors: &[String],
        max_hops: usize,
        kinds: GraphKinds,
        caps: GraphCaps,
    ) -> Result<GraphFacts> {
        if !(1..=MAX_IMPORT_HOPS).contains(&max_hops) {
            anyhow::bail!("max_hops must be between 1 and {MAX_IMPORT_HOPS}, got {max_hops}");
        }
        if !kinds.import && !kinds.cochange {
            anyhow::bail!("kinds must include at least one of \"import\", \"cochange\"");
        }
        if caps.max_nodes == 0 || caps.max_edges == 0 || caps.cochange_per_anchor == 0 {
            anyhow::bail!("graph caps must be at least 1");
        }
        let mut anchors: Vec<String> = anchors.to_vec();
        anchors.sort();
        anchors.dedup();
        if anchors.is_empty() {
            anyhow::bail!("at least one anchor path is required");
        }
        // Checked before any anchor is looked up, so an oversized request is
        // refused without touching the index.
        if anchors.len() > caps.max_nodes {
            anyhow::bail!(
                "{} anchor paths exceed the {}-node cap; pass fewer paths",
                anchors.len(),
                caps.max_nodes
            );
        }
        for path in &anchors {
            self.require_indexed(path)?;
        }
        let anchor_set: HashSet<&str> = anchors.iter().map(String::as_str).collect();

        // Import neighbourhood: BFS levels over import edges in either
        // direction. `levels[d]` is sorted, so every walk below is in a
        // deterministic order.
        let mut import_dist: HashMap<String, usize> =
            anchors.iter().map(|a| (a.clone(), 0)).collect();
        let mut levels: Vec<Vec<String>> = vec![anchors.clone()];
        if kinds.import {
            for depth in 0..max_hops {
                let mut next: BTreeSet<String> = BTreeSet::new();
                for node in &levels[depth] {
                    for other in self
                        .graph
                        .import_targets(node)
                        .into_iter()
                        .chain(self.graph.import_sources(node))
                    {
                        if !import_dist.contains_key(&other) {
                            import_dist.insert(other.clone(), depth + 1);
                            next.insert(other);
                        }
                    }
                }
                levels.push(next.into_iter().collect());
            }
        }
        let mut dist: BTreeMap<String, usize> =
            import_dist.iter().map(|(p, d)| (p.clone(), *d)).collect();

        // Co-change facts: each anchor's strongest rows plus the partner's
        // row back, keyed (source, target). Every read below happens inside one
        // snapshot, so the rows, their denominators and the hidden count are
        // all of one generation; if it is not this engine's, the engine is
        // latched stale and the response says so rather than mixing the two.
        let view = store.begin_read()?;
        if view.generation() != self.built_generation {
            self.mark_stale();
        }
        let totals = store.cochange_totals()?;
        let mut cochange: BTreeMap<(String, String), GraphEdge> = BTreeMap::new();
        let mut cochange_total = 0usize;
        if kinds.cochange {
            let window = cochange_window();
            let mut opportunities: HashMap<String, Option<u32>> = HashMap::new();
            let mut touching = |path: &str| -> Result<Option<u32>> {
                if let Some(known) = opportunities.get(path) {
                    return Ok(*known);
                }
                let n = store.cochange_commits_touching(path)?;
                opportunities.insert(path.to_owned(), n);
                Ok(n)
            };
            for anchor in &anchors {
                let neighbors = store.cochange_neighbors(anchor, caps.cochange_per_anchor)?;
                for row in neighbors.rows {
                    let partner = row.path_b.clone();
                    if partner == *anchor || !self.by_path.contains_key(&partner) {
                        continue;
                    }
                    // The anchor's own row, and the partner's row back to it.
                    let reverse = store.cochange_pair(&partner, anchor)?;
                    for stored in std::iter::once(row).chain(reverse) {
                        let key = (stored.path_a.clone(), stored.path_b.clone());
                        if cochange.contains_key(&key) {
                            continue;
                        }
                        let edge = GraphEdge::Cochange {
                            edge_id: cochange_edge_id(
                                &self.snapshot,
                                &stored.path_a,
                                &stored.path_b,
                            ),
                            opportunities_source: touching(&stored.path_a)?,
                            opportunities_target: touching(&stored.path_b)?,
                            source: stored.path_a,
                            target: stored.path_b,
                            direction: CoChangeDirection::SourceToTarget,
                            count: stored.count,
                            strength: stored.strength,
                            window,
                        };
                        cochange.insert(key, edge);
                    }
                    dist.entry(partner)
                        .and_modify(|d| *d = (*d).min(1))
                        .or_insert(1);
                }
            }
            // Every such row there is, read or not: the total the hidden
            // count is taken against.
            cochange_total = store.cochange_fact_count(&anchors)?;
        }
        drop(view);
        let cochange_hidden = cochange_total.saturating_sub(cochange.len());

        // Node cap: anchors always stay; neighbours are kept nearest first,
        // then by path, so the same cut is made on every run.
        let mut neighbors: Vec<(&String, &usize)> = dist
            .iter()
            .filter(|(path, _)| !anchor_set.contains(path.as_str()))
            .collect();
        neighbors.sort_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));
        let node_room = caps.max_nodes - anchors.len();
        let omitted_nodes = neighbors.len().saturating_sub(node_room);
        let kept: BTreeSet<&str> = anchors
            .iter()
            .map(String::as_str)
            .chain(
                neighbors
                    .iter()
                    .take(node_room)
                    .map(|(path, _)| path.as_str()),
            )
            .collect();

        // Edge caps. First drop edges touching a node that did not make it,
        // then keep the nearest `max_edges` (co-change rows, all at distance
        // 0, before imports at the same distance; ties by the output order).
        let cochange_found = cochange.len();
        let mut cochange_kept: Vec<GraphEdge> = cochange
            .into_iter()
            .filter(|((source, target), _)| {
                kept.contains(source.as_str()) && kept.contains(target.as_str())
            })
            .map(|(_, edge)| edge)
            .collect();
        let cochange_dropped_by_node_cap = cochange_found - cochange_kept.len();
        let cochange_dropped_by_edge_cap = cochange_kept.len().saturating_sub(caps.max_edges);
        cochange_kept.truncate(caps.max_edges);

        // Imports are walked once more, counting every edge and holding only
        // the best `import_room` of them: a hub with thousands of importers
        // costs a counter, not thousands of built edges. An edge belongs to
        // the endpoint nearer to the anchors (the smaller path when equally
        // near), so each is seen exactly once.
        let import_room = caps.max_edges - cochange_kept.len();
        let mut import_total = 0usize;
        let mut import_dropped_by_node_cap = 0usize;
        let mut import_candidates = 0usize;
        let mut best: BinaryHeap<(usize, String, String)> = BinaryHeap::new();
        if kinds.import {
            for (depth, level) in levels.iter().enumerate().take(max_hops) {
                for node in level {
                    let outgoing = self.graph.import_targets(node);
                    let incoming = self.graph.import_sources(node);
                    let seen = outgoing
                        .iter()
                        .map(|imported| (node, imported, imported))
                        .chain(incoming.iter().map(|importer| (importer, node, importer)));
                    for (importer, imported, other) in seen {
                        let other_depth = import_dist.get(other).copied().unwrap_or(usize::MAX);
                        let owned = other_depth > depth
                            || (other_depth == depth && node.as_str() < other.as_str());
                        if !owned {
                            continue;
                        }
                        import_total += 1;
                        if !(kept.contains(importer.as_str()) && kept.contains(imported.as_str())) {
                            import_dropped_by_node_cap += 1;
                            continue;
                        }
                        import_candidates += 1;
                        if import_room == 0 {
                            continue;
                        }
                        let candidate = (depth, importer.as_str(), imported.as_str());
                        if best.len() < import_room {
                            best.push((depth, importer.clone(), imported.clone()));
                        } else if best.peek().is_some_and(|worst| {
                            candidate < (worst.0, worst.1.as_str(), worst.2.as_str())
                        }) {
                            best.pop();
                            best.push((depth, importer.clone(), imported.clone()));
                        }
                    }
                }
            }
        }
        let import_dropped_by_edge_cap = import_candidates - best.len();
        let mut imports: Vec<(String, String)> = best
            .into_iter()
            .map(|(_, importer, imported)| (importer, imported))
            .collect();
        imports.sort();

        let mut emitted: Vec<GraphEdge> = cochange_kept;
        emitted.extend(
            imports
                .into_iter()
                .map(|(importer, imported)| GraphEdge::Import {
                    edge_id: import_edge_id(&self.snapshot, &[&importer, &imported]),
                    source: importer,
                    target: imported,
                    direction: ImportDirection::SourceImportsTarget,
                    hops: 1,
                    resolver: IMPORT_RESOLVER,
                    resolution_status: "candidate",
                }),
        );

        let dropped_by_node_cap = cochange_dropped_by_node_cap + import_dropped_by_node_cap;
        let dropped_by_edge_cap = cochange_dropped_by_edge_cap + import_dropped_by_edge_cap;
        let omitted_edges = cochange_hidden + dropped_by_node_cap + dropped_by_edge_cap;
        debug_assert_eq!(
            emitted.len() + omitted_edges,
            cochange_total + import_total,
            "emitted + omitted must be every fact there is"
        );
        let truncated = omitted_edges > 0 || omitted_nodes > 0;
        let reason = if truncated {
            let mut parts = Vec::new();
            if cochange_hidden > 0 {
                parts.push(format!(
                    "{cochange_hidden} co-change row(s) past the per-anchor limit of {} partners \
                     (including the reverse rows of the hidden partners)",
                    caps.cochange_per_anchor
                ));
            }
            if omitted_nodes > 0 {
                parts.push(format!(
                    "{omitted_nodes} neighbouring file(s) past the {}-node cap",
                    caps.max_nodes
                ));
            }
            if dropped_by_node_cap > 0 {
                parts.push(format!(
                    "{dropped_by_node_cap} edge(s) touching a file past the node cap"
                ));
            }
            if dropped_by_edge_cap > 0 {
                parts.push(format!(
                    "{dropped_by_edge_cap} edge(s) past the {}-edge cap (nearest to an anchor \
                     kept first)",
                    caps.max_edges
                ));
            }
            format!("truncated: {}", parts.join("; "))
        } else {
            "complete: no cap cut anything".to_owned()
        };

        let nodes: Vec<GraphNode> = kept
            .iter()
            .map(|path| {
                let language = Language::from_path(path);
                GraphNode {
                    path: (*path).to_owned(),
                    role: if anchor_set.contains(path) {
                        GraphNodeRole::Anchor
                    } else {
                        GraphNodeRole::Neighbor
                    },
                    distance: dist.get(*path).copied().unwrap_or(0),
                    language: (language != Language::Other).then(|| language.as_str()),
                    is_test: engram_repo_map::inventory::is_test_path(path),
                }
            })
            .collect();

        let window = cochange_window();
        let mut scope_kinds = Vec::new();
        if kinds.import {
            scope_kinds.push("import");
        }
        if kinds.cochange {
            scope_kinds.push("cochange");
        }
        Ok(GraphFacts {
            snapshot: self.snapshot.clone(),
            anchors,
            nodes,
            edges: emitted,
            limits: GraphLimits {
                max_nodes: caps.max_nodes,
                max_edges: caps.max_edges,
                cochange_per_anchor: caps.cochange_per_anchor,
                truncated,
                omitted_edges,
                omitted_nodes,
                reason,
            },
            scope: GraphScope {
                kinds: scope_kinds,
                import_max_hops: max_hops,
                cochange_window: window,
                commits_scanned: totals.map(|(scanned, _)| scanned),
                bulk_commits_excluded: totals.map(|(_, bulk)| bulk),
                note: format!(
                    "Facts only, not a ranking and not a call graph: import edges are \
                     heuristic module-name matches (single hop, either direction, from the \
                     anchors out to {max_hops} hop(s)); co-change edges are stored git-history \
                     rows between each anchor and its strongest partners (last {} commits, \
                     pairs sharing at least {} commits, commits touching more than {} files \
                     ignored). Symbol-level call edges are not recorded. Absence of an edge \
                     means none was recorded within this scope.",
                    window.max_commits, window.min_count, window.max_files_per_commit
                ),
            },
        })
    }

    /// Shared deterministic expansion: co-change graph (git history fact) +
    /// import graph (static-analysis fact, up to `max_hops` hops), excluding
    /// files already in `seeds`/`likely`. Returns (cochange, imports,
    /// discovered_tests, dropped), where `dropped` lists every candidate a cap
    /// withheld. The caps themselves are unchanged.
    #[allow(clippy::type_complexity)]
    fn expand_from_seeds(
        &self,
        store: &mut Store,
        seeds: &[String],
        max_hops: usize,
    ) -> Result<(
        Vec<ScoredPath>,
        Vec<ScoredPath>,
        Vec<String>,
        Vec<DroppedCandidate>,
    )> {
        let mut likely_tests = Vec::new();
        // Per partner: the strongest row's numbers (which set `confidence` and
        // `reason`), and the id of every kept row from any anchor.
        struct Expansion {
            strength: f32,
            count: u32,
            anchor: String,
            reason: String,
            evidence: Vec<String>,
        }
        let mut expansions: HashMap<String, Expansion> = HashMap::new();
        let mut beyond_seed_cap: Vec<String> = Vec::new();
        // The co-change rows of every seed come from one snapshot; a generation
        // other than this engine's latches it stale (the response says so).
        let view = store.begin_read()?;
        if view.generation() != self.built_generation {
            self.mark_stale();
        }
        for seed in seeds {
            // Read every row, order it totally (strength desc, then path), and
            // keep the first COCHANGE_PER_SEED as the cap always did. The rows
            // past the cap are what gets disclosed.
            let mut rows = store.cochange_for(seed, COCHANGE_SCAN_LIMIT)?;
            rows.sort_by(|a, b| {
                b.strength
                    .total_cmp(&a.strength)
                    .then_with(|| a.path_b.cmp(&b.path_b))
            });
            let cut = rows.len().min(COCHANGE_PER_SEED);
            for edge in &rows[..cut] {
                if self.by_path.contains_key(&edge.path_b) && !seeds.contains(&edge.path_b) {
                    let e = expansions
                        .entry(edge.path_b.clone())
                        .or_insert_with(|| Expansion {
                            strength: 0.0,
                            count: edge.count,
                            anchor: seed.clone(),
                            reason: String::new(),
                            evidence: Vec::new(),
                        });
                    e.evidence
                        .push(cochange_edge_id(&self.snapshot, seed, &edge.path_b));
                    if edge.strength > e.strength {
                        e.strength = edge.strength;
                        e.count = edge.count;
                        e.anchor = seed.clone();
                        e.reason = format!(
                            "changes with {seed} in {}% of its commits",
                            (edge.strength * 100.0).round() as u32
                        );
                    }
                }
            }
            for edge in &rows[cut..] {
                if self.by_path.contains_key(&edge.path_b) && !seeds.contains(&edge.path_b) {
                    beyond_seed_cap.push(edge.path_b.clone());
                }
            }
        }
        drop(view);
        // A path cut for one anchor but kept for another was not dropped.
        let mut dropped_cochange: Vec<String> = beyond_seed_cap
            .into_iter()
            .filter(|p| !expansions.contains_key(p))
            .collect();

        let mut cochange_expansions: Vec<ScoredPath> = expansions
            .into_iter()
            .map(|(path, mut e)| {
                if engram_repo_map::inventory::is_test_path(&path) {
                    likely_tests.push(path.clone());
                }
                e.evidence.sort();
                e.evidence.dedup();
                ScoredPath {
                    path,
                    confidence: e.strength,
                    reason: e.reason,
                    evidence_ids: e.evidence,
                    score_components: Some(ScoreComponents::CochangeStrength {
                        strength: e.strength,
                        count: e.count,
                        anchor: e.anchor,
                    }),
                }
            })
            .collect();
        cochange_expansions.sort_by(by_confidence_then_path);
        if cochange_expansions.len() > EXPANSION_CAP {
            dropped_cochange.extend(
                cochange_expansions
                    .split_off(EXPANSION_CAP)
                    .into_iter()
                    .map(|c| c.path),
            );
        }

        // Import-graph expansion via the in-memory code graph: files that
        // (transitively, up to `max_hops` hops) import a seed. SQL stores the
        // edges; we traverse them in memory (see docs/adr/0001).
        let seed_paths: HashSet<&str> = seeds.iter().map(|s| s.as_str()).collect();
        let cochange_paths: HashSet<&str> = cochange_expansions
            .iter()
            .map(|c| c.path.as_str())
            .collect();
        let mut import_expansions: Vec<ScoredPath> = Vec::new();
        for hit in self.graph.dependents(seeds, &[EdgeKind::Imports], max_hops) {
            if seed_paths.contains(hit.path.as_str()) || cochange_paths.contains(hit.path.as_str())
            {
                continue;
            }
            if engram_repo_map::inventory::is_test_path(&hit.path) {
                likely_tests.push(hit.path.clone());
            }
            let confidence = if hit.hops <= 1 { 0.6 } else { 0.35 };
            // One id per hop of the route back to an anchor, each the id of
            // that single import edge.
            let evidence_ids: Vec<String> = hit
                .chain
                .windows(2)
                .map(|w| import_edge_id(&self.snapshot, w))
                .collect();
            import_expansions.push(ScoredPath {
                path: hit.path,
                confidence,
                evidence_ids,
                score_components: Some(ScoreComponents::ImportHops {
                    hops: hit.hops,
                    weight: confidence,
                }),
                reason: if hit.hops <= 1 {
                    "imports a likely-changed file (1 hop)".to_string()
                } else {
                    format!(
                        "imports a file that imports a likely-changed file ({} hops)",
                        hit.hops
                    )
                },
            });
        }
        import_expansions.sort_by(by_confidence_then_path);
        let dropped_import: Vec<String> = if import_expansions.len() > EXPANSION_CAP {
            import_expansions
                .split_off(EXPANSION_CAP)
                .into_iter()
                .map(|c| c.path)
                .collect()
        } else {
            Vec::new()
        };

        dropped_cochange.sort();
        dropped_cochange.dedup();
        let mut dropped: Vec<DroppedCandidate> = dropped_cochange
            .into_iter()
            .map(|path| DroppedCandidate {
                path,
                kind: "cochange",
            })
            .collect();
        dropped.extend(dropped_import.into_iter().map(|path| DroppedCandidate {
            path,
            kind: "import",
        }));

        Ok((
            cochange_expansions,
            import_expansions,
            likely_tests,
            dropped,
        ))
    }
}

/// Confidence descending, then path ascending: a total order, so equal
/// confidences never fall back to hash-map or traversal order.
fn by_confidence_then_path(a: &ScoredPath, b: &ScoredPath) -> std::cmp::Ordering {
    b.confidence
        .total_cmp(&a.confidence)
        .then_with(|| a.path.cmp(&b.path))
}

/// What [`Engine::search_with_outcome`] returns: the packets `search` returns,
/// plus what its two result caps cut.
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub packets: Vec<EvidencePacket>,
    /// Packets that were built and ranked but fell past the final cap
    /// (`top_k + 3`), in rank order.
    pub cap_dropped: Vec<EvidencePacket>,
    /// Fused candidates cut by the first cap (`top_k`) before any packet was
    /// built. Counted, not enumerated. This counts every fused candidate past
    /// `top_k` (BM25 and vector hits are each fetched 50 deep), so a search
    /// usually has some, and `pack.truncated`/`omitted_count` read `true` / `> 0`
    /// even with no budget.
    pub beyond_first_cap: usize,
    /// The paths of the best of those candidates (rank order, at most
    /// `budget::MAX_OMITTED_LISTED`, without any that came back as evidence), so
    /// that the pack can name some of what it cut instead of only counting it.
    pub beyond_first_cap_paths: Vec<String>,
}

/// A symbol's `(start, end)` line span, or `None` for the zero spans a
/// pre-span database holds.
fn line_span(start: usize, end: usize) -> Option<(usize, usize)> {
    (start > 0 && end >= start).then_some((start, end))
}

fn best_symbols_for(store: &Store, path: &str, q_tokens: &[String]) -> Vec<SymbolRecord> {
    let mut hits = Vec::new();
    for t in q_tokens {
        if let Ok(mut syms) = store.symbols_matching(t, 5) {
            syms.retain(|s| s.path == path);
            hits.extend(syms);
        }
    }
    hits
}

#[cfg(test)]
mod lazy_extraction_tests {
    use super::*;

    /// Throwaway repo root, removed on drop. Not `tempfile`: CI builds
    /// `--locked`, so a dev-dependency would need a Cargo.lock regeneration.
    struct TempRepo(PathBuf);

    impl TempRepo {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let path = std::env::temp_dir().join(format!(
                "engram-retrieval-lazy-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create temp repo");
            TempRepo(path)
        }

        fn write(&self, rel: &str, contents: &str) {
            std::fs::write(self.0.join(rel), contents).expect("write");
        }
    }

    impl Drop for TempRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn importers_of(engine: &Engine, path: &str) -> Vec<String> {
        engine
            .graph
            .dependents(&[path.to_owned()], &[EdgeKind::Imports], 1)
            .into_iter()
            .map(|hit| hit.path)
            .collect()
    }

    #[test]
    fn lazy_extraction_marks_the_engine_stale_until_it_is_rebuilt() {
        let repo = TempRepo::new();
        repo.write(
            "app.py",
            "import helpers\n\ndef frobnicate_widgets():\n    return helpers.work()\n",
        );
        repo.write("helpers.py", "def work():\n    return 1\n");

        // Inventory only: no file has been Tier-1 extracted yet, which is the
        // state lazy extraction exists for (a repo past the eager limit).
        let mut store = Store::open(&repo.0).expect("open store");
        store
            .upsert_fingerprinted(&engram_repo_map::inventory::scan(&repo.0))
            .expect("upsert files");
        let mut engine = Engine::build(&repo.0, &mut store).expect("build engine");
        assert!(!engine.is_stale(), "a fresh engine reflects the store");
        assert!(importers_of(&engine, "helpers.py").is_empty());

        let packets = engine
            .search(&mut store, "frobnicate_widgets", 8)
            .expect("search");
        assert!(packets.iter().any(|p| p.path == "app.py"), "{packets:?}");
        // The search extracted app.py's imports into the store, after the
        // graph was built, so this engine can no longer see the whole truth.
        assert!(engine.is_stale());
        assert!(
            importers_of(&engine, "helpers.py").is_empty(),
            "the old graph cannot know about the new edge"
        );

        let rebuilt = Engine::build(&repo.0, &mut store).expect("rebuild engine");
        assert!(!rebuilt.is_stale());
        assert_eq!(importers_of(&rebuilt, "helpers.py"), vec!["app.py"]);
    }

    #[test]
    fn a_search_that_extracts_nothing_new_leaves_the_engine_fresh() {
        let repo = TempRepo::new();
        repo.write("app.py", "def frobnicate_widgets():\n    return 1\n");
        let mut store = Store::open(&repo.0).expect("open store");
        engram_repo_map::index_repo(&repo.0, 100).expect("index");
        let mut engine = Engine::build(&repo.0, &mut store).expect("build engine");

        engine
            .search(&mut store, "frobnicate_widgets", 8)
            .expect("search");
        assert!(!engine.is_stale(), "everything was already extracted");
    }

    #[test]
    fn only_files_with_extractable_facts_count_as_ahead_of_the_index() {
        let repo = TempRepo::new();
        repo.write("app.py", "def frobnicate_widgets():\n    return 1\n");
        repo.write("notes.md", "# notes\n");
        repo.write("conf.json", "{}\n");
        engram_repo_map::index_repo(&repo.0, 100).expect("index");
        let mut store = Store::open(&repo.0).expect("open store");

        // A prose and a data file edited after indexing: they have no symbols
        // or imports, so nothing about them can be stale and nothing is said.
        repo.write("notes.md", "# notes, edited\n");
        repo.write("conf.json", "{\"a\": 1}\n");
        let engine = Engine::build(&repo.0, &mut store).expect("build engine");
        assert_eq!(engine.files_ahead_of_index(), 0);

        // A source file edited after indexing is exactly what the warning is for.
        repo.write("app.py", "def frobnicate_widgets():\n    return 2\n");
        let engine = Engine::build(&repo.0, &mut store).expect("build engine");
        assert_eq!(engine.files_ahead_of_index(), 1);
    }

    fn batch_replacing_symbols(
        path: &str,
        symbol_name: &str,
    ) -> engram_repo_map::store::IndexBatch {
        use engram_repo_map::store::{Extraction, ExtractionUpdate, IndexBatch};
        IndexBatch {
            extractions: vec![ExtractionUpdate {
                expected_hash: None,
                path: path.to_owned(),
                extraction: Extraction::Extracted {
                    symbols: vec![SymbolRecord {
                        name: symbol_name.to_owned(),
                        kind: engram_domain::SymbolKind::Function,
                        path: path.to_owned(),
                        start_line: 1,
                        end_line: 2,
                        signature: format!("def {symbol_name}()"),
                    }],
                    imports: Vec::new(),
                },
            }],
            ..IndexBatch::default()
        }
    }

    #[test]
    fn an_engine_records_the_generation_it_was_built_from_and_goes_stale_when_one_is_published() {
        let repo = TempRepo::new();
        repo.write("app.py", "def frobnicate_widgets():\n    return 1\n");
        repo.write("lib.py", "def helper():\n    return 2\n");
        engram_repo_map::index_repo(&repo.0, 100).expect("index");
        let mut store = Store::open(&repo.0).expect("open store");
        let engine = Engine::build(&repo.0, &mut store).expect("build engine");

        let built = store.generation().unwrap();
        assert!(built > 0);
        assert_eq!(engine.generation(), built);
        assert!(!engine.is_stale_in(&store));
        assert!(!engine.is_stale());

        // Another connection publishes (as the watcher does).
        let mut other = Store::open(&repo.0).expect("second handle");
        repo.write("lib.py", "def helper_renamed():\n    return 2\n");
        assert!(engram_repo_map::reindex_file(&mut other, &repo.0, "lib.py").unwrap());
        assert_eq!(other.generation().unwrap(), built + 1);

        assert!(
            engine.is_stale_in(&store),
            "the store moved past the engine"
        );
        assert!(engine.is_stale(), "and the engine remembers it");
        assert_eq!(engine.generation(), built, "its own facts did not move");

        let rebuilt = Engine::build(&repo.0, &mut store).expect("rebuild");
        assert_eq!(rebuilt.generation(), built + 1);
        assert!(!rebuilt.is_stale_in(&store));
    }

    #[test]
    fn a_search_after_a_foreign_publish_is_answered_from_the_engines_facts_only() {
        let repo = TempRepo::new();
        repo.write("app.py", "def frobnicate_widgets():\n    return 1\n");
        repo.write("lib.py", "def helper():\n    return 2\n");
        engram_repo_map::index_repo(&repo.0, 100).expect("index");
        let mut store = Store::open(&repo.0).expect("open store");
        let mut engine = Engine::build(&repo.0, &mut store).expect("build engine");

        // A different connection publishes a symbol this engine has never seen.
        let mut other = Store::open(&repo.0).expect("second handle");
        other
            .publish_batch(batch_replacing_symbols("lib.py", "zzz_unheard_of"))
            .unwrap();

        let packets = engine
            .search(&mut store, "zzz_unheard_of", 8)
            .expect("search");
        assert!(
            !packets
                .iter()
                .any(|p| p.symbol.as_deref() == Some("zzz_unheard_of")),
            "a newer generation's symbols must not be mixed into an older engine's answer: \
             {packets:?}"
        );
        assert!(engine.is_stale(), "and the answer says it is out of date");

        // The control: an engine built after the publish does find it.
        let mut current = Engine::build(&repo.0, &mut store).expect("rebuild");
        let packets = current
            .search(&mut store, "zzz_unheard_of", 8)
            .expect("search");
        assert!(
            packets
                .iter()
                .any(|p| p.symbol.as_deref() == Some("zzz_unheard_of")),
            "{packets:?}"
        );
        assert!(!current.is_stale());
    }

    #[test]
    fn co_change_reads_of_a_newer_generation_are_flagged_not_silently_mixed() {
        let repo = TempRepo::new();
        repo.write("a.py", "def alpha():\n    return 1\n");
        repo.write("b.py", "def beta():\n    return 2\n");
        engram_repo_map::index_repo(&repo.0, 100).expect("index");
        let mut store = Store::open(&repo.0).expect("open store");
        let mut engine = Engine::build(&repo.0, &mut store).expect("build engine");
        let fresh = engine
            .explain_connection(&mut store, "a.py", "b.py")
            .expect("explain");
        assert!(fresh.reasons.is_empty());
        assert!(
            !engine.is_stale(),
            "every read was of the engine's generation"
        );

        let mut other = Store::open(&repo.0).expect("second handle");
        other
            .replace_cochange(&[engram_domain::CoChange {
                path_a: "a.py".to_owned(),
                path_b: "b.py".to_owned(),
                count: 3,
                strength: 0.9,
            }])
            .unwrap();

        // The explanation is still produced, and says what it is made of.
        let explained = engine
            .explain_connection(&mut store, "a.py", "b.py")
            .expect("explain");
        assert_eq!(explained.reasons.len(), 1, "{explained:?}");
        assert!(
            engine.is_stale(),
            "co-change rows of generation N+1 under engine N"
        );
    }

    #[test]
    fn get_graph_and_expansion_flag_a_store_that_moved_past_the_engine() {
        let repo = TempRepo::new();
        repo.write("a.py", "def alpha():\n    return 1\n");
        repo.write("b.py", "def beta():\n    return 2\n");
        engram_repo_map::index_repo(&repo.0, 100).expect("index");
        let mut store = Store::open(&repo.0).expect("open store");
        let mut engine = Engine::build(&repo.0, &mut store).expect("build engine");
        let anchors = ["a.py".to_owned()];
        engine
            .graph_facts(
                &store,
                &anchors,
                2,
                GraphKinds::default(),
                GraphCaps::default(),
            )
            .expect("graph");
        engine
            .impact_from_files(&mut store, &anchors)
            .expect("impact");
        assert!(!engine.is_stale());

        let mut other = Store::open(&repo.0).expect("second handle");
        other
            .publish_batch(batch_replacing_symbols("b.py", "beta_two"))
            .unwrap();

        engine
            .graph_facts(
                &store,
                &anchors,
                2,
                GraphKinds::default(),
                GraphCaps::default(),
            )
            .expect("graph");
        assert!(engine.is_stale(), "get_graph read a newer generation");

        let mut second = Engine::build(&repo.0, &mut store).expect("rebuild");
        assert!(!second.is_stale());
        other
            .publish_batch(batch_replacing_symbols("b.py", "beta_three"))
            .unwrap();
        second
            .impact_from_files(&mut store, &anchors)
            .expect("impact");
        assert!(second.is_stale(), "expansion read a newer generation");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_domain::SymbolKind;

    const SRC: &str = "\
use std::io;

fn first() {
    let x = 1;
}

fn second() {
    let y = 2;
}
";

    fn symbol(name: &str, start_line: usize, end_line: usize) -> SymbolRecord {
        SymbolRecord {
            name: name.to_owned(),
            kind: SymbolKind::Function,
            path: "a.rs".to_owned(),
            start_line,
            end_line,
            signature: format!("fn {name}()"),
        }
    }

    #[test]
    fn query_identifiers_keeps_whole_symbol_names() {
        // The exact failure from langchain #38366: the query names merge_dicts,
        // and the tokenizer used to shatter it into merge + dicts.
        let ids = query_identifiers("merge_dicts doubles model_name and finish_reason");
        assert!(ids.contains(&"merge_dicts".to_owned()), "{ids:?}");
        assert!(ids.contains(&"model_name".to_owned()), "{ids:?}");
    }

    #[test]
    fn query_identifiers_skips_plain_english_words() {
        // Bare words are not symbol lookups; only identifier-shaped tokens are.
        let ids = query_identifiers("fix the retry logic in the parser");
        assert!(ids.is_empty(), "{ids:?}");
    }

    #[test]
    fn query_identifiers_recovers_a_dotted_tail() {
        let ids = query_identifiers("utils.merge_dicts is wrong");
        assert!(ids.contains(&"merge_dicts".to_owned()), "{ids:?}");
    }

    #[test]
    fn a_file_without_symbols_still_gets_one_chunk() {
        // A doc with no chunks could never be returned by vector search, and
        // symbol extraction is lazy, so this is a normal state not an edge case.
        let planned = plan_chunks(SRC, SRC, &[]);
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].0, 0, "head chunk is keyed at line 0");
        assert!(planned[0].1.is_none());
        assert!(planned[0].2.contains("use std::io;"));
    }

    #[test]
    fn each_symbol_becomes_its_own_chunk_holding_its_definition() {
        let planned = plan_chunks(SRC, SRC, &[symbol("first", 3, 5), symbol("second", 7, 9)]);
        assert_eq!(planned.len(), 3, "head chunk plus one per symbol");

        let second = planned
            .iter()
            .find(|(_, name, _)| name.as_deref() == Some("second"))
            .expect("second was planned");
        assert_eq!(second.0, 7, "keyed by its start line");
        assert!(second.2.starts_with("fn second()"), "got: {}", second.2);
        assert!(second.2.contains("let y = 2;"), "got: {}", second.2);
        assert!(
            !second.2.contains("first"),
            "bled into the neighbour: {}",
            second.2
        );
    }

    #[test]
    fn unusable_spans_are_skipped_rather_than_panicking() {
        // start_line 0 comes from a pre-span database; inverted and
        // past-the-end spans come from a file edited since extraction.
        let planned = plan_chunks(
            SRC,
            SRC,
            &[
                symbol("spanless", 0, 0),
                symbol("inverted", 9, 3),
                symbol("past_eof", 900, 950),
            ],
        );
        assert_eq!(planned.len(), 1, "only the head chunk survives");
    }

    #[test]
    fn a_span_running_past_a_shrunken_file_is_clamped() {
        let planned = plan_chunks(SRC, SRC, &[symbol("truncated", 7, 9_999)]);
        let chunk = &planned[1];
        assert_eq!(chunk.1.as_deref(), Some("truncated"));
        assert!(chunk.2.contains("let y = 2;"));
    }

    #[test]
    fn chunk_text_is_capped() {
        let huge = format!("fn big() {{\n{}\n}}\n", "    let x = 1;\n".repeat(2_000));
        let planned = plan_chunks(&huge, &huge, &[symbol("big", 1, 2_002)]);
        assert!(planned[1].2.chars().count() <= CHUNK_BYTES);
    }

    #[test]
    fn a_matching_cached_hash_is_reused_and_a_stale_one_is_not() {
        let embedder = HashedNgramEmbedder::default();
        let vector = embedder.embed("fn retry()");
        let bytes = embed::vector_to_bytes(&vector);

        let (reused, from_cache) = embed_or_reuse(
            &embedder,
            Some(&("h1".to_owned(), bytes.clone())),
            "fn retry()",
            "h1",
        );
        assert!(from_cache);
        assert_eq!(reused, vector);

        let (_, from_cache) = embed_or_reuse(
            &embedder,
            Some(&("stale".to_owned(), bytes)),
            "fn retry()",
            "h1",
        );
        assert!(!from_cache, "a changed hash must force a re-embed");

        let (_, from_cache) = embed_or_reuse(&embedder, None, "fn retry()", "h1");
        assert!(!from_cache);
    }

    #[test]
    fn corrupt_cached_bytes_fall_back_to_embedding() {
        let embedder = HashedNgramEmbedder::default();
        let (_, from_cache) = embed_or_reuse(
            &embedder,
            Some(&("h1".to_owned(), vec![1u8, 2, 3])),
            "fn retry()",
            "h1",
        );
        assert!(!from_cache, "wrong-length blob must not be trusted");
    }

    /// Throwaway repo root, removed on drop. Not `tempfile`: CI builds
    /// `--locked`, so a dev-dependency would need a Cargo.lock regeneration.
    struct TempRepo(PathBuf);

    impl TempRepo {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "engram-retrieval-test-{}-{tag}",
                std::process::id()
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

    /// Four modules: `alpha` imports `beta`, `beta` imports `gamma`, and
    /// `delta` has no import edges.
    fn connection_fixture(tag: &str) -> (TempRepo, Store, Engine) {
        use engram_domain::{FileRecord, Language};
        let repo = TempRepo::new(tag);
        std::fs::create_dir_all(repo.0.join("app")).unwrap();
        let mut files = Vec::new();
        for n in ["alpha", "beta", "gamma", "delta"] {
            std::fs::write(
                repo.0.join(format!("app/{n}.py")),
                format!("def {n}(): pass\n"),
            )
            .unwrap();
            files.push(FileRecord {
                path: format!("app/{n}.py"),
                language: Language::Python,
                size_bytes: 10,
                is_test: false,
            });
        }
        let mut store = Store::open(&repo.0).expect("open store");
        store.upsert_files(&files).expect("upsert files");
        store
            .replace_imports_for_file("app/alpha.py", &["beta".to_owned()])
            .unwrap();
        store
            .replace_imports_for_file("app/beta.py", &["gamma".to_owned()])
            .unwrap();
        let engine = Engine::build(&repo.0, &mut store).expect("build engine");
        (repo, store, engine)
    }

    fn cochange(a: &str, b: &str, count: u32, strength: f32) -> engram_domain::CoChange {
        engram_domain::CoChange {
            path_a: a.to_owned(),
            path_b: b.to_owned(),
            count,
            strength,
        }
    }

    fn run_git(dir: &std::path::Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args([
                "-c",
                "commit.gpgsign=false",
                "-c",
                "init.defaultBranch=main",
            ])
            .args(args)
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "Anonymous")
            .env("GIT_AUTHOR_EMAIL", "anonymous@example.invalid")
            .env("GIT_COMMITTER_NAME", "Anonymous")
            .env("GIT_COMMITTER_EMAIL", "anonymous@example.invalid")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .output()
            .expect("run git");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// A real repository: `a.py` and `b.py` change together in three commits,
    /// `a.py` changes alone once, one empty commit lists no files, and one
    /// bulk commit touches 31 files (past the cap), `a.py` and `b.py` among
    /// them. Returns `None` when `git` is unavailable.
    fn history_fixture(tag: &str) -> Option<TempRepo> {
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("SKIPPED: git is not available");
            return None;
        }
        let repo = TempRepo::new(tag);
        let dir = repo.0.as_path();
        std::fs::create_dir_all(dir.join("app")).unwrap();
        run_git(dir, &["init", "-q"]);
        let write = |name: &str, body: &str| {
            std::fs::write(dir.join("app").join(name), body).unwrap();
        };
        for round in 1..=3 {
            write("a.py", &format!("def a(): return {round}\n"));
            write("b.py", &format!("def b(): return {round}\n"));
            run_git(dir, &["add", "app/a.py", "app/b.py"]);
            run_git(dir, &["commit", "-q", "-m", &format!("together {round}")]);
        }
        write("a.py", "def a(): return 'alone'\n");
        run_git(dir, &["add", "app/a.py"]);
        run_git(dir, &["commit", "-q", "-m", "a alone"]);
        run_git(dir, &["commit", "-q", "--allow-empty", "-m", "no files"]);
        write("a.py", "def a(): return 'bulk'\n");
        write("b.py", "def b(): return 'bulk'\n");
        for n in 0..29 {
            write(&format!("bulk{n:02}.py"), "x = 1\n");
        }
        run_git(dir, &["add", "app"]);
        run_git(dir, &["commit", "-q", "-m", "mass change"]);
        Some(repo)
    }

    fn engine_over(repo: &TempRepo) -> (Store, Engine) {
        engram_repo_map::index_repo(&repo.0, 100).expect("index");
        let mut store = Store::open(&repo.0).expect("open store");
        let engine = Engine::build(&repo.0, &mut store).expect("build engine");
        (store, engine)
    }

    #[test]
    fn the_cochange_percent_comes_from_the_counts_when_the_denominator_is_known() {
        // The counts decide, not the f32 strength (here deliberately off).
        assert_eq!(cochange_percent(29, Some(100), 0.2899), 29);
        // Rounded to nearest, in integers.
        assert_eq!(cochange_percent(1, Some(3), 1.0 / 3.0), 33);
        assert_eq!(cochange_percent(2, Some(3), 2.0 / 3.0), 67);
        assert_eq!(cochange_percent(3, Some(4), 0.75), 75);
        assert_eq!(cochange_percent(1, Some(8), 0.125), 13);
        assert_eq!(cochange_percent(4, Some(4), 1.0), 100);
        // Unknown denominator (or a degenerate zero): the stored strength, rounded.
        assert_eq!(cochange_percent(29, None, 0.29), 29);
        assert_eq!(cochange_percent(2, Some(0), 0.666), 67);
    }

    #[test]
    fn explain_cochange_reports_opportunities_and_bulk_exclusions_from_real_history() {
        let Some(repo) = history_fixture("cochange-facts") else {
            return;
        };
        let (mut store, mut engine) = engine_over(&repo);
        let explanation = engine
            .explain_connection(&mut store, "app/a.py", "app/b.py")
            .unwrap();

        // 3 together + 1 alone + 1 empty + 1 bulk = 6 commits walked.
        assert_eq!(explanation.scope.commits_scanned, Some(6));
        assert_eq!(explanation.scope.bulk_commits_excluded, Some(1));

        let json = serde_json::to_value(&explanation).unwrap();
        assert_eq!(json["scope"]["commits_scanned"], 6);
        assert_eq!(json["scope"]["bulk_commits_excluded"], 1);
        let co: Vec<&serde_json::Value> = json["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["type"] == "historical_cochange")
            .collect();
        assert_eq!(co.len(), 2, "{json}");
        let forward = co.iter().find(|r| r["direction"] == "source_to_target");
        let forward = forward.expect("a -> b");
        // a.py: 3 shared + 1 alone = 4 non-bulk commits; b.py: 3. The bulk
        // commit counts for neither.
        assert_eq!(forward["count"], 3);
        assert_eq!(forward["opportunities_source"], 4);
        assert_eq!(forward["opportunities_target"], 3);
        assert_eq!(forward["strength"], 0.75);
        let detail = forward["detail"].as_str().unwrap();
        assert!(
            detail.contains(
                "changed together in 3 of the 4 non-bulk commits that touched app/a.py (75%)"
            ),
            "{detail}"
        );
        let reverse = co
            .iter()
            .find(|r| r["direction"] == "target_to_source")
            .expect("b -> a");
        assert_eq!(reverse["opportunities_source"], 4);
        assert_eq!(reverse["opportunities_target"], 3);
        assert_eq!(reverse["strength"], 1.0);
        assert!(reverse["detail"].as_str().unwrap().contains(
            "changed together in 3 of the 3 non-bulk commits that touched app/b.py (100%)"
        ));
        // The wording is a statement of fact, not of cause.
        for r in &co {
            let d = r["detail"].as_str().unwrap();
            assert!(!d.contains("because") && !d.contains("caused"), "{d}");
        }
    }

    #[test]
    fn explain_cochange_facts_are_null_not_zero_on_an_index_that_predates_them() {
        let Some(repo) = history_fixture("cochange-old-db") else {
            return;
        };
        let (mut store, mut engine) = engine_over(&repo);
        // Make the database look like one written before the facts existed.
        store
            .conn
            .execute_batch(
                "DROP TABLE cochange_file_commits;
                 DELETE FROM meta WHERE key IN
                   ('cochange_commits_scanned', 'cochange_bulk_commits_excluded');",
            )
            .unwrap();

        let old = serde_json::to_value(
            engine
                .explain_connection(&mut store, "app/a.py", "app/b.py")
                .unwrap(),
        )
        .unwrap();
        assert!(old["scope"]["commits_scanned"].is_null(), "{old}");
        assert!(old["scope"]["bulk_commits_excluded"].is_null(), "{old}");
        let first = old["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["type"] == "historical_cochange")
            .expect("the pair itself is still reported");
        assert!(first["opportunities_source"].is_null(), "{first}");
        assert!(first["opportunities_target"].is_null(), "{first}");
        assert_eq!(first["count"], 3);
        // The wording falls back to the one that claims nothing it cannot back.
        let fallback = first["detail"].as_str().unwrap();
        assert!(
            fallback.contains("of the non-bulk commits that touched")
                && fallback.contains("not recorded in this index"),
            "{fallback}"
        );

        // Reopening migrates the schema; the next history refresh repopulates.
        drop(store);
        let mut store = Store::open(&repo.0).expect("reopen migrates");
        engram_repo_map::refresh_history(&mut store, &repo.0).unwrap();
        let fresh = serde_json::to_value(
            engine
                .explain_connection(&mut store, "app/a.py", "app/b.py")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(fresh["scope"]["commits_scanned"], 6);
        assert_eq!(fresh["scope"]["bulk_commits_excluded"], 1);
    }

    #[test]
    fn explain_reports_a_direct_import_in_each_direction() {
        let (_repo, mut store, mut engine) = connection_fixture("direct");

        let fwd = engine
            .explain_connection(&mut store, "app/alpha.py", "app/beta.py")
            .unwrap()
            .reasons;
        assert_eq!(fwd.len(), 1, "{fwd:?}");
        let ConnectionReason::Import {
            direction,
            hops,
            chain,
            weight,
            resolution,
            detail,
            ..
        } = &fwd[0]
        else {
            panic!("expected an import reason: {fwd:?}");
        };
        assert_eq!(*direction, ImportDirection::SourceImportsTarget);
        assert_eq!(*hops, 1);
        assert_eq!(chain, &["app/alpha.py", "app/beta.py"]);
        assert_eq!(*weight, 0.6);
        assert_eq!(*resolution, "heuristic");
        assert!(
            detail.contains("imports") && detail.contains("heuristic"),
            "{detail}"
        );

        // Same pair, arguments swapped: the importer is now the target.
        let rev = engine
            .explain_connection(&mut store, "app/beta.py", "app/alpha.py")
            .unwrap()
            .reasons;
        assert_eq!(rev.len(), 1, "{rev:?}");
        let ConnectionReason::Import {
            direction, chain, ..
        } = &rev[0]
        else {
            panic!("expected an import reason: {rev:?}");
        };
        assert_eq!(*direction, ImportDirection::TargetImportsSource);
        assert_eq!(chain, &["app/alpha.py", "app/beta.py"]);
    }

    #[test]
    fn explain_names_the_intermediate_file_of_a_two_hop_chain() {
        let (_repo, mut store, mut engine) = connection_fixture("twohop");
        let rs = engine
            .explain_connection(&mut store, "app/alpha.py", "app/gamma.py")
            .unwrap()
            .reasons;
        assert_eq!(rs.len(), 1, "{rs:?}");
        let ConnectionReason::Import {
            hops,
            chain,
            weight,
            detail,
            direction,
            ..
        } = &rs[0]
        else {
            panic!("expected an import reason: {rs:?}");
        };
        assert_eq!(*hops, 2);
        assert_eq!(chain, &["app/alpha.py", "app/beta.py", "app/gamma.py"]);
        assert_eq!(*weight, 0.35);
        assert_eq!(*direction, ImportDirection::SourceImportsTarget);
        assert!(
            detail.contains("app/beta.py"),
            "intermediate missing: {detail}"
        );
        assert!(
            !detail.contains("imports app/gamma.py directly"),
            "a 2-hop chain must not read as a direct import: {detail}"
        );
    }

    #[test]
    fn explain_finds_a_cochange_pair_ranked_beyond_the_top_five() {
        let (_repo, mut store, mut engine) = connection_fixture("rank");
        // Seven stronger partners for alpha; delta ranks eighth by strength.
        let mut edges: Vec<_> = (0..7)
            .map(|i| cochange("app/alpha.py", &format!("app/other{i}.py"), 40, 0.9))
            .collect();
        edges.push(cochange("app/alpha.py", "app/delta.py", 3, 0.12));
        store.replace_cochange(&edges).unwrap();

        // The capped recommendation path cannot see delta...
        let impact = engine
            .impact_from_files(&mut store, &["app/alpha.py".to_owned()])
            .unwrap();
        assert!(!impact
            .cochange_expansions
            .iter()
            .any(|p| p.path == "app/delta.py"));

        // ...the pairwise lookup does.
        let rs = engine
            .explain_connection(&mut store, "app/alpha.py", "app/delta.py")
            .unwrap()
            .reasons;
        assert_eq!(rs.len(), 1, "{rs:?}");
        let ConnectionReason::CoChange {
            direction,
            count,
            strength,
            weight,
            ..
        } = &rs[0]
        else {
            panic!("expected a co-change reason: {rs:?}");
        };
        assert_eq!(*direction, CoChangeDirection::SourceToTarget);
        assert_eq!(*count, 3);
        assert_eq!(*strength, 0.12);
        assert_eq!(*weight, 0.12);
    }

    #[test]
    fn explain_keeps_both_the_import_and_the_cochange_for_one_pair() {
        let (_repo, mut store, mut engine) = connection_fixture("both");
        store
            .replace_cochange(&[cochange("app/alpha.py", "app/beta.py", 5, 0.5)])
            .unwrap();
        let rs = engine
            .explain_connection(&mut store, "app/alpha.py", "app/beta.py")
            .unwrap()
            .reasons;
        assert_eq!(rs.len(), 2, "{rs:?}");
        assert!(rs
            .iter()
            .any(|r| matches!(r, ConnectionReason::Import { hops: 1, .. })));
        assert!(rs
            .iter()
            .any(|r| matches!(r, ConnectionReason::CoChange { count: 5, .. })));
    }

    #[test]
    fn explain_reports_the_reverse_cochange_row_with_its_own_numbers() {
        let (_repo, mut store, mut engine) = connection_fixture("reverse");
        // Only delta -> alpha was recorded; asking alpha -> delta must find it.
        store
            .replace_cochange(&[cochange("app/delta.py", "app/alpha.py", 4, 0.8)])
            .unwrap();
        let rs = engine
            .explain_connection(&mut store, "app/alpha.py", "app/delta.py")
            .unwrap()
            .reasons;
        assert_eq!(rs.len(), 1, "{rs:?}");
        assert!(matches!(
            &rs[0],
            ConnectionReason::CoChange {
                direction: CoChangeDirection::TargetToSource,
                count: 4,
                ..
            }
        ));

        // Both rows present: both directions are reported, distinctly.
        store
            .replace_cochange(&[
                cochange("app/delta.py", "app/alpha.py", 4, 0.8),
                cochange("app/alpha.py", "app/delta.py", 6, 0.1),
            ])
            .unwrap();
        let rs = engine
            .explain_connection(&mut store, "app/alpha.py", "app/delta.py")
            .unwrap()
            .reasons;
        assert_eq!(rs.len(), 2, "{rs:?}");
        assert!(rs.iter().any(|r| matches!(
            r,
            ConnectionReason::CoChange {
                direction: CoChangeDirection::SourceToTarget,
                count: 6,
                ..
            }
        )));
    }

    #[test]
    fn explain_serializes_with_the_original_keys_plus_typed_fields() {
        let (_repo, mut store, mut engine) = connection_fixture("json");
        store
            .replace_cochange(&[cochange("app/alpha.py", "app/gamma.py", 2, 0.25)])
            .unwrap();
        let rs = engine
            .explain_connection(&mut store, "app/alpha.py", "app/gamma.py")
            .unwrap()
            .reasons;
        let v = serde_json::to_value(&rs).unwrap();
        let by_type = |t: &str| {
            v.as_array()
                .unwrap()
                .iter()
                .find(|r| r["type"] == t)
                .unwrap()
        };
        let imp = by_type("import_edge");
        assert_eq!(imp["direction"], "source_imports_target");
        assert_eq!(imp["hops"], 2);
        assert_eq!(imp["resolution"], "heuristic");
        assert_eq!(
            imp["chain"],
            serde_json::json!(["app/alpha.py", "app/beta.py", "app/gamma.py"])
        );
        assert!(imp["detail"].is_string() && imp["weight"].is_number());
        let co = by_type("historical_cochange");
        assert_eq!(co["direction"], "source_to_target");
        assert_eq!(co["count"], 2);
        assert_eq!(co["strength"], 0.25);
        assert!(co["detail"].is_string() && co["weight"].is_number());
    }

    #[test]
    fn explain_with_no_recorded_facts_states_what_was_searched() {
        let (_repo, mut store, mut engine) = connection_fixture("none");
        let ex = engine
            .explain_connection(&mut store, "app/delta.py", "app/gamma.py")
            .unwrap();
        assert!(ex.reasons.is_empty(), "{ex:?}");
        assert_eq!(ex.status, ConnectionStatus::NotFoundWithinScope);

        let v = serde_json::to_value(&ex).unwrap();
        assert_eq!(v["status"], "not_found_within_scope");
        let scope = &v["scope"];
        assert_eq!(scope["import_max_hops"], 2);
        assert_eq!(
            scope["import_directions_searched"],
            serde_json::json!(["source_imports_target", "target_imports_source"])
        );
        assert_eq!(
            scope["import_search_source_imports_target"],
            "none_within_bound"
        );
        assert_eq!(scope["cochange_lookup"], "exact_pair");
        assert_eq!(
            scope["cochange_directions_searched"],
            serde_json::json!(["source_to_target", "target_to_source"])
        );
        assert_eq!(
            scope["cochange_window"],
            serde_json::json!({
                "max_commits": engram_repo_map::cochange::MAX_COMMITS,
                "min_count": engram_repo_map::cochange::MIN_PAIR_COUNT,
                "max_files_per_commit": engram_repo_map::cochange::MAX_FILES_PER_COMMIT,
            })
        );
        assert!(scope["note"]
            .as_str()
            .unwrap()
            .contains("not that the files are unrelated"));
    }

    /// Five modules in a line: aaa imports bbb imports ccc imports ddd imports eee.
    fn long_chain_fixture(tag: &str) -> (TempRepo, Store) {
        use engram_domain::{FileRecord, Language};
        let repo = TempRepo::new(tag);
        std::fs::create_dir_all(repo.0.join("app")).unwrap();
        let names = ["aaa", "bbb", "ccc", "ddd", "eee"];
        let mut files = Vec::new();
        for n in names {
            std::fs::write(repo.0.join(format!("app/{n}.py")), "pass\n").unwrap();
            files.push(FileRecord {
                path: format!("app/{n}.py"),
                language: Language::Python,
                size_bytes: 5,
                is_test: false,
            });
        }
        let mut store = Store::open(&repo.0).expect("open store");
        store.upsert_files(&files).unwrap();
        for w in names.windows(2) {
            store
                .replace_imports_for_file(&format!("app/{}.py", w[0]), &[w[1].to_owned()])
                .unwrap();
        }
        (repo, store)
    }

    #[test]
    fn explain_is_unknown_beyond_bound_not_absent_when_the_chain_is_too_long() {
        let (repo, mut store) = long_chain_fixture("beyond");
        let mut engine = Engine::build(&repo.0, &mut store).unwrap();
        // aaa -> bbb -> ccc -> ddd is 3 hops: past the 2-hop bound.
        let ex = engine
            .explain_connection(&mut store, "app/aaa.py", "app/ddd.py")
            .unwrap();
        assert!(ex.reasons.is_empty(), "{ex:?}");
        assert_eq!(ex.status, ConnectionStatus::UnknownBeyondBound);
        assert_eq!(
            ex.scope.import_search_source_imports_target,
            ImportSearchOutcome::BeyondBound
        );
        assert_eq!(ex.scope.import_max_hops, 2);
        let v = serde_json::to_value(&ex).unwrap();
        assert_eq!(v["status"], "unknown_beyond_bound");
        assert_eq!(v["scope"]["import_max_hops"], 2);

        // Opposite direction has nothing to explore, but the pair as a whole
        // is still not conclusively unconnected.
        assert_eq!(
            ex.scope.import_search_target_imports_source,
            ImportSearchOutcome::NoneWithinBound
        );

        // Within the bound the same file is reachable and reported as found.
        let ex = engine
            .explain_connection(&mut store, "app/aaa.py", "app/ccc.py")
            .unwrap();
        assert_eq!(ex.status, ConnectionStatus::Found);
    }

    #[test]
    fn explain_two_hop_route_carries_typed_edge_evidence() {
        let (repo, mut store) = long_chain_fixture("typed");
        let mut engine = Engine::build(&repo.0, &mut store).unwrap();
        let ex = engine
            .explain_connection(&mut store, "app/bbb.py", "app/ddd.py")
            .unwrap();
        assert_eq!(ex.status, ConnectionStatus::Found);
        let v = serde_json::to_value(&ex).unwrap();
        let imp = &v["reasons"][0];
        assert_eq!(imp["type"], "import_edge");
        assert_eq!(imp["edge_kind"], "import");
        assert_eq!(imp["resolver"], "heuristic_module_segments");
        assert_eq!(imp["resolution_status"], "candidate");
        assert_eq!(imp["hops"], 2);
        assert_eq!(imp["chain"].as_array().unwrap().len(), 3);
        assert_eq!(imp["chain"][1], "app/ccc.py");
        // Each hop also carries the id of that single edge: the one
        // `get_graph` emits for it, not the id of the whole route.
        let snapshot = engine.snapshot_id();
        assert_eq!(
            imp["edges"],
            serde_json::json!([
                {
                    "from": "app/bbb.py",
                    "to": "app/ccc.py",
                    "edge_id": import_edge_id(snapshot, &["app/bbb.py", "app/ccc.py"]),
                },
                {
                    "from": "app/ccc.py",
                    "to": "app/ddd.py",
                    "edge_id": import_edge_id(snapshot, &["app/ccc.py", "app/ddd.py"]),
                },
            ])
        );
        assert_eq!(
            imp["edge_id"],
            import_edge_id(snapshot, &["app/bbb.py", "app/ccc.py", "app/ddd.py"])
        );
    }

    #[test]
    fn explain_returns_imports_in_both_directions_and_cochange_alongside() {
        let (repo, mut store) = long_chain_fixture("bidir");
        // aaa already imports bbb; make bbb import aaa as well.
        store
            .replace_imports_for_file("app/bbb.py", &["aaa".to_owned(), "ccc".to_owned()])
            .unwrap();
        store
            .replace_cochange(&[cochange("app/aaa.py", "app/bbb.py", 7, 0.7)])
            .unwrap();
        let mut engine = Engine::build(&repo.0, &mut store).unwrap();
        let rs = engine
            .explain_connection(&mut store, "app/aaa.py", "app/bbb.py")
            .unwrap()
            .reasons;
        let dirs: Vec<_> = rs
            .iter()
            .filter_map(|r| match r {
                ConnectionReason::Import { direction, .. } => Some(*direction),
                _ => None,
            })
            .collect();
        assert_eq!(dirs.len(), 2, "{rs:?}");
        assert!(dirs.contains(&ImportDirection::SourceImportsTarget));
        assert!(dirs.contains(&ImportDirection::TargetImportsSource));
        let co = rs
            .iter()
            .find_map(|r| match r {
                ConnectionReason::CoChange {
                    detail,
                    edge_kind,
                    window,
                    ..
                } => Some((detail, *edge_kind, *window)),
                _ => None,
            })
            .expect("co-change fact kept alongside imports");
        assert_eq!(co.1, "cochange");
        assert_eq!(co.2.min_count, engram_repo_map::cochange::MIN_PAIR_COUNT);
        assert!(co.0.contains("changed together in 7 commits"), "{}", co.0);
        assert!(
            !co.0.contains("must") && !co.0.contains("required"),
            "{}",
            co.0
        );
    }

    #[test]
    fn explain_rejects_unknown_paths_and_identical_endpoints() {
        let (_repo, mut store, mut engine) = connection_fixture("errors");
        let err = engine
            .explain_connection(&mut store, "app/alpha.py", "app/missing.py")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("app/missing.py")
                && err.contains("not an indexed text file in the Engram index"),
            "{err}"
        );
        let err = engine
            .explain_connection(&mut store, "nope.py", "app/alpha.py")
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope.py"), "{err}");
        let err = engine
            .explain_connection(&mut store, "app/alpha.py", "app/alpha.py")
            .unwrap_err()
            .to_string();
        assert!(err.contains("same file"), "{err}");
    }

    #[test]
    fn expansion_does_not_describe_a_two_hop_dependent_as_a_direct_importer() {
        let (_repo, mut store, mut engine) = connection_fixture("expansion");
        let impact = engine
            .impact_from_files(&mut store, &["app/gamma.py".to_owned()])
            .unwrap();
        let reason = |p: &str| {
            impact
                .import_expansions
                .iter()
                .find(|s| s.path == p)
                .unwrap_or_else(|| panic!("{p} missing: {impact:?}"))
                .reason
                .clone()
        };
        let one = reason("app/beta.py");
        assert!(one.contains("1 hop"), "{one}");
        let two = reason("app/alpha.py");
        assert!(
            two.contains("2 hops") && two.contains("imports a file that imports"),
            "{two}"
        );
    }

    /// What the history walk stores: every pair in both directions, the row
    /// back carrying its own strength (relative to its own source's commits).
    fn mirrored(forward: Vec<engram_domain::CoChange>) -> Vec<engram_domain::CoChange> {
        let back: Vec<engram_domain::CoChange> = forward
            .iter()
            .map(|e| {
                cochange(
                    &e.path_b,
                    &e.path_a,
                    e.count,
                    e.count as f32 / (e.count as f32 + 4.0),
                )
            })
            .collect();
        forward.into_iter().chain(back).collect()
    }

    /// A real repository where `app/anchor.py` co-changed with `partners`
    /// files, partner `i` in `partners + 2 - i` commits (strictly decreasing),
    /// and `importers` files import the anchor. The walk that indexing runs
    /// turns this into the table production holds. `None` when `git` is
    /// unavailable.
    fn partner_history_repo(tag: &str, partners: usize, importers: usize) -> Option<TempRepo> {
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("SKIPPED: git is not available");
            return None;
        }
        let repo = TempRepo::new(tag);
        let dir = repo.0.as_path();
        std::fs::create_dir_all(dir.join("app")).unwrap();
        run_git(dir, &["init", "-q"]);
        let write = |name: &str, body: &str| {
            std::fs::write(dir.join("app").join(name), body).unwrap();
        };
        for i in 0..importers {
            write(
                &format!("imp{i:02}.py"),
                "from app.anchor import anchor\n\ndef use(): return anchor()\n",
            );
        }
        for round in 0..=partners + 1 {
            write("anchor.py", &format!("def anchor(): return {round}\n"));
            for i in 0..partners {
                if round == 0 || partners + 1 - i >= round {
                    write(
                        &format!("p{i:02}.py"),
                        &format!("def p{i}(): return {round}\n"),
                    );
                }
            }
            run_git(dir, &["add", "app"]);
            run_git(dir, &["commit", "-q", "-m", &format!("round {round}")]);
        }
        Some(repo)
    }

    /// One anchor, `partners` indexed files it co-changed with (strengths
    /// strictly decreasing, so the ranking is unambiguous; each pair stored in
    /// both directions, as production stores it) and `importers` indexed files
    /// that import it.
    fn fan_fixture(tag: &str, partners: usize, importers: usize) -> (TempRepo, Store, Engine) {
        use engram_domain::{FileRecord, Language};
        let repo = TempRepo::new(tag);
        std::fs::create_dir_all(repo.0.join("app")).unwrap();
        let mut names = vec!["anchor".to_owned(), "other".to_owned()];
        names.extend((0..partners).map(|i| format!("p{i:02}")));
        names.extend((0..importers).map(|i| format!("imp{i:02}")));
        let mut files = Vec::new();
        for n in &names {
            std::fs::write(
                repo.0.join(format!("app/{n}.py")),
                format!("def {n}(): pass\n"),
            )
            .unwrap();
            files.push(FileRecord {
                path: format!("app/{n}.py"),
                language: Language::Python,
                size_bytes: 10,
                is_test: false,
            });
        }
        let mut store = Store::open(&repo.0).expect("open store");
        store.upsert_files(&files).expect("upsert files");
        for i in 0..importers {
            store
                .replace_imports_for_file(&format!("app/imp{i:02}.py"), &["anchor".to_owned()])
                .unwrap();
        }
        let edges: Vec<_> = (0..partners)
            .map(|i| {
                cochange(
                    "app/anchor.py",
                    &format!("app/p{i:02}.py"),
                    50 - i as u32,
                    0.9 - i as f32 * 0.01,
                )
            })
            .collect();
        store.replace_cochange(&mirrored(edges)).unwrap();
        let engine = Engine::build(&repo.0, &mut store).expect("build engine");
        (repo, store, engine)
    }

    fn dropped_of(impact: &ImpactPrediction, kind: &str) -> Vec<String> {
        impact
            .dropped
            .iter()
            .filter(|d| d.kind == kind)
            .map(|d| d.path.clone())
            .collect()
    }

    #[test]
    fn cap_disclosure_counts_the_cochange_rows_past_the_per_anchor_cap() {
        let (_repo, mut store, mut engine) = fan_fixture("cap-per-anchor", 12, 0);
        let impact = engine
            .impact_from_files(&mut store, &["app/anchor.py".to_owned()])
            .unwrap();
        // Unchanged behaviour: the top five by strength are returned.
        let returned: Vec<&str> = impact
            .cochange_expansions
            .iter()
            .map(|p| p.path.as_str())
            .collect();
        assert_eq!(
            returned,
            [
                "app/p00.py",
                "app/p01.py",
                "app/p02.py",
                "app/p03.py",
                "app/p04.py"
            ]
        );
        // Disclosure: exactly the seven that were cut, none of the returned.
        let dropped = dropped_of(&impact, "cochange");
        assert_eq!(dropped.len(), 7, "{dropped:?}");
        assert_eq!(dropped.len() + returned.len(), 12);
        assert!(dropped.iter().all(|d| !returned.contains(&d.as_str())));
        assert!(dropped.contains(&"app/p11.py".to_owned()));
        assert!(dropped_of(&impact, "import").is_empty());
    }

    #[test]
    fn cap_disclosure_counts_entries_past_the_per_kind_limit() {
        // Two anchors with five distinct partners each: ten unique candidates
        // survive the per-anchor cap, eight are returned, two are cut.
        let (_repo, mut store, mut engine) = fan_fixture("cap-per-kind", 10, 11);
        store
            .replace_cochange(
                &(0..5)
                    .map(|i| cochange("app/anchor.py", &format!("app/p{i:02}.py"), 9, 0.9))
                    .chain(
                        (5..10)
                            .map(|i| cochange("app/other.py", &format!("app/p{i:02}.py"), 9, 0.5)),
                    )
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let impact = engine
            .impact_from_files(
                &mut store,
                &["app/anchor.py".to_owned(), "app/other.py".to_owned()],
            )
            .unwrap();
        assert_eq!(impact.cochange_expansions.len(), 8);
        let dropped_co = dropped_of(&impact, "cochange");
        assert_eq!(dropped_co, ["app/p08.py", "app/p09.py"]);

        // Eleven direct importers, eight returned, three cut.
        assert_eq!(impact.import_expansions.len(), 8);
        let dropped_imp = dropped_of(&impact, "import");
        assert_eq!(dropped_imp.len(), 3, "{dropped_imp:?}");
        let returned: Vec<&str> = impact
            .import_expansions
            .iter()
            .map(|p| p.path.as_str())
            .collect();
        assert!(dropped_imp.iter().all(|d| !returned.contains(&d.as_str())));
        // Equal confidence is ordered by path, so the cut is the last three.
        assert_eq!(
            dropped_imp,
            ["app/imp08.py", "app/imp09.py", "app/imp10.py"]
        );
    }

    #[test]
    fn nothing_is_disclosed_when_no_cap_applied() {
        let (_repo, mut store, mut engine) = fan_fixture("cap-none", 3, 2);
        let impact = engine
            .impact_from_files(&mut store, &["app/anchor.py".to_owned()])
            .unwrap();
        assert_eq!(impact.cochange_expansions.len(), 3);
        assert_eq!(impact.import_expansions.len(), 2);
        assert!(impact.dropped.is_empty(), "{:?}", impact.dropped);
        // The disclosure is not part of the prediction's own JSON.
        let json = serde_json::to_value(&impact).unwrap();
        assert!(json.get("dropped").is_none());
    }

    #[test]
    fn expansion_ignores_anchor_order_and_duplicates() {
        let (_repo, mut store, mut engine) = fan_fixture("anchor-order", 12, 3);
        let run = |engine: &mut Engine, store: &mut Store, anchors: &[&str]| {
            let anchors: Vec<String> = anchors.iter().map(|s| (*s).to_owned()).collect();
            let impact = engine.impact_from_files(store, &anchors).unwrap();
            (
                serde_json::to_string(&impact).unwrap(),
                impact
                    .dropped
                    .iter()
                    .map(|d| (d.kind, d.path.clone()))
                    .collect::<Vec<_>>(),
            )
        };
        let a = run(&mut engine, &mut store, &["app/anchor.py", "app/other.py"]);
        let b = run(
            &mut engine,
            &mut store,
            &["app/other.py", "app/anchor.py", "app/anchor.py"],
        );
        assert_eq!(a, b);
    }

    #[test]
    fn max_hops_one_excludes_two_hop_dependents() {
        let (_repo, mut store, mut engine) = connection_fixture("hops");
        let anchors = ["app/gamma.py".to_owned()];
        let paths = |impact: &ImpactPrediction| -> Vec<String> {
            impact
                .import_expansions
                .iter()
                .map(|p| p.path.clone())
                .collect()
        };
        let default = engine.impact_from_files(&mut store, &anchors).unwrap();
        assert_eq!(paths(&default), ["app/beta.py", "app/alpha.py"]);
        let two = engine
            .impact_from_files_with(&mut store, &anchors, 2)
            .unwrap();
        assert_eq!(paths(&two), paths(&default));
        let one = engine
            .impact_from_files_with(&mut store, &anchors, 1)
            .unwrap();
        assert_eq!(paths(&one), ["app/beta.py"]);
    }

    #[test]
    fn max_hops_outside_one_to_two_is_an_error() {
        let (_repo, mut store, mut engine) = connection_fixture("hops-bad");
        for bad in [0, 3, 10] {
            let err = engine
                .impact_from_files_with(&mut store, &["app/gamma.py".to_owned()], bad)
                .expect_err("out-of-range max_hops must fail");
            assert!(err.to_string().contains("max_hops"), "{err}");
        }
    }

    /// Twelve files with identical contents: every score ties, so the order
    /// can only come from the tie-break rules.
    fn tied_corpus(tag: &str, reversed: bool) -> (TempRepo, Store, Engine) {
        use engram_domain::{FileRecord, Language};
        let repo = TempRepo::new(tag);
        std::fs::create_dir_all(repo.0.join("src")).unwrap();
        let mut files: Vec<FileRecord> = (0..12)
            .map(|i| FileRecord {
                path: format!("src/twin{i:02}.py"),
                language: Language::Python,
                size_bytes: 10,
                is_test: false,
            })
            .collect();
        for f in &files {
            std::fs::write(
                repo.0.join(&f.path),
                "def retry_handler():\n    return 'retry the webhook'\n",
            )
            .unwrap();
        }
        if reversed {
            files.reverse();
        }
        let mut store = Store::open(&repo.0).expect("open store");
        store.upsert_files(&files).expect("upsert files");
        let engine = Engine::build(&repo.0, &mut store).expect("build engine");
        (repo, store, engine)
    }

    #[test]
    fn search_order_is_total_and_ignores_the_order_files_were_indexed_in() {
        let render = |outcome: &SearchOutcome| {
            serde_json::to_string(&(
                &outcome.packets,
                &outcome.cap_dropped,
                &outcome.beyond_first_cap_paths,
            ))
            .unwrap()
        };
        let (_a, mut store_a, mut engine_a) = tied_corpus("tied-forward", false);
        let first = engine_a
            .search_with_outcome(&mut store_a, "retry webhook handler", 8)
            .unwrap();
        let again = engine_a
            .search_with_outcome(&mut store_a, "retry webhook handler", 8)
            .unwrap();
        assert_eq!(render(&first), render(&again));

        let (_b, mut store_b, mut engine_b) = tied_corpus("tied-reversed", true);
        let other = engine_b
            .search_with_outcome(&mut store_b, "retry webhook handler", 8)
            .unwrap();
        assert_eq!(render(&first), render(&other));

        // Ordered by the documented total order, ties broken by path.
        let mut sorted = first.packets.clone();
        sorted.sort_by(budget::compare_packets);
        assert_eq!(
            sorted.iter().map(|p| &p.id).collect::<Vec<_>>(),
            first.packets.iter().map(|p| &p.id).collect::<Vec<_>>()
        );
        assert!(first.packets.len() > 1);
    }

    /// `n` near-identical files written to disk and registered in `store` in the
    /// order `order` gives them (a permutation of `0..n`).
    fn twin_corpus(
        tag: &str,
        n: usize,
        order: impl Fn(Vec<usize>) -> Vec<usize>,
    ) -> (TempRepo, Store, Engine) {
        use engram_domain::{FileRecord, Language};
        let repo = TempRepo::new(tag);
        std::fs::create_dir_all(repo.0.join("src")).unwrap();
        let files: Vec<FileRecord> = order((0..n).collect())
            .into_iter()
            .map(|i| FileRecord {
                path: format!("src/twin{i:03}.py"),
                language: Language::Python,
                size_bytes: 10,
                is_test: false,
            })
            .collect();
        for f in &files {
            std::fs::write(
                repo.0.join(&f.path),
                "def retry_handler():\n    return 'retry the webhook'\n",
            )
            .unwrap();
        }
        let mut store = Store::open(&repo.0).expect("open store");
        store.upsert_files(&files).expect("upsert files");
        let engine = Engine::build(&repo.0, &mut store).expect("build engine");
        (repo, store, engine)
    }

    #[test]
    fn indexing_the_same_files_in_different_orders_gives_identical_search_bytes() {
        // More files than the 50-deep BM25 and vector cuts, so the cut itself is
        // exercised, not just the final sort. The corpus is indexed in natural,
        // reversed and hash-shuffled order.
        type Order = fn(Vec<usize>) -> Vec<usize>;
        let orders: [(&str, Order); 3] = [
            ("natural", |v| v),
            ("reversed", |mut v| {
                v.reverse();
                v
            }),
            ("shuffled", |mut v| {
                v.sort_by_key(|i| embed::content_hash(&format!("shuffle{i}")));
                v
            }),
        ];
        let mut outputs = Vec::new();
        for (tag, order) in orders {
            let (_repo, mut store, mut engine) = twin_corpus(&format!("order-{tag}"), 70, order);
            // The store itself reports files in path order, whatever went in.
            let paths: Vec<String> = store
                .all_files()
                .unwrap()
                .into_iter()
                .map(|f| f.path)
                .collect();
            let mut sorted = paths.clone();
            sorted.sort();
            assert_eq!(paths, sorted, "{tag}: all_files is path-ordered");

            let outcome = engine
                .search_with_outcome(&mut store, "retry webhook handler", 8)
                .unwrap();
            assert!(
                outcome.beyond_first_cap > 0,
                "{tag}: the first cap should bite"
            );
            // ... and the best of what it cut is named, never a path that was
            // returned as evidence.
            assert!(
                !outcome.beyond_first_cap_paths.is_empty()
                    && outcome.beyond_first_cap_paths.len()
                        <= outcome.beyond_first_cap.min(budget::MAX_OMITTED_LISTED),
                "{tag}: {:?}",
                outcome.beyond_first_cap_paths
            );
            for path in &outcome.beyond_first_cap_paths {
                assert!(outcome
                    .packets
                    .iter()
                    .chain(&outcome.cap_dropped)
                    .all(|p| &p.path != path));
            }
            let bytes = serde_json::to_string(&(
                &outcome.packets,
                &outcome.cap_dropped,
                outcome.beyond_first_cap,
                &outcome.beyond_first_cap_paths,
            ))
            .unwrap();
            let ranked = engine
                .rank(&mut store, "retry webhook handler", RankMode::Vector, 50)
                .unwrap();
            outputs.push((tag, bytes, ranked));
        }
        for (tag, bytes, ranked) in &outputs[1..] {
            assert_eq!(bytes, &outputs[0].1, "{tag} differs from natural order");
            assert_eq!(ranked, &outputs[0].2, "{tag}: vector top-k differs");
        }
    }

    // ----- edge ids, graph facts and cited evidence -------------------------

    /// A plain (non-git) repository holding exactly `files`, indexed for real
    /// (tree-sitter imports, content hashes), so the snapshot id follows the
    /// content.
    fn plain_repo(tag: &str, files: &[(&str, &str)]) -> TempRepo {
        let repo = TempRepo::new(tag);
        for (path, body) in files {
            let full = repo.0.join(path);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(full, body).unwrap();
        }
        repo
    }

    const CHAIN_FILES: [(&str, &str); 3] = [
        (
            "app/alpha.py",
            "from app.beta import beta\n\ndef alpha(): pass\n",
        ),
        (
            "app/beta.py",
            "from app.gamma import gamma\n\ndef beta(): pass\n",
        ),
        ("app/gamma.py", "def gamma(): pass\n"),
    ];

    /// The import route in an explanation: (whole-route id, per-hop ids).
    fn route_ids(ex: &ConnectionExplanation) -> (String, Vec<String>) {
        let v = serde_json::to_value(ex).unwrap();
        let imp = v["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["type"] == "import_edge")
            .unwrap_or_else(|| panic!("no import reason in {v}"))
            .clone();
        (
            imp["edge_id"].as_str().unwrap().to_owned(),
            imp["edges"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e["edge_id"].as_str().unwrap().to_owned())
                .collect(),
        )
    }

    fn graph_ids(facts: &GraphFacts) -> HashSet<String> {
        facts.edges.iter().map(|e| e.edge_id().to_owned()).collect()
    }

    fn cited_ids(impact: &ImpactPrediction) -> Vec<String> {
        impact
            .cochange_expansions
            .iter()
            .chain(&impact.import_expansions)
            .flat_map(|p| p.evidence_ids.clone())
            .collect()
    }

    fn all_graph(engine: &Engine, store: &Store, anchors: &[&str], max_hops: usize) -> GraphFacts {
        let anchors: Vec<String> = anchors.iter().map(|a| (*a).to_owned()).collect();
        engine
            .graph_facts(
                store,
                &anchors,
                max_hops,
                GraphKinds::default(),
                GraphCaps::default(),
            )
            .unwrap()
    }

    #[test]
    fn edge_ids_are_stable_across_builds_and_change_with_the_snapshot() {
        let a = plain_repo("ids-a", &CHAIN_FILES);
        let b = plain_repo("ids-b", &CHAIN_FILES);
        let (mut store_a, mut engine_a) = engine_over(&a);
        let (mut store_b, mut engine_b) = engine_over(&b);
        assert_eq!(engine_a.snapshot_id(), engine_b.snapshot_id());

        let explain = |engine: &mut Engine, store: &mut Store| {
            engine
                .explain_connection(store, "app/alpha.py", "app/gamma.py")
                .unwrap()
        };
        let (route_a, hops_a) = route_ids(&explain(&mut engine_a, &mut store_a));
        let (route_b, hops_b) = route_ids(&explain(&mut engine_b, &mut store_b));
        assert_eq!(
            (&route_a, &hops_a),
            (&route_b, &hops_b),
            "same repo, same ids"
        );

        // The id is the documented function of snapshot, kind and route.
        let snapshot = engine_a.snapshot_id().to_owned();
        let chain = ["app/alpha.py", "app/beta.py", "app/gamma.py"];
        assert_eq!(route_a, import_edge_id(&snapshot, &chain));
        assert_eq!(hops_a[0], import_edge_id(&snapshot, &chain[0..2]));
        assert_eq!(hops_a[1], import_edge_id(&snapshot, &chain[1..3]));
        // A route and its hops are different facts with different ids, and a
        // reversed hop is a different edge again.
        assert!(!hops_a.contains(&route_a));
        assert_ne!(
            hops_a[0],
            import_edge_id(&snapshot, &["app/beta.py", "app/alpha.py"])
        );

        // Change one file's content: the snapshot moves, and so does every id.
        std::fs::write(b.0.join("app/gamma.py"), "def gamma(): return 'changed'\n").unwrap();
        let (mut store_c, mut engine_c) = engine_over(&b);
        assert_ne!(engine_c.snapshot_id(), snapshot);
        let (route_c, hops_c) = route_ids(&explain(&mut engine_c, &mut store_c));
        assert_ne!(route_c, route_a);
        assert!(hops_c.iter().all(|id| !hops_a.contains(id)), "{hops_c:?}");
    }

    #[test]
    fn explain_hop_ids_are_the_ids_get_graph_emits_for_those_single_edges() {
        let (_repo, mut store, mut engine) = connection_fixture("hop-ids");
        let ex = engine
            .explain_connection(&mut store, "app/alpha.py", "app/gamma.py")
            .unwrap();
        let (route, hops) = route_ids(&ex);
        assert_eq!(hops.len(), 2);
        // From either end, every hop is in the graph under the same id.
        for anchor in ["app/alpha.py", "app/gamma.py", "app/beta.py"] {
            let ids = graph_ids(&all_graph(&engine, &store, &[anchor], 2));
            assert!(hops.iter().all(|h| ids.contains(h)), "{anchor}: {hops:?}");
            assert!(!ids.contains(&route), "a whole route is not a graph edge");
        }
        // A direct import: the route is its only hop.
        let direct = engine
            .explain_connection(&mut store, "app/alpha.py", "app/beta.py")
            .unwrap();
        let (route, hops) = route_ids(&direct);
        assert_eq!(hops, vec![route.clone()]);
        assert!(graph_ids(&all_graph(&engine, &store, &["app/alpha.py"], 1)).contains(&route));
    }

    #[test]
    fn explain_cochange_ids_match_get_graph_in_both_directions() {
        let (_repo, mut store, mut engine) = connection_fixture("co-ids");
        store
            .replace_cochange(&[
                cochange("app/alpha.py", "app/delta.py", 4, 0.5),
                cochange("app/delta.py", "app/alpha.py", 4, 0.8),
            ])
            .unwrap();
        let ex = engine
            .explain_connection(&mut store, "app/alpha.py", "app/delta.py")
            .unwrap();
        let v = serde_json::to_value(&ex).unwrap();
        let by_dir = |dir: &str| -> String {
            v["reasons"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["type"] == "historical_cochange" && r["direction"] == dir)
                .unwrap()["edge_id"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        let snapshot = engine.snapshot_id().to_owned();
        assert_eq!(
            by_dir("source_to_target"),
            cochange_edge_id(&snapshot, "app/alpha.py", "app/delta.py")
        );
        assert_eq!(
            by_dir("target_to_source"),
            cochange_edge_id(&snapshot, "app/delta.py", "app/alpha.py")
        );
        assert_ne!(by_dir("source_to_target"), by_dir("target_to_source"));

        // Asked the other way round, the same rows keep the same ids.
        let swapped = engine
            .explain_connection(&mut store, "app/delta.py", "app/alpha.py")
            .unwrap();
        let swapped = serde_json::to_value(&swapped).unwrap();
        let ids = |v: &serde_json::Value| -> HashSet<String> {
            v["reasons"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["type"] == "historical_cochange")
                .map(|r| r["edge_id"].as_str().unwrap().to_owned())
                .collect()
        };
        assert_eq!(ids(&v), ids(&swapped));

        // And get_graph emits both rows, from either file as the anchor.
        for anchor in ["app/alpha.py", "app/delta.py"] {
            let in_graph = graph_ids(&all_graph(&engine, &store, &[anchor], 1));
            assert!(ids(&v).is_subset(&in_graph), "{anchor}: {in_graph:?}");
        }
    }

    #[test]
    fn graph_facts_lists_a_direct_import_importer_to_imported() {
        let (_repo, store, engine) = connection_fixture("graph-direct");
        let facts = all_graph(&engine, &store, &["app/beta.py"], 1);
        let imports: Vec<(&str, &str)> = facts
            .edges
            .iter()
            .filter_map(|e| match e {
                GraphEdge::Import {
                    source,
                    target,
                    hops,
                    direction,
                    resolver,
                    resolution_status,
                    ..
                } => {
                    assert_eq!(*hops, 1);
                    assert_eq!(*direction, ImportDirection::SourceImportsTarget);
                    assert_eq!(*resolver, "heuristic_module_segments");
                    assert_eq!(*resolution_status, "candidate");
                    Some((source.as_str(), target.as_str()))
                }
                GraphEdge::Cochange { .. } => None,
            })
            .collect();
        // alpha imports beta (incoming), beta imports gamma (outgoing); each
        // once, with the importer as the source.
        assert_eq!(
            imports,
            vec![
                ("app/alpha.py", "app/beta.py"),
                ("app/beta.py", "app/gamma.py")
            ]
        );
        assert_eq!(facts.snapshot, engine.snapshot_id());
        assert_eq!(facts.anchors, vec!["app/beta.py"]);
        let roles: Vec<(&str, GraphNodeRole, usize)> = facts
            .nodes
            .iter()
            .map(|n| (n.path.as_str(), n.role, n.distance))
            .collect();
        assert_eq!(
            roles,
            vec![
                ("app/alpha.py", GraphNodeRole::Neighbor, 1),
                ("app/beta.py", GraphNodeRole::Anchor, 0),
                ("app/gamma.py", GraphNodeRole::Neighbor, 1),
            ]
        );
        assert!(!facts.limits.truncated);
        assert_eq!(facts.limits.omitted_edges + facts.limits.omitted_nodes, 0);
    }

    #[test]
    fn a_two_hop_neighbour_appears_only_with_max_hops_two() {
        let (_repo, store, engine) = connection_fixture("graph-hops");
        let one = all_graph(&engine, &store, &["app/alpha.py"], 1);
        let paths =
            |f: &GraphFacts| -> Vec<String> { f.nodes.iter().map(|n| n.path.clone()).collect() };
        assert_eq!(paths(&one), vec!["app/alpha.py", "app/beta.py"]);
        assert_eq!(one.edges.len(), 1, "{:?}", one.edges);

        let two = all_graph(&engine, &store, &["app/alpha.py"], 2);
        assert_eq!(
            paths(&two),
            vec!["app/alpha.py", "app/beta.py", "app/gamma.py"]
        );
        let gamma = two.nodes.iter().find(|n| n.path == "app/gamma.py").unwrap();
        assert_eq!((gamma.role, gamma.distance), (GraphNodeRole::Neighbor, 2));
        assert_eq!(two.edges.len(), 2, "{:?}", two.edges);
        // The one-hop result is a subset of the two-hop one, ids and all.
        assert!(graph_ids(&one).is_subset(&graph_ids(&two)));
        assert!(engine
            .graph_facts(
                &store,
                &["app/alpha.py".to_owned()],
                3,
                GraphKinds::default(),
                GraphCaps::default()
            )
            .is_err());
    }

    #[test]
    fn graph_facts_include_a_cochange_pair_the_recommendations_cut() {
        // Seven strong partners (p00..p06) and `other`, the weakest of all.
        let (_repo, mut store, mut engine) = fan_fixture("graph-beyond", 7, 0);
        let mut edges: Vec<_> = (0..7)
            .map(|i| {
                cochange(
                    "app/anchor.py",
                    &format!("app/p{i:02}.py"),
                    40,
                    0.9 - i as f32 * 0.01,
                )
            })
            .collect();
        edges.push(cochange("app/anchor.py", "app/other.py", 3, 0.12));
        store.replace_cochange(&edges).unwrap();

        let impact = engine
            .impact_from_files(&mut store, &["app/anchor.py".to_owned()])
            .unwrap();
        let recommended: Vec<&str> = impact
            .cochange_expansions
            .iter()
            .map(|p| p.path.as_str())
            .collect();
        assert_eq!(recommended.len(), 5);
        assert!(!recommended.contains(&"app/other.py"), "{recommended:?}");

        let facts = all_graph(&engine, &store, &["app/anchor.py"], 1);
        let weak = facts
            .edges
            .iter()
            .find(|e| matches!(e, GraphEdge::Cochange { target, .. } if target == "app/other.py"));
        let Some(GraphEdge::Cochange {
            edge_id,
            count,
            strength,
            source,
            direction,
            ..
        }) = weak
        else {
            panic!(
                "the weak pair is a recorded fact and must be listed: {:?}",
                facts.edges
            );
        };
        assert_eq!((*count, *strength), (3, 0.12));
        assert_eq!(source, "app/anchor.py");
        assert_eq!(*direction, CoChangeDirection::SourceToTarget);
        assert_eq!(
            edge_id,
            &cochange_edge_id(engine.snapshot_id(), "app/anchor.py", "app/other.py")
        );
        // Every partner is listed, so the cap disclosure is honest: nothing cut.
        let partners = facts
            .edges
            .iter()
            .filter(|e| matches!(e, GraphEdge::Cochange { .. }))
            .count();
        assert_eq!(partners, 8);
        assert!(!facts.limits.truncated);
        assert_eq!(facts.limits.omitted_edges, 0);
    }

    #[test]
    fn graph_facts_reject_unknown_anchors_and_bad_arguments() {
        let (_repo, store, engine) = connection_fixture("graph-errors");
        let call = |anchors: &[&str], hops: usize, kinds: GraphKinds, caps: GraphCaps| {
            let anchors: Vec<String> = anchors.iter().map(|a| (*a).to_owned()).collect();
            engine.graph_facts(&store, &anchors, hops, kinds, caps)
        };
        let both = GraphKinds::default();
        let caps = GraphCaps::default();
        let err = call(&["app/alpha.py", "app/nope.py"], 1, both, caps)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("app/nope.py") && err.contains("not an indexed text file"),
            "{err}"
        );
        assert!(call(&[], 1, both, caps).is_err());
        assert!(call(&["app/alpha.py"], 0, both, caps).is_err());
        assert!(call(&["app/alpha.py"], 3, both, caps).is_err());
        let none = GraphKinds {
            import: false,
            cochange: false,
        };
        assert!(call(&["app/alpha.py"], 1, none, caps).is_err());
        let tiny = GraphCaps {
            max_nodes: 1,
            ..caps
        };
        let err = call(&["app/alpha.py", "app/beta.py"], 1, both, tiny)
            .unwrap_err()
            .to_string();
        assert!(err.contains("node cap"), "{err}");
    }

    #[test]
    fn graph_facts_are_byte_identical_on_repeat_and_ignore_anchor_order() {
        let (_repo, mut store, engine) = connection_fixture("graph-bytes");
        store
            .replace_cochange(&[
                cochange("app/alpha.py", "app/delta.py", 4, 0.5),
                cochange("app/delta.py", "app/alpha.py", 4, 0.8),
                cochange("app/beta.py", "app/delta.py", 2, 0.3),
            ])
            .unwrap();
        let bytes = |anchors: &[&str]| {
            serde_json::to_string(&all_graph(&engine, &store, anchors, 2)).unwrap()
        };
        let first = bytes(&["app/alpha.py", "app/beta.py"]);
        assert_eq!(first, bytes(&["app/alpha.py", "app/beta.py"]));
        assert_eq!(
            first,
            bytes(&["app/beta.py", "app/alpha.py", "app/beta.py"]),
            "order and repetition of anchors must not matter"
        );
        // Edges come out in (kind, source, target, direction) order.
        let facts = all_graph(&engine, &store, &["app/alpha.py", "app/beta.py"], 2);
        let keys: Vec<(String, String, String)> = facts
            .edges
            .iter()
            .map(|e| match e {
                GraphEdge::Import { source, target, .. } => {
                    ("import".to_owned(), source.clone(), target.clone())
                }
                GraphEdge::Cochange { source, target, .. } => {
                    ("cochange".to_owned(), source.clone(), target.clone())
                }
            })
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
        let node_paths: Vec<&String> = facts.nodes.iter().map(|n| &n.path).collect();
        let mut sorted_nodes = node_paths.clone();
        sorted_nodes.sort();
        assert_eq!(node_paths, sorted_nodes);
    }

    #[test]
    fn graph_facts_serialize_with_the_documented_shape() {
        let (_repo, mut store, engine) = connection_fixture("graph-json");
        store
            .replace_cochange(&[cochange("app/alpha.py", "app/delta.py", 4, 0.5)])
            .unwrap();
        let v = serde_json::to_value(all_graph(&engine, &store, &["app/alpha.py"], 1)).unwrap();
        assert_eq!(v["anchors"], serde_json::json!(["app/alpha.py"]));
        let node = &v["nodes"][0];
        assert_eq!(node["path"], "app/alpha.py");
        assert_eq!(node["role"], "anchor");
        assert_eq!(node["language"], "python");
        assert_eq!(node["is_test"], false);
        let edge = |kind: &str| {
            v["edges"]
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["kind"] == kind)
                .unwrap_or_else(|| panic!("no {kind} edge in {v}"))
                .clone()
        };
        let imp = edge("import");
        assert_eq!(imp["direction"], "source_imports_target");
        assert_eq!(imp["hops"], 1);
        assert_eq!(imp["resolver"], "heuristic_module_segments");
        assert_eq!(imp["resolution_status"], "candidate");
        assert!(imp["edge_id"].as_str().unwrap().starts_with("e_"));
        let co = edge("cochange");
        assert_eq!(co["direction"], "source_to_target");
        assert_eq!(co["count"], 4);
        assert_eq!(co["strength"], 0.5);
        assert!(co["window"]["max_commits"].is_u64());
        // Hand-built index: no history totals were recorded, so null, not 0.
        assert!(co["opportunities_source"].is_null());
        assert!(co["opportunities_target"].is_null());
        for key in [
            "max_nodes",
            "max_edges",
            "truncated",
            "omitted_edges",
            "omitted_nodes",
        ] {
            assert!(v["limits"].get(key).is_some(), "limits.{key}");
        }
        assert!(v["limits"]["reason"].is_string());
        assert_eq!(
            v["scope"]["kinds"],
            serde_json::json!(["import", "cochange"])
        );
        assert!(v["scope"]["note"].as_str().unwrap().contains("call"));
    }

    fn caps(max_nodes: usize, max_edges: usize, per_anchor: usize) -> GraphCaps {
        GraphCaps {
            max_nodes,
            max_edges,
            cochange_per_anchor: per_anchor,
        }
    }

    fn capped(engine: &Engine, store: &Store, caps: GraphCaps) -> GraphFacts {
        engine
            .graph_facts(
                store,
                &["app/anchor.py".to_owned()],
                1,
                GraphKinds::default(),
                caps,
            )
            .unwrap()
    }

    #[test]
    fn caps_truncate_graph_facts_with_exact_omitted_counts() {
        // A hub: 12 co-change partners (both directions, like production) and
        // 6 importers: 24 co-change rows + 6 imports = 30 edges, 19 files.
        let (_repo, store, engine) = fan_fixture("graph-caps", 12, 6);
        let full = capped(&engine, &store, GraphCaps::default());
        assert_eq!((full.nodes.len(), full.edges.len()), (19, 30));
        assert!(!full.limits.truncated);
        assert_eq!(full.limits.reason, "complete: no cap cut anything");
        let full_ids = graph_ids(&full);

        // Co-change read limit: 5 of 12 partners read, so 7 partners are
        // hidden and each hides two rows (its own and its row back): 14.
        let facts = capped(&engine, &store, caps(200, 500, 5));
        assert_eq!((facts.nodes.len(), facts.edges.len()), (12, 16));
        assert!(facts.limits.truncated);
        assert_eq!(facts.limits.omitted_edges, 14);
        assert_eq!(facts.limits.omitted_nodes, 0);
        assert!(
            facts.limits.reason.contains("14 co-change row(s)"),
            "{}",
            facts.limits.reason
        );
        assert_eq!(facts.limits.cochange_per_anchor, 5);
        // The strongest five by (strength, count, path) are the ones kept.
        let kept: Vec<&str> = facts
            .edges
            .iter()
            .filter_map(|e| match e {
                GraphEdge::Cochange { source, target, .. } if source == "app/anchor.py" => {
                    Some(target.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            kept,
            [
                "app/p00.py",
                "app/p01.py",
                "app/p02.py",
                "app/p03.py",
                "app/p04.py"
            ]
        );

        // Node cap: 10 nodes = the anchor + 9 neighbours, nearest then by path
        // (the 6 importers, then p00..p02). The 18 co-change rows to and from
        // p03..p11 lose an endpoint and are counted; so are the 9 files.
        let facts = capped(&engine, &store, caps(10, 500, 50));
        assert_eq!((facts.nodes.len(), facts.edges.len()), (10, 12));
        assert_eq!(facts.limits.omitted_nodes, 9);
        assert_eq!(facts.limits.omitted_edges, 18);
        assert!(facts.limits.truncated);
        let node_paths: Vec<&str> = facts.nodes.iter().map(|n| n.path.as_str()).collect();
        assert!(node_paths.contains(&"app/p02.py") && !node_paths.contains(&"app/p03.py"));
        assert!(graph_ids(&facts).is_subset(&full_ids));

        // Edge cap: all 30 candidates touch the anchor, so ties fall back to
        // (kind, source, target) order: the first 7 are the anchor's rows to
        // p00..p06.
        let facts = capped(&engine, &store, caps(200, 7, 50));
        assert_eq!((facts.nodes.len(), facts.edges.len()), (19, 7));
        assert_eq!(facts.limits.omitted_edges, 23);
        assert_eq!(facts.limits.omitted_nodes, 0);
        assert!(
            facts.limits.reason.contains("23 edge(s)"),
            "{}",
            facts.limits.reason
        );
        assert!(graph_ids(&facts).is_subset(&full_ids));

        // Whatever the caps, emitted + omitted accounts for every fact.
        for c in [
            caps(200, 500, 5),
            caps(10, 500, 50),
            caps(200, 7, 50),
            caps(4, 3, 2),
        ] {
            let f = capped(&engine, &store, c);
            assert_eq!(f.edges.len() + f.limits.omitted_edges, 30, "{c:?}");
            assert!(
                f.nodes.len() <= c.max_nodes && f.edges.len() <= c.max_edges,
                "{c:?}"
            );
        }
    }

    /// Caps nothing can reach in these fixtures: the reference "no caps" run.
    fn raised() -> GraphCaps {
        caps(1000, 100_000, 100_000)
    }

    /// Under every cap in a grid, `emitted + omitted_edges` is exactly the
    /// number of edges the same call emits with the caps raised.
    fn assert_exact_accounting(engine: &Engine, store: &Store, anchors: &[&str], hops: usize) {
        let anchors: Vec<String> = anchors.iter().map(|a| (*a).to_owned()).collect();
        let run = |c: GraphCaps| {
            engine
                .graph_facts(store, &anchors, hops, GraphKinds::default(), c)
                .unwrap()
        };
        let full = run(raised());
        assert!(!full.limits.truncated, "{}", full.limits.reason);
        assert_eq!(full.limits.omitted_edges, 0);
        let total = full.edges.len();
        assert!(total > 0);
        let full_ids = graph_ids(&full);
        assert_eq!(full_ids.len(), total, "every edge once");
        for c in [
            caps(200, 500, 1),
            caps(200, 500, 3),
            caps(200, 500, 5),
            caps(200, 500, 25),
            caps(200, 5, 25),
            caps(200, 1, 100),
            caps(8, 500, 100),
            caps(8, 6, 3),
            caps(3, 500, 2),
            caps(anchors.len(), 500, 100),
            caps(anchors.len(), 1, 1),
            caps(200, 500, 100_000),
        ] {
            if c.max_nodes < anchors.len() {
                continue;
            }
            let f = run(c);
            assert_eq!(
                f.edges.len() + f.limits.omitted_edges,
                total,
                "{anchors:?} hops {hops} {c:?}: {}",
                f.limits.reason
            );
            assert_eq!(
                f.limits.truncated,
                f.limits.omitted_edges + f.limits.omitted_nodes > 0,
                "{c:?}"
            );
            assert!(
                f.edges.len() <= c.max_edges && f.nodes.len() <= c.max_nodes,
                "{c:?}"
            );
            assert!(graph_ids(&f).is_subset(&full_ids), "{c:?}");
            assert_eq!(graph_ids(&f).len(), f.edges.len(), "every edge once");
            if c.cochange_per_anchor >= 100 {
                // Nothing was hidden by the per-anchor limit, so every file is
                // either listed or counted.
                assert_eq!(
                    f.nodes.len() + f.limits.omitted_nodes,
                    full.nodes.len(),
                    "{c:?}"
                );
            }
        }
    }

    #[test]
    fn omitted_edges_is_exact_on_a_symmetric_cochange_table() {
        // The fixture stores every pair in both directions, as the history
        // walk does, so a partner hidden by the per-anchor limit also hides the
        // row back from it.
        let (_repo, mut store, engine) = fan_fixture("graph-exact", 12, 6);
        for hops in [1, 2] {
            assert_exact_accounting(&engine, &store, &["app/anchor.py"], hops);
        }
        // Several anchors that are partners of each other and share partners.
        let mut forward: Vec<engram_domain::CoChange> = (0..12)
            .map(|i| cochange("app/anchor.py", &format!("app/p{i:02}.py"), 40 - i, 0.9))
            .collect();
        forward.extend(
            (6..12).map(|i| cochange("app/other.py", &format!("app/p{i:02}.py"), 20 + i, 0.5)),
        );
        forward.push(cochange("app/anchor.py", "app/other.py", 9, 0.3));
        store.replace_cochange(&mirrored(forward)).unwrap();
        for hops in [1, 2] {
            assert_exact_accounting(&engine, &store, &["app/anchor.py", "app/other.py"], hops);
            assert_exact_accounting(&engine, &store, &["app/p07.py", "app/other.py"], hops);
        }
        // A lone reverse row (no forward row from the anchor) is not a fact of
        // the anchor's neighbourhood until the anchor's own row exists.
        store
            .replace_cochange(&[cochange("app/p00.py", "app/anchor.py", 5, 0.5)])
            .unwrap();
        let facts = all_graph(&engine, &store, &["app/anchor.py"], 1);
        assert!(!facts
            .edges
            .iter()
            .any(|e| matches!(e, GraphEdge::Cochange { .. })));
        assert_eq!(facts.limits.omitted_edges, 0);
    }

    #[test]
    fn omitted_edges_is_exact_on_a_history_built_by_the_real_walk() {
        let Some(repo) = partner_history_repo("graph-exact-git", 9, 3) else {
            return;
        };
        let (store, engine) = engine_over(&repo);
        // The walk wrote both directions of every pair that co-changed.
        let both = store.cochange_pair("app/anchor.py", "app/p00.py").unwrap();
        let back = store.cochange_pair("app/p00.py", "app/anchor.py").unwrap();
        assert!(both.is_some() && back.is_some(), "{both:?} {back:?}");
        let rows = all_graph(&engine, &store, &["app/anchor.py"], 1);
        let cochange_rows = rows
            .edges
            .iter()
            .filter(|e| matches!(e, GraphEdge::Cochange { .. }))
            .count();
        assert_eq!(cochange_rows, 18, "9 partners, both directions");
        for hops in [1, 2] {
            assert_exact_accounting(&engine, &store, &["app/anchor.py"], hops);
            assert_exact_accounting(&engine, &store, &["app/anchor.py", "app/p03.py"], hops);
        }
    }

    #[test]
    fn a_hub_with_many_importers_is_counted_exactly_and_only_the_best_edges_are_built() {
        // 4 partners (8 co-change rows) and 300 importers of the anchor.
        let (_repo, store, engine) = fan_fixture("graph-hub", 4, 300);
        let full = capped(&engine, &store, caps(400, 100_000, 100));
        assert_eq!(full.edges.len(), 8 + 300);
        assert!(!full.limits.truncated);
        let mut importers: Vec<String> = (0..300).map(|i| format!("app/imp{i:02}.py")).collect();
        importers.sort();
        let imports_of = |f: &GraphFacts| -> Vec<String> {
            f.edges
                .iter()
                .filter_map(|e| match e {
                    GraphEdge::Import { source, .. } => Some(source.clone()),
                    GraphEdge::Cochange { .. } => None,
                })
                .collect()
        };

        // Edge cap 5: only co-change rows fit; every import is still counted.
        let few = capped(&engine, &store, caps(400, 5, 25));
        assert_eq!(few.edges.len(), 5);
        assert!(imports_of(&few).is_empty());
        assert_eq!(few.limits.omitted_edges, 303);
        assert_eq!(few.limits.omitted_nodes, 0);

        // Edge cap 12: the 8 co-change rows, then the 4 imports first in path
        // order (the same cut on every run).
        let some = capped(&engine, &store, caps(400, 12, 25));
        assert_eq!(some.edges.len(), 12);
        assert_eq!(imports_of(&some), importers[..4].to_vec());
        assert_eq!(some.limits.omitted_edges, 296);
        assert_eq!(
            serde_json::to_string(&some).unwrap(),
            serde_json::to_string(&capped(&engine, &store, caps(400, 12, 25))).unwrap()
        );

        // Node cap 200 as well: 199 neighbours are kept, nearest then by path,
        // which here is the first 199 importers and none of the partners. The
        // 8 co-change rows and the importer edges of the 101 files past the
        // cap are counted with the edges past the edge cap; the total holds.
        let both = capped(&engine, &store, caps(200, 12, 25));
        assert_eq!(both.nodes.len(), 200);
        assert_eq!(both.limits.omitted_nodes, 304 - 199);
        assert_eq!(both.edges.len(), 12);
        assert_eq!(imports_of(&both), importers[..12].to_vec());
        assert_eq!(both.edges.len() + both.limits.omitted_edges, 308);

        assert_exact_accounting(&engine, &store, &["app/anchor.py"], 2);
    }

    #[test]
    fn graph_facts_refuse_too_many_anchors_before_looking_any_of_them_up() {
        let (_repo, store, engine) = connection_fixture("graph-anchor-cap");
        // None of these is an indexed file; the count is what is refused.
        let anchors: Vec<String> = (0..5).map(|i| format!("app/ghost{i}.py")).collect();
        let err = engine
            .graph_facts(&store, &anchors, 1, GraphKinds::default(), caps(4, 500, 25))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("node cap") && !err.contains("not an indexed"),
            "{err}"
        );
    }

    /// A repository where `alpha` reaches `gamma` by two routes of equal
    /// length, through `beta1` and `beta2`. `order` is the order the import
    /// rows are written in; the chain must not depend on it.
    fn diamond(tag: &str, order: &[(&str, &[&str])]) -> (TempRepo, Store, Engine) {
        use engram_domain::{FileRecord, Language};
        let repo = TempRepo::new(tag);
        std::fs::create_dir_all(repo.0.join("app")).unwrap();
        let mut files = Vec::new();
        for n in ["alpha", "beta1", "beta2", "gamma"] {
            std::fs::write(
                repo.0.join(format!("app/{n}.py")),
                format!("def {n}(): pass\n"),
            )
            .unwrap();
            files.push(FileRecord {
                path: format!("app/{n}.py"),
                language: Language::Python,
                size_bytes: 10,
                is_test: false,
            });
        }
        let mut store = Store::open(&repo.0).expect("open store");
        store.upsert_files(&files).expect("upsert files");
        for (file, targets) in order {
            let targets: Vec<String> = targets.iter().map(|t| (*t).to_owned()).collect();
            store
                .replace_imports_for_file(&format!("app/{file}.py"), &targets)
                .unwrap();
        }
        let engine = Engine::build(&repo.0, &mut store).expect("build engine");
        (repo, store, engine)
    }

    #[test]
    fn equal_length_import_routes_are_chosen_the_same_way_whatever_the_row_order() {
        let forward: [(&str, &[&str]); 3] = [
            ("alpha", &["beta1", "beta2"]),
            ("beta1", &["gamma"]),
            ("beta2", &["gamma"]),
        ];
        let reversed: [(&str, &[&str]); 3] = [
            ("beta2", &["gamma"]),
            ("beta1", &["gamma"]),
            ("alpha", &["beta2", "beta1"]),
        ];
        let observe = |tag: &str, order: &[(&str, &[&str])]| {
            let (_repo, mut store, mut engine) = diamond(tag, order);
            let explained = engine
                .explain_connection(&mut store, "app/alpha.py", "app/gamma.py")
                .unwrap();
            let route = route_ids(&explained);
            let v = serde_json::to_value(&explained).unwrap();
            let chain = v["reasons"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["type"] == "import_edge")
                .unwrap()["chain"]
                .clone();
            let impact = engine
                .impact_from_files(&mut store, &["app/gamma.py".to_owned()])
                .unwrap();
            let alpha = impact
                .import_expansions
                .iter()
                .find(|p| p.path == "app/alpha.py")
                .expect("alpha depends on gamma in two hops")
                .evidence_ids
                .clone();
            (engine.snapshot_id().to_owned(), chain, route, alpha)
        };
        let a = observe("diamond-forward", &forward);
        let b = observe("diamond-reversed", &reversed);
        assert_eq!(a, b, "row order must not change the route or its ids");
        // And it is the alphabetically first of the equal routes.
        assert_eq!(
            a.1,
            serde_json::json!(["app/alpha.py", "app/beta1.py", "app/gamma.py"])
        );
        assert_eq!(
            a.3,
            vec![
                import_edge_id(&a.0, &["app/alpha.py", "app/beta1.py"]),
                import_edge_id(&a.0, &["app/beta1.py", "app/gamma.py"]),
            ]
        );
    }

    #[test]
    fn a_cited_id_is_in_the_graph_only_while_the_hops_reach_it_and_nothing_cut_it() {
        let (_repo, mut store, mut engine) = connection_fixture("cite-conditional");
        let anchors = vec!["app/gamma.py".to_owned()];
        let impact = engine.impact_from_files(&mut store, &anchors).unwrap();
        let cited = cited_ids(&impact);
        let snapshot = engine.snapshot_id().to_owned();
        let outer = import_edge_id(&snapshot, &["app/alpha.py", "app/beta.py"]);
        assert!(cited.contains(&outer), "alpha cites its outer hop");

        // (a) With max_hops 2, the depth the expansion used, nothing is cut
        // and every cited id is present.
        let two = all_graph(&engine, &store, &["app/gamma.py"], 2);
        assert!(!two.limits.truncated);
        assert!(cited.iter().all(|id| graph_ids(&two).contains(id)));

        // (b) With max_hops 1 the outer-hop id is absent, and the graph does
        // not claim to be truncated: the guarantee needs max_hops >= the hop.
        let one = all_graph(&engine, &store, &["app/gamma.py"], 1);
        assert!(!graph_ids(&one).contains(&outer));
        assert!(!one.limits.truncated);

        // (c) A cap that cuts a cited edge is reported as truncation.
        let cut = engine
            .graph_facts(&store, &anchors, 2, GraphKinds::default(), caps(200, 1, 25))
            .unwrap();
        assert_eq!(cut.edges.len(), 1);
        assert!(cited.iter().any(|id| !graph_ids(&cut).contains(id)));
        assert!(cut.limits.truncated);
        assert_eq!(cut.limits.omitted_edges, 1);
    }

    #[test]
    fn every_id_a_recommendation_cites_is_a_fact_get_graph_emits() {
        let (_repo, mut store, mut engine) = fan_fixture("cite-fan", 12, 10);
        let anchors = vec!["app/anchor.py".to_owned()];
        let impact = engine.impact_from_files(&mut store, &anchors).unwrap();
        let cited = cited_ids(&impact);
        assert!(!cited.is_empty());
        // 5 co-change rows and 8 of the 10 importers are recommended.
        assert_eq!(impact.cochange_expansions.len(), 5);
        assert_eq!(impact.import_expansions.len(), 8);
        let facts = all_graph(&engine, &store, &["app/anchor.py"], 2);
        let emitted = graph_ids(&facts);
        for id in &cited {
            assert!(emitted.contains(id), "cited {id} is not in get_graph");
        }
        // Each recommendation cites exactly the edges its own numbers rest on.
        let snapshot = engine.snapshot_id().to_owned();
        for p in &impact.cochange_expansions {
            assert_eq!(
                p.evidence_ids,
                vec![cochange_edge_id(&snapshot, "app/anchor.py", &p.path)]
            );
            let Some(ScoreComponents::CochangeStrength {
                strength,
                count,
                anchor,
            }) = &p.score_components
            else {
                panic!("co-change entry without its numbers: {p:?}");
            };
            assert_eq!(*strength, p.confidence);
            assert_eq!(anchor, "app/anchor.py");
            assert!(*count >= 40, "{count}");
        }
        for p in &impact.import_expansions {
            assert_eq!(
                p.evidence_ids,
                vec![import_edge_id(
                    &snapshot,
                    &[p.path.as_str(), "app/anchor.py"]
                )]
            );
            assert_eq!(
                p.score_components,
                Some(ScoreComponents::ImportHops {
                    hops: 1,
                    weight: 0.6
                })
            );
        }
        // Anchors and search-free entries cite nothing.
        assert!(impact
            .likely_files
            .iter()
            .all(|p| p.evidence_ids.is_empty()));
    }

    #[test]
    fn a_two_hop_dependent_cites_each_hop_of_its_route() {
        let (_repo, mut store, mut engine) = connection_fixture("cite-two-hop");
        let impact = engine
            .impact_from_files(&mut store, &["app/gamma.py".to_owned()])
            .unwrap();
        let snapshot = engine.snapshot_id().to_owned();
        let alpha = impact
            .import_expansions
            .iter()
            .find(|p| p.path == "app/alpha.py")
            .expect("alpha is a two-hop dependent of gamma");
        assert_eq!(
            alpha.evidence_ids,
            vec![
                import_edge_id(&snapshot, &["app/alpha.py", "app/beta.py"]),
                import_edge_id(&snapshot, &["app/beta.py", "app/gamma.py"]),
            ]
        );
        assert_eq!(
            alpha.score_components,
            Some(ScoreComponents::ImportHops {
                hops: 2,
                weight: 0.35
            })
        );
        let emitted = graph_ids(&all_graph(&engine, &store, &["app/gamma.py"], 2));
        for id in cited_ids(&impact) {
            assert!(emitted.contains(&id), "{id}");
        }
        // With one hop only the direct importer is recommended or cited, and
        // the one-hop graph still holds everything cited.
        let one = engine
            .impact_from_files_with(&mut store, &["app/gamma.py".to_owned()], 1)
            .unwrap();
        assert_eq!(one.import_expansions.len(), 1);
        let emitted = graph_ids(&all_graph(&engine, &store, &["app/gamma.py"], 1));
        for id in cited_ids(&one) {
            assert!(emitted.contains(&id), "{id}");
        }
    }

    #[test]
    fn multi_anchor_cochange_cites_every_kept_row_and_the_strongest_sets_the_numbers() {
        let (_repo, mut store, mut engine) = connection_fixture("cite-multi");
        store
            .replace_cochange(&[
                cochange("app/alpha.py", "app/delta.py", 3, 0.3),
                cochange("app/beta.py", "app/delta.py", 9, 0.9),
            ])
            .unwrap();
        let anchors = vec!["app/alpha.py".to_owned(), "app/beta.py".to_owned()];
        let impact = engine.impact_from_files(&mut store, &anchors).unwrap();
        let delta = impact
            .cochange_expansions
            .iter()
            .find(|p| p.path == "app/delta.py")
            .unwrap();
        let snapshot = engine.snapshot_id().to_owned();
        let mut expected = vec![
            cochange_edge_id(&snapshot, "app/alpha.py", "app/delta.py"),
            cochange_edge_id(&snapshot, "app/beta.py", "app/delta.py"),
        ];
        expected.sort();
        assert_eq!(delta.evidence_ids, expected);
        assert_eq!(
            delta.score_components,
            Some(ScoreComponents::CochangeStrength {
                strength: 0.9,
                count: 9,
                anchor: "app/beta.py".to_owned()
            })
        );
        let emitted = graph_ids(&all_graph(
            &engine,
            &store,
            &["app/alpha.py", "app/beta.py"],
            2,
        ));
        for id in cited_ids(&impact) {
            assert!(emitted.contains(&id), "{id}");
        }
    }
}
