//! engram-retrieval: hybrid BM25 (Tantivy) + vector (cosine) retrieval with
//! score fusion, lazy Tier-1 extraction on miss, and co-change impact expansion.

pub mod budget;
pub mod embed;
pub mod stopwords;
pub mod weights;

use anyhow::Result;
use embed::{cosine, Embedder, HashedNgramEmbedder};
use engram_domain::{
    CoChangeDirection, CoChangeWindow, ConnectionExplanation, ConnectionReason, ConnectionScope,
    ConnectionStatus, DroppedCandidate, EvidenceKind, EvidencePacket, ImpactPrediction,
    ImportDirection, ImportEdge, ImportSearchOutcome, ScoredPath, SymbolRecord,
};
use engram_repo_map::graph::{CodeGraph, EdgeKind, ImportSearch};
use engram_repo_map::store::Store;
use std::collections::{HashMap, HashSet};
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
const COCHANGE_SCAN_LIMIT: usize = 100_000;
/// Import edges come from matching module names against file paths, not from a
/// compiler, so every import fact is labelled with how it was resolved.
const IMPORT_RESOLUTION: &str = "heuristic";
const IMPORT_RESOLVER: &str = "heuristic_module_segments";

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
    pub fn build(repo_root: &Path, store: &mut Store) -> Result<Engine> {
        let mut schema_builder = Schema::builder();
        let f_path = schema_builder.add_text_field("path", TEXT | STORED);
        let f_body = schema_builder.add_text_field("body", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer(64_000_000)?;

        let stopwords = std::sync::Arc::new(stopwords::load(repo_root));
        let embedder = HashedNgramEmbedder::new(stopwords.clone());
        let mut docs = Vec::new();
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut by_path = HashMap::new();
        let recency = store.recency_map()?;
        let cached = store.load_chunk_vectors()?;
        let mut fresh: Vec<(String, usize, String, Vec<u8>)> = Vec::new();
        let mut present: HashSet<(String, usize)> = HashSet::new();
        let (mut reused, mut embedded) = (0usize, 0usize);

        for f in store.all_files()? {
            let full = repo_root.join(&f.path);
            let Ok(src) = std::fs::read_to_string(&full) else {
                continue;
            };
            let body: String = src.chars().take(INDEX_BODY_BYTES).collect();
            let doc_index = docs.len();

            // Symbols are extracted lazily, so a file may legitimately have
            // none yet; plan_chunks always yields at least the head chunk.
            let symbols = store.symbols_for_path(&f.path).unwrap_or_default();
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
        let snapshot = engram_repo_map::snapshot_id(store, repo_root)?;
        let graph = CodeGraph::build(store, repo_root)?;
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
        })
    }

    /// Identity of the indexed content (commit, parser version, and every
    /// file's path and content hash) this engine was built from.
    pub fn snapshot_id(&self) -> &str {
        &self.snapshot
    }

    /// Whether the store now holds symbols or imports this engine never saw:
    /// lazy extraction on a search hit writes them after the chunks and the
    /// code graph were built. The owner should discard the engine and
    /// rebuild it before the next query.
    pub fn is_stale(&self) -> bool {
        self.stale.load(std::sync::atomic::Ordering::SeqCst)
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
        // Total order: equal scores fall back to the doc index (files are
        // indexed in a fixed order), never to HashMap iteration order.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
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
        ranked.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.docs[a.0].path.cmp(&self.docs[b.0].path))
                .then_with(|| a.0.cmp(&b.0))
        });
        let beyond_first_cap = ranked.len().saturating_sub(top_k);
        ranked.truncate(top_k);

        // lazy Tier-1: ensure symbols exist for top hits, then attach best symbols
        let mut packets = Vec::new();
        for (rank, (i, score, signals)) in ranked.iter().enumerate() {
            let d = &self.docs[*i];
            if engram_repo_map::ensure_tier1(store, &self.repo_root, &d.path)? {
                self.stale.store(true, std::sync::atomic::Ordering::SeqCst);
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
                None => match best_symbols_for(store, &d.path, &q_tokens).first() {
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
        for ident in query_identifiers(query) {
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
        for t in &q_tokens {
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
        Ok(SearchOutcome {
            packets,
            cap_dropped,
            beyond_first_cap,
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
    /// it is touching (e.g. the current diff, per `engram impact --diff current`),
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
            if !self.by_path.contains_key(path) {
                anyhow::bail!(
                    "{path} is not an indexed text file in the Engram index; use a \
                     repo-relative path returned by search_context/expand_connections"
                );
            }
        }
        if source == target {
            anyhow::bail!(
                "source and target are the same file ({source}); \
                 pass two different files to explain how they are connected"
            );
        }

        let window = CoChangeWindow {
            max_commits: engram_repo_map::cochange::MAX_COMMITS,
            min_count: engram_repo_map::cochange::MIN_PAIR_COUNT,
            max_files_per_commit: engram_repo_map::cochange::MAX_FILES_PER_COMMIT,
        };
        let mut reasons = Vec::new();

        for (direction, from, to) in [
            (CoChangeDirection::SourceToTarget, source, target),
            (CoChangeDirection::TargetToSource, target, source),
        ] {
            if let Some(row) = store.cochange_pair(from, to)? {
                reasons.push(ConnectionReason::CoChange {
                    detail: format!(
                        "{from} and {to} changed together in {} commit{}, \
                         {}% of the commits that touched {from}",
                        row.count,
                        if row.count == 1 { "" } else { "s" },
                        (row.strength * 100.0) as u32
                    ),
                    weight: row.strength,
                    edge_kind: "cochange",
                    direction,
                    count: row.count,
                    strength: row.strength,
                    window,
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
                        })
                        .collect();
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
            note: format!(
                "Import chains up to {MAX_IMPORT_HOPS} hops were searched in both directions \
                 over heuristically resolved import edges. Co-change covers only the last {} \
                 commits, pairs sharing at least {} commits, ignoring commits touching more \
                 than {} files. Absence of a fact means none was recorded within this scope, \
                 not that the files are unrelated.",
                window.max_commits, window.min_count, window.max_files_per_commit
            ),
        };
        Ok(ConnectionExplanation {
            status,
            scope,
            reasons,
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
        let mut expansions: HashMap<String, (f32, String)> = HashMap::new();
        let mut beyond_seed_cap: Vec<String> = Vec::new();
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
                        .or_insert((0.0, String::new()));
                    if edge.strength > e.0 {
                        *e = (
                            edge.strength,
                            format!(
                                "changes with {seed} in {}% of its commits",
                                (edge.strength * 100.0) as u32
                            ),
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
        // A path cut for one anchor but kept for another was not dropped.
        let mut dropped_cochange: Vec<String> = beyond_seed_cap
            .into_iter()
            .filter(|p| !expansions.contains_key(p))
            .collect();

        let mut cochange_expansions: Vec<ScoredPath> = expansions
            .into_iter()
            .map(|(path, (conf, reason))| {
                if engram_repo_map::inventory::is_test_path(&path) {
                    likely_tests.push(path.clone());
                }
                ScoredPath {
                    path,
                    confidence: conf,
                    reason,
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
            import_expansions.push(ScoredPath {
                path: hit.path,
                confidence,
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
    /// built. Counted, not enumerated.
    pub beyond_first_cap: usize,
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
        assert_eq!(
            imp["edges"],
            serde_json::json!([
                {"from": "app/bbb.py", "to": "app/ccc.py"},
                {"from": "app/ccc.py", "to": "app/ddd.py"},
            ])
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

    /// One anchor, `partners` indexed files it co-changed with (strengths
    /// strictly decreasing, so the ranking is unambiguous) and `importers`
    /// indexed files that import it.
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
        store.replace_cochange(&edges).unwrap();
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
            serde_json::to_string(&(&outcome.packets, &outcome.cap_dropped)).unwrap()
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
}
