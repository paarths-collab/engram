//! engram-retrieval: hybrid BM25 (Tantivy) + vector (cosine) retrieval with
//! score fusion, lazy Tier-1 extraction on miss, and co-change impact expansion.

pub mod embed;
pub mod stopwords;
pub mod weights;

use anyhow::Result;
use embed::{cosine, Embedder, HashedNgramEmbedder};
use engram_domain::{EvidenceKind, EvidencePacket, ImpactPrediction, ScoredPath, SymbolRecord};
use engram_repo_map::graph::{CodeGraph, EdgeKind};
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
                chunks.push(Chunk {
                    doc: doc_index,
                    symbol,
                    start_line,
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
        })
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
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(top_k);

        // lazy Tier-1: ensure symbols exist for top hits, then attach best symbols
        let mut packets = Vec::new();
        for (rank, (i, score, signals)) in ranked.iter().enumerate() {
            let d = &self.docs[*i];
            engram_repo_map::ensure_tier1(store, &self.repo_root, &d.path)?;
            // Prefer the span the vector side actually matched: its text is the
            // definition itself, where the symbol table only holds a signature
            // line and the file preview is just the first few hundred bytes.
            let matched_span = best_chunk
                .get(i)
                .map(|&index| &self.chunks[index])
                .filter(|chunk| chunk.symbol.is_some());
            let (title, symbol, snippet) = match matched_span {
                Some(chunk) => {
                    let name = chunk.symbol.clone().unwrap_or_default();
                    // path:line, so the agent can open the definition directly
                    // instead of searching the file for the name again.
                    (
                        format!("{} in {}:{}", name, d.path, chunk.start_line),
                        Some(name),
                        Some(chunk.text.clone()),
                    )
                }
                None => match best_symbols_for(store, &d.path, &q_tokens).first() {
                    Some(s) => (
                        format!("{} in {}", s.name, d.path),
                        Some(s.name.clone()),
                        Some(s.signature.clone()),
                    ),
                    None => (d.path.clone(), None, Some(d.preview.clone())),
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
                    score: self.config.weights.symbol_exact,
                    signals: vec!["symbol".into()],
                });
            }
        }

        // Re-rank before cutting. The symbol hits above are appended, not
        // inserted in score order, so without this an exact match sits below
        // whatever fusion put in the first top_k slots and gets truncated away
        // — which is exactly how the langchain `merge_dicts` miss happened.
        packets.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        packets.truncate(top_k + 3);
        Ok(packets)
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

        let (cochange_expansions, import_expansions, mut expansion_tests) =
            self.expand_from_seeds(store, &seeds)?;
        likely_tests.append(&mut expansion_tests);
        likely_tests.sort();
        likely_tests.dedup();

        Ok(ImpactPrediction {
            likely_files,
            likely_tests,
            cochange_expansions,
            import_expansions,
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
        let known: Vec<String> = anchors
            .iter()
            .filter(|p| self.by_path.contains_key(p.as_str()))
            .cloned()
            .collect();
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

        let (cochange_expansions, import_expansions, mut expansion_tests) =
            self.expand_from_seeds(store, &known)?;
        likely_tests.append(&mut expansion_tests);
        likely_tests.sort();
        likely_tests.dedup();

        Ok(ImpactPrediction {
            likely_files,
            likely_tests,
            cochange_expansions,
            import_expansions,
        })
    }

    /// Shared deterministic expansion: co-change graph (git history fact) +
    /// import graph (static-analysis fact, up to 2 hops), excluding files
    /// already in `seeds`/`likely`. Returns (cochange, imports, discovered_tests).
    fn expand_from_seeds(
        &self,
        store: &mut Store,
        seeds: &[String],
    ) -> Result<(Vec<ScoredPath>, Vec<ScoredPath>, Vec<String>)> {
        let mut likely_tests = Vec::new();
        let mut expansions: HashMap<String, (f32, String)> = HashMap::new();
        for seed in seeds {
            for edge in store.cochange_for(seed, 5)? {
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
        }

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
        cochange_expansions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        cochange_expansions.truncate(8);

        // Import-graph expansion via the in-memory code graph: files that
        // (transitively, up to 2 hops) import a seed. SQL stores the edges;
        // we traverse them in memory (see docs/adr/0001).
        let seed_paths: HashSet<&str> = seeds.iter().map(|s| s.as_str()).collect();
        let cochange_paths: HashSet<&str> = cochange_expansions
            .iter()
            .map(|c| c.path.as_str())
            .collect();
        let mut import_expansions: Vec<ScoredPath> = Vec::new();
        for hit in self.graph.dependents(seeds, &[EdgeKind::Imports], 2) {
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
                reason: format!(
                    "imports a likely-changed file ({} hop{})",
                    hit.hops,
                    if hit.hops == 1 { "" } else { "s" }
                ),
            });
        }
        import_expansions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        import_expansions.truncate(8);

        Ok((cochange_expansions, import_expansions, likely_tests))
    }
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
}
