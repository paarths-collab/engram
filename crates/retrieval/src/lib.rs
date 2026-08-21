//! engram-retrieval: hybrid BM25 (Tantivy) + vector (cosine) retrieval with
//! score fusion, lazy Tier-1 extraction on miss, and co-change impact expansion.

pub mod embed;
pub mod stopwords;
pub mod weights;

use anyhow::Result;
use embed::{cosine, Embedder, HashedNgramEmbedder};
use engram_domain::{
    EvidenceKind, EvidencePacket, ImpactPrediction, ReuseAssessment, ReuseCandidate, ReuseState,
    ScoredPath, SymbolKind, SymbolRecord,
};
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
    /// Inclusive one-based end line; `0` for the head chunk.
    end_line: usize,
    symbol_kind: Option<SymbolKind>,
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
    /// Whether every parser-supported file had Tier-1 symbols extracted when
    /// this engine was built. An honest negative reuse result requires this.
    symbol_index_complete: bool,
}

const PREVIEW_BYTES: usize = 400;
const INDEX_BODY_BYTES: usize = 60_000;
/// Cap on the source text embedded and quoted for one chunk. Long enough for a
/// normal definition, short enough that one sprawling function cannot dominate.
const CHUNK_BYTES: usize = 4_000;
const REUSE_CANDIDATE_LIMIT: usize = 3;

struct PlannedChunk {
    start_line: usize,
    end_line: usize,
    symbol: Option<String>,
    symbol_kind: Option<SymbolKind>,
    text: String,
    truncated: bool,
}

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

fn query_requests_tests(query: &str) -> bool {
    query
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .any(|token| {
            matches!(
                token.as_str(),
                "test" | "tests" | "testing" | "spec" | "specs"
            )
        })
}

fn identifier_parts(identifier: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut previous_was_lower_or_digit = false;
    for ch in identifier.chars() {
        if !ch.is_alphanumeric() {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            previous_was_lower_or_digit = false;
            continue;
        }
        if ch.is_uppercase() && previous_was_lower_or_digit && !current.is_empty() {
            parts.push(std::mem::take(&mut current));
        }
        current.extend(ch.to_lowercase());
        previous_was_lower_or_digit = ch.is_lowercase() || ch.is_ascii_digit();
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn symbol_has_token(symbol: &str, query_token: &str) -> bool {
    identifier_parts(symbol)
        .iter()
        .any(|part| part == query_token)
}

fn evidence_id(path: &str, symbol: Option<&str>, start_line: Option<usize>) -> String {
    let identity = format!(
        "{}\0{}\0{}",
        path,
        symbol.unwrap_or(""),
        start_line.unwrap_or(0)
    );
    format!("ev_{}", embed::content_hash(&identity))
}

fn same_candidate(a: &EvidencePacket, b: &EvidencePacket) -> bool {
    a.path == b.path && a.symbol == b.symbol && a.start_line == b.start_line
}

fn merge_packet(packets: &mut Vec<EvidencePacket>, mut candidate: EvidencePacket) {
    if let Some(existing) = packets
        .iter_mut()
        .find(|existing| same_candidate(existing, &candidate))
    {
        existing.score = existing.score.max(candidate.score);
        existing.bm25_score = match (existing.bm25_score, candidate.bm25_score) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        existing.vector_score = match (existing.vector_score, candidate.vector_score) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        if existing.start_line.is_none() {
            existing.start_line = candidate.start_line;
        }
        if existing.end_line.is_none() {
            existing.end_line = candidate.end_line;
        }
        if existing.symbol_kind.is_none() {
            existing.symbol_kind = candidate.symbol_kind;
        }
        if candidate
            .snippet
            .as_ref()
            .is_some_and(|snippet| snippet.len() > existing.snippet.as_deref().unwrap_or("").len())
        {
            existing.snippet = candidate.snippet.take();
        }
        for signal in candidate.signals {
            if !existing.signals.contains(&signal) {
                existing.signals.push(signal);
            }
        }
        existing.signals.sort();
        return;
    }
    candidate.signals.sort();
    candidate.signals.dedup();
    packets.push(candidate);
}

fn compare_packets(a: &EvidencePacket, b: &EvidencePacket) -> std::cmp::Ordering {
    b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.path.cmp(&b.path))
        .then_with(|| a.start_line.cmp(&b.start_line))
        .then_with(|| a.symbol.cmp(&b.symbol))
}

/// Plan a file's chunks: the head, then one span per extracted symbol.
///
/// The head chunk always exists. A file with no chunks could never be returned
/// by vector search, and symbol extraction is lazy, so plenty of files have no
/// symbols yet on any given build.
fn plan_chunks(source: &str, body: &str, symbols: &[SymbolRecord]) -> Vec<PlannedChunk> {
    let mut planned = vec![PlannedChunk {
        start_line: 0,
        end_line: 0,
        symbol: None,
        symbol_kind: None,
        text: body.chars().take(CHUNK_BYTES).collect(),
        truncated: source.chars().count() > CHUNK_BYTES,
    }];
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
        let full_text = lines[symbol.start_line - 1..end].join("\n");
        let truncated = full_text.chars().count() > CHUNK_BYTES;
        let text: String = full_text.chars().take(CHUNK_BYTES).collect();
        planned.push(PlannedChunk {
            start_line: symbol.start_line,
            end_line: end,
            symbol: Some(symbol.name.clone()),
            symbol_kind: Some(symbol.kind),
            text,
            truncated,
        });
    }
    planned
}

fn reuse_state_for_packet(packet: &EvidencePacket, tests_requested: bool) -> Option<ReuseState> {
    let exact_symbol = packet.signals.iter().any(|signal| signal == "symbol_exact");
    let fuzzy_symbol = packet.signals.iter().any(|signal| signal == "symbol");
    let symbol_evidence = packet.symbol.is_some() && (exact_symbol || fuzzy_symbol);
    let path = packet.signals.iter().any(|signal| signal == "path");

    // The current "vector" backend is hashed token/character n-grams, not a
    // semantic model. BM25 and that vector therefore form one lexical family,
    // not two independent votes. Treating them independently recreates the
    // old false-positive bug for nearly every vocabulary overlap.
    // One fuzzy name/path signal is useful enough to inspect. BM25 plus the
    // hashed-ngram backend is still text-only overlap, so without an identity
    // signal it cannot establish that the matching span is an implementation.
    if !symbol_evidence && !path {
        return None;
    }

    // Symbol-name, path, BM25, and hashed n-grams are all lexical. Only exact
    // identity, or agreement between the separately indexed symbol and path
    // identity surfaces, is independent enough for a strong claim today.
    let mut state = if exact_symbol || (symbol_evidence && path) {
        ReuseState::ReuseLikely
    } else {
        ReuseState::PossibleReuse
    };
    if packet.kind == EvidenceKind::Test && !tests_requested {
        state = ReuseState::PossibleReuse;
    }
    Some(state)
}

fn has_precise_source_evidence(packet: &EvidencePacket) -> bool {
    let Some(symbol) = packet.symbol.as_deref().filter(|symbol| !symbol.is_empty()) else {
        return false;
    };
    packet.start_line.is_some_and(|line| line > 0)
        && packet
            .snippet
            .as_deref()
            .is_some_and(|snippet| !snippet.trim().is_empty() && snippet.contains(symbol))
}

fn is_explicitly_deprecated(packet: &EvidencePacket) -> bool {
    packet.snippet.as_deref().is_some_and(|snippet| {
        let lower = snippet.to_ascii_lowercase();
        lower.contains("#[deprecated")
            || lower.contains("@deprecated")
            || lower.contains("deprecationwarning")
    })
}

fn reuse_state_priority(state: ReuseState) -> u8 {
    match state {
        ReuseState::ReuseLikely => 0,
        ReuseState::PossibleReuse => 1,
        ReuseState::NoEvidence => 2,
        ReuseState::IndexIncomplete => 3,
    }
}

fn assess_reuse_packets(
    pool: Vec<EvidencePacket>,
    indexed_files: usize,
    index_complete: bool,
    tests_requested: bool,
) -> ReuseAssessment {
    let mut candidates: Vec<ReuseCandidate> = pool
        .into_iter()
        .filter(|packet| classify_path(&packet.path) == DocClass::Code)
        .filter(has_precise_source_evidence)
        .filter(|packet| !is_explicitly_deprecated(packet))
        .filter_map(|evidence| {
            reuse_state_for_packet(&evidence, tests_requested)
                .map(|state| ReuseCandidate { state, evidence })
        })
        .collect();
    candidates.sort_by(|a, b| {
        reuse_state_priority(a.state)
            .cmp(&reuse_state_priority(b.state))
            .then_with(|| compare_packets(&a.evidence, &b.evidence))
    });
    candidates.truncate(REUSE_CANDIDATE_LIMIT);

    let has_likely = candidates
        .iter()
        .any(|candidate| candidate.state == ReuseState::ReuseLikely);
    let state = if has_likely {
        ReuseState::ReuseLikely
    } else if !index_complete {
        // Weak candidates cannot override missing coverage. Suppress them so
        // callers cannot accidentally treat an incomplete result as evidence.
        candidates.clear();
        ReuseState::IndexIncomplete
    } else if !candidates.is_empty() {
        ReuseState::PossibleReuse
    } else {
        ReuseState::NoEvidence
    };

    ReuseAssessment {
        state,
        candidates,
        indexed_files,
        index_complete,
    }
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
        let mut symbol_index_complete = true;

        for f in store.all_files()? {
            let full = repo_root.join(&f.path);
            let Ok(src) = std::fs::read_to_string(&full) else {
                symbol_index_complete = false;
                continue;
            };
            let body: String = src.chars().take(INDEX_BODY_BYTES).collect();
            let doc_index = docs.len();
            if engram_repo_map::symbols::supports(f.language) && !store.is_tier1_done(&f.path) {
                symbol_index_complete = false;
            }
            if f.language != engram_domain::Language::Other
                && !engram_repo_map::symbols::supports(f.language)
            {
                symbol_index_complete = false;
            }

            // Symbols are extracted lazily, so a file may legitimately have
            // none yet; plan_chunks always yields at least the head chunk.
            let symbols = store.symbols_for_path(&f.path).unwrap_or_default();
            for planned in plan_chunks(&src, &body, &symbols) {
                if planned.truncated && (planned.symbol.is_some() || symbols.is_empty()) {
                    symbol_index_complete = false;
                }
                let embed_text = match &planned.symbol {
                    Some(name) => format!("{} {} {}", f.path, name, planned.text),
                    None => format!("{} {}", f.path, planned.text),
                };
                let hash = embed::content_hash(&embed_text);
                let key = (f.path.clone(), planned.start_line);
                let (vector, from_cache) =
                    embed_or_reuse(&embedder, cached.get(&key), &embed_text, &hash);
                if from_cache {
                    reused += 1;
                } else {
                    embedded += 1;
                    fresh.push((
                        f.path.clone(),
                        planned.start_line,
                        hash,
                        embed::vector_to_bytes(&vector),
                    ));
                }
                present.insert(key);
                chunks.push(Chunk {
                    doc: doc_index,
                    symbol: planned.symbol,
                    start_line: planned.start_line,
                    end_line: planned.end_line,
                    symbol_kind: planned.symbol_kind,
                    text: planned.text,
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
            symbol_index_complete,
        })
    }

    pub fn indexed_file_count(&self) -> usize {
        self.docs.len()
    }

    pub fn symbol_index_complete(&self) -> bool {
        self.symbol_index_complete
    }

    /// Assess whether indexed evidence supports reusing an implementation.
    ///
    /// Unlike generic search, this API can abstain. It excludes documentation,
    /// refuses vector-only evidence, distinguishes likely from merely possible
    /// reuse, and returns at most three deterministic candidates.
    pub fn assess_reuse(&mut self, store: &mut Store, concept: &str) -> Result<ReuseAssessment> {
        let tests_requested = query_requests_tests(concept);
        let pool = self.search(store, concept, 50)?;
        Ok(assess_reuse_packets(
            pool,
            self.indexed_file_count(),
            self.symbol_index_complete,
            tests_requested,
        ))
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

    fn chunk_for_symbol(&self, path: &str, start_line: usize) -> Option<&Chunk> {
        self.chunks
            .iter()
            .find(|chunk| chunk.start_line == start_line && self.docs[chunk.doc].path == path)
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
        let bm25_raw: HashMap<usize, f32> = bm25.iter().copied().collect();
        let vector_raw: HashMap<usize, f32> = vecs.iter().copied().collect();

        let query_wants_tests = query_requests_tests(query);

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
            let path_tokens: HashSet<&str> = pl
                .split(|c: char| !c.is_alphanumeric())
                .filter(|token| !token.is_empty())
                .collect();
            if q_tokens.iter().any(|t| path_tokens.contains(t.as_str())) {
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
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.docs[a.0].path.cmp(&self.docs[b.0].path))
        });
        ranked.truncate(top_k);

        // lazy Tier-1: ensure symbols exist for top hits, then attach best symbols
        let mut packets = Vec::new();
        for (i, score, signals) in &ranked {
            let d = &self.docs[*i];
            engram_repo_map::ensure_tier1(store, &self.repo_root, &d.path)?;
            // Prefer the span the vector side actually matched: its text is the
            // definition itself, where the symbol table only holds a signature
            // line and the file preview is just the first few hundred bytes.
            let matched_span = best_chunk
                .get(i)
                .map(|&index| &self.chunks[index])
                .filter(|chunk| chunk.symbol.is_some());
            let (title, symbol, start_line, end_line, symbol_kind, snippet) = match matched_span {
                Some(chunk) => {
                    let name = chunk.symbol.clone().unwrap_or_default();
                    // path:line, so the agent can open the definition directly
                    // instead of searching the file for the name again.
                    (
                        format!("{} in {}:{}", name, d.path, chunk.start_line),
                        Some(name),
                        Some(chunk.start_line),
                        Some(chunk.end_line),
                        chunk.symbol_kind,
                        Some(chunk.text.clone()),
                    )
                }
                None => match best_symbols_for(store, &d.path, &q_tokens).first() {
                    Some(s) => (
                        format!("{} in {}:{}", s.name, d.path, s.start_line),
                        Some(s.name.clone()),
                        Some(s.start_line),
                        Some(s.end_line),
                        Some(s.kind),
                        Some(s.signature.clone()),
                    ),
                    None => (
                        d.path.clone(),
                        None,
                        None,
                        None,
                        None,
                        Some(d.preview.clone()),
                    ),
                },
            };
            let start_for_id = start_line;
            let symbol_for_id = symbol.as_deref();
            packets.push(EvidencePacket {
                id: evidence_id(&d.path, symbol_for_id, start_for_id),
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
                start_line,
                end_line,
                symbol_kind,
                snippet,
                score: *score,
                bm25_score: bm25_raw.get(i).copied(),
                vector_score: vector_raw.get(i).copied(),
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
                let chunk = self.chunk_for_symbol(&s.path, s.start_line);
                merge_packet(
                    &mut packets,
                    EvidencePacket {
                        id: evidence_id(&s.path, Some(&s.name), Some(s.start_line)),
                        kind: if engram_repo_map::inventory::is_test_path(&s.path) {
                            EvidenceKind::Test
                        } else {
                            EvidenceKind::Symbol
                        },
                        title: format!("{} in {}:{}", s.name, s.path, s.start_line),
                        path: s.path.clone(),
                        symbol: Some(s.name.clone()),
                        start_line: Some(s.start_line),
                        end_line: Some(s.end_line),
                        symbol_kind: Some(s.kind),
                        snippet: Some(
                            chunk
                                .map(|matched| matched.text.clone())
                                .unwrap_or_else(|| s.signature.clone()),
                        ),
                        score: exact_score,
                        bm25_score: None,
                        vector_score: None,
                        signals: vec!["symbol_exact".into()],
                    },
                );
            }
        }

        // Weaker whole identifier-token hits that BM25/vector may have missed.
        // SQL supplies a broad substring pool; the token filter prevents tiny
        // words such as `ion` from matching `configuration`.
        for t in &q_tokens {
            for s in store
                .symbols_matching(t, 25)?
                .into_iter()
                .filter(|symbol| symbol_has_token(&symbol.name, t))
                .take(3)
            {
                let chunk = self.chunk_for_symbol(&s.path, s.start_line);
                merge_packet(
                    &mut packets,
                    EvidencePacket {
                        id: evidence_id(&s.path, Some(&s.name), Some(s.start_line)),
                        kind: if engram_repo_map::inventory::is_test_path(&s.path) {
                            EvidenceKind::Test
                        } else {
                            EvidenceKind::Symbol
                        },
                        title: format!("{} in {}:{}", s.name, s.path, s.start_line),
                        path: s.path.clone(),
                        symbol: Some(s.name.clone()),
                        start_line: Some(s.start_line),
                        end_line: Some(s.end_line),
                        symbol_kind: Some(s.kind),
                        snippet: Some(
                            chunk
                                .map(|matched| matched.text.clone())
                                .unwrap_or_else(|| s.signature.clone()),
                        ),
                        score: self.config.weights.symbol_exact,
                        bm25_score: None,
                        vector_score: None,
                        signals: vec!["symbol".into()],
                    },
                );
            }
        }

        // Re-rank before cutting. The symbol hits above are appended, not
        // inserted in score order, so without this an exact match sits below
        // whatever fusion put in the first top_k slots and gets truncated away
        // — which is exactly how the langchain `merge_dicts` miss happened.
        packets.sort_by(compare_packets);
        packets.truncate(top_k);
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

    /// Explain, from recorded facts only, why two files are connected. Returns a
    /// list of structured reasons (import edge with direction/hops, or historical
    /// co-change), each traceable to a concrete edge. Empty means "no recorded
    /// connection found" — never a guess. Reuses the same deterministic
    /// expansion as `impact_from_files`.
    pub fn explain_connection(
        &mut self,
        store: &mut Store,
        source: &str,
        target: &str,
    ) -> Result<Vec<serde_json::Value>> {
        let mut reasons = Vec::new();
        // Expansions of `source`: co-change is symmetric; import_expansions here
        // are files that import `source`, so `target` appearing means target->source.
        let from_source =
            self.impact_from_files(store, std::slice::from_ref(&source.to_string()))?;
        for sp in from_source
            .cochange_expansions
            .iter()
            .filter(|p| p.path == target)
        {
            reasons.push(serde_json::json!({
                "type": "historical_cochange",
                "detail": sp.reason,
                "weight": sp.confidence,
            }));
        }
        for sp in from_source
            .import_expansions
            .iter()
            .filter(|p| p.path == target)
        {
            reasons.push(serde_json::json!({
                "type": "import_edge",
                "detail": format!("{target} imports {source}"),
                "weight": sp.confidence,
            }));
        }
        // Reverse direction: files that import `target`; `source` appearing means
        // source->target.
        let from_target =
            self.impact_from_files(store, std::slice::from_ref(&target.to_string()))?;
        for sp in from_target
            .import_expansions
            .iter()
            .filter(|p| p.path == source)
        {
            reasons.push(serde_json::json!({
                "type": "import_edge",
                "detail": format!("{source} imports {target}"),
                "weight": sp.confidence,
            }));
        }
        Ok(reasons)
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
        if let Ok(mut syms) = store.symbols_matching(t, 25) {
            syms.retain(|s| s.path == path && symbol_has_token(&s.name, t));
            syms.truncate(5);
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

    fn packet(
        path: &str,
        symbol: Option<&str>,
        kind: EvidenceKind,
        score: f32,
        signals: &[&str],
    ) -> EvidencePacket {
        let start_line = symbol.map(|_| 10);
        EvidencePacket {
            id: evidence_id(path, symbol, start_line),
            kind,
            title: path.to_owned(),
            path: path.to_owned(),
            symbol: symbol.map(str::to_owned),
            start_line,
            end_line: symbol.map(|_| 14),
            symbol_kind: symbol.map(|_| SymbolKind::Function),
            snippet: symbol.map(|name| format!("fn {name}() {{}}")),
            score,
            bm25_score: None,
            vector_score: None,
            signals: signals.iter().map(|signal| (*signal).to_owned()).collect(),
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
    fn test_intent_uses_tokens_not_substrings() {
        assert!(query_requests_tests("find the unit tests for retries"));
        assert!(query_requests_tests("reuse a spec helper"));
        assert!(!query_requests_tests("use the latest retry helper"));
    }

    #[test]
    fn symbol_token_matching_rejects_embedded_substrings() {
        assert!(symbol_has_token("retry_policy", "retry"));
        assert!(symbol_has_token("RetryPolicy", "policy"));
        assert!(!symbol_has_token("configuration", "ion"));
        assert!(!symbol_has_token("latest_value", "test"));
    }

    #[test]
    fn exact_symbol_is_reuse_likely() {
        let assessment = assess_reuse_packets(
            vec![packet(
                "src/retry.rs",
                Some("retry_policy"),
                EvidenceKind::Symbol,
                4.0,
                &["symbol_exact"],
            )],
            1,
            true,
            false,
        );
        assert_eq!(assessment.state, ReuseState::ReuseLikely);
        assert_eq!(assessment.candidates[0].state, ReuseState::ReuseLikely);
    }

    #[test]
    fn unrelated_vector_only_result_abstains() {
        let mut noisy = packet(
            "src/recent.rs",
            Some("unrelated"),
            EvidenceKind::Symbol,
            1.2,
            &["vector", "recent"],
        );
        noisy.vector_score = Some(0.91);
        let assessment = assess_reuse_packets(vec![noisy], 1, true, false);
        assert_eq!(assessment.state, ReuseState::NoEvidence);
        assert!(assessment.candidates.is_empty());
    }

    #[test]
    fn same_named_symbols_in_different_files_are_preserved() {
        let mut packets = Vec::new();
        merge_packet(
            &mut packets,
            packet(
                "src/a.rs",
                Some("retry_policy"),
                EvidenceKind::Symbol,
                2.0,
                &["symbol_exact"],
            ),
        );
        merge_packet(
            &mut packets,
            packet(
                "src/b.rs",
                Some("retry_policy"),
                EvidenceKind::Symbol,
                2.0,
                &["symbol_exact"],
            ),
        );
        assert_eq!(packets.len(), 2);
        assert_ne!(packets[0].id, packets[1].id);
    }

    #[test]
    fn exact_and_fuzzy_evidence_for_one_symbol_is_merged() {
        let mut packets = vec![packet(
            "src/retry.rs",
            Some("retry_policy"),
            EvidenceKind::Symbol,
            1.0,
            &["symbol"],
        )];
        merge_packet(
            &mut packets,
            packet(
                "src/retry.rs",
                Some("retry_policy"),
                EvidenceKind::Symbol,
                3.0,
                &["symbol_exact"],
            ),
        );
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].score, 3.0);
        assert_eq!(packets[0].signals, vec!["symbol", "symbol_exact"]);
    }

    #[test]
    fn documentation_cannot_be_called_reusable_code() {
        let assessment = assess_reuse_packets(
            vec![packet(
                "README.md",
                Some("retry_policy"),
                EvidenceKind::Symbol,
                5.0,
                &["symbol_exact"],
            )],
            1,
            true,
            false,
        );
        assert_eq!(assessment.state, ReuseState::NoEvidence);
        assert!(assessment.candidates.is_empty());
    }

    #[test]
    fn tests_are_possible_not_likely_unless_explicitly_requested() {
        let test = packet(
            "tests/retry.rs",
            Some("retry_policy"),
            EvidenceKind::Test,
            5.0,
            &["symbol_exact"],
        );
        let ordinary = assess_reuse_packets(vec![test.clone()], 1, true, false);
        assert_eq!(ordinary.state, ReuseState::PossibleReuse);
        assert_eq!(ordinary.candidates[0].state, ReuseState::PossibleReuse);

        let explicitly_requested = assess_reuse_packets(vec![test], 1, true, true);
        assert_eq!(explicitly_requested.state, ReuseState::ReuseLikely);
    }

    #[test]
    fn weak_single_non_vector_signal_is_only_possible_reuse() {
        let assessment = assess_reuse_packets(
            vec![packet(
                "src/retry.rs",
                Some("retry_policy"),
                EvidenceKind::Symbol,
                1.5,
                &["symbol"],
            )],
            1,
            true,
            false,
        );
        assert_eq!(assessment.state, ReuseState::PossibleReuse);
        assert_eq!(assessment.candidates[0].state, ReuseState::PossibleReuse);
    }

    #[test]
    fn correlated_text_only_signals_abstain() {
        let mut text_only = packet(
            "src/retry.rs",
            Some("some_span"),
            EvidenceKind::Symbol,
            2.0,
            &["bm25", "vector"],
        );
        text_only.bm25_score = Some(2.0);
        text_only.vector_score = Some(0.7);
        let abstained = assess_reuse_packets(vec![text_only], 1, true, false);
        assert_eq!(abstained.state, ReuseState::NoEvidence);
        assert!(abstained.candidates.is_empty());

        let mut symbol_and_vector = packet(
            "src/retry.rs",
            Some("retry_policy"),
            EvidenceKind::Symbol,
            2.0,
            &["symbol", "vector"],
        );
        symbol_and_vector.vector_score = Some(0.7);
        let possible = assess_reuse_packets(vec![symbol_and_vector], 1, true, false);
        assert_eq!(possible.state, ReuseState::PossibleReuse);

        let symbol_and_path = packet(
            "src/retry.rs",
            Some("retry_policy"),
            EvidenceKind::Symbol,
            2.0,
            &["symbol", "path"],
        );
        let likely = assess_reuse_packets(vec![symbol_and_path], 1, true, false);
        assert_eq!(likely.state, ReuseState::ReuseLikely);
    }

    #[test]
    fn candidates_without_exact_symbol_evidence_are_not_returned() {
        let mut head = packet(
            "src/retry.rs",
            None,
            EvidenceKind::ExistingCode,
            2.0,
            &["bm25", "vector"],
        );
        head.bm25_score = Some(2.0);
        head.vector_score = Some(0.7);
        let assessment = assess_reuse_packets(vec![head], 1, true, false);
        assert_eq!(assessment.state, ReuseState::NoEvidence);
        assert!(assessment.candidates.is_empty());
    }

    #[test]
    fn explicitly_deprecated_symbols_do_not_guide_reuse() {
        let mut deprecated = packet(
            "src/legacy.rs",
            Some("legacy_retry"),
            EvidenceKind::Symbol,
            5.0,
            &["symbol_exact"],
        );
        deprecated.snippet =
            Some("#[deprecated(note = \"unsafe\")] pub fn legacy_retry() {}".to_owned());
        let assessment = assess_reuse_packets(vec![deprecated], 1, true, false);
        assert_eq!(assessment.state, ReuseState::NoEvidence);
        assert!(assessment.candidates.is_empty());
    }

    #[test]
    fn incomplete_coverage_overrides_weak_candidates() {
        let weak = packet(
            "src/retry.rs",
            Some("retry_policy"),
            EvidenceKind::Symbol,
            1.5,
            &["symbol"],
        );
        let assessment = assess_reuse_packets(vec![weak], 100, false, false);
        assert_eq!(assessment.state, ReuseState::IndexIncomplete);
        assert!(assessment.candidates.is_empty());
    }

    #[test]
    fn empty_result_reports_incomplete_index_honestly() {
        let assessment = assess_reuse_packets(Vec::new(), 100, false, false);
        assert_eq!(assessment.state, ReuseState::IndexIncomplete);
        assert!(!assessment.index_complete);
    }

    #[test]
    fn reuse_candidates_are_stable_and_capped_at_three() {
        let pool = ["e.rs", "d.rs", "c.rs", "b.rs", "a.rs"]
            .into_iter()
            .map(|path| {
                packet(
                    path,
                    Some("retry_policy"),
                    EvidenceKind::Symbol,
                    4.0,
                    &["symbol_exact"],
                )
            })
            .collect();
        let assessment = assess_reuse_packets(pool, 5, true, false);
        let paths: Vec<&str> = assessment
            .candidates
            .iter()
            .map(|candidate| candidate.evidence.path.as_str())
            .collect();
        assert_eq!(paths, vec!["a.rs", "b.rs", "c.rs"]);
    }

    #[test]
    fn a_file_without_symbols_still_gets_one_chunk() {
        // A doc with no chunks could never be returned by vector search, and
        // symbol extraction is lazy, so this is a normal state not an edge case.
        let planned = plan_chunks(SRC, SRC, &[]);
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].start_line, 0, "head chunk is keyed at line 0");
        assert!(planned[0].symbol.is_none());
        assert!(planned[0].text.contains("use std::io;"));
    }

    #[test]
    fn each_symbol_becomes_its_own_chunk_holding_its_definition() {
        let planned = plan_chunks(SRC, SRC, &[symbol("first", 3, 5), symbol("second", 7, 9)]);
        assert_eq!(planned.len(), 3, "head chunk plus one per symbol");

        let second = planned
            .iter()
            .find(|chunk| chunk.symbol.as_deref() == Some("second"))
            .expect("second was planned");
        assert_eq!(second.start_line, 7, "keyed by its start line");
        assert_eq!(second.end_line, 9);
        assert_eq!(second.symbol_kind, Some(SymbolKind::Function));
        assert!(
            second.text.starts_with("fn second()"),
            "got: {}",
            second.text
        );
        assert!(second.text.contains("let y = 2;"), "got: {}", second.text);
        assert!(
            !second.text.contains("first"),
            "bled into the neighbour: {}",
            second.text
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
        assert_eq!(chunk.symbol.as_deref(), Some("truncated"));
        assert!(chunk.text.contains("let y = 2;"));
    }

    #[test]
    fn chunk_text_is_capped() {
        let huge = format!("fn big() {{\n{}\n}}\n", "    let x = 1;\n".repeat(2_000));
        let planned = plan_chunks(&huge, &huge, &[symbol("big", 1, 2_002)]);
        assert!(planned[1].text.chars().count() <= CHUNK_BYTES);
        assert!(planned[1].truncated);
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
