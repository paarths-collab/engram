//! engram-retrieval: hybrid BM25 (Tantivy) + vector (cosine) retrieval with
//! score fusion, lazy Tier-1 extraction on miss, and co-change impact expansion.

pub mod embed;
pub mod stopwords;
pub mod weights;

use anyhow::Result;
use embed::{cosine, Embedder, HashedNgramEmbedder};
use engram_domain::{ConnectionMap, EvidenceKind, EvidencePacket, ScoredPath, SymbolRecord};
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

#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    pub top_k: usize,
    pub prefer_tests: bool,
    pub prefer_docs: bool,
}

impl SearchOptions {
    pub fn standard(top_k: usize) -> Self {
        Self {
            top_k,
            prefer_tests: false,
            prefer_docs: false,
        }
    }
}

struct DocMeta {
    path: String,
    is_test: bool,
    preview: String,
    vector: Vec<f32>,
    last_commit_ts: Option<i64>,
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
    by_path: HashMap<String, usize>,
    embedder: HashedNgramEmbedder,
    config: WeightsConfig,
    graph: CodeGraph,
    stopwords: std::sync::Arc<HashSet<String>>,
}

const PREVIEW_BYTES: usize = 400;
const INDEX_BODY_BYTES: usize = 60_000;

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
        let mut by_path = HashMap::new();
        let recency = store.recency_map()?;
        let cached = store.load_vectors()?;
        let mut fresh: Vec<(String, String, Vec<u8>)> = Vec::new();
        let mut present: HashSet<String> = HashSet::new();
        let (mut reused, mut embedded) = (0usize, 0usize);

        for f in store.all_files()? {
            let full = repo_root.join(&f.path);
            let Ok(src) = std::fs::read_to_string(&full) else {
                continue;
            };
            let body: String = src.chars().take(INDEX_BODY_BYTES).collect();
            // Embed path + a head slice; heads carry imports/signatures = high signal.
            let embed_text = format!("{} {}", f.path, body.chars().take(4000).collect::<String>());
            let hash = embed::content_hash(&embed_text);
            let vector = match cached.get(&f.path) {
                Some((h, bytes)) if *h == hash => match embed::bytes_to_vector(bytes) {
                    Some(v) => {
                        reused += 1;
                        v
                    }
                    None => {
                        embedded += 1;
                        let v = embedder.embed(&embed_text);
                        fresh.push((f.path.clone(), hash, embed::vector_to_bytes(&v)));
                        v
                    }
                },
                _ => {
                    embedded += 1;
                    let v = embedder.embed(&embed_text);
                    fresh.push((f.path.clone(), hash, embed::vector_to_bytes(&v)));
                    v
                }
            };
            present.insert(f.path.clone());
            writer.add_document(doc!(f_path => f.path.clone(), f_body => body.clone()))?;
            by_path.insert(f.path.clone(), docs.len());
            docs.push(DocMeta {
                preview: body.chars().take(PREVIEW_BYTES).collect(),
                last_commit_ts: recency.get(&f.path).copied(),
                path: f.path,
                is_test: f.is_test,
                vector,
            });
        }
        writer.commit()?;
        if !fresh.is_empty() {
            store.upsert_vectors(&fresh)?;
        }
        store.prune_vectors(&present)?;
        eprintln!("[engram] vectors: {reused} reused, {embedded} embedded");
        let graph = CodeGraph::build(store)?;
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

    fn vector_candidates(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let qv = self.embedder.embed(query);
        let mut scored: Vec<(usize, f32)> = self
            .docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i, cosine(&qv, &d.vector)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(k);
        scored
    }

    /// Hybrid search: BM25 + vector fused with weighted normalized scores,
    /// plus symbol exact-match boost. Lazily Tier-1 extracts top hits on miss.
    pub fn search(
        &mut self,
        store: &mut Store,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<EvidencePacket>> {
        self.search_with_options(store, query, SearchOptions::standard(top_k))
    }

    pub fn search_with_options(
        &mut self,
        store: &mut Store,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<EvidencePacket>> {
        // Dev hot-reload: pick up config/scoring.toml edits without a restart.
        self.config.reload_if_changed();
        let bm25 = self.bm25_candidates(query, 50);
        let vecs = self.vector_candidates(query, 50);
        let now = now_unix();
        let w = &self.config.weights;

        let norm = |list: &[(usize, f32)]| -> HashMap<usize, f32> {
            let max = list.iter().map(|x| x.1).fold(0f32, f32::max).max(1e-6);
            list.iter().map(|(i, s)| (*i, s / max)).collect()
        };
        let bm25_n = norm(&bm25);
        let vec_n = norm(&vecs);

        let query_wants_tests = options.prefer_tests || query.to_lowercase().contains("test");

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
                    if options.prefer_docs {
                        *score *= 0.8;
                        signals.push("documentation_task".into());
                    } else {
                        *score *= w.changelog_penalty;
                        signals.push("changelog_demoted".into());
                    }
                }
                DocClass::Doc => {
                    if options.prefer_docs {
                        *score += w.path_match * 0.5;
                        signals.push("documentation_task".into());
                    } else {
                        *score *= w.doc_file_penalty;
                        signals.push("doc_demoted".into());
                    }
                }
                DocClass::Code => {}
            }
        }

        let mut ranked: Vec<(usize, f32, Vec<String>)> =
            fused.into_iter().map(|(i, (s, sig))| (i, s, sig)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(options.top_k);

        // lazy Tier-1: ensure symbols exist for top hits, then attach best symbols
        let mut packets = Vec::new();
        for (rank, (i, score, signals)) in ranked.iter().enumerate() {
            let d = &self.docs[*i];
            engram_repo_map::ensure_tier1(store, &self.repo_root, &d.path)?;
            let symbols = best_symbols_for(store, &d.path, &q_tokens);
            let (title, symbol, snippet) = match symbols.first() {
                Some(s) => (
                    format!("{} in {}", s.name, d.path),
                    Some(s.name.clone()),
                    Some(s.signature.clone()),
                ),
                None => (d.path.clone(), None, Some(d.preview.clone())),
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

        // symbol-name exact hits that BM25/vector may have missed
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
        packets.truncate(options.top_k + 3);
        Ok(packets)
    }

    /// Expand explicit file anchors using only source and history evidence.
    pub fn expand_connections(
        &self,
        store: &Store,
        anchors: &[String],
        max_hops: usize,
    ) -> Result<ConnectionMap> {
        let anchors: Vec<String> = anchors
            .iter()
            .filter(|path| self.by_path.contains_key(path.as_str()))
            .cloned()
            .collect();
        let anchor_set: HashSet<&str> = anchors.iter().map(String::as_str).collect();
        let mut historical: HashMap<String, (f32, String)> = HashMap::new();

        for anchor in &anchors {
            for edge in store.cochange_for(anchor, 10)? {
                if self.by_path.contains_key(&edge.path_b)
                    && !anchor_set.contains(edge.path_b.as_str())
                {
                    let entry = historical
                        .entry(edge.path_b.clone())
                        .or_insert((0.0, String::new()));
                    if edge.strength > entry.0 {
                        *entry = (
                            edge.strength,
                            format!(
                                "historically changed with {anchor} in {}% of commits touching that anchor",
                                (edge.strength * 100.0) as u32
                            ),
                        );
                    }
                }
            }
        }

        let mut historical_connections: Vec<ScoredPath> = historical
            .into_iter()
            .map(|(path, (confidence, reason))| ScoredPath {
                path,
                confidence,
                reason,
            })
            .collect();
        historical_connections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        historical_connections.truncate(10);

        let mut import_dependents = Vec::new();
        for hit in self
            .graph
            .dependents(&anchors, &[EdgeKind::Imports], max_hops.clamp(1, 4))
        {
            if !engram_repo_map::inventory::is_test_path(&hit.path) {
                import_dependents.push(ScoredPath {
                    path: hit.path,
                    confidence: if hit.hops == 1 { 1.0 } else { 0.7 },
                    reason: format!(
                        "static import dependent ({} hop{})",
                        hit.hops,
                        if hit.hops == 1 { "" } else { "s" }
                    ),
                });
            }
        }
        import_dependents.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        import_dependents.truncate(10);

        let mut related_tests: Vec<ScoredPath> = self
            .graph
            .dependents(&anchors, &[EdgeKind::Imports], max_hops.clamp(1, 4))
            .into_iter()
            .filter(|hit| engram_repo_map::inventory::is_test_path(&hit.path))
            .map(|hit| ScoredPath {
                path: hit.path,
                confidence: if hit.hops == 1 { 1.0 } else { 0.7 },
                reason: format!(
                    "static test dependent ({} hop{})",
                    hit.hops,
                    if hit.hops == 1 { "" } else { "s" }
                ),
            })
            .collect();
        related_tests.extend(
            historical_connections
                .iter()
                .filter(|connection| engram_repo_map::inventory::is_test_path(&connection.path))
                .cloned(),
        );
        related_tests.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let mut seen_tests = HashSet::new();
        related_tests.retain(|test| seen_tests.insert(test.path.clone()));
        related_tests.truncate(10);

        Ok(ConnectionMap {
            anchors,
            import_dependents,
            historical_connections,
            related_tests,
        })
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
