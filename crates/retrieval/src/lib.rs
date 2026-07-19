//! engram-retrieval: hybrid BM25 (Tantivy) + vector (cosine) retrieval with
//! score fusion, lazy Tier-1 extraction on miss, and co-change impact expansion.

pub mod embed;

use anyhow::Result;
use embed::{cosine, Embedder, HashedNgramEmbedder};
use engram_domain::{EvidenceKind, EvidencePacket, ImpactPrediction, ScoredPath, SymbolRecord};
use engram_repo_map::store::Store;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, STORED, TEXT};
use tantivy::{doc, Index, IndexReader};

/// Fusion weights — the section-7 formula, trimmed to the signals we have today.
/// Externalize to config/scoring.toml once numbers stabilize.
pub struct Weights {
    pub bm25: f32,
    pub vector: f32,
    pub symbol_exact: f32,
    pub path_match: f32,
    pub test_bonus_for_tests_query: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            bm25: 1.0,
            vector: 1.2,
            symbol_exact: 1.5,
            path_match: 0.6,
            test_bonus_for_tests_query: 0.5,
        }
    }
}

struct DocMeta {
    path: String,
    is_test: bool,
    preview: String,
    vector: Vec<f32>,
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
    pub weights: Weights,
}

const PREVIEW_BYTES: usize = 400;
const INDEX_BODY_BYTES: usize = 60_000;

impl Engine {
    /// Build the in-memory hybrid index from the current repo + store.
    pub fn build(repo_root: &Path, store: &Store) -> Result<Engine> {
        let mut schema_builder = Schema::builder();
        let f_path = schema_builder.add_text_field("path", TEXT | STORED);
        let f_body = schema_builder.add_text_field("body", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer(64_000_000)?;

        let embedder = HashedNgramEmbedder;
        let mut docs = Vec::new();
        let mut by_path = HashMap::new();

        for f in store.all_files()? {
            let full = repo_root.join(&f.path);
            let Ok(src) = std::fs::read_to_string(&full) else {
                continue;
            };
            let body: String = src.chars().take(INDEX_BODY_BYTES).collect();
            // Embed path + a head slice; heads carry imports/signatures = high signal.
            let embed_text = format!("{} {}", f.path, body.chars().take(4000).collect::<String>());
            let vector = embedder.embed(&embed_text);
            writer.add_document(doc!(f_path => f.path.clone(), f_body => body.clone()))?;
            by_path.insert(f.path.clone(), docs.len());
            docs.push(DocMeta {
                preview: body.chars().take(PREVIEW_BYTES).collect(),
                path: f.path,
                is_test: f.is_test,
                vector,
            });
        }
        writer.commit()?;
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
            weights: Weights::default(),
        })
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
        &self,
        store: &mut Store,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<EvidencePacket>> {
        let bm25 = self.bm25_candidates(query, 50);
        let vecs = self.vector_candidates(query, 50);

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
            e.0 += self.weights.bm25 * s;
            e.1.push("bm25".into());
        }
        for (i, s) in &vec_n {
            let e = fused.entry(*i).or_insert((0.0, Vec::new()));
            e.0 += self.weights.vector * s;
            e.1.push("vector".into());
        }

        // path token match + test handling
        let q_tokens: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 2)
            .map(|t| t.to_lowercase())
            .collect();
        for (i, (score, signals)) in fused.iter_mut() {
            let d = &self.docs[*i];
            let pl = d.path.to_lowercase();
            if q_tokens.iter().any(|t| pl.contains(t)) {
                *score += self.weights.path_match;
                signals.push("path".into());
            }
            if d.is_test {
                if query_wants_tests {
                    *score += self.weights.test_bonus_for_tests_query;
                } else {
                    *score *= 0.8; // slight demotion, tests still surfaced via impact
                }
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
                    score: self.weights.symbol_exact,
                    signals: vec!["symbol".into()],
                });
            }
        }
        packets.truncate(top_k + 3);
        Ok(packets)
    }

    /// predict_impact: hybrid hits = direct; co-change graph = historical expansion.
    pub fn predict_impact(&self, store: &mut Store, query: &str) -> Result<ImpactPrediction> {
        let direct = self.search(store, query, 8)?;
        let mut likely_files = Vec::new();
        let mut likely_tests = Vec::new();
        let mut expansions: HashMap<String, (f32, String)> = HashMap::new();

        let max_score = direct.iter().map(|p| p.score).fold(1e-6, f32::max);
        for p in &direct {
            if p.kind == EvidenceKind::Test {
                likely_tests.push(p.path.clone());
            } else {
                likely_files.push(ScoredPath {
                    path: p.path.clone(),
                    confidence: (p.score / max_score).min(1.0),
                    reason: format!("matched task via {}", p.signals.join("+")),
                });
            }
            for edge in store.cochange_for(&p.path, 5)? {
                if self.by_path.contains_key(&edge.path_b)
                    && !direct.iter().any(|d| d.path == edge.path_b)
                {
                    let e = expansions
                        .entry(edge.path_b.clone())
                        .or_insert((0.0, String::new()));
                    if edge.strength > e.0 {
                        *e = (
                            edge.strength,
                            format!(
                                "changes with {} in {}% of its commits",
                                p.path,
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
        likely_tests.sort();
        likely_tests.dedup();

        Ok(ImpactPrediction {
            likely_files,
            likely_tests,
            cochange_expansions,
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
