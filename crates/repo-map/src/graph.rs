//! In-memory code graph over the SQL edge tables (see docs/adr/0001).
//!
//! Relationships are *stored* as edge rows (`file_imports`, `cochange`); this
//! module *materializes* them into a `petgraph` so we can do multi-hop traversal
//! in memory — which is where dense/deep relationship reasoning belongs. No graph
//! database required: SQL stores the edges, petgraph walks them.

use crate::imports::module_needle;
use crate::store::Store;
use anyhow::Result;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction::Incoming;
use std::collections::{HashMap, HashSet, VecDeque};

/// Kind of relationship an edge represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// `src` imports `dst`.
    Imports,
    /// `src` and `dst` historically change together.
    CoChange,
}

/// One file reached during graph traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphHit {
    pub path: String,
    pub hops: usize,
    pub kind: EdgeKind,
}

/// Directed code graph: an edge `src -> dst` means `src` depends on `dst`
/// (imports it) or co-changes with it. Traversal against the edge direction
/// (`dependents`) answers "what would be affected if `dst` changes".
pub struct CodeGraph {
    g: DiGraph<(), EdgeKind>,
    idx: HashMap<String, NodeIndex>,
    rev: Vec<String>,
}

impl CodeGraph {
    /// Build directly from an edge list. `(src, dst, kind)`.
    pub fn from_edges<I: IntoIterator<Item = (String, String, EdgeKind)>>(edges: I) -> Self {
        let mut g = DiGraph::new();
        let mut idx: HashMap<String, NodeIndex> = HashMap::new();
        let mut rev: Vec<String> = Vec::new();
        let node = |g: &mut DiGraph<(), EdgeKind>,
                    idx: &mut HashMap<String, NodeIndex>,
                    rev: &mut Vec<String>,
                    path: &str|
         -> NodeIndex {
            if let Some(&n) = idx.get(path) {
                return n;
            }
            let n = g.add_node(());
            idx.insert(path.to_string(), n);
            rev.push(path.to_string());
            n
        };
        for (src, dst, kind) in edges {
            if src == dst {
                continue;
            }
            let s = node(&mut g, &mut idx, &mut rev, &src);
            let d = node(&mut g, &mut idx, &mut rev, &dst);
            g.add_edge(s, d, kind);
        }
        CodeGraph { g, idx, rev }
    }

    /// Build the graph from the store: resolved import edges + co-change edges.
    pub fn build(store: &Store) -> Result<Self> {
        let files = store.all_files()?;
        let file_set: HashSet<String> = files.iter().map(|f| f.path.clone()).collect();

        // needle -> files that expose it (an importable module key)
        let mut needle_map: HashMap<String, Vec<String>> = HashMap::new();
        for f in &files {
            if let Some(n) = module_needle(&f.path) {
                needle_map.entry(n).or_default().push(f.path.clone());
            }
        }

        let mut edges: Vec<(String, String, EdgeKind)> = Vec::new();

        // Resolve import targets to concrete files: an imported file's needle is
        // a single segment or an adjacent pair of the normalized target.
        for (importer, target) in store.all_imports()? {
            for key in candidate_keys(&target) {
                if let Some(imported) = needle_map.get(&key) {
                    for f in imported {
                        if *f != importer {
                            edges.push((importer.clone(), f.clone(), EdgeKind::Imports));
                        }
                    }
                }
            }
        }

        // Co-change edges (only between known files).
        for (a, b, _strength) in store.all_cochange_edges()? {
            if file_set.contains(&a) && file_set.contains(&b) {
                edges.push((a, b, EdgeKind::CoChange));
            }
        }

        Ok(Self::from_edges(edges))
    }

    pub fn node_count(&self) -> usize {
        self.g.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.g.edge_count()
    }

    /// Files that (transitively) depend on any of `seeds`, up to `max_hops`,
    /// following edges of the given `kinds` in reverse (a `dependent -> seed`
    /// edge means the dependent is affected when the seed changes). Seeds are
    /// excluded from the result; each file is reported at its shortest distance.
    pub fn dependents(
        &self,
        seeds: &[String],
        kinds: &[EdgeKind],
        max_hops: usize,
    ) -> Vec<GraphHit> {
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        for s in seeds {
            if let Some(&n) = self.idx.get(s) {
                if visited.insert(n) {
                    queue.push_back((n, 0));
                }
            }
        }
        let mut hits: Vec<GraphHit> = Vec::new();
        while let Some((node, hops)) = queue.pop_front() {
            if hops >= max_hops {
                continue;
            }
            for edge in self.g.edges_directed(node, Incoming) {
                let kind = *edge.weight();
                if !kinds.contains(&kind) {
                    continue;
                }
                use petgraph::visit::EdgeRef;
                let src = edge.source();
                if visited.insert(src) {
                    hits.push(GraphHit {
                        path: self.rev[src.index()].clone(),
                        hops: hops + 1,
                        kind,
                    });
                    queue.push_back((src, hops + 1));
                }
            }
        }
        hits
    }
}

/// Candidate module keys contained in a normalized import target `a/b/c`:
/// each single segment and each adjacent pair (matching `module_needle` shapes).
fn candidate_keys(target: &str) -> Vec<String> {
    let segs: Vec<&str> = target.split('/').filter(|s| !s.is_empty()).collect();
    let mut keys = Vec::new();
    for w in segs.windows(2) {
        keys.push(format!("{}/{}", w[0], w[1]));
    }
    for s in &segs {
        keys.push((*s).to_string());
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(a: &str, b: &str, k: EdgeKind) -> (String, String, EdgeKind) {
        (a.to_string(), b.to_string(), k)
    }

    #[test]
    fn multi_hop_dependents() {
        // a imports b, b imports c  =>  changing c affects b (1 hop) and a (2 hops)
        let g = CodeGraph::from_edges([
            e("a.rs", "b.rs", EdgeKind::Imports),
            e("b.rs", "c.rs", EdgeKind::Imports),
        ]);
        let hits = g.dependents(&["c.rs".to_string()], &[EdgeKind::Imports], 2);
        let mut got: Vec<(String, usize)> = hits.iter().map(|h| (h.path.clone(), h.hops)).collect();
        got.sort();
        assert_eq!(got, vec![("a.rs".to_string(), 2), ("b.rs".to_string(), 1)]);
    }

    #[test]
    fn max_hops_bounds_traversal() {
        let g = CodeGraph::from_edges([
            e("a.rs", "b.rs", EdgeKind::Imports),
            e("b.rs", "c.rs", EdgeKind::Imports),
        ]);
        // one hop only reaches b
        let hits = g.dependents(&["c.rs".to_string()], &[EdgeKind::Imports], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "b.rs");
    }

    #[test]
    fn kind_filter_excludes_other_edges() {
        let g = CodeGraph::from_edges([
            e("a.rs", "c.rs", EdgeKind::CoChange),
            e("b.rs", "c.rs", EdgeKind::Imports),
        ]);
        let imports_only = g.dependents(&["c.rs".to_string()], &[EdgeKind::Imports], 3);
        assert_eq!(imports_only.len(), 1);
        assert_eq!(imports_only[0].path, "b.rs");
    }

    #[test]
    fn candidate_keys_are_segments_and_pairs() {
        let keys = candidate_keys("utils/retry/backoff");
        assert!(keys.contains(&"utils/retry".to_string()));
        assert!(keys.contains(&"retry/backoff".to_string()));
        assert!(keys.contains(&"retry".to_string()));
    }
}
