//! Vector side of hybrid retrieval.
//!
//! `Embedder` is a trait so the implementation is swappable:
//!   - `HashedNgramEmbedder` (default): deterministic, zero-dependency, local.
//!     Hashed character-trigram + token TF vectors. Surprisingly effective for
//!     code because identifiers share subword structure (RetryPolicy ~ retry_policy).
//!   - Later: fastembed-rs (local ONNX bge/minilm) or any OpenAI-compatible API,
//!     behind the same trait. Nothing else in the pipeline changes.

pub const DIM: usize = 512;

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Vec<f32>;
}

pub struct HashedNgramEmbedder;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn tokenize(text: &str) -> Vec<String> {
    // split camelCase, snake_case, paths; lowercase everything
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            if ch.is_uppercase() && prev_lower && !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
            prev_lower = ch.is_lowercase() || ch.is_numeric();
            cur.extend(ch.to_lowercase());
        } else {
            prev_lower = false;
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

impl Embedder for HashedNgramEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; DIM];
        let tokens = tokenize(text);
        for tok in &tokens {
            // whole-token feature (weighted higher)
            let h = fnv1a(tok.as_bytes()) as usize % DIM;
            v[h] += 2.0;
            // char trigrams for fuzzy/subword match
            let chars: Vec<char> = tok.chars().collect();
            if chars.len() >= 3 {
                for w in chars.windows(3) {
                    let s: String = w.iter().collect();
                    let h = fnv1a(s.as_bytes()) as usize % DIM;
                    v[h] += 1.0;
                }
            }
        }
        // l2 normalize
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
