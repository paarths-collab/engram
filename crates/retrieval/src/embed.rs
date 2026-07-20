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

/// Stable content hash (FNV-1a, hex) used to invalidate cached embeddings when
/// the embedded text changes. Deterministic across runs and machines.
pub fn content_hash(text: &str) -> String {
    format!("{:016x}", fnv1a(text.as_bytes()))
}

/// Encode an embedding vector as little-endian `f32` bytes for BLOB storage.
pub fn vector_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode little-endian `f32` bytes back into a vector. Returns `None` if the
/// byte length is not a multiple of 4 or the dimension is wrong.
pub fn bytes_to_vector(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) || bytes.len() / 4 != DIM {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_roundtrips_through_bytes() {
        let v = HashedNgramEmbedder.embed("fn retry_with_backoff(policy: RetryPolicy)");
        let bytes = vector_to_bytes(&v);
        assert_eq!(bytes.len(), DIM * 4);
        let back = bytes_to_vector(&bytes).expect("decodes");
        assert_eq!(v, back);
    }

    #[test]
    fn bytes_to_vector_rejects_bad_length() {
        assert!(bytes_to_vector(&[0u8; 3]).is_none());
        assert!(bytes_to_vector(&[0u8; (DIM - 1) * 4]).is_none());
    }

    #[test]
    fn content_hash_is_stable_and_sensitive() {
        assert_eq!(content_hash("hello world"), content_hash("hello world"));
        assert_ne!(content_hash("hello world"), content_hash("hello worlds"));
    }
}
