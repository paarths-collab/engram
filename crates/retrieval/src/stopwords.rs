//! Stopword "word DB" — the low-signal words dropped from queries and indexed
//! text before scoring, so noise like "the", "is", "a", "for" can't match or
//! dilute results. Deterministic and in-process (no LLM): the coding agent is
//! the AI that phrases the query; this list keeps that query clean.
//!
//! The list is data: `config/stopwords.txt` (one word per line, `#` comments).
//! If present it is unioned with the built-in defaults, so a repo can add
//! domain-specific noise words without losing the basics.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Built-in English function-word stoplist. Deliberately conservative — only
/// articles, prepositions, pronouns, conjunctions, and auxiliaries, never
/// domain verbs like "add"/"retry" that carry retrieval signal.
const DEFAULT: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "nor", "not", "of", "off",
    "on", "onto", "or", "our", "ours", "out", "over", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this",
    "those", "to", "too", "up", "us", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
];

/// Resolve the stopwords file path: `<repo>/config/stopwords.txt`, else the
/// process-relative `config/stopwords.txt`.
pub fn stopwords_path(repo_root: &Path) -> PathBuf {
    let repo_file = repo_root.join("config").join("stopwords.txt");
    if repo_file.exists() {
        repo_file
    } else {
        PathBuf::from("config").join("stopwords.txt")
    }
}

/// Load the stopword set: built-in defaults unioned with `config/stopwords.txt`
/// (if present). All entries are lowercased.
pub fn load(repo_root: &Path) -> HashSet<String> {
    let mut set: HashSet<String> = DEFAULT.iter().map(|s| s.to_string()).collect();
    if let Ok(text) = std::fs::read_to_string(stopwords_path(repo_root)) {
        for line in text.lines() {
            let word = line
                .split('#')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase();
            if !word.is_empty() {
                set.insert(word);
            }
        }
    }
    set
}

/// The built-in defaults alone (used when no repo is available).
pub fn default_set() -> HashSet<String> {
    DEFAULT.iter().map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_contain_common_function_words() {
        let s = default_set();
        for w in ["the", "is", "a", "for", "of", "to"] {
            assert!(s.contains(w), "missing stopword {w}");
        }
        // domain-signal words must NOT be stopped
        for w in ["retry", "webhook", "cancel", "refund"] {
            assert!(!s.contains(w), "over-stopped {w}");
        }
    }
}
