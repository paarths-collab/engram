//! Retrieval scoring weights, externalized to `config/scoring.toml`.
//!
//! Loaded at startup and hot-reloadable in dev: [`WeightsConfig::reload_if_changed`]
//! re-reads the file when its mtime changes, so weights can be tuned without a
//! recompile or restart. Every key is optional and falls back to [`Weights::default`].

use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Fusion + demotion + recency weights. Deserialized from `config/scoring.toml`;
/// missing keys use the defaults below (`#[serde(default)]`).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Weights {
    // Signal fusion
    pub bm25: f32,
    pub vector: f32,
    pub symbol_exact: f32,
    pub path_match: f32,
    pub test_bonus_for_tests_query: f32,
    // Low-signal document demotions (multipliers < 1.0)
    pub doc_file_penalty: f32,
    pub changelog_penalty: f32,
    // Recency
    pub recency_boost: f32,
    pub recency_halflife_days: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            bm25: 1.0,
            vector: 1.2,
            symbol_exact: 1.5,
            path_match: 0.6,
            test_bonus_for_tests_query: 0.5,
            doc_file_penalty: 0.6,
            changelog_penalty: 0.35,
            recency_boost: 0.4,
            recency_halflife_days: 90.0,
        }
    }
}

impl Weights {
    /// Read weights from a TOML file. On missing file or parse error, log and
    /// fall back to defaults so the server always starts.
    pub fn load(path: &Path) -> Weights {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => return Weights::default(),
        };
        match toml::from_str::<Weights>(&text) {
            Ok(w) => w,
            Err(e) => {
                eprintln!(
                    "[engram] {}: parse error ({e}); using default weights",
                    path.display()
                );
                Weights::default()
            }
        }
    }
}

/// Resolve the scoring config path: `<repo>/config/scoring.toml`, else the
/// process-relative `config/scoring.toml`.
pub fn config_path(repo_root: &Path) -> PathBuf {
    let repo_cfg = repo_root.join("config").join("scoring.toml");
    if repo_cfg.exists() {
        repo_cfg
    } else {
        PathBuf::from("config").join("scoring.toml")
    }
}

/// Owns the loaded [`Weights`] plus enough state to hot-reload them in dev.
pub struct WeightsConfig {
    path: PathBuf,
    mtime: Option<SystemTime>,
    pub weights: Weights,
}

impl WeightsConfig {
    /// Load weights for a repo at startup.
    pub fn load(repo_root: &Path) -> Self {
        let path = config_path(repo_root);
        let mtime = file_mtime(&path);
        let weights = Weights::load(&path);
        WeightsConfig {
            path,
            mtime,
            weights,
        }
    }

    /// Re-read the config if the file changed on disk since last load. Returns
    /// true when weights were reloaded.
    pub fn reload_if_changed(&mut self) -> bool {
        let current = file_mtime(&self.path);
        if current != self.mtime {
            self.mtime = current;
            self.weights = Weights::load(&self.path);
            true
        } else {
            false
        }
    }
}

fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Coarse document class used to demote low-signal files during scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocClass {
    /// CHANGES / CHANGELOG / HISTORY / NEWS — mention every feature, demote hardest.
    Changelog,
    /// Prose docs (.md/.txt/.rst/.adoc).
    Doc,
    /// Everything else (source code).
    Code,
}

/// Classify a repo-relative path for demotion purposes.
pub fn classify_path(path: &str) -> DocClass {
    let file = path.rsplit('/').next().unwrap_or(path);
    let stem = file.split('.').next().unwrap_or(file).to_ascii_uppercase();
    if matches!(stem.as_str(), "CHANGES" | "CHANGELOG" | "HISTORY" | "NEWS") {
        return DocClass::Changelog;
    }
    let lower = file.to_ascii_lowercase();
    if [".md", ".markdown", ".txt", ".rst", ".adoc"]
        .iter()
        .any(|ext| lower.ends_with(ext))
    {
        return DocClass::Doc;
    }
    DocClass::Code
}

/// Additive recency boost for a file given its last-commit unix timestamp.
/// Decays exponentially with age (half every `recency_halflife_days`).
pub fn recency_boost(weights: &Weights, last_commit_ts: Option<i64>, now_unix: i64) -> f32 {
    let Some(ts) = last_commit_ts else {
        return 0.0;
    };
    if weights.recency_boost <= 0.0 || weights.recency_halflife_days <= 0.0 {
        return 0.0;
    }
    let age_days = ((now_unix - ts).max(0) as f32) / 86_400.0;
    let decay = 0.5f32.powf(age_days / weights.recency_halflife_days);
    weights.recency_boost * decay
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_changelog_and_docs() {
        assert_eq!(classify_path("CHANGES.md"), DocClass::Changelog);
        assert_eq!(classify_path("docs/CHANGELOG.rst"), DocClass::Changelog);
        assert_eq!(classify_path("HISTORY"), DocClass::Changelog);
        assert_eq!(classify_path("README.md"), DocClass::Doc);
        assert_eq!(classify_path("notes.txt"), DocClass::Doc);
        assert_eq!(classify_path("src/main.rs"), DocClass::Code);
    }

    #[test]
    fn recency_decays_with_age() {
        let w = Weights::default();
        let now = 1_000_000_000;
        let fresh = recency_boost(&w, Some(now), now);
        let old = recency_boost(&w, Some(now - 90 * 86_400), now); // one half-life
        assert!((fresh - w.recency_boost).abs() < 1e-6);
        assert!((old - w.recency_boost * 0.5).abs() < 1e-3);
        assert_eq!(recency_boost(&w, None, now), 0.0);
    }

    #[test]
    fn defaults_when_file_missing() {
        let w = Weights::load(Path::new("/nonexistent/scoring.toml"));
        assert_eq!(w.bm25, 1.0);
        assert_eq!(w.changelog_penalty, 0.35);
    }
}
