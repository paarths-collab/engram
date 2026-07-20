//! engram-verify: build a verification checklist for a set of changed files.
//!
//! Per-domain checklists live as YAML data in `config/verification/*.yaml`
//! (files decide, not code). Each profile declares glob detection rules and a
//! list of checks; a profile applies when any changed file matches its globs.
//! Test/verify commands are detected from the repo's manifests.

use globset::{Glob, GlobSetBuilder};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// One domain verification profile, loaded from a YAML file.
#[derive(Debug, Clone, Deserialize)]
pub struct Profile {
    /// Domain name (backend, frontend, database, infra, ...).
    pub name: String,
    /// Globs; the profile applies if any changed file matches one.
    #[serde(default)]
    pub detect: Vec<String>,
    /// Checklist items contributed when the profile applies.
    #[serde(default)]
    pub checks: Vec<String>,
}

/// Resolve the verification profile directory: `<repo>/config/verification`,
/// else the process-relative `config/verification`.
pub fn profiles_dir(repo_root: &Path) -> PathBuf {
    let repo_dir = repo_root.join("config").join("verification");
    if repo_dir.is_dir() {
        repo_dir
    } else {
        PathBuf::from("config").join("verification")
    }
}

/// Load all `*.yaml` / `*.yml` profiles from the verification directory.
/// Unparseable files are skipped with a warning rather than failing the call.
pub fn load_profiles(repo_root: &Path) -> Vec<Profile> {
    let dir = profiles_dir(repo_root);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut profiles = Vec::new();
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("yaml") | Some("yml")
            )
        })
        .collect();
    paths.sort(); // deterministic profile order
    for path in paths {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        match serde_yaml::from_str::<Profile>(&text) {
            Ok(p) => profiles.push(p),
            Err(e) => eprintln!(
                "[engram] {}: verification profile parse error ({e})",
                path.display()
            ),
        }
    }
    profiles
}

/// Result of matching changed files against profiles.
pub struct MatchResult {
    pub matched_profiles: Vec<String>,
    pub checklist: Vec<String>,
}

/// Match changed files against profiles: a profile applies if any changed file
/// matches one of its globs. Returns matched profile names and the merged,
/// de-duplicated checklist (profile order preserved).
pub fn match_profiles(profiles: &[Profile], changed_files: &[String]) -> MatchResult {
    let mut matched_profiles = Vec::new();
    let mut checklist = Vec::new();
    for profile in profiles {
        let mut builder = GlobSetBuilder::new();
        for pat in &profile.detect {
            if let Ok(glob) = Glob::new(pat) {
                builder.add(glob);
            }
        }
        let Ok(set) = builder.build() else { continue };
        if changed_files.iter().any(|f| set.is_match(f)) {
            matched_profiles.push(profile.name.clone());
            for check in &profile.checks {
                if !checklist.contains(check) {
                    checklist.push(check.clone());
                }
            }
        }
    }
    MatchResult {
        matched_profiles,
        checklist,
    }
}

/// Detect test/verify commands from manifests at the repo root.
pub fn detect_test_commands(repo_root: &Path) -> Vec<String> {
    let exists = |name: &str| repo_root.join(name).exists();
    let mut cmds = Vec::new();
    if exists("Cargo.toml") {
        cmds.push("cargo test".to_string());
    }
    if exists("go.mod") {
        cmds.push("go test ./...".to_string());
    }
    if exists("pyproject.toml")
        || exists("setup.py")
        || exists("setup.cfg")
        || exists("pytest.ini")
        || exists("tox.ini")
        || exists("requirements.txt")
    {
        cmds.push("pytest".to_string());
    }
    if exists("package.json") {
        cmds.push("npm test".to_string());
    }
    cmds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(name: &str, detect: &[&str], checks: &[&str]) -> Profile {
        Profile {
            name: name.to_string(),
            detect: detect.iter().map(|s| s.to_string()).collect(),
            checks: checks.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn matches_by_glob_and_merges_dedup_checklist() {
        let profiles = vec![
            profile("backend", &["**/*.rs"], &["unit tests", "error handling"]),
            profile("frontend", &["**/*.tsx"], &["a11y", "unit tests"]),
            profile("database", &["**/migrations/**"], &["rollback path"]),
        ];
        let changed = vec![
            "src/billing/cancel.rs".to_string(),
            "web/app/Page.tsx".to_string(),
        ];
        let r = match_profiles(&profiles, &changed);
        assert_eq!(r.matched_profiles, vec!["backend", "frontend"]);
        // "unit tests" appears in both profiles but only once in the checklist
        assert_eq!(r.checklist, vec!["unit tests", "error handling", "a11y"]);
    }

    #[test]
    fn no_match_yields_empty() {
        let profiles = vec![profile("backend", &["**/*.rs"], &["unit tests"])];
        let r = match_profiles(&profiles, &["README.md".to_string()]);
        assert!(r.matched_profiles.is_empty());
        assert!(r.checklist.is_empty());
    }
}
