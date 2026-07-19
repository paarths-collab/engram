//! Tier-0: cheap structural inventory of the repository.

use engram_domain::{FileRecord, Language};
use std::path::Path;
use walkdir::WalkDir;

const IGNORED_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "dist", "build", ".venv", "venv",
    "__pycache__", ".next", ".cache", "vendor", ".idea", ".vscode",
];

const MAX_FILE_BYTES: u64 = 1_500_000; // skip generated blobs / bundles

pub fn is_test_path(path: &str) -> bool {
    let p = path.to_lowercase();
    p.contains("/tests/")
        || p.contains("/test/")
        || p.contains("/__tests__/")
        || p.ends_with("_test.rs")
        || p.ends_with("_test.go")
        || p.ends_with("_test.py")
        || p.starts_with("tests/")
        || p.starts_with("test/")
        || {
            let file = p.rsplit('/').next().unwrap_or(&p);
            file.starts_with("test_")
                || file.ends_with(".test.ts")
                || file.ends_with(".test.tsx")
                || file.ends_with(".test.js")
                || file.ends_with(".spec.ts")
                || file.ends_with(".spec.js")
        }
}

/// Walk the repository and produce the Tier-0 file inventory.
pub fn scan(repo_root: &Path) -> Vec<FileRecord> {
    let mut out = Vec::new();
    for entry in WalkDir::new(repo_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(e.file_type().is_dir() && IGNORED_DIRS.contains(&name.as_ref()))
        })
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.len() > MAX_FILE_BYTES {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(repo_root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .replace('\\', "/");
        let language = Language::from_path(&rel);
        if language == Language::Other {
            // keep only source-adjacent docs for now
            if !(rel.ends_with(".md") || rel.ends_with(".toml") || rel.ends_with(".json")) {
                continue;
            }
        }
        out.push(FileRecord {
            is_test: is_test_path(&rel),
            path: rel,
            language,
            size_bytes: meta.len(),
        });
    }
    out
}
