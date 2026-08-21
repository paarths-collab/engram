//! Tier-0: cheap structural inventory of the repository.

use engram_domain::{FileRecord, Language};
use std::path::Path;
use walkdir::WalkDir;

const IGNORED_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".cache",
    "vendor",
    ".idea",
    ".vscode",
];

const MAX_FILE_BYTES: u64 = 1_500_000; // avoid indexing generated blobs / bundles
const OVERSIZED_REASON: &str = "file_exceeds_1_500_000_bytes";

/// Stable FNV-1a digest used to detect edits across process restarts.
///
/// This matches the zero-dependency content hashing already used by the
/// retrieval cache. It is an invalidation fingerprint, not a security digest.
pub(crate) fn content_hash(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn record(path: &Path, relative_path: String, size_bytes: u64) -> Option<FileRecord> {
    let language = Language::from_path(&relative_path);
    if language == Language::Other
        && !(relative_path.ends_with(".md")
            || relative_path.ends_with(".toml")
            || relative_path.ends_with(".json"))
    {
        return None;
    }

    if size_bytes > MAX_FILE_BYTES {
        return Some(FileRecord {
            is_test: is_test_path(&relative_path),
            path: relative_path,
            language,
            size_bytes,
            // An oversized file is not read merely to fingerprint it. The
            // sentinel still differs from any eligible content digest, which
            // invalidates derived data if a formerly eligible file grows.
            content_hash: format!("oversized:{size_bytes}"),
            indexing_ineligibility: Some(OVERSIZED_REASON.to_owned()),
        });
    }

    let contents = match std::fs::read(path) {
        Ok(contents) => contents,
        Err(_) => {
            return Some(FileRecord {
                is_test: is_test_path(&relative_path),
                path: relative_path,
                language,
                size_bytes,
                content_hash: format!("unreadable:{size_bytes}"),
                indexing_ineligibility: Some("file_could_not_be_read".to_owned()),
            });
        }
    };
    Some(FileRecord {
        is_test: is_test_path(&relative_path),
        path: relative_path,
        language,
        size_bytes,
        content_hash: content_hash(&contents),
        indexing_ineligibility: None,
    })
}

/// Inventory one known path using the same eligibility and hashing rules as a
/// full scan. Used by the file watcher so warm and cold reindexing agree.
pub(crate) fn scan_path(repo_root: &Path, relative_path: &str) -> Option<FileRecord> {
    let full_path = repo_root.join(relative_path);
    let metadata = std::fs::metadata(&full_path).ok()?;
    metadata
        .is_file()
        .then(|| record(&full_path, relative_path.replace('\\', "/"), metadata.len()))?
}

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
        let rel = entry
            .path()
            .strip_prefix(repo_root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .replace('\\', "/");
        if let Some(file) = record(entry.path(), rel, meta.len()) {
            out.push(file);
        }
    }
    out
}
