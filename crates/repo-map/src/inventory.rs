//! Tier-0: cheap structural inventory of the repository.

use engram_domain::{FileRecord, Language};
use std::path::Path;
use walkdir::WalkDir;

const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".engram",
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

/// Whether a directory with this name is never indexed, wherever it appears.
/// The single ignore rule shared by the full scan, [`is_indexable`], and the
/// file watcher, so they can never disagree about what is covered.
///
/// Changing the list changes what is indexed: bump
/// `store::PARSER_CONFIG_VERSION` when you do.
pub fn is_ignored_component(name: &str) -> bool {
    IGNORED_DIRS.contains(&name)
}

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

/// A file's last-modification time as nanoseconds since the Unix epoch, or
/// `None` where the platform cannot report one (or it predates the epoch).
pub fn mtime_ns(meta: &std::fs::Metadata) -> Option<i64> {
    let since_epoch = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    i64::try_from(since_epoch.as_nanos()).ok()
}

/// FNV-1a, 64-bit. A change fingerprint, not a security boundary: stable
/// across platforms and releases, with no dependency.
pub struct Fnv64(u64);

impl Default for Fnv64 {
    fn default() -> Self {
        Fnv64(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv64 {
    pub fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    /// Lowercase 16-digit hex of the digest.
    pub fn finish_hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// Content fingerprint of a byte string (see [`Fnv64`]).
pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut h = Fnv64::default();
    h.write(bytes);
    h.finish_hex()
}

/// Content fingerprint of a file on disk, or `None` if it cannot be read.
pub fn hash_file(path: &Path) -> Option<String> {
    std::fs::read(path).ok().map(|bytes| hash_bytes(&bytes))
}

/// One inventoried file with the fingerprint that tells the store whether it
/// changed since it was last extracted.
///
/// A file counts as changed when its size, its mtime, or its content hash
/// differs from what is stored, so any one of the three is enough. The content
/// hash is what catches a same-size edit whose mtime was put back (or that
/// landed in the same timestamp tick); every indexed file is re-hashed on each
/// scan, one read per file, which indexing does anyway for BM25. The converse
/// also holds: a change of mtime alone (`touch`, or a checkout that rewrites
/// identical content) invalidates the file and re-extracts it. That is safe,
/// and only costs time.
#[derive(Debug, Clone)]
pub struct ScannedFile {
    pub record: FileRecord,
    pub mtime_ns: Option<i64>,
    pub content_hash: Option<String>,
}

/// Whether a repo-relative file belongs in the inventory. The single source of
/// truth for both the full scan and the watcher's single-file reindex, so a
/// live-updated index and a freshly built one cover exactly the same files
/// (and therefore produce the same snapshot id).
///
/// Changing these rules changes what is indexed: bump
/// `store::PARSER_CONFIG_VERSION` when you do.
pub fn is_indexable(rel: &str, size_bytes: u64) -> bool {
    if size_bytes > MAX_FILE_BYTES {
        return false;
    }
    let mut parts: Vec<&str> = rel.split('/').collect();
    parts.pop(); // the file itself
    if parts.iter().any(|dir| is_ignored_component(dir)) {
        return false;
    }
    // keep only source-adjacent docs for now
    Language::from_path(rel) != Language::Other
        || rel.ends_with(".md")
        || rel.ends_with(".toml")
        || rel.ends_with(".json")
}

/// Walk the repository and produce the Tier-0 file inventory.
pub fn scan(repo_root: &Path) -> Vec<ScannedFile> {
    let mut out = Vec::new();
    for entry in WalkDir::new(repo_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(e.file_type().is_dir() && is_ignored_component(&name))
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
        if !is_indexable(&rel, meta.len()) {
            continue;
        }
        out.push(ScannedFile {
            record: FileRecord {
                is_test: is_test_path(&rel),
                language: Language::from_path(&rel),
                size_bytes: meta.len(),
                path: rel,
            },
            mtime_ns: mtime_ns(&meta),
            content_hash: hash_file(entry.path()),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_matches_published_vectors() {
        // Reference values for 64-bit FNV-1a.
        assert_eq!(hash_bytes(b""), "cbf29ce484222325");
        assert_eq!(hash_bytes(b"a"), "af63dc4c8601ec8c");
        assert_eq!(hash_bytes(b"foobar"), "85944171f73967e8");
    }

    #[test]
    fn indexability_rules() {
        assert!(is_indexable("src/main.rs", 10));
        assert!(is_indexable("README.md", 10));
        assert!(!is_indexable("notes.txt", 10));
        assert!(!is_indexable("src/main.rs", MAX_FILE_BYTES + 1));
        assert!(!is_indexable("web/node_modules/x/index.js", 10));
        assert!(!is_indexable("a/target/debug/x.rs", 10));
        assert!(!is_indexable(".engram/state.json", 10));
        // A file merely named like an ignored directory is still a file.
        assert!(is_indexable("src/build.rs", 10));
        // The scan, `is_indexable`, and the watcher share one ignore rule.
        for dir in ["vendor", ".next", ".cache", "venv", ".idea", ".vscode"] {
            assert!(is_ignored_component(dir), "{dir}");
            assert!(!is_indexable(&format!("{dir}/x.rs"), 10), "{dir}");
        }
        assert!(!is_ignored_component("src"));
    }
}
