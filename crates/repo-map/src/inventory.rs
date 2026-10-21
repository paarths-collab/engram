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
/// Editing the list needs no manual version bump: it is part of
/// [`inventory_rules_fingerprint`], which invalidates stale indexes.
pub fn is_ignored_component(name: &str) -> bool {
    IGNORED_DIRS.contains(&name)
}

const MAX_FILE_BYTES: u64 = 1_500_000; // skip generated blobs / bundles

/// Non-source files kept in the inventory (docs and config), as file-name
/// suffixes. Source files are decided by `engram_domain::LANGUAGE_EXTENSIONS`.
const EXTRA_SUFFIXES: &[&str] = &[".md", ".toml", ".json"];

/// Fingerprint of everything that decides *which files* the inventory
/// includes, over explicit rule lists so a test can change one.
///
/// Stable FNV-1a over the sorted, de-duplicated ignored directory names, the
/// sorted indexable rules (each a self-describing string such as
/// `lang:rs=rust` or `suffix:.md`), the size cap, and the parser/config
/// version, so listing order never matters and any real change to a list does.
pub fn rules_fingerprint(
    ignored: &[&str],
    indexable_rules: &[String],
    max_file_bytes: u64,
    parser_config_version: u32,
) -> String {
    fn sorted<T: AsRef<str>>(items: &[T]) -> Vec<&str> {
        let mut v: Vec<&str> = items.iter().map(AsRef::as_ref).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
    let mut h = Fnv64::default();
    // 0xff cannot occur in UTF-8, so fields cannot run into each other.
    let mut field = |s: &str| {
        h.write(s.as_bytes());
        h.write(&[0xff]);
    };
    field("engram-inventory-rules-v1");
    field("ignored");
    for name in sorted(ignored) {
        field(name);
    }
    field("indexable");
    for rule in sorted(indexable_rules) {
        field(rule);
    }
    field("max_file_bytes");
    field(&max_file_bytes.to_string());
    field("parser_config_version");
    field(&parser_config_version.to_string());
    h.finish_hex()
}

/// The version recorded in the database and folded into the snapshot id:
/// [`rules_fingerprint`] over the production ignore list, extension rules,
/// size cap, and `store::PARSER_CONFIG_VERSION`. Editing any of those changes
/// it, so a database written under other rules is re-extracted on open without
/// anyone remembering to bump a number. `PARSER_CONFIG_VERSION` remains for
/// changes this cannot see (parser queries, extraction logic).
pub fn inventory_rules_fingerprint() -> String {
    let mut rules: Vec<String> = engram_domain::LANGUAGE_EXTENSIONS
        .iter()
        .map(|(ext, lang)| format!("lang:{ext}={}", lang.as_str()))
        .collect();
    rules.extend(EXTRA_SUFFIXES.iter().map(|s| format!("suffix:{s}")));
    rules_fingerprint(
        IGNORED_DIRS,
        &rules,
        MAX_FILE_BYTES,
        crate::store::PARSER_CONFIG_VERSION,
    )
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

// The FNV-1a implementation lives in the domain crate so every crate can use it
// (edge ids hash with it too); it stays reachable here for the store.
pub use engram_domain::Fnv64;

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
/// Changing these rules changes what is indexed; the change is picked up
/// through [`inventory_rules_fingerprint`].
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
        || EXTRA_SUFFIXES.iter().any(|suffix| rel.ends_with(suffix))
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

    fn rules(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn rules_fingerprint_is_deterministic_and_order_independent() {
        let a = rules_fingerprint(
            &["target", ".git"],
            &rules(&["lang:rs=rust", "suffix:.md"]),
            10,
            1,
        );
        let b = rules_fingerprint(
            &["target", ".git"],
            &rules(&["lang:rs=rust", "suffix:.md"]),
            10,
            1,
        );
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
        let reordered = rules_fingerprint(
            &[".git", "target", ".git"],
            &rules(&["suffix:.md", "lang:rs=rust"]),
            10,
            1,
        );
        assert_eq!(a, reordered, "order and duplicates must not matter");
        assert_eq!(inventory_rules_fingerprint(), inventory_rules_fingerprint());
    }

    #[test]
    fn changing_any_rule_changes_the_fingerprint() {
        let base_ignored = ["target", ".git"];
        let base_rules = rules(&["lang:rs=rust", "suffix:.md"]);
        let base = rules_fingerprint(&base_ignored, &base_rules, 10, 1);
        let ignore_added = rules_fingerprint(&["target", ".git", "out"], &base_rules, 10, 1);
        let ignore_removed = rules_fingerprint(&["target"], &base_rules, 10, 1);
        let ext_added = rules_fingerprint(
            &base_ignored,
            &rules(&["lang:rs=rust", "suffix:.md", "suffix:.yml"]),
            10,
            1,
        );
        let ext_renamed = rules_fingerprint(
            &base_ignored,
            &rules(&["lang:rs=python", "suffix:.md"]),
            10,
            1,
        );
        let size = rules_fingerprint(&base_ignored, &base_rules, 11, 1);
        let version = rules_fingerprint(&base_ignored, &base_rules, 10, 2);
        let all = [
            &base,
            &ignore_added,
            &ignore_removed,
            &ext_added,
            &ext_renamed,
            &size,
            &version,
        ];
        for (i, x) in all.iter().enumerate() {
            for y in &all[i + 1..] {
                assert_ne!(x, y);
            }
        }
        // Names cannot run together: ["ab"] is not ["a", "b"].
        assert_ne!(
            rules_fingerprint(&["ab"], &[], 0, 0),
            rules_fingerprint(&["a", "b"], &[], 0, 0)
        );
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
