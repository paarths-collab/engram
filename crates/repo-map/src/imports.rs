//! Import-graph helpers: normalize raw import strings into comparable module
//! keys, and derive the "needle" used to find files that import a given file.
//!
//! Resolution is deliberately heuristic and language-agnostic: both an import
//! target (`use a::b::c`, `from a.b import c`, `import x from './a/b'`) and a
//! repo file path are reduced to slash-joined lowercase segments, then matched
//! by substring. Favors precision over recall — a structural hint, not a proof.

/// File-extension suffixes stripped from JS/TS-style import paths.
const STRIPPABLE_EXTS: &[&str] = &[".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json"];

/// Segments that carry no locating information at the head of an import path.
fn is_noise_segment(seg: &str) -> bool {
    matches!(seg, "" | "." | ".." | "crate" | "self" | "super")
}

/// Directory-module filenames whose real module name is the parent directory.
fn is_dir_module(stem: &str) -> bool {
    matches!(stem, "mod" | "index" | "__init__" | "lib" | "main")
}

/// Normalize a raw import string into `a/b/c` lowercase form.
/// Returns an empty string if nothing locating remains.
pub fn normalize_target(raw: &str) -> String {
    let mut s = raw.trim().to_ascii_lowercase();
    for ext in STRIPPABLE_EXTS {
        if let Some(stripped) = s.strip_suffix(ext) {
            s = stripped.to_string();
            break;
        }
    }
    let unified = s.replace("::", "/").replace(['.', '\\'], "/");
    let segs: Vec<&str> = unified
        .split('/')
        .skip_while(|seg| is_noise_segment(seg))
        .filter(|seg| !is_noise_segment(seg))
        .collect();
    segs.join("/")
}

/// Build the lookup key that identifies a repo file as an import target.
/// Prefers `parent/stem` for specificity; falls back to a distinctive stem.
/// Returns `None` for files too generic to match on (short or empty stems).
pub fn module_needle(path: &str) -> Option<String> {
    let p = path.to_ascii_lowercase();
    let no_ext = p.rsplit_once('.').map(|(a, _)| a).unwrap_or(&p);
    let mut segs: Vec<&str> = no_ext.split('/').filter(|s| !s.is_empty()).collect();
    if segs.last().is_some_and(|s| is_dir_module(s)) {
        segs.pop();
    }
    let stem = *segs.last()?;
    if stem.len() < 3 {
        return None;
    }
    let n = segs.len();
    let needle = if n >= 2 {
        format!("{}/{}", segs[n - 2], segs[n - 1])
    } else {
        stem.to_string()
    };
    Some(needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_across_languages() {
        assert_eq!(
            normalize_target("engram_domain::cochange"),
            "engram_domain/cochange"
        );
        assert_eq!(normalize_target("a.b.c"), "a/b/c");
        assert_eq!(normalize_target("./utils/retry.ts"), "utils/retry");
        assert_eq!(normalize_target("../../lib/http"), "lib/http");
        assert_eq!(normalize_target("crate::store::Store"), "store/store");
    }

    #[test]
    fn needle_prefers_parent_stem_and_handles_dir_modules() {
        assert_eq!(
            module_needle("src/utils/retry.py").as_deref(),
            Some("utils/retry")
        );
        assert_eq!(module_needle("retry.rs").as_deref(), Some("retry"));
        // directory-module files resolve to the directory
        assert_eq!(module_needle("http/mod.rs").as_deref(), Some("http"));
        assert_eq!(module_needle("api/index.ts").as_deref(), Some("api"));
        // too generic to match on
        assert_eq!(module_needle("a/io.rs"), None);
    }

    #[test]
    fn needle_matches_normalized_target() {
        let needle = module_needle("src/utils/retry.py").unwrap();
        let target = normalize_target(
            "from utils.retry import backoff"
                .trim_start_matches("from ")
                .split(" import")
                .next()
                .unwrap(),
        );
        assert!(target.contains(&needle));
    }
}
