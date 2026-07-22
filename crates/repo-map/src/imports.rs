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

/// Directory names that describe project layout rather than a module. Left in
/// place they swallow the real name: `crates/retrieval/src/lib.rs` reduces to
/// the stem `src`, which matches nothing an importer would ever write.
fn is_structural_segment(seg: &str) -> bool {
    matches!(seg, "src" | "lib" | "crates" | "packages" | "pkg")
}

/// Lookup keys that identify a repo file as an import target, most specific
/// first. Returns empty for files too generic to match on.
pub fn module_needles(path: &str) -> Vec<String> {
    let p = path.to_ascii_lowercase();
    let no_ext = p.rsplit_once('.').map(|(a, _)| a).unwrap_or(&p);
    let mut segs: Vec<&str> = no_ext.split('/').filter(|s| !s.is_empty()).collect();
    if segs.last().is_some_and(|s| is_dir_module(s)) {
        segs.pop();
    }
    // A trailing layout directory is not the module's name. Keep climbing, but
    // never to nothing.
    while segs.len() > 1 && segs.last().is_some_and(|s| is_structural_segment(s)) {
        segs.pop();
    }
    let Some(&stem) = segs.last() else {
        return Vec::new();
    };
    if stem.len() < 3 {
        return Vec::new();
    }
    let n = segs.len();
    let mut needles = Vec::new();
    if n >= 2 {
        needles.push(format!("{}/{}", segs[n - 2], segs[n - 1]));
    }
    needles.push(stem.to_string());
    needles
}

/// The manifest that would declare the crate rooted at this file, if the file
/// looks like a crate root (`<dir>/src/lib.rs` or `<dir>/src/main.rs`).
///
/// Rust code imports a crate by the name in its `Cargo.toml`, never by its
/// path. `use engram_retrieval::Engine` carries no trace of
/// `crates/retrieval/src/lib.rs`, so path-derived keys alone can never connect
/// the two.
pub fn crate_manifest_for(path: &str) -> Option<String> {
    let lower = path.to_ascii_lowercase();
    let dir = lower
        .strip_suffix("/src/lib.rs")
        .or_else(|| lower.strip_suffix("/src/main.rs"))?;
    if dir.is_empty() {
        return None;
    }
    Some(format!("{dir}/Cargo.toml"))
}

/// The `name` field of a Cargo manifest's `[package]` section, normalized the
/// way Rust code refers to it (hyphens become underscores).
///
/// Hand-scanned rather than parsed: this needs two lines out of the file, and
/// a TOML dependency would have to be justified for that alone.
pub fn package_name(manifest: &str) -> Option<String> {
    let mut in_package = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != "name" {
            continue;
        }
        let name = value.trim().trim_matches('"').trim_matches('\'');
        if name.is_empty() {
            return None;
        }
        return Some(name.to_ascii_lowercase().replace('-', "_"));
    }
    None
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
        assert!(module_needles("src/utils/retry.py").contains(&"utils/retry".to_owned()));
        assert!(module_needles("retry.rs").contains(&"retry".to_owned()));
        // directory-module files resolve to the directory
        assert!(module_needles("http/mod.rs").contains(&"http".to_owned()));
        assert!(module_needles("api/index.ts").contains(&"api".to_owned()));
        // too generic to match on
        assert!(module_needles("a/io.rs").is_empty());
    }

    #[test]
    fn needle_matches_normalized_target() {
        let needles = module_needles("src/utils/retry.py");
        let target = normalize_target(
            "from utils.retry import backoff"
                .trim_start_matches("from ")
                .split(" import")
                .next()
                .unwrap(),
        );
        assert!(needles.iter().any(|n| target.contains(n)));
    }

    #[test]
    fn layout_directories_do_not_become_the_module_name() {
        // The bug this fixes: crates/retrieval/src/lib.rs used to reduce to the
        // stem "src", because popping the dir-module "lib" left a layout
        // directory behind. Nothing an importer writes ever says "src".
        let needles = module_needles("crates/retrieval/src/lib.rs");
        assert!(
            !needles.iter().any(|n| n == "src" || n == "retrieval/src"),
            "layout directory leaked into the needles: {needles:?}"
        );
        assert!(needles.contains(&"retrieval".to_owned()), "{needles:?}");
    }

    #[test]
    fn crate_roots_are_recognised_from_their_path() {
        assert_eq!(
            crate_manifest_for("crates/retrieval/src/lib.rs").as_deref(),
            Some("crates/retrieval/Cargo.toml")
        );
        assert_eq!(
            crate_manifest_for("crates/mcp-server/src/main.rs").as_deref(),
            Some("crates/mcp-server/Cargo.toml")
        );
        // Not a crate root: an ordinary module inside one.
        assert_eq!(crate_manifest_for("crates/retrieval/src/embed.rs"), None);
        assert_eq!(crate_manifest_for("src/lib.rs"), None);
    }

    #[test]
    fn package_name_is_read_and_normalized_the_way_rust_refers_to_it() {
        // Verbatim shape of this repo's own manifests, including the workspace
        // keys that follow the name and the [dependencies] section after it.
        let manifest = r#"
[package]
name = "engram-retrieval"
version = "0.1.0"
edition = "2021"
license.workspace = true

[dependencies]
name = "not-the-package-name"
"#;
        assert_eq!(package_name(manifest).as_deref(), Some("engram_retrieval"));
    }

    #[test]
    fn package_name_ignores_manifests_without_a_package_section() {
        let workspace_only = "[workspace]\nmembers = [\"crates/domain\"]\n";
        assert_eq!(package_name(workspace_only), None);
    }

    #[test]
    fn a_crate_name_import_can_reach_the_crate_root() {
        // End to end on the exact values that were failing: the importer writes
        // `use engram_retrieval::Engine`, and the crate root must expose a key
        // that the resulting candidate keys actually contain.
        let target = normalize_target("engram_retrieval::Engine");
        let manifest = "[package]\nname = \"engram-retrieval\"\n";
        let alias = package_name(manifest).expect("package name");
        assert!(
            target.starts_with(&alias),
            "target {target:?} should carry the crate alias {alias:?}"
        );
    }
}
