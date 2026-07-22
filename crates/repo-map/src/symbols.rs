//! Tier-1: structural parsing — extract symbols with tree-sitter.
//! Lazy-friendly: call `extract_file` per file; caching lives in the store.

use engram_domain::{Language, SymbolKind, SymbolRecord};
use tree_sitter::{Node, Parser};

fn parser_for(lang: Language) -> Option<Parser> {
    let mut p = Parser::new();
    let ok = match lang {
        Language::Rust => p.set_language(tree_sitter_rust::language()).is_ok(),
        Language::Python => p.set_language(tree_sitter_python::language()).is_ok(),
        Language::TypeScript => p
            .set_language(tree_sitter_typescript::language_typescript())
            .is_ok(),
        Language::JavaScript => p.set_language(tree_sitter_javascript::language()).is_ok(),
        _ => false,
    };
    if ok {
        Some(p)
    } else {
        None
    }
}

fn kind_for(lang: Language, node_kind: &str) -> Option<SymbolKind> {
    match (lang, node_kind) {
        (Language::Rust, "function_item") => Some(SymbolKind::Function),
        (Language::Rust, "struct_item") => Some(SymbolKind::Struct),
        (Language::Rust, "enum_item") => Some(SymbolKind::Enum),
        (Language::Rust, "trait_item") => Some(SymbolKind::Trait),
        (Language::Rust, "impl_item") => None, // methods found via nested function_item
        (Language::Python, "function_definition") => Some(SymbolKind::Function),
        (Language::Python, "class_definition") => Some(SymbolKind::Class),
        (Language::TypeScript | Language::JavaScript, "function_declaration") => {
            Some(SymbolKind::Function)
        }
        (Language::TypeScript | Language::JavaScript, "class_declaration") => {
            Some(SymbolKind::Class)
        }
        (Language::TypeScript, "interface_declaration") => Some(SymbolKind::Interface),
        (Language::TypeScript | Language::JavaScript, "method_definition") => {
            Some(SymbolKind::Method)
        }
        _ => None,
    }
}

fn first_line(src: &str, node: &Node) -> String {
    let text = &src[node.byte_range()];
    text.lines()
        .next()
        .unwrap_or("")
        .trim()
        .chars()
        .take(200)
        .collect()
}

fn name_of(src: &str, node: &Node) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    Some(src[name_node.byte_range()].to_string())
}

/// Extract all symbols from one file. Returns empty vec for unsupported languages.
pub fn extract_file(path: &str, source: &str, lang: Language) -> Vec<SymbolRecord> {
    let Some(mut parser) = parser_for(lang) else {
        return Vec::new();
    };
    let Some(tree) = parser.parse(source, None) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if let Some(kind) = kind_for(lang, node.kind()) {
            if let Some(name) = name_of(source, &node) {
                out.push(SymbolRecord {
                    name,
                    kind,
                    path: path.to_string(),
                    start_line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    signature: first_line(source, &node),
                });
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
    }
    out
}

/// Extract raw import targets from one file (module paths / specifiers).
/// Normalization into comparable keys is done by `crate::imports::normalize_target`.
pub fn extract_imports(source: &str, lang: Language) -> Vec<String> {
    let Some(mut parser) = parser_for(lang) else {
        return Vec::new();
    };
    let Some(tree) = parser.parse(source, None) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if let Some(target) = import_target(source, &node, lang) {
            if !target.is_empty() {
                out.push(target);
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
    }
    out
}

/// Pull the module path/specifier out of an import-like node, as raw text.
fn import_target(src: &str, node: &Node, lang: Language) -> Option<String> {
    let text = &src[node.byte_range()];
    match (lang, node.kind()) {
        (Language::Rust, "use_declaration") => {
            // "use a::b::c;" / "use a::b::{c, d};" -> keep the common prefix path
            let s = text
                .trim_start_matches("use")
                .trim()
                .trim_end_matches(';')
                .trim();
            let s = s.split('{').next().unwrap_or(s);
            let s = s.trim().trim_end_matches("::").trim();
            (!s.is_empty()).then(|| s.to_string())
        }
        (Language::Python, "import_from_statement") => {
            // "from a.b import c"
            let after = text.strip_prefix("from")?.trim_start();
            after.split_whitespace().next().map(|m| m.to_string())
        }
        (Language::Python, "import_statement") => {
            // "import a.b.c" / "import a.b as x"
            let after = text.strip_prefix("import")?.trim_start();
            after
                .split([',', ' ', '\n'])
                .next()
                .map(|m| m.trim().to_string())
        }
        (Language::TypeScript | Language::JavaScript, "import_statement") => first_quoted(text),
        (Language::TypeScript | Language::JavaScript, "call_expression") => text
            .starts_with("require")
            .then(|| first_quoted(text))
            .flatten(),
        _ => None,
    }
}

/// Return the contents of the first quoted string in `s` (', ", or `).
fn first_quoted(s: &str) -> Option<String> {
    let mut chars = s.char_indices();
    for (i, c) in chars.by_ref() {
        if c == '"' || c == '\'' || c == '`' {
            let start = i + c.len_utf8();
            if let Some(end) = s[start..].find(c) {
                return Some(s[start..start + end].to_string());
            }
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUST_SRC: &str = "\
fn first() {
    let x = 1;
}

fn second() {
    let y = 2;
}
";

    fn find<'a>(symbols: &'a [SymbolRecord], name: &str) -> &'a SymbolRecord {
        symbols
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no symbol named {name}"))
    }

    /// Slice a symbol's span back out of the source, the way retrieval will.
    fn span_text(source: &str, symbol: &SymbolRecord) -> String {
        source.lines().collect::<Vec<_>>()[symbol.start_line - 1..symbol.end_line].join("\n")
    }

    #[test]
    fn rust_functions_get_one_based_spans() {
        let symbols = extract_file("a.rs", RUST_SRC, Language::Rust);
        let first = find(&symbols, "first");
        assert_eq!((first.start_line, first.end_line), (1, 3));
        let second = find(&symbols, "second");
        assert_eq!((second.start_line, second.end_line), (5, 7));
    }

    #[test]
    fn spans_slice_back_to_exactly_one_definition() {
        // This is the property chunk-level embedding depends on: the span must
        // cover the whole definition and none of its neighbour.
        let symbols = extract_file("a.rs", RUST_SRC, Language::Rust);
        let body = span_text(RUST_SRC, find(&symbols, "second"));
        assert!(body.starts_with("fn second()"), "got: {body}");
        assert!(body.trim_end().ends_with('}'), "got: {body}");
        assert!(!body.contains("first"), "bled into the neighbour: {body}");
        assert!(body.contains("let y = 2;"), "lost the body: {body}");
    }

    #[test]
    fn a_nested_python_method_is_contained_by_its_class() {
        let src = "\
class Greeter:
    def hello(self):
        return \"hi\"
";
        let symbols = extract_file("g.py", src, Language::Python);
        let class = find(&symbols, "Greeter");
        let method = find(&symbols, "hello");
        assert!(
            class.start_line <= method.start_line && method.end_line <= class.end_line,
            "class {:?} should contain method {:?}",
            (class.start_line, class.end_line),
            (method.start_line, method.end_line)
        );
        assert!(span_text(src, method).contains("return"));
    }

    #[test]
    fn unsupported_languages_yield_nothing_rather_than_guessing() {
        assert!(extract_file("a.go", "func main() {}", Language::Go).is_empty());
        assert!(extract_file("a.txt", "hello", Language::Other).is_empty());
    }
}
