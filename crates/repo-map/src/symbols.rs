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
    text.lines().next().unwrap_or("").trim().chars().take(200).collect()
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
