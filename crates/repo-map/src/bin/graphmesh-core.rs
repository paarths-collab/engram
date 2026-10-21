//! GraphMesh's native RPC entrypoint in the existing engram-repo-map package.
//! No Python launcher: this binary owns persistence and graph queries.
//! Source capture, language-specific semantic adapters, UI and MCP are clients.
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::Path;

#[path = "../mesh_queries.rs"]
mod mesh_queries;
#[path = "../mesh_store.rs"]
mod mesh_store;
use mesh_store::MeshStore;

const VERSION: &str = "0.3.0";
const MAX_REQUEST: u64 = 128 * 1024 * 1024;

fn text<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("{key} must be a string"))
}
fn num(args: &Value, key: &str, default: usize) -> Result<usize> {
    match args.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .ok_or_else(|| anyhow::anyhow!("{key} must be a nonnegative integer")),
    }
}
fn capabilities() -> Value {
    json!({"name":"graphmesh-core", "version":VERSION, "protocol":1,
      "package":"engram-repo-map", "native_storage":true, "native_graph_queries":true,
      "syntax_languages":["Rust","Python","TypeScript","JavaScript"],
      "deep_dependence":false, "llm_calls":0})
}

fn dispatch(store: &mut MeshStore, op: &str, a: &Value) -> Result<Value> {
    match op {
        "ping" | "capabilities" => Ok(capabilities()),
        "cache_get" => store.cache_get(text(a, "key")?),
        "cache_put" => {
            store.cache_put(text(a, "key")?, &a["value"])?;
            Ok(json!({}))
        }
        "publish" => store.publish(a),
        "resolve" => Ok(json!(store.resolve(text(a, "name")?)?)),
        "graph" => store.graph(text(a, "name")?),
        "snapshots" => store.snapshots(),
        "source" => store.source(
            text(a, "name")?,
            text(a, "repo")?,
            text(a, "path")?,
            num(a, "line", 1)?,
            num(a, "context", 8)?,
        ),
        "add_observation" => store.add_observation(text(a, "name")?, &a["observation"]),
        "observations" => store.observations(text(a, "name")?),
        "storage_stats" => store.stats(),
        "search" => mesh_queries::search(
            store,
            text(a, "snapshot")?,
            text(a, "query")?,
            num(a, "limit", 50)?,
        ),
        "diff" => mesh_queries::diff(store, text(a, "before")?, text(a, "after")?),
        "impact" => mesh_queries::impact(
            store,
            text(a, "before")?,
            text(a, "after")?,
            num(a, "max_hops", 5)?,
            num(a, "limit", 200)?,
        ),
        "why" => mesh_queries::why(
            store,
            text(a, "snapshot")?,
            text(a, "source")?,
            text(a, "target")?,
            num(a, "max_hops", 6)?,
            num(a, "limit", 500)?,
        ),
        "extract" => {
            use engram_domain::Language;
            let language = match text(a, "language")? {
                "Rust" => Language::Rust,
                "Python" => Language::Python,
                "TypeScript" => Language::TypeScript,
                "JavaScript" => Language::JavaScript,
                _ => bail!("Unsupported native syntax language"),
            };
            let path = text(a, "path")?;
            let source = text(a, "source")?;
            if source.len() > 1_000_000 {
                bail!("Source file exceeds 1 MB");
            }
            let symbols = engram_repo_map::symbols::extract_file(path, source, language);
            let imports = engram_repo_map::symbols::extract_imports(source, language);
            Ok(
                json!({"symbols":symbols,"imports":imports,"analyzer":"engram-repo-map/tree-sitter",
                "scope":"syntax_only", "deep_dependence":false}),
            )
        }
        _ => bail!("Unknown native operation: {op}"),
    }
}

/// Serve JSON-lines requests until `input` closes. Responses are ASCII-only JSON ending in a bare
/// LF: Windows clients decode stdout with the ANSI code page, so raw UTF-8 would arrive as mojibake.
fn serve(store: &mut MeshStore, mut input: impl BufRead, mut output: impl Write) -> Result<()> {
    loop {
        let mut bytes = Vec::new();
        // Bound allocations before parsing an untrusted line. Oversize closes transport.
        let read =
            std::io::Read::take(&mut input, MAX_REQUEST + 1).read_until(b'\n', &mut bytes)?;
        if read == 0 {
            break;
        }
        if bytes.len() as u64 > MAX_REQUEST {
            bail!("Native request size limit exceeded");
        }
        if bytes.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let (id, result) = match serde_json::from_slice::<Value>(&bytes) {
            Ok(req) => {
                let id = req.get("id").cloned().unwrap_or(Value::Null);
                let result = (|| {
                    let op = text(&req, "op")?;
                    let a = req.get("args").cloned().unwrap_or_else(|| json!({}));
                    if !a.is_object() {
                        bail!("args must be an object");
                    }
                    dispatch(store, op, &a)
                })();
                (id, result)
            }
            Err(e) => (Value::Null, Err(e.into())),
        };
        let response = match result {
            Ok(value) => json!({"id":id,"ok":true,"result":value}),
            Err(e) => json!({"id":id,"ok":false,"error":{
                "code": if e.downcast_ref::<rusqlite::Error>().is_some(){"DATABASE"}else{"INVALID_INPUT"},
                "message":e.to_string()}}),
        };
        output.write_all(mesh_store::canonical(&response)?.as_bytes())?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.as_slice() == ["--version"] {
        println!("graphmesh-core {VERSION}");
        return Ok(());
    }
    if args.as_slice() == ["--probe"] {
        println!("{}", capabilities());
        return Ok(());
    }
    if args.len() != 2 || args[0] != "--db" {
        bail!("Usage: graphmesh-core --db PATH | --probe | --version");
    }
    let mut store = MeshStore::open(Path::new(&args[1]))?;
    serve(&mut store, io::stdin().lock(), io::stdout().lock())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("graphmesh-core: {e}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Scratch dir with a space and a non-ASCII character, like many real Windows profiles.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "graphmesh space \u{e9} {}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            std::fs::create_dir_all(&path).expect("create temp dir");
            TempDir(path)
        }

        /// Nested so `MeshStore::open` has to create the parent directories.
        fn db(&self) -> PathBuf {
            self.0.join("nested").join("dir").join("state.db")
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn line(id: u32, op: &str, args: Value) -> String {
        format!("{}\n", json!({"id": id, "op": op, "args": args}))
    }

    fn serve_bytes(store: &mut MeshStore, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        serve(store, input, &mut out).expect("serve");
        out
    }

    fn snapshot(alias: &str, second_node: &str, lib_source: &str) -> Value {
        json!({
            "alias": alias,
            "created_at": "2026-01-01T00:00:00Z",
            "manifest": {"repos": {"app": {}}},
            "summary": {},
            "nodes": [
                {"id": "n1", "repo": "app", "path": "src/lib.rs", "kind": "file", "name": "lib.rs"},
                {"id": second_node, "repo": "app", "path": "src/caf\u{e9}.rs", "kind": "file", "name": "caf\u{e9}.rs"}
            ],
            "edges": [{"id": "e1", "source": "n1", "target": second_node, "kind": "IMPORTS"}],
            "diagnostics": [],
            "source_texts": {"app": {"src/lib.rs": lib_source, "src/caf\u{e9}.rs": "// caf\u{e9}\n"}}
        })
    }

    #[test]
    fn responses_are_ascii_so_any_client_codec_decodes_them_identically() {
        // Windows Python decodes subprocess output as cp1252; raw UTF-8 became "cafÃ© ðŸ˜€".
        let dir = TempDir::new();
        let mut store = MeshStore::open(&dir.db()).unwrap();
        let input = [
            line(
                1,
                "cache_put",
                json!({"key": "k", "value": {"text": "caf\u{e9} \u{1F600}"}}),
            ),
            line(2, "cache_get", json!({"key": "k"})),
            line(3, "resolve", json!({"name": "caf\u{e9}"})),
        ]
        .concat();

        let out = serve_bytes(&mut store, input.as_bytes());

        assert!(
            out.is_ascii(),
            "wire output must be ASCII: {}",
            String::from_utf8_lossy(&out)
        );
        assert!(!out.contains(&b'\r'), "responses end in a bare LF");
        let as_latin1: String = out.iter().map(|&b| char::from(b)).collect();
        assert_eq!(
            as_latin1,
            String::from_utf8(out).unwrap(),
            "codec must not matter"
        );
        let rows: Vec<Value> = as_latin1
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(rows[1]["result"]["text"], "caf\u{e9} \u{1F600}");
        assert_eq!(rows[2]["ok"], false);
        assert!(rows[2]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("caf\u{e9}"));
    }

    #[test]
    fn lf_crlf_and_unterminated_requests_get_identical_answers() {
        let ping = r#"{"id":7,"op":"ping","args":{}}"#;
        let answer = |input: String| {
            let dir = TempDir::new();
            let mut store = MeshStore::open(&dir.db()).unwrap();
            serve_bytes(&mut store, input.as_bytes())
        };
        let expected = answer(format!("{ping}\n"));
        assert!(!expected.is_empty());
        for framing in [
            format!("{ping}\r\n"),
            ping.to_string(),
            format!("\r\n{ping}\r\n\r\n"),
            format!("\n{ping}\n"),
        ] {
            assert_eq!(answer(framing.clone()), expected, "framing {framing:?}");
        }
    }

    #[test]
    fn snapshots_aliases_and_cache_survive_restart_and_release_the_files() {
        let dir = TempDir::new();
        let db = dir.db();

        let first = {
            let mut store = MeshStore::open(&db).unwrap();
            store.cache_put("k", &json!({"v": 1})).unwrap();
            store
                .publish(&snapshot("main", "n2", "fn one() {}\n"))
                .unwrap()
        };

        let second = {
            let mut store = MeshStore::open(&db).unwrap();
            assert_eq!(store.resolve("main").unwrap(), first.as_str().unwrap());
            assert_eq!(store.cache_get("k").unwrap(), json!({"v": 1}));
            assert_eq!(
                store.graph("main").unwrap()["nodes"]
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
            let src = store
                .source("main", "app", "src/caf\u{e9}.rs", 1, 0)
                .unwrap();
            assert_eq!(src["available"], true);
            assert_eq!(src["lines"][0], "// caf\u{e9}");
            store
                .publish(&snapshot("main", "n3", "fn two() {}\n"))
                .unwrap()
        };
        assert_ne!(first, second);

        let store = MeshStore::open(&db).unwrap();
        assert_eq!(
            store.resolve("main").unwrap(),
            second.as_str().unwrap(),
            "alias follows the replacement"
        );
        assert_eq!(
            store.resolve(first.as_str().unwrap()).unwrap(),
            first.as_str().unwrap(),
            "old snapshot stays addressable"
        );
        assert_eq!(store.snapshots().unwrap().as_array().unwrap().len(), 1);
        drop(store);

        // Windows cannot delete a file that still has an open handle.
        std::fs::remove_dir_all(&dir.0).expect("db files must be released when the store drops");
    }

    #[test]
    fn captured_source_paths_are_opaque_keys_never_filesystem_paths() {
        let dir = TempDir::new();
        let mut store = MeshStore::open(&dir.db()).unwrap();

        for bad in [
            "a\\b.rs",
            "C:\\Windows\\x.rs",
            "\\\\server\\share\\x.rs",
            "../x.rs",
            "/etc/passwd",
            "a/../b.rs",
        ] {
            let mut args = snapshot("main", "n2", "x");
            args["source_texts"] = json!({"app": {(bad): "x"}});
            assert!(store.publish(&args).is_err(), "{bad:?} must be rejected");
        }
        assert!(
            store.snapshots().unwrap().as_array().unwrap().is_empty(),
            "rejected publishes leave nothing behind"
        );

        // Case-folding filesystems must not change what the store does: both names stay distinct.
        let mut args = snapshot("main", "n2", "x");
        args["source_texts"] = json!({"app": {"Mod.rs": "upper", "mod.rs": "lower"}});
        store.publish(&args).unwrap();
        assert_eq!(
            store.source("main", "app", "Mod.rs", 1, 0).unwrap()["lines"][0],
            "upper"
        );
        assert_eq!(
            store.source("main", "app", "mod.rs", 1, 0).unwrap()["lines"][0],
            "lower"
        );
    }
}
