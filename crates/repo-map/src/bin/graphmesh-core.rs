//! GraphMesh's native RPC entrypoint in the existing engram-repo-map package.
//! No Python launcher: this binary owns persistence and graph queries.
//! Source capture, language-specific semantic adapters, UI and MCP are clients.
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::Path;

#[path = "../mesh_store.rs"]
mod mesh_store;
#[path = "../mesh_queries.rs"]
mod mesh_queries;
use mesh_store::MeshStore;

const VERSION: &str = "0.3.0";
const MAX_REQUEST: u64 = 128 * 1024 * 1024;

fn text<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key).and_then(Value::as_str).ok_or_else(|| anyhow::anyhow!("{key} must be a string"))
}
fn num(args: &Value, key: &str, default: usize) -> Result<usize> {
    match args.get(key) {
        None => Ok(default),
        Some(v) => v.as_u64().and_then(|x| usize::try_from(x).ok())
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
        "cache_get" => store.cache_get(text(a,"key")?),
        "cache_put" => { store.cache_put(text(a,"key")?, &a["value"])?; Ok(json!({})) },
        "publish" => store.publish(a),
        "resolve" => Ok(json!(store.resolve(text(a,"name")?)?)),
        "graph" => store.graph(text(a,"name")?),
        "snapshots" => store.snapshots(),
        "source" => store.source(text(a,"name")?,text(a,"repo")?,text(a,"path")?,num(a,"line",1)?,num(a,"context",8)?),
        "add_observation" => store.add_observation(text(a,"name")?, &a["observation"]),
        "observations" => store.observations(text(a,"name")?),
        "storage_stats" => store.stats(),
        "search" => mesh_queries::search(store,text(a,"snapshot")?,text(a,"query")?,num(a,"limit",50)?),
        "diff" => mesh_queries::diff(store,text(a,"before")?,text(a,"after")?),
        "impact" => mesh_queries::impact(store,text(a,"before")?,text(a,"after")?,num(a,"max_hops",5)?,num(a,"limit",200)?),
        "why" => mesh_queries::why(store,text(a,"snapshot")?,text(a,"source")?,text(a,"target")?,num(a,"max_hops",6)?,num(a,"limit",500)?),
        "extract" => {
            use engram_domain::Language;
            let language = match text(a,"language")? {
                "Rust"=>Language::Rust, "Python"=>Language::Python,
                "TypeScript"=>Language::TypeScript, "JavaScript"=>Language::JavaScript,
                _ => bail!("Unsupported native syntax language")
            };
            let path=text(a,"path")?; let source=text(a,"source")?;
            if source.len()>1_000_000 {bail!("Source file exceeds 1 MB");}
            let symbols=engram_repo_map::symbols::extract_file(path,source,language);
            let imports=engram_repo_map::symbols::extract_imports(source,language);
            Ok(json!({"symbols":symbols,"imports":imports,"analyzer":"engram-repo-map/tree-sitter",
                "scope":"syntax_only", "deep_dependence":false}))
        }
        _ => bail!("Unknown native operation: {op}"),
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.as_slice()==["--version"] {println!("graphmesh-core {VERSION}");return Ok(());}
    if args.as_slice()==["--probe"] {println!("{}",capabilities());return Ok(());}
    if args.len()!=2 || args[0]!="--db" {bail!("Usage: graphmesh-core --db PATH | --probe | --version");}
    let mut store=MeshStore::open(Path::new(&args[1]))?;
    let mut input=io::stdin().lock(); let mut output=io::stdout().lock();
    loop {
        let mut bytes=Vec::new();
        // Bound allocations before parsing an untrusted line. Oversize closes transport.
        let read=std::io::Read::take(&mut input,MAX_REQUEST+1).read_until(b'\n',&mut bytes)?;
        if read==0 {break;}
        if bytes.len() as u64>MAX_REQUEST {bail!("Native request size limit exceeded");}
        if bytes.iter().all(u8::is_ascii_whitespace) {continue;}
        let (id,result)=match serde_json::from_slice::<Value>(&bytes) {
            Ok(req) => {
                let id=req.get("id").cloned().unwrap_or(Value::Null);
                let result=(||{
                    let op=text(&req,"op")?;
                    let a=req.get("args").cloned().unwrap_or_else(||json!({}));
                    if !a.is_object() {bail!("args must be an object");}
                    dispatch(&mut store,op,&a)
                })();
                (id,result)
            },
            Err(e)=>(Value::Null,Err(e.into()))
        };
        let response=match result {
            Ok(value)=>json!({"id":id,"ok":true,"result":value}),
            Err(e)=>json!({"id":id,"ok":false,"error":{
                "code": if e.downcast_ref::<rusqlite::Error>().is_some(){"DATABASE"}else{"INVALID_INPUT"},
                "message":e.to_string()}}),
        };
        serde_json::to_writer(&mut output,&response)?;output.write_all(b"\n")?;output.flush()?;
    }
    Ok(())
}
fn main() {
    if let Err(e)=run() {eprintln!("graphmesh-core: {e}");std::process::exit(2);}
}
