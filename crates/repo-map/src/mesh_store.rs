//! Versioned GraphMesh tables on Engram's existing Store connection.
//! This replaces v0.2's Python database implementation; the schema is compatible.
use anyhow::{bail, Context, Result};
use engram_repo_map::store::Store;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap,BTreeSet};
use std::path::Path;
use std::time::Duration;

pub struct MeshStore {pub inner: Store}

/// Canonical JSON matching the adapter's sorted ASCII JSON for these graph values.
pub fn canonical(v:&Value)->Result<String>{
    let encoded=serde_json::to_string(v)?;
    let mut out=String::with_capacity(encoded.len());
    for c in encoded.chars(){
        if c.is_ascii(){out.push(c);}else{
            let mut buf=[0;2];
            for unit in c.encode_utf16(&mut buf){out.push_str(&format!("\\u{unit:04x}"));}
        }
    }
    Ok(out)
}
pub fn digest(v:&Value)->Result<String>{Ok(format!("{:x}",Sha256::digest(canonical(v)?.as_bytes())))}
pub fn field<'a>(v:&'a Value,key:&str)->Result<&'a str>{v[key].as_str().ok_or_else(||anyhow::anyhow!("{key} must be a string"))}
pub fn array<'a>(v:&'a Value,key:&str)->Result<&'a Vec<Value>>{v[key].as_array().ok_or_else(||anyhow::anyhow!("{key} must be an array"))}
fn path_ok(s:&str)->bool{!s.is_empty() && !s.starts_with('/') && !s.contains('\\') && !s.contains('\0') && !s.split('/').any(|x|x=="..")}
fn alias_ok(s:&str)->bool{!s.is_empty() && s.len()<=128 && s.bytes().all(|c|c.is_ascii_alphanumeric()||b"_.-/".contains(&c))}

impl MeshStore{
 pub fn open(path:&Path)->Result<Self>{
    if let Some(parent)=path.parent(){if !parent.as_os_str().is_empty(){std::fs::create_dir_all(parent)?;}}
    if path.symlink_metadata().is_ok_and(|m|m.file_type().is_symlink()){bail!("Database path may not be a symlink");}
    let conn=Connection::open(path)?;
    conn.busy_timeout(Duration::from_secs(30))?;
    conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);")?;
    let version:Option<String>=conn.query_row("SELECT value FROM metadata WHERE key='schema_version'",[],|r|r.get(0)).optional()?;
    if version.as_deref().is_some_and(|v|v!="1"){bail!("Unsupported database schema; preserve it and use a new database");}
    conn.execute_batch("CREATE TABLE IF NOT EXISTS objects(hash TEXT PRIMARY KEY,body TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY,created_at TEXT NOT NULL,manifest TEXT NOT NULL,summary TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS members(snapshot TEXT NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,kind TEXT NOT NULL,object_hash TEXT NOT NULL REFERENCES objects(hash),PRIMARY KEY(snapshot,kind,object_hash));
      CREATE TABLE IF NOT EXISTS aliases(name TEXT PRIMARY KEY,snapshot TEXT NOT NULL REFERENCES snapshots(id));
      CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,body TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY,snapshot TEXT NOT NULL REFERENCES snapshots(id),body TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS source_blobs(hash TEXT PRIMARY KEY,body TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS source_members(snapshot TEXT NOT NULL REFERENCES snapshots(id),repo TEXT NOT NULL,path TEXT NOT NULL,source_hash TEXT NOT NULL REFERENCES source_blobs(hash),PRIMARY KEY(snapshot,repo,path));
      CREATE INDEX IF NOT EXISTS members_by_snapshot ON members(snapshot,kind);
      CREATE INDEX IF NOT EXISTS observations_by_snapshot ON observations(snapshot);
      INSERT OR IGNORE INTO metadata VALUES('schema_version','1');")?;
    Ok(Self{inner:Store{conn}})
 }
 pub fn cache_get(&self,key:&str)->Result<Value>{
    let body:Option<String>=self.inner.conn.query_row("SELECT body FROM cache WHERE key=?",[key],|r|r.get(0)).optional()?;
    match body{Some(s)=>Ok(serde_json::from_str(&s)?),None=>Ok(Value::Null)}
 }
 pub fn cache_put(&self,key:&str,value:&Value)->Result<()>{
    self.inner.conn.execute("INSERT OR REPLACE INTO cache VALUES(?,?)",params![key,canonical(value)?])?;Ok(())
 }
 pub fn resolve(&self,name:&str)->Result<String>{
    let alias:Option<String>=self.inner.conn.query_row("SELECT snapshot FROM aliases WHERE name=?",[name],|r|r.get(0)).optional()?;
    if let Some(s)=alias{return Ok(s);}
    let present=self.inner.conn.query_row("SELECT 1 FROM snapshots WHERE id=?",[name],|r|r.get::<_,i64>(0)).optional()?.is_some();
    if present{Ok(name.into())}else{bail!("Unknown snapshot or alias: {name}")}
 }
 pub fn publish(&mut self,a:&Value)->Result<Value>{
    let alias=field(a,"alias")?;if !alias_ok(alias){bail!("Invalid snapshot alias");}
    if !a["manifest"].is_object()||!a["summary"].is_object(){bail!("Manifest and summary must be objects");}
    let nodes=array(a,"nodes")?;let edges=array(a,"edges")?;let ds=array(a,"diagnostics")?;
    if nodes.len()>1_000_000||edges.len()>3_000_000{bail!("Graph size limit exceeded");}
    let mut ids=BTreeSet::new();
    for n in nodes{let id=field(n,"id")?;if !ids.insert(id){bail!("Node identity collision");}for key in ["repo","path","kind","name"]{field(n,key)?;}}
    let mut eids=BTreeSet::new();
    for e in edges{
        if !ids.contains(field(e,"source")?)||!ids.contains(field(e,"target")?){bail!("Dangling graph edges; snapshot not published");}
        if !eids.insert(field(e,"id")?){bail!("Duplicate edge ID");}field(e,"kind")?;
    }
    let groups=[("node",nodes),("edge",edges),("diagnostic",ds)];
    let mut hashes=BTreeMap::new();
    for (kind,objects) in &groups{hashes.insert(*kind,objects.iter().map(digest).collect::<Result<BTreeSet<_>>>()?);}
    let sid=format!("g:{}",digest(&json!({"schema":1,"manifest":a["manifest"],"objects":hashes}))?);
    let tx=self.inner.conn.transaction()?;
    tx.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?)",params![sid,a["created_at"].as_str().unwrap_or("unknown"),canonical(&a["manifest"])?,canonical(&a["summary"])?])?;
    for (kind,objects) in groups{for obj in objects{
        let h=digest(obj)?;
        tx.execute("INSERT OR IGNORE INTO objects VALUES(?,?)",params![h,canonical(obj)?])?;
        tx.execute("INSERT OR IGNORE INTO members VALUES(?,?,?)",params![sid,kind,h])?;
    }}
    if let Some(repos)=a["source_texts"].as_object(){for (repo,files) in repos{
        let fs=files.as_object().context("Source files must be an object")?;
        for (path,body) in fs{
            if !path_ok(path){bail!("Unsafe captured source path");}let text=body.as_str().context("Source must be a string")?;
            let h=digest(body)?;
            tx.execute("INSERT OR IGNORE INTO source_blobs VALUES(?,?)",params![h,text])?;
            let existing:Option<String>=tx.query_row("SELECT source_hash FROM source_members WHERE snapshot=? AND repo=? AND path=?",params![sid,repo,path],|r|r.get(0)).optional()?;
            if existing.as_ref().is_some_and(|v|v!=&h){bail!("Attempt to modify immutable source evidence");}
            tx.execute("INSERT OR IGNORE INTO source_members VALUES(?,?,?,?)",params![sid,repo,path,h])?;
        }
    }}
    tx.execute("INSERT INTO aliases VALUES(?,?) ON CONFLICT(name) DO UPDATE SET snapshot=excluded.snapshot",params![alias,sid])?;
    tx.commit()?;Ok(json!(sid))
 }
 pub fn graph(&self,name:&str)->Result<Value>{
    let sid=self.resolve(name)?;
    let (at,m,s):(String,String,String)=self.inner.conn.query_row("SELECT created_at,manifest,summary FROM snapshots WHERE id=?",[&sid],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
    let mut result=json!({"snapshot_id":sid,"created_at":at,"manifest":serde_json::from_str::<Value>(&m)?,"summary":serde_json::from_str::<Value>(&s)?});
    for (kind,key) in [("node","nodes"),("edge","edges"),("diagnostic","diagnostics")]{
        let mut q=self.inner.conn.prepare("SELECT o.body FROM members m JOIN objects o ON o.hash=m.object_hash WHERE m.snapshot=? AND m.kind=? ORDER BY o.hash")?;
        let rows=q.query_map(params![sid,kind],|r|r.get::<_,String>(0))?;
        let values=rows.map(|r|Ok(serde_json::from_str::<Value>(&r?)?)).collect::<Result<Vec<_>>>()?;
        result[key]=json!(values);
    }
    Ok(result)
 }
 pub fn snapshots(&self)->Result<Value>{
    let mut q=self.inner.conn.prepare("SELECT a.name,s.id,s.created_at,s.summary FROM aliases a JOIN snapshots s ON s.id=a.snapshot ORDER BY a.name")?;
    let rows=q.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,String>(3)?)))?;
    let values=rows.map(|r|{let (alias,sid,at,s)=r?;Ok(json!({"alias":alias,"snapshot_id":sid,"created_at":at,"summary":serde_json::from_str::<Value>(&s)?}))}).collect::<Result<Vec<_>>>()?;
    Ok(json!(values))
 }
 pub fn source(&self,name:&str,repo:&str,path:&str,line:usize,context:usize)->Result<Value>{
    if line==0||context>80{bail!("Invalid source window");}
    let sid=self.resolve(name)?;
    let row:Option<(String,String)>=self.inner.conn.query_row("SELECT b.body,b.hash FROM source_members m JOIN source_blobs b ON b.hash=m.source_hash WHERE snapshot=? AND repo=? AND path=?",params![sid,repo,path],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
    match row{
        None=>Ok(json!({"snapshot_id":sid,"available":false,"repo":repo,"path":path,"reason":"Source not captured for this snapshot"})),
        Some((body,h))=>{
            let lines:Vec<&str>=body.lines().collect();let start=std::cmp::max(1,line.saturating_sub(context));let end=std::cmp::min(lines.len(),line.saturating_add(context));
            let window=if start<=end{lines[start-1..end].to_vec()}else{vec![]};
            Ok(json!({"snapshot_id":sid,"available":true,"repo":repo,"path":path,"source_hash":h,"start_line":start,"end_line":end,"lines":window}))
        }
    }
 }
 pub fn add_observation(&mut self,name:&str,o:&Value)->Result<Value>{
    let sid=self.resolve(name)?;if !o.is_object()||o["snapshot_id"]!=sid{bail!("Artifact snapshot_id must exactly match immutable analyzed snapshot");}
    let run=field(o,"run_id")?;if run.trim().is_empty()||run.len()>512{bail!("Invalid run_id");}
    let tests=array(o,"tests")?;if tests.len()>10000{bail!("Too many tests");}
    let g=self.graph(&sid)?;let mut seen=BTreeSet::new();
    for t in tests{
        let id=field(t,"id")?;if id.trim().is_empty(){bail!("Test ID cannot be empty");}
        if !["passed","failed","skipped","error"].contains(&field(t,"status")?){bail!("Invalid test status");}
        if t.get("repo").is_some()!=t.get("path").is_some(){bail!("Test source needs repo and path");}
        if t.get("repo").is_some(){
            let repo=field(t,"repo")?;if !path_ok(field(t,"path")?){bail!("Unsafe test path");}
            if g["manifest"]["repos"].get(repo).is_none(){bail!("Unknown test repository");}
        }
        if !seen.insert(canonical(&json!([t.get("repo"),t.get("path"),id]))?){bail!("Duplicate test identity");}
    }
    let mut clean=o.clone();clean.as_object_mut().unwrap().remove("observation_id");
    clean["trust"]=json!("untrusted_import");clean["claims_production_safety"]=json!(false);
    if canonical(&clean)?.len()>10_000_000{bail!("Artifact size limit exceeded");}
    let oid=format!("run:{}",digest(&clean)?);
    self.inner.conn.execute("INSERT OR IGNORE INTO observations VALUES(?,?,?)",params![oid,sid,canonical(&clean)?])?;
    clean["observation_id"]=json!(oid);Ok(clean)
 }
 pub fn observations(&self,name:&str)->Result<Value>{
    let sid=self.resolve(name)?;let mut q=self.inner.conn.prepare("SELECT id,body FROM observations WHERE snapshot=? ORDER BY id")?;
    let rows=q.query_map([sid],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?)))?;
    let out=rows.map(|r|{let (id,body)=r?;let mut v:Value=serde_json::from_str(&body)?;v["observation_id"]=json!(id);Ok(v)}).collect::<Result<Vec<_>>>()?;
    Ok(json!(out))
 }
 pub fn stats(&self)->Result<Value>{
    let mut v=json!({});for name in ["objects","members","snapshots","source_blobs","observations"]{
        // Constant identifiers, never user-controlled SQL.
        let count:i64=self.inner.conn.query_row(&format!("SELECT count(*) FROM {name}"),[],|r|r.get(0))?;
        v[name]=json!(count);
    }Ok(v)
 }
}

#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn canonical_unicode(){assert_eq!(canonical(&json!({"b":"é😀","a":1})).unwrap(),"{\"a\":1,\"b\":\"\\u00e9\\ud83d\\ude00\"}");}
 #[test]fn hashes_match_python(){assert_eq!(digest(&json!("hello")).unwrap(),"5aa762ae383fbb727af3c7a36d4940a5b8c40a989452d2304fc958ff3f354e7a");}
 #[test]fn invalid_paths(){for p in ["../x","/etc/passwd","a/../b","a\\b",""]{assert!(!path_ok(p));}}
}
