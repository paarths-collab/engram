//! Bounded graph queries over exact snapshots. Never mix revisions in one path.
//! This is dependency reachability, NOT a whole-program dependence/safety proof.
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap,BTreeSet,VecDeque};
use super::mesh_store::{array,field,MeshStore};

fn limits(hops:usize,limit:usize)->Result<()>{if !(1..=12).contains(&hops)||!(1..=2000).contains(&limit){bail!("Query limits out of range");}Ok(())}
fn dependency(e:&Value)->bool{matches!(e["kind"].as_str(),Some("IMPORTS"|"MAY_CALL"|"REFERENCES"|"FILE_DEPENDS_ON"|"CONSUMES_CONTRACT"|"IMPLEMENTED_BY"))}
fn key(n:&Value)->(String,String){(n["repo"].as_str().unwrap_or("").into(),n["path"].as_str().unwrap_or("").into())}
fn keyed(values:&[Value])->BTreeMap<String,Value>{values.iter().filter_map(|v|v["id"].as_str().map(|id|(id.into(),v.clone()))).collect()}
fn relation(e:&Value)->(String,String,String){(e["source"].as_str().unwrap_or("").into(),e["target"].as_str().unwrap_or("").into(),e["kind"].as_str().unwrap_or("").into())}
fn select(g:&Value,selector:&str)->Result<Value>{
 let nodes=array(g,"nodes")?;
 if let Some(n)=nodes.iter().find(|n|n["id"]==selector){return Ok(n.clone());}
 let matched:Vec<_>=nodes.iter().filter(|n|{
    if n["kind"]=="callsite"{return false;}
    let (repo,path)=key(n);let qualified=n["qualified_name"].as_str().unwrap_or("");
    (n["kind"]=="file"&&format!("{repo}:{path}")==selector)||(!qualified.is_empty()&&(qualified==selector||format!("{repo}:{path}#{qualified}")==selector))
 }).collect();
 if matched.len()!=1{bail!("Selector must identify exactly one node ({} matched); use a node ID or repo:path#qualified_name.",matched.len());}
 Ok(matched[0].clone())
}

pub fn search(s:&MeshStore,snapshot:&str,query:&str,limit:usize)->Result<Value>{
 limits(1,limit)?;let g=s.graph(snapshot)?;let q=query.to_lowercase();
 let mut ns:Vec<_>=array(&g,"nodes")?.iter().filter(|n|n["name"].as_str().unwrap_or("").to_lowercase().contains(&q)||n["qualified_name"].as_str().unwrap_or("").to_lowercase().contains(&q)).cloned().collect();
 ns.sort_by_key(|n|(n["kind"]=="callsite",n["kind"]=="file",key(n),n["start_line"].as_u64().unwrap_or(1),n["id"].as_str().unwrap_or("").to_owned()));
 let truncated=ns.len()>limit;ns.truncate(limit);
 Ok(json!({"snapshot_id":g["snapshot_id"],"nodes":ns,"truncated":truncated}))
}

pub fn diff(s:&MeshStore,before:&str,after:&str)->Result<Value>{
 let a=s.graph(before)?;let b=s.graph(after)?;let na=keyed(array(&a,"nodes")?);let nb=keyed(array(&b,"nodes")?);
 let ea=keyed(array(&a,"edges")?);let eb=keyed(array(&b,"edges")?);
 let fa:BTreeMap<_,_>=na.values().filter(|n|n["kind"]=="file").map(|n|(key(n),n)).collect();
 let fb:BTreeMap<_,_>=nb.values().filter(|n|n["kind"]=="file").map(|n|(key(n),n)).collect();
 let mut changes=vec![];
 for k in fa.keys().chain(fb.keys()).collect::<BTreeSet<_>>(){
    let old=fa.get(k);let new=fb.get(k);
    if old.is_some()&&new.is_some()&&old.unwrap()["content_hash"]==new.unwrap()["content_hash"]{continue;}
    // First differing captured line. Do not claim a full semantic diff.
    let left=s.source(field(&a,"snapshot_id")?,&k.0,&k.1,1,80)?;
    let right=s.source(field(&b,"snapshot_id")?,&k.0,&k.1,1,80)?;
    let mut line=1;
    if left["available"]==true&&right["available"]==true{
       let ll=array(&left,"lines")?;let rr=array(&right,"lines")?;
       line=ll.iter().zip(rr).take_while(|(x,y)|x==y).count()+1;
       if line>ll.len()&&line>rr.len(){line=1;}
    }
    changes.push(json!({"repo":k.0,"path":k.1,"before_line":line,"after_line":line,
       "change":if old.is_none(){"added"}else if new.is_none(){"deleted"}else{"modified"},
       "before_id":old.map(|n|n["id"].clone()),"after_id":new.map(|n|n["id"].clone())}));
 }
 let ra:BTreeMap<_,_>=ea.values().map(|e|(relation(e),e)).collect();
 let rb:BTreeMap<_,_>=eb.values().map(|e|(relation(e),e)).collect();
 let added_rel:Vec<_>=rb.iter().filter(|(k,_)|!ra.contains_key(*k)).map(|(_,v)|*v).collect();
 let removed_rel:Vec<_>=ra.iter().filter(|(k,_)|!rb.contains_key(*k)).map(|(_,v)|*v).collect();
 Ok(json!({"before":a["snapshot_id"],"after":b["snapshot_id"],"changed_files":changes,
   "relationship_delta":{"added":added_rel,"removed":removed_rel},
   "edge_evidence_updates":ra.iter().filter(|(k,v)|rb.get(*k).is_some_and(|w|w!=*v)).count(),
   "added_nodes":nb.iter().filter(|(k,_)|!na.contains_key(*k)).map(|(_,v)|v).collect::<Vec<_>>(),
   "removed_nodes":na.iter().filter(|(k,_)|!nb.contains_key(*k)).map(|(_,v)|v).collect::<Vec<_>>(),
   "modified_nodes":na.iter().filter_map(|(k,v)|nb.get(k).filter(|w|*w!=v).map(|w|json!({"before":v,"after":w}))).collect::<Vec<_>>(),
   "added_edges":eb.iter().filter(|(k,_)|!ea.contains_key(*k)).map(|(_,v)|v).collect::<Vec<_>>(),
   "removed_edges":ea.iter().filter(|(k,_)|!eb.contains_key(*k)).map(|(_,v)|v).collect::<Vec<_>>(),
   "note":"Text edits seed conservative file impact; removing an edge is not itself a failure."}))
}

fn adjacency(g:&Value,incoming:bool)->Result<BTreeMap<String,Vec<Value>>>{
 let mut map:BTreeMap<String,Vec<Value>>=BTreeMap::new();
 for e in array(g,"edges")?.iter().filter(|e|dependency(e)){
    let from=if incoming{"target"}else{"source"};
    map.entry(field(e,from)?.into()).or_default().push(e.clone());
 }
 for v in map.values_mut(){v.sort_by_key(|e|e["id"].as_str().unwrap_or("").to_owned());}Ok(map)
}

pub fn why(s:&MeshStore,snapshot:&str,source:&str,target:&str,hops:usize,limit:usize)->Result<Value>{
 limits(hops,limit)?;let g=s.graph(snapshot)?;let start=select(&g,source)?;let end=select(&g,target)?;
 let adj=adjacency(&g,false)?;let start_id=field(&start,"id")?.to_string();
 let mut visited=BTreeSet::from([start_id.clone()]);let mut q=VecDeque::from([(start_id,Vec::<Value>::new())]);let mut truncated=false;
 while let Some((node,path))=q.pop_front(){
    if node==field(&end,"id")?{return Ok(json!({"snapshot_id":g["snapshot_id"],"found":true,"source":start,"target":end,"path":path,"direction":"dependent_to_dependency","truncated":false,"claim":"Recorded dependency path, not proof of a runtime failure."}));}
    if let Some(edges)=adj.get(&node){
       if path.len()>=hops{if !edges.is_empty(){truncated=true;}continue;}
       for e in edges{let next=field(e,"target")?.to_string();if visited.contains(&next){continue;}
          if visited.len()>=limit{truncated=true;continue;}
          visited.insert(next.clone());let mut p=path.clone();p.push(e.clone());q.push_back((next,p));
       }
    }
 }
 Ok(json!({"snapshot_id":g["snapshot_id"],"found":false,"path":[],"truncated":truncated,"unknown":true,"claim":"No path found in the modeled, bounded graph; independence is not established."}))
}

pub fn impact(s:&MeshStore,before:&str,after:&str,hops:usize,limit:usize)->Result<Value>{
 limits(hops,limit)?;let mut delta=diff(s,before,after)?;
 let mut changed:BTreeSet<_>=array(&delta,"changed_files")?.iter().map(key).collect();
 let a=s.graph(field(&delta,"before")?)?;let b=s.graph(field(&delta,"after")?)?;
 if a["manifest"]["bindings"]!=b["manifest"]["bindings"]{
    for g in [&a,&b]{if let Some(bs)=g["manifest"]["bindings"].as_array(){for binding in bs{for endpoint in ["consumer","producer"]{
       let k=key(&binding[endpoint]);if changed.insert(k.clone()){delta["changed_files"].as_array_mut().unwrap().push(json!({"repo":k.0,"path":k.1,"change":"binding_changed","before_id":null,"after_id":null}));}
    }}}}
 }
 let mut results:BTreeMap<(String,String),Value>=BTreeMap::new();let mut truncated=false;
 for g in [&a,&b]{
    let nodes=keyed(array(g,"nodes")?);let incoming=adjacency(g,true)?;
    let seeds:BTreeSet<String>=nodes.values().filter(|n|changed.contains(&key(n))&&n["kind"]!="callsite").map(|n|n["id"].as_str().unwrap().into()).collect();
    let mut visited=seeds.clone();let mut q:VecDeque<_>=seeds.iter().map(|n|(n.clone(),Vec::<Value>::new())).collect();
    let budget=std::cmp::max(500,limit*20);
    while let Some((target,path))=q.pop_front(){
       if let Some(es)=incoming.get(&target){
          if path.len()>=hops{if !es.is_empty(){truncated=true;}continue;}
          for e in es{
             let source=field(e,"source")?.to_owned();if visited.contains(&source){continue;}
             if visited.len().saturating_sub(seeds.len())>=budget{truncated=true;continue;}
             visited.insert(source.clone());let mut p=path.clone();p.push(e.clone());q.push_back((source.clone(),p.clone()));
             let n=nodes.get(&source).ok_or_else(||anyhow::anyhow!("Dangling dependency source"))?;let k=key(n);
             if changed.contains(&k)||n["repo"]=="@contracts"{continue;}
             let candidate=json!({"repo":k.0,"path":k.1,"node":n,"hops":p.len(),"evidence_snapshot":g["snapshot_id"],"dependency_path":p,"classification":"potentially_affected"});
             let better=results.get(&k).is_none_or(|old|(candidate["hops"].as_u64(),candidate["evidence_snapshot"].as_str())<(old["hops"].as_u64(),old["evidence_snapshot"].as_str()));
             if better{results.insert(k,candidate);}
          }
       }
    }
 }
 let mut ordered:Vec<_>=results.into_values().collect();ordered.sort_by_key(|v|(v["hops"].as_u64(),key(v)));
 truncated|=ordered.len()>limit;ordered.truncate(limit);
 Ok(json!({"before":delta["before"],"after":delta["after"],"changed_files":delta["changed_files"],"candidates":ordered,
   "truncated":truncated,"analysis_complete":false,"deep_dependence":false,
   "claim":"Potential static impact only. Unresolved calls, excluded files and missing bindings may hide consumers."}))
}
