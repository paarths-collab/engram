//! End-to-end: `get_graph` over a real MCP session against the standard git
//! fixture. The tool exposes raw graph facts with citable `edge_id`s; these
//! tests check that the ids it emits are the very ids `explain_connection`
//! reports and `expand_connections` cites, and that its disclosure is honest.
//!
//! Set `ENGRAM_E2E_GRAPH_OUT=<path>` to write the `get_graph` payload of the
//! first test as sanitized JSON (used for the documentation example).

mod common;

use common::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// The edge of `kind` from `source` to `target`, failing with the payload.
fn edge<'a>(graph: &'a Value, kind: &str, source: &str, target: &str) -> &'a Value {
    graph["edges"]
        .as_array()
        .expect("edges array")
        .iter()
        .find(|e| e["kind"] == kind && e["source"] == source && e["target"] == target)
        .unwrap_or_else(|| {
            panic!(
                "no {kind} edge {source} -> {target} in {}",
                serde_json::to_string_pretty(&graph["edges"]).unwrap()
            )
        })
}

fn node<'a>(graph: &'a Value, path: &str) -> &'a Value {
    graph["nodes"]
        .as_array()
        .expect("nodes array")
        .iter()
        .find(|n| n["path"] == path)
        .unwrap_or_else(|| panic!("no node {path} in {}", graph["nodes"]))
}

fn edge_ids(graph: &Value) -> BTreeSet<String> {
    graph["edges"]
        .as_array()
        .expect("edges array")
        .iter()
        .map(|e| e["edge_id"].as_str().expect("edge_id").to_owned())
        .collect()
}

#[test]
fn get_graph_returns_the_api_service_facts_under_the_ids_explain_reports() {
    if skip_without_git() {
        return;
    }
    let mut s = Session::start("graph");

    // 1. It is advertised, as a supporting tool, without a budget.
    let tools = s.client.list_tools().expect("tools/list");
    let tool = tools["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .find(|t| t["name"] == "get_graph")
        .unwrap_or_else(|| panic!("get_graph missing from tools/list: {tools}"));
    assert!(tool["description"]
        .as_str()
        .is_some_and(|d| d.starts_with("SUPPORTING")));
    assert!(tool["inputSchema"]["properties"]
        .get("budget_tokens")
        .is_none());

    // 2. The facts around the service.
    let graph = s.client.ok("get_graph", json!({ "paths": [SERVICE] }));
    assert_eq!(snapshot_of(&graph), snapshot_of(s.ready.output.payload()));
    assert_eq!(graph["anchors"], json!([SERVICE]));
    assert!(graph.get("snapshot_mismatch").is_none(), "{graph}");
    assert_eq!(node(&graph, SERVICE)["role"], "anchor");
    assert_eq!(node(&graph, API)["role"], "neighbor");
    assert_eq!(node(&graph, API)["language"], "python");
    assert_eq!(node(&graph, TEST_SERVICE)["is_test"], true);

    // The import: api imports service (importer is the source).
    let import = edge(&graph, "import", API, SERVICE);
    assert_eq!(import["direction"], "source_imports_target");
    assert_eq!(import["hops"], 1);
    assert_eq!(import["resolver"], "heuristic_module_segments");
    assert_eq!(import["resolution_status"], "candidate");
    // The import runs one way only: nothing records service importing api.
    assert!(graph["edges"]
        .as_array()
        .unwrap()
        .iter()
        .all(|e| !(e["kind"] == "import" && e["source"] == SERVICE && e["target"] == API)));

    // The co-change pair, as the two stored rows it is.
    let forward = edge(&graph, "cochange", API, SERVICE);
    let backward = edge(&graph, "cochange", SERVICE, API);
    for row in [forward, backward] {
        assert_eq!(row["direction"], "source_to_target");
        assert!(row["count"].as_u64().unwrap() >= 2, "{row}");
        assert!(row["strength"].as_f64().unwrap() > 0.0, "{row}");
        assert!(row["window"]["max_commits"].as_u64().is_some());
        assert!(row["opportunities_source"].as_u64().is_some(), "{row}");
        assert!(row["opportunities_target"].as_u64().is_some(), "{row}");
    }
    assert_ne!(forward["edge_id"], backward["edge_id"]);

    // 3. Not capped like the recommendations: repo.py is the service's sixth or
    // seventh co-change partner, absent from expand_connections' top five.
    let expand = s
        .client
        .ok("expand_connections", json!({ "paths": [SERVICE] }));
    assert!(!paths_in(&expand, "cochange_expansions")
        .iter()
        .any(|p| p == REPO));
    edge(&graph, "cochange", SERVICE, REPO);
    assert_eq!(graph["limits"]["truncated"], false);
    assert_eq!(graph["limits"]["omitted_edges"], 0);
    assert_eq!(graph["limits"]["omitted_nodes"], 0);

    // 4. The ids are the ones explain_connection reports for the same pair.
    let explain = s.client.ok(
        "explain_connection",
        json!({ "source": API, "target": SERVICE }),
    );
    let import_reason = reasons_of(&explain, "import_edge")[0];
    assert_eq!(import_reason["edge_id"], import["edge_id"]);
    assert_eq!(import_reason["edges"][0]["edge_id"], import["edge_id"]);
    let cochange_reasons = reasons_of(&explain, "historical_cochange");
    let reason_id = |direction: &str| -> Value {
        cochange_reasons
            .iter()
            .find(|r| r["direction"] == direction)
            .unwrap_or_else(|| panic!("no {direction} co-change in {explain}"))["edge_id"]
            .clone()
    };
    assert_eq!(reason_id("source_to_target"), forward["edge_id"]);
    assert_eq!(reason_id("target_to_source"), backward["edge_id"]);

    // A two-hop route: each hop is individually citeable and is a graph edge.
    let route = s.client.ok(
        "explain_connection",
        json!({ "source": API, "target": REPO }),
    );
    let route = reasons_of(&route, "import_edge")[0];
    let hops = route["edges"].as_array().unwrap();
    assert_eq!(hops.len(), 2);
    let wide = s
        .client
        .ok("get_graph", json!({ "paths": [API], "max_hops": 2 }));
    for hop in hops {
        let from = hop["from"].as_str().unwrap();
        let to = hop["to"].as_str().unwrap();
        assert_eq!(edge(&wide, "import", from, to)["edge_id"], hop["edge_id"]);
    }
    assert_ne!(route["edge_id"], hops[0]["edge_id"]);
    assert!(!edge_ids(&wide).contains(route["edge_id"].as_str().unwrap()));

    // 5. Everything the recommendations cite is a fact get_graph emits, at
    // its default arguments (nothing cut, and as deep as the expansion).
    let emitted = edge_ids(&s.client.ok("get_graph", json!({ "paths": [SERVICE] })));
    let mut cited = 0;
    for key in ["cochange_expansions", "import_expansions"] {
        for entry in expand[key].as_array().unwrap() {
            let ids = entry["evidence_ids"]
                .as_array()
                .unwrap_or_else(|| panic!("{key} entry without evidence_ids: {entry}"));
            assert!(!ids.is_empty(), "{entry}");
            assert!(entry["score_components"]["basis"].is_string(), "{entry}");
            for id in ids {
                cited += 1;
                assert!(emitted.contains(id.as_str().unwrap()), "{id} of {entry}");
            }
        }
    }
    assert!(
        cited >= 5,
        "at least the 5 co-change rows are cited: {cited}"
    );

    // 6. Same question, same bytes.
    let again = s.client.ok("get_graph", json!({ "paths": [SERVICE] }));
    assert_eq!(again, graph);

    if let Ok(path) = std::env::var("ENGRAM_E2E_GRAPH_OUT") {
        let text = serde_json::to_string_pretty(&graph).unwrap();
        std::fs::write(path, s.fx.sanitize(&text)).expect("write graph sample");
    }
}

#[test]
fn get_graph_rejects_an_unknown_path_and_flags_a_foreign_snapshot() {
    if skip_without_git() {
        return;
    }
    let mut s = Session::start("graph-errors");

    let out = s
        .client
        .call_tool("get_graph", json!({ "paths": [SERVICE, "app/nope.py"] }))
        .expect("call");
    assert!(out.is_error, "{}", out.text);
    assert!(
        out.text.contains("app/nope.py") && out.text.contains("not an indexed text file"),
        "{}",
        out.text
    );

    let current = snapshot_of(s.ready.output.payload());
    let graph = s.client.ok(
        "get_graph",
        json!({ "paths": [SERVICE], "kinds": ["cochange"], "snapshot": "snap-from-yesterday" }),
    );
    assert_eq!(
        graph["snapshot_mismatch"],
        json!({ "requested": "snap-from-yesterday", "current": current })
    );
    assert_eq!(graph["snapshot"], current.as_str());
    // Only co-change edges were asked for, and the scope says so.
    assert_eq!(graph["scope"]["kinds"], json!(["cochange"]));
    assert!(graph["edges"]
        .as_array()
        .unwrap()
        .iter()
        .all(|e| e["kind"] == "cochange"));
}
