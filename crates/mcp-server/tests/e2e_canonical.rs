//! End-to-end: a real MCP client drives the compiled `engram` binary over stdio
//! against a real git fixture. Covers the canonical flow
//! (search_context -> expand_connections -> explain_connection) and the first two
//! acceptance cases (a co-change outside the top-five recommendations, and
//! multi-hop / multi-fact explanations).
//!
//! Set `ENGRAM_E2E_TRANSCRIPT=<path>` to write the canonical run as a sanitized
//! Markdown transcript (`ENGRAM_E2E_COMMIT` names the Engram commit it records).

mod common;

use common::*;
use serde_json::{json, Value};
use std::fmt::Write as _;

const SEARCH_QUERY: &str = "execute order";

/// Find the single reason of a type, failing with the whole payload if absent.
fn only_reason<'a>(explain: &'a Value, ty: &str) -> &'a Value {
    let found = reasons_of(explain, ty);
    assert!(
        !found.is_empty(),
        "expected a {ty} reason in {}",
        serde_json::to_string_pretty(explain).unwrap()
    );
    found[0]
}

#[test]
fn canonical_flow_search_expand_explain() {
    if skip_without_git() {
        return;
    }
    let fx = Fixture::standard("canonical");
    let mut client = McpClient::spawn(&fx.root);

    // 1. Handshake.
    let init = client.initialize().expect("initialize");
    assert_eq!(init["serverInfo"]["name"], "engram");
    assert_eq!(init["protocolVersion"], "2024-11-05");

    // 2. The three canonical tools are advertised.
    let tools = client.list_tools().expect("tools/list");
    let names: Vec<&str> = tools["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    for canonical in ["search_context", "expand_connections", "explain_connection"] {
        assert!(
            names.contains(&canonical),
            "{canonical} missing from tools/list: {names:?}"
        );
    }

    // 3. Cold start: indexing runs in the background, so poll until it is done.
    let ready = wait_ready(&mut client);
    let cold_start = ready.since_spawn;
    let snapshot = snapshot_of(ready.output.payload());

    // 4. search_context finds the service with a symbol hit.
    let search = client.ok("search_context", json!({ "query": SEARCH_QUERY }));
    let evidence = search["evidence"].as_array().expect("evidence array");
    assert!(
        evidence
            .iter()
            .any(|p| p["path"] == SERVICE && p["symbol"] == "execute"),
        "expected a symbol hit for execute in {SERVICE}, got: {}",
        serde_json::to_string_pretty(&search["evidence"]).unwrap()
    );
    assert_eq!(snapshot_of(&search), snapshot, "one index, one snapshot");

    // 5. expand_connections from the service reaches the api.
    let expand = client.ok("expand_connections", json!({ "paths": [SERVICE] }));
    let connected: Vec<String> = ["likely_files", "cochange_expansions", "import_expansions"]
        .iter()
        .flat_map(|k| paths_in(&expand, k))
        .collect();
    assert!(
        connected.iter().any(|p| p == API),
        "{API} should be connected to {SERVICE}: {connected:?}"
    );
    assert!(
        expand["likely_tests"]
            .as_array()
            .is_some_and(|t| t.iter().any(|p| p == TEST_SERVICE)),
        "the service's test should be listed: {}",
        expand["likely_tests"]
    );

    // 6. explain_connection says why, with typed facts.
    let explain = client.ok(
        "explain_connection",
        json!({ "source": API, "target": SERVICE }),
    );
    assert_eq!(explain["connected"], true);
    assert_eq!(explain["status"], "found");
    let import = only_reason(&explain, "import_edge");
    assert_eq!(import["edge_kind"], "import");
    assert_eq!(import["direction"], "source_imports_target");
    assert_eq!(import["hops"], 1);
    assert_eq!(import["chain"], json!([API, SERVICE]));
    assert_eq!(import["resolution"], "heuristic");
    assert_eq!(import["resolution_status"], "candidate");
    let cochange = only_reason(&explain, "historical_cochange");
    assert_eq!(cochange["edge_kind"], "cochange");
    assert!(cochange["count"].as_u64().unwrap() >= 2, "{cochange}");
    assert!(cochange["strength"].as_f64().unwrap() > 0.0, "{cochange}");
    assert!(cochange["window"]["max_commits"].as_u64().is_some());

    // 7. Report timings, and the transcript when asked.
    eprintln!(
        "[e2e] summary: cold start {} ms; calls: {}",
        cold_start.as_millis(),
        client
            .log
            .iter()
            .map(|e| format!("{} {} ms", e.label, e.elapsed.as_millis()))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if let Ok(path) = std::env::var("ENGRAM_E2E_TRANSCRIPT") {
        let commit = std::env::var("ENGRAM_E2E_COMMIT").unwrap_or_else(|_| "unknown".into());
        let doc = render_transcript(&fx, &commit, &ready, &client.log);
        std::fs::write(&path, doc).expect("write transcript");
        eprintln!("[e2e] transcript written");
    }
}

/// Acceptance 1: a connected pair outside the top-five recommendations.
///
/// `expand_connections` is a capped, ranked list (five co-change partners per
/// anchor). `explain_connection` is a direct pairwise lookup. A weaker pair that
/// the list omits must still be explainable.
#[test]
fn acceptance_1_pair_outside_top_five_is_still_explained() {
    if skip_without_git() {
        return;
    }
    let mut s = Session::start("acc1");

    // The list: exactly five co-change partners, and repo.py is not one.
    let expand = s
        .client
        .ok("expand_connections", json!({ "paths": [SERVICE] }));
    let cochange = paths_in(&expand, "cochange_expansions");
    assert_eq!(
        cochange.len(),
        5,
        "the service has more than five partners, so the list is capped at five: {cochange:?}"
    );
    assert!(cochange.iter().any(|p| p == API), "{cochange:?}");
    let everything: Vec<String> = ["likely_files", "cochange_expansions", "import_expansions"]
        .iter()
        .flat_map(|k| paths_in(&expand, k))
        .collect();
    assert!(
        !everything.iter().any(|p| p == REPO),
        "{REPO} must be omitted from the recommendations: {everything:?}"
    );

    // The pairwise lookup: the recorded co-change is there anyway.
    let explain = s.client.ok(
        "explain_connection",
        json!({ "source": SERVICE, "target": REPO }),
    );
    assert_eq!(explain["connected"], true);
    let cochange_facts = reasons_of(&explain, "historical_cochange");
    let forward = cochange_facts
        .iter()
        .find(|r| r["direction"] == "source_to_target")
        .unwrap_or_else(|| panic!("no source_to_target co-change in {explain}"));
    assert!(forward["count"].as_u64().unwrap() >= 2, "{forward}");
    assert_eq!(forward["edge_kind"], "cochange");
    // The same pair also has a real import (service imports repo); that is a
    // separate fact, not a substitute.
    let import = only_reason(&explain, "import_edge");
    assert_eq!(import["direction"], "source_imports_target");
    assert_eq!(import["hops"], 1);
}

/// Acceptance 2: a two-hop import route, and a pair with both an import and a
/// co-change fact.
#[test]
fn acceptance_2_two_hop_route_and_separate_typed_facts() {
    if skip_without_git() {
        return;
    }
    let mut s = Session::start("acc2");

    // api -> service -> repo. api never imports repo directly.
    let explain = s.client.ok(
        "explain_connection",
        json!({ "source": API, "target": REPO }),
    );
    assert_eq!(explain["connected"], true);
    assert_eq!(explain["status"], "found");
    let imports = reasons_of(&explain, "import_edge");
    assert_eq!(imports.len(), 1, "exactly one route, got {explain}");
    let route = imports[0];
    assert_eq!(route["hops"], 2);
    assert_eq!(route["direction"], "source_imports_target");
    assert_eq!(route["chain"], json!([API, SERVICE, REPO]));
    let edges = route["edges"].as_array().expect("edges");
    assert_eq!(edges.len(), 2);
    // Each hop keeps its `from`/`to` and gains the id of that single edge (the
    // one `get_graph` emits for it); the two hops have distinct ids.
    assert_eq!(edges[0]["from"], API);
    assert_eq!(edges[0]["to"], SERVICE);
    assert_eq!(edges[1]["from"], SERVICE);
    assert_eq!(edges[1]["to"], REPO);
    for edge in edges {
        assert_eq!(edge.as_object().map(|o| o.len()), Some(3), "{edge}");
    }
    let hop_ids: Vec<&str> = edges
        .iter()
        .map(|e| e["edge_id"].as_str().expect("hop edge_id"))
        .collect();
    assert!(hop_ids.iter().all(|id| id.starts_with("e_")), "{hop_ids:?}");
    assert_ne!(hop_ids[0], hop_ids[1]);
    assert_eq!(route["resolver"], "heuristic_module_segments");
    // A 2-hop route is weaker evidence than a direct import.
    assert!(route["weight"].as_f64().unwrap() < 0.6, "{route}");

    // api <-> service: a direct import AND a co-change, as separate reasons.
    let explain = s.client.ok(
        "explain_connection",
        json!({ "source": API, "target": SERVICE }),
    );
    let reasons = explain["reasons"].as_array().expect("reasons");
    let types: Vec<&str> = reasons.iter().filter_map(|r| r["type"].as_str()).collect();
    assert!(types.contains(&"import_edge"), "{types:?}");
    assert!(types.contains(&"historical_cochange"), "{types:?}");
    let import = only_reason(&explain, "import_edge");
    let cochange = only_reason(&explain, "historical_cochange");
    assert_eq!(import["edge_kind"], "import");
    assert_eq!(import["hops"], 1);
    assert_eq!(cochange["edge_kind"], "cochange");
    assert!(cochange["count"].as_u64().unwrap() >= 2);
    // Neither fact is folded into the other.
    assert!(import.get("count").is_none(), "{import}");
    assert!(cochange.get("chain").is_none(), "{cochange}");
}

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------

fn render_transcript(fx: &Fixture, commit: &str, ready: &Settled, log: &[Exchange]) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Engram demo transcript\n");
    let _ = writeln!(
        md,
        "Recorded fixture output: this is not mocked. It is the captured JSON-RPC session of a \
         real MCP client (`crates/mcp-server/tests/e2e_canonical.rs`, test \
         `canonical_flow_search_expand_explain`) driving the compiled `engram` binary over stdio \
         against a small real git repository built for the test.\n"
    );
    let _ = writeln!(md, "- Engram commit: `{commit}`");
    let _ = writeln!(
        md,
        "- Fixture: a Python repository with {} commits (fixed identities and dates)",
        HISTORY.len()
    );
    let _ = writeln!(
        md,
        "- Regenerate: run the test with `ENGRAM_E2E_TRANSCRIPT=<path>` and \
         `ENGRAM_E2E_COMMIT=<short sha>` set. Timings are from one run on a shared machine \
         (a Docker container), not a benchmark.\n"
    );

    let _ = writeln!(md, "## Fixture history\n");
    let _ = writeln!(md, "| # | Commit | Files |");
    let _ = writeln!(md, "|---|---|---|");
    for (i, (message, files)) in HISTORY.iter().enumerate() {
        let _ = writeln!(md, "| {} | {} | `{}` |", i + 1, message, files.join("`, `"));
    }
    let _ = writeln!(
        md,
        "\n`app/api.py` imports `app/service.py`, which imports `app/repo.py`; `app/api.py` does \
         not import `app/repo.py`. `tests/test_service.py` imports `app/service.py`.\n"
    );

    let _ = writeln!(md, "## Timings\n");
    let _ = writeln!(md, "| Step | Time |");
    let _ = writeln!(md, "|---|---|");
    let _ = writeln!(
        md,
        "| Cold start: process spawn to first answer (background index + first engine build, {} \
         poll{}) | {} ms |",
        ready.polls,
        if ready.polls == 1 { "" } else { "s" },
        ready.since_spawn.as_millis()
    );
    for e in log {
        let _ = writeln!(md, "| {} | {} ms |", e.label, e.elapsed.as_millis());
    }

    let _ = writeln!(md, "\n## Session\n");
    let _ = writeln!(
        md,
        "Requests are shown as the `params` of the JSON-RPC call. For `tools/call`, the response \
         is the tool's JSON payload exactly as returned in `result.content[0].text`.\n"
    );
    for (n, e) in log.iter().enumerate() {
        let _ = writeln!(md, "### {}. {}\n", n + 1, e.label);
        let _ = writeln!(
            md,
            "`{}`{}\n",
            e.method,
            if e.is_error { " (isError)" } else { "" }
        );
        let _ = writeln!(md, "Request:\n\n```json\n{}\n```\n", pretty(fx, &e.request));
        let _ = writeln!(
            md,
            "Response:\n\n```json\n{}\n```\n",
            pretty(fx, &e.response)
        );
    }
    md
}

fn pretty(fx: &Fixture, value: &Value) -> String {
    fx.sanitize(&serde_json::to_string_pretty(value).unwrap_or_default())
}
