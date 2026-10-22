//! End-to-end: how the server behaves across restarts, what identifies an
//! index, and how it reports a repository it cannot index. Each test drives the
//! compiled `engram` binary over stdio against real git fixtures.

mod common;

use common::*;
use serde_json::json;
use std::time::Duration;

/// Start a server on `fx`, wait until it is ready, and return the snapshot id
/// its first answer carried. The server is stopped before returning.
fn snapshot_for(fx: &Fixture) -> String {
    let mut client = McpClient::spawn(&fx.root);
    client.initialize().expect("initialize");
    let ready = wait_ready(&mut client);
    let id = snapshot_of(ready.output.payload());
    client.stop();
    id
}

/// Acceptance 3: restart freshness.
///
/// The server caches its index in `<repo>/.engram`. Files edited, removed, or
/// renamed while it is stopped must not leak stale symbols, edges, or paths
/// from that cache into the next run.
#[test]
fn acceptance_3_restart_drops_stale_symbols_edges_and_files() {
    if skip_without_git() {
        return;
    }
    let fx = Fixture::standard("restart");

    // Run 1: the fixture as committed.
    let first_snapshot = {
        let mut client = McpClient::spawn(&fx.root);
        client.initialize().expect("initialize");
        let ready = wait_ready(&mut client);
        let snapshot = snapshot_of(ready.output.payload());

        let search = client.ok("search_context", json!({ "query": "execute order" }));
        assert!(
            search["evidence"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["path"] == SERVICE && p["symbol"] == "execute"),
            "run 1 should find execute in {SERVICE}: {}",
            search["evidence"]
        );
        let explain = client.ok(
            "explain_connection",
            json!({ "source": API, "target": SERVICE }),
        );
        assert!(
            !reasons_of(&explain, "import_edge").is_empty(),
            "run 1: api imports service: {explain}"
        );
        client.stop();
        snapshot
    };

    // Offline edits, with the server stopped.
    // 1. Rename a function.
    let service = fx.read(SERVICE).replace("def execute(", "def run_execute(");
    fx.write(SERVICE, &service);
    // 2. Remove an import (api no longer touches the service at all).
    let api = fx
        .read(API)
        .replace("from app.service import execute\n", "import json\n")
        .replace("return execute(payload)", "return json.dumps(payload)");
    fx.write(API, &api);
    // 3. Delete a file.
    fx.remove(AUDIT);
    // 4. Rename a file (and update what it calls).
    let renamed_test = "tests/test_run_service.py";
    let test_body = fx.read(TEST_SERVICE).replace("execute", "run_execute");
    fx.remove(TEST_SERVICE);
    fx.write(renamed_test, &test_body);

    // Run 2: restart over the same cache directory.
    let mut client = McpClient::spawn(&fx.root);
    client.initialize().expect("initialize");
    let ready = wait_ready(&mut client);
    let second_snapshot = snapshot_of(ready.output.payload());
    assert_ne!(
        first_snapshot, second_snapshot,
        "the snapshot id must change when the indexed content changed"
    );

    // The old symbol is gone, the new one is found.
    let search = client.ok("search_context", json!({ "query": "execute order" }));
    let symbols: Vec<(&str, &str)> = search["evidence"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|p| Some((p["path"].as_str()?, p["symbol"].as_str()?)))
        .collect();
    assert!(
        !symbols.iter().any(|(_, s)| *s == "execute"),
        "the old symbol `execute` must not be returned as a symbol hit: {symbols:?}"
    );
    let search = client.ok("search_context", json!({ "query": "run_execute" }));
    assert!(
        search["evidence"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["path"] == SERVICE && p["symbol"] == "run_execute"),
        "the new symbol must be found in {SERVICE}: {}",
        search["evidence"]
    );
    for p in search["evidence"].as_array().unwrap() {
        let path = p["path"].as_str().unwrap();
        assert!(
            path != AUDIT && path != TEST_SERVICE,
            "removed path {path} was returned as evidence"
        );
    }

    // The removed import is gone from the graph: api and service are now linked
    // only by history.
    let explain = client.ok(
        "explain_connection",
        json!({ "source": API, "target": SERVICE }),
    );
    assert!(
        reasons_of(&explain, "import_edge").is_empty(),
        "the removed api -> service import is still reported: {explain}"
    );
    assert!(
        !reasons_of(&explain, "historical_cochange").is_empty(),
        "git history is unchanged, so the co-change fact stays: {explain}"
    );

    // The deleted file and the old name of the renamed file are no longer
    // indexed, so asking about them is an error rather than "not connected".
    for gone in [AUDIT, TEST_SERVICE] {
        let out = client
            .call_tool(
                "explain_connection",
                json!({ "source": gone, "target": SERVICE }),
            )
            .expect("call");
        assert!(out.is_error, "{gone} should no longer be indexed");
        assert!(out.text.contains("not an indexed"), "{}", out.text);
    }

    // The renamed file is indexed under its new path, with its import edge.
    let explain = client.ok(
        "explain_connection",
        json!({ "source": renamed_test, "target": SERVICE }),
    );
    let imports = reasons_of(&explain, "import_edge");
    assert_eq!(imports.len(), 1, "{explain}");
    assert_eq!(imports[0]["hops"], 1);

    // Neither removed path leaks into the expansion of an anchor that used to
    // reach them.
    let expand = client
        .call_tool("expand_connections", json!({ "paths": [SERVICE] }))
        .expect("call");
    assert!(!expand.is_error, "{}", expand.text);
    for gone in [AUDIT, TEST_SERVICE] {
        assert!(
            !expand.text.contains(gone),
            "{gone} leaked into expand_connections: {}",
            expand.text
        );
    }
}

/// Acceptance 4: what a snapshot id does and does not depend on.
///
/// (Invalidation by parser/config version is covered by the unit tests in
/// `engram-repo-map`: `snapshot_ids_follow_content_commit_and_version`. It
/// needs a store opened under another version, which a black-box client cannot
/// arrange.)
#[test]
fn acceptance_4_snapshot_identity() {
    if skip_without_git() {
        return;
    }
    let files: &[(&str, &str)] = &[
        ("app/alpha.py", "def alpha():\n    return 1\n"),
        ("app/beta.py", "def beta():\n    return 2\n"),
        ("README.md", "# demo\n"),
    ];

    // Identical content, built twice in different directories: identical id.
    let a1 = Fixture::mini("snap-a1", files);
    let a2 = Fixture::mini("snap-a2", files);
    assert_eq!(
        a1.git(&["rev-parse", "HEAD"], &[]),
        a2.git(&["rev-parse", "HEAD"], &[]),
        "fixture precondition: identical content, message, dates and identities give the same commit"
    );
    let id_a1 = snapshot_for(&a1);
    let id_a2 = snapshot_for(&a2);
    assert_eq!(
        id_a1, id_a2,
        "identical content must give an identical snapshot id regardless of location"
    );

    // The same server run twice over the same repo: the id is stable.
    assert_eq!(
        id_a1,
        snapshot_for(&a1),
        "restarting over an unchanged repo must not change the id"
    );

    // Same paths, same commit message and dates, different content: different id.
    let b = Fixture::mini(
        "snap-b",
        &[
            ("app/alpha.py", "def alpha():\n    return 1\n"),
            ("app/beta.py", "def beta():\n    return 3\n"),
            ("README.md", "# demo\n"),
        ],
    );
    assert_ne!(
        id_a1,
        snapshot_for(&b),
        "different content at the same paths must give a different id"
    );

    // Same commit, but an uncommitted edit: the id follows the content on disk,
    // not just HEAD.
    let dirty = Fixture::mini("snap-dirty", files);
    dirty.write("app/beta.py", "def beta():\n    return 99\n");
    assert_ne!(
        id_a1,
        snapshot_for(&dirty),
        "an uncommitted content change must change the id"
    );

    // Files in ignored directories are not part of the index, so they are not
    // part of its identity.
    let ignored = Fixture::mini("snap-ignored", files);
    ignored.write("vendor/lib.py", "def vendored():\n    return 1\n");
    ignored.write("node_modules/dep/index.js", "module.exports = 1;\n");
    ignored.write(".venv/lib/site.py", "x = 1\n");
    assert_eq!(
        id_a1,
        snapshot_for(&ignored),
        "files under vendor/, node_modules/ and .venv/ must not affect the snapshot id"
    );
}

/// Acceptance 6: an index that cannot be built is reported as a failure with
/// recovery guidance, not as "still indexing, retry".
#[test]
fn index_failure_is_reported_not_retried() {
    if skip_without_git() {
        return;
    }
    let canonical_calls: [(&str, serde_json::Value); 3] = [
        ("search_context", json!({ "query": "alpha" })),
        ("expand_connections", json!({ "paths": ["app/alpha.py"] })),
        (
            "explain_connection",
            json!({ "source": "app/alpha.py", "target": "app/beta.py" }),
        ),
    ];

    let check = |client: &mut McpClient, what: &str| {
        // After a bounded wait the server has settled into the failed state.
        let settled =
            settle(client, Duration::from_secs(30)).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(
            settled.output.is_error,
            "{what}: expected an error, got {}",
            settled.output.text
        );
        for (name, args) in &canonical_calls {
            let out = client
                .call_tool(name, args.clone())
                .unwrap_or_else(|e| panic!("{what}: {name}: {e}"));
            assert!(
                out.is_error,
                "{what}: {name} should be an error: {}",
                out.text
            );
            assert!(
                out.text.contains("restart the Engram MCP server"),
                "{what}: {name} gives no recovery guidance: {}",
                out.text
            );
            assert!(
                out.text.contains("will not help"),
                "{what}: {name} should say that calling again will not help: {}",
                out.text
            );
            assert!(
                !out.text.contains("Retry this tool call"),
                "{what}: {name} wrongly invites a retry: {}",
                out.text
            );
            assert!(
                !out.text.contains("still indexing"),
                "{what}: {name} must not claim indexing is in progress: {}",
                out.text
            );
        }
        // The server itself stays up and keeps answering protocol requests.
        let tools = client.list_tools().expect("tools/list after failure");
        assert!(tools["tools"].as_array().is_some_and(|t| !t.is_empty()));
    };

    // `<repo>/.engram` is a regular file, so the cache directory cannot be
    // created. (Unlike chmod, this fails the same way for root.)
    let blocked = Fixture::mini(
        "blocked",
        &[
            ("app/alpha.py", "def alpha():\n    return 1\n"),
            ("app/beta.py", "def beta():\n    return 2\n"),
        ],
    );
    blocked.write(".engram", "this is a file, not a directory\n");
    {
        let mut client = McpClient::spawn(&blocked.root);
        client.initialize().expect("initialize");
        check(&mut client, ".engram is a file");
        client.stop();
    }

    // A --repo that does not exist.
    let missing =
        std::env::temp_dir().join(format!("engram-e2e-does-not-exist-{}", std::process::id()));
    assert!(!missing.exists());
    let mut client = McpClient::spawn(&missing);
    client.initialize().expect("initialize");
    check(&mut client, "--repo does not exist");
}
