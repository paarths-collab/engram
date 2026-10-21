//! End-to-end: index generations. Every change to the indexed facts is
//! published as one generation (a counter in the cache database, bumped in the
//! same transaction as the change), and every engine-backed response says which
//! generation it was answered at. Each test drives the compiled `engram` binary
//! over stdio against a real git fixture and looks at the cache it leaves.

mod common;

use common::*;
use engram_repo_map::store::Store;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

/// The generation recorded in the fixture's cache database right now.
fn stored_generation(fx: &Fixture) -> u64 {
    Store::open(&fx.root)
        .expect("open the cache")
        .generation()
        .expect("read the generation")
}

/// The `generation` field of an engine-backed response.
fn generation_of(payload: &Value) -> u64 {
    payload["generation"]
        .as_u64()
        .unwrap_or_else(|| panic!("response has no integer generation: {payload}"))
}

/// Start a server, wait until it is ready, and return the generation its first
/// answer carried. The server is stopped before returning.
fn run_once(fx: &Fixture) -> u64 {
    let mut client = McpClient::spawn(&fx.root);
    client.initialize().expect("initialize");
    let ready = wait_ready(&mut client);
    let generation = generation_of(ready.output.payload());
    client.stop();
    generation
}

/// A restart after offline edits (a rename, a removed import, a deleted file,
/// a renamed file) reconciles the cache as ONE generation, and by the time the
/// server answers, the stale facts are already gone from it.
#[test]
fn restart_after_offline_edits_is_exactly_one_generation_and_the_old_facts_are_gone() {
    if skip_without_git() {
        return;
    }
    let fx = Fixture::standard("generations-restart");

    let first = run_once(&fx);
    let before = stored_generation(&fx);
    assert!(first >= 1, "the first index is a generation: {first}");
    assert_eq!(
        before, first,
        "nothing published after the answer that was served"
    );

    // Offline, with the server stopped: the same edits acceptance 3 makes.
    let service = fx.read(SERVICE).replace("def execute(", "def run_execute(");
    fx.write(SERVICE, &service);
    let api = fx
        .read(API)
        .replace("from app.service import execute\n", "import json\n")
        .replace("return execute(payload)", "return json.dumps(payload)");
    fx.write(API, &api);
    fx.remove(AUDIT);
    let renamed_test = "tests/test_run_service.py";
    let test_body = fx.read(TEST_SERVICE).replace("execute", "run_execute");
    fx.remove(TEST_SERVICE);
    fx.write(renamed_test, &test_body);

    let mut client = McpClient::spawn(&fx.root);
    client.initialize().expect("initialize");
    let ready = wait_ready(&mut client);
    let payload = ready.output.payload();

    // The reconcile of every edit, removal and addition above was one publish.
    assert_eq!(
        generation_of(payload),
        before + 1,
        "{}",
        client.stderr_text()
    );
    assert_eq!(stored_generation(&fx), before + 1, "and nothing else did");

    // The cache itself, read directly: the old facts are gone, the new are in.
    let store = Store::open(&fx.root).expect("open the cache");
    assert!(
        store.symbols_exact("execute", 5).unwrap().is_empty(),
        "the old symbol survived the restart"
    );
    assert!(
        !store.symbols_exact("run_execute", 5).unwrap().is_empty(),
        "the new symbol is missing"
    );
    let paths: Vec<String> = store
        .all_files()
        .unwrap()
        .into_iter()
        .map(|f| f.path)
        .collect();
    for gone in [AUDIT, TEST_SERVICE] {
        assert!(!paths.contains(&gone.to_owned()), "{gone} is still indexed");
    }
    assert!(paths.contains(&renamed_test.to_owned()));
    assert!(
        !store
            .all_imports()
            .unwrap()
            .iter()
            .any(|(importer, target)| importer == API && target.contains("service")),
        "the removed api -> service import edge survived the restart"
    );

    // And the answer is of that generation, plainly current.
    let explain = client.ok(
        "explain_connection",
        json!({ "source": API, "target": SERVICE }),
    );
    assert_eq!(generation_of(&explain), before + 1);
    assert!(explain.get("index_state").is_none(), "{explain}");
    client.stop();
}

/// A restart that finds nothing changed must not invent a generation.
#[test]
fn restart_without_edits_does_not_move_the_generation() {
    if skip_without_git() {
        return;
    }
    let fx = Fixture::standard("generations-steady");
    let first = run_once(&fx);
    let second = run_once(&fx);
    let third = run_once(&fx);
    assert_eq!(first, second);
    assert_eq!(second, third);
    assert_eq!(stored_generation(&fx), first);
}

/// Every tool served from the engine reports the generation next to the
/// snapshot; a live edit published by the watcher moves it forward.
#[test]
fn engine_backed_tools_report_the_generation_and_a_live_edit_moves_it() {
    if skip_without_git() {
        return;
    }
    let mut s = Session::start("generations-live");
    let g0 = generation_of(s.ready.output.payload());
    assert_eq!(stored_generation(&s.fx), g0);

    let calls: Vec<(&str, Value)> = vec![
        ("search_context", json!({ "query": "execute order" })),
        ("get_task_context", json!({ "task": "execute order" })),
        (
            "find_existing_implementation",
            json!({ "concept": "execute" }),
        ),
        ("predict_impact", json!({ "task": "execute order" })),
        ("expand_connections", json!({ "paths": [SERVICE] })),
        ("find_connected_files", json!({ "files": [SERVICE] })),
        (
            "explain_connection",
            json!({ "source": API, "target": SERVICE }),
        ),
        ("get_graph", json!({ "paths": [SERVICE] })),
    ];
    for (tool, args) in &calls {
        let r = s.client.ok(tool, args.clone());
        assert_eq!(generation_of(&r), g0, "{tool}");
        assert!(r["snapshot"].is_string(), "{tool}");
    }

    // A save while the server runs: the watcher stages and publishes it, and
    // the next call rebuilds and answers at the newer generation.
    let edited =
        s.fx.read(SERVICE)
            .replace("def execute(", "def run_execute(");
    s.fx.write(SERVICE, &edited);
    let deadline = Instant::now() + Duration::from_secs(60);
    let moved = loop {
        let r = s.client.ok(
            "explain_connection",
            json!({ "source": API, "target": SERVICE }),
        );
        if generation_of(&r) > g0 {
            break r;
        }
        assert!(
            Instant::now() < deadline,
            "the edit never produced a new generation; stderr:\n{}",
            s.client.stderr_text()
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    // A burst of events is one publish; at most a handful even if the editor's
    // write was split across debounce windows.
    assert!(generation_of(&moved) <= g0 + 3, "{moved}");
    assert!(generation_of(&moved) <= stored_generation(&s.fx));
}
