//! End-to-end: the token budget (`budget_tokens` in, `pack` out) on
//! `search_context` and `expand_connections`.
//!
//! The size is measured on `content[0].text` exactly as it comes back over stdio.

mod common;

use common::*;
use serde_json::json;

/// A module with a long, query-relevant body, so many files are candidates and
/// the full answer is far larger than the budget.
fn candidate_module(n: usize) -> String {
    let mut body = format!("\"\"\"Handler {n}: process the order batch.\n\n");
    for line in 0..14 {
        body.push_str(&format!(
            "Step {line}: validate, process the order batch, and record the batch result for handler {n}.\n"
        ));
    }
    body.push_str("\"\"\"\n\n\n");
    body.push_str(&format!(
        "def process_order_batch_{n:02}(batch):\n    return [process_one_{n:02}(o) for o in batch]\n\n\n"
    ));
    body.push_str(&format!(
        "def process_one_{n:02}(order):\n    return order\n"
    ));
    body
}

/// Estimated tokens for a tool-result text under the named estimator.
fn estimated_tokens(estimator: &str, text: &str) -> usize {
    assert_eq!(estimator, "bytes3-v1", "unexpected estimator {estimator:?}");
    text.len().div_ceil(3)
}

fn handler_fixture(tag: &str) -> Fixture {
    let fx = Fixture::init(tag);
    for n in 0..40 {
        fx.write(&format!("app/handler_{n:02}.py"), &candidate_module(n));
    }
    fx.commit_all("Add many handler modules", 1);
    fx
}

#[test]
fn acceptance_5_budget_tokens_bounds_search_context() {
    if skip_without_git() {
        return;
    }
    let fx = handler_fixture("budget");

    let mut client = McpClient::spawn(&fx.root);
    client.initialize().expect("initialize");
    wait_ready(&mut client);

    let args = json!({ "query": "process order batch", "budget_tokens": 2000 });
    let first = client
        .call_tool("search_context", args.clone())
        .expect("search_context");
    assert!(!first.is_error, "{}", first.text);
    let pack = &first.payload()["pack"];
    assert!(
        pack.is_object(),
        "search_context returned no `pack` object for budget_tokens: {}",
        first.text
    );

    // Measure the text of the tool result as returned over stdio.
    let estimator = pack["estimator"].as_str().unwrap_or("");
    let estimate = estimated_tokens(estimator, &first.text);
    assert!(
        estimate <= 2000,
        "estimated {estimate} tokens ({} bytes, estimator {estimator:?}) exceeds the 2000 budget",
        first.text.len()
    );
    assert_eq!(pack["budget_tokens"], 2000, "{pack}");

    // 40 candidate files cannot fit in 2000 tokens, so the pack says it cut.
    assert_eq!(pack["truncated"], true, "{pack}");
    assert!(
        pack["omitted_count"].as_u64().unwrap_or(0) > 0,
        "omitted_count should be positive: {pack}"
    );
    let omitted = pack["omitted"].as_array().expect("omitted array");
    assert!(!omitted.is_empty(), "omitted list is empty: {pack}");
    for entry in omitted {
        assert!(
            entry["path"].as_str().is_some_and(|p| !p.is_empty()),
            "omitted entry without a path: {entry}"
        );
    }

    // Packing is deterministic: the same call gives the same bytes.
    let second = client
        .call_tool("search_context", args)
        .expect("search_context again");
    assert_eq!(first.text, second.text, "identical calls must be identical");

    // Without a budget the evidence set is still returned, and the pack says
    // there was no budget.
    let unbudgeted = client.ok("search_context", json!({ "query": "process order batch" }));
    assert!(
        unbudgeted["evidence"]
            .as_array()
            .is_some_and(|e| !e.is_empty()),
        "unbudgeted search returned no evidence: {unbudgeted}"
    );
    assert!(
        unbudgeted["pack"]["budget_tokens"].is_null(),
        "unbudgeted pack should have a null budget_tokens: {}",
        unbudgeted["pack"]
    );
}

#[test]
fn budget_tokens_bounds_expand_connections() {
    if skip_without_git() {
        return;
    }
    let fx = handler_fixture("budget-expand");

    let mut client = McpClient::spawn(&fx.root);
    client.initialize().expect("initialize");
    wait_ready(&mut client);

    let paths: Vec<String> = (0..20).map(|n| format!("app/handler_{n:02}.py")).collect();
    let budget = 600;
    let args = json!({ "paths": paths, "budget_tokens": budget });
    let first = client
        .call_tool("expand_connections", args.clone())
        .expect("expand_connections");
    assert!(!first.is_error, "{}", first.text);
    let pack = &first.payload()["pack"];
    assert!(
        pack.is_object(),
        "expand_connections returned no `pack` object for budget_tokens: {}",
        first.text
    );
    let estimator = pack["estimator"].as_str().unwrap_or("");
    let estimate = estimated_tokens(estimator, &first.text);
    assert!(
        estimate <= budget,
        "estimated {estimate} tokens ({} bytes, estimator {estimator:?}) exceeds the {budget} budget",
        first.text.len()
    );
    assert_eq!(pack["budget_tokens"], budget, "{pack}");

    let second = client
        .call_tool("expand_connections", args)
        .expect("expand_connections again");
    assert_eq!(first.text, second.text, "identical calls must be identical");
}
