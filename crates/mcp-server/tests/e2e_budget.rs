//! End-to-end: the token budget on `search_context`.
//!
//! Written against the budget contract (`budget_tokens` in, `pack` out) ahead of
//! the change that implements it, so it is ignored until that change is merged.

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

#[test]
#[ignore = "needs pack support from the budget change; run after merge"]
fn acceptance_5_budget_tokens_bounds_search_context() {
    if skip_without_git() {
        return;
    }
    let fx = Fixture::init("budget");
    for n in 0..40 {
        fx.write(&format!("app/handler_{n:02}.py"), &candidate_module(n));
    }
    fx.commit_all("Add many handler modules", 1);

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

    // The estimator names the formula; the default is ceil(bytes / 4).
    let estimator = pack["estimator"].as_str().unwrap_or("");
    let divisor = if estimator.contains("bytes3") { 3 } else { 4 };
    let estimate = first.text.len().div_ceil(divisor);
    assert!(
        estimate <= 2000,
        "estimated {estimate} tokens ({} bytes, estimator {estimator:?}) exceeds the 2000 budget",
        first.text.len()
    );

    // 40 candidate files cannot fit in 2000 tokens, so the pack says it cut.
    assert_eq!(pack["truncated"], true, "{pack}");
    assert!(
        pack["omitted_count"].as_u64().unwrap_or(0) > 0,
        "omitted_count should be positive: {pack}"
    );

    // Packing is deterministic: the same call gives the same bytes.
    let second = client
        .call_tool("search_context", args)
        .expect("search_context again");
    assert_eq!(first.text, second.text, "identical calls must be identical");
}
