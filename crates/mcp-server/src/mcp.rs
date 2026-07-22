//! Minimal MCP (Model Context Protocol) stdio server — hand-rolled JSON-RPC 2.0.
//! No SDK dependency: MCP over stdio is newline-delimited JSON-RPC, nothing more.
//! Works with Claude Code, Cursor, Codex CLI, and any MCP-compatible client.

use serde_json::{json, Value};
use std::io::{BufRead, Write};

pub trait ToolHandler {
    /// Return the tools/list payload (tool descriptions are the steering surface).
    fn list_tools(&self) -> Value;
    /// Execute a tool call; return content value or error string.
    fn call_tool(&mut self, name: &str, args: &Value) -> Result<Value, String>;
}

pub fn serve(handler: &mut dyn ToolHandler) {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    serve_io(handler, stdin.lock(), stdout.lock());
}

/// The protocol loop over arbitrary streams.
///
/// [`serve`] binds this to stdio; tests bind it to in-memory buffers. Keeping
/// the loop generic is the only reason it can be tested at all — stdin and
/// stdout are process-global and cannot be driven from a unit test.
pub fn serve_io<R: BufRead, W: Write>(handler: &mut dyn ToolHandler, reader: R, mut writer: W) {
    for line in reader.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");

        let response = match method {
            "initialize" => Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "engram", "version": env!("CARGO_PKG_VERSION") }
            })),
            "notifications/initialized" => None, // notification, no reply
            "tools/list" => Some(handler.list_tools()),
            "tools/call" => {
                let name = msg
                    .pointer("/params/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let default_args = json!({});
                let args = msg.pointer("/params/arguments").unwrap_or(&default_args);
                Some(call_tool_caught(handler, name, args))
            }
            "ping" => Some(json!({})),
            _ => {
                if id.is_some() {
                    // unknown request → JSON-RPC error
                    let err = json!({
                        "jsonrpc": "2.0", "id": id,
                        "error": { "code": -32601, "message": format!("method not found: {method}") }
                    });
                    let _ = writeln!(writer, "{err}");
                    let _ = writer.flush();
                }
                continue;
            }
        };

        // Only requests (with id) get responses; notifications don't.
        if let (Some(id), Some(result)) = (id, response) {
            let reply = json!({ "jsonrpc": "2.0", "id": id, "result": result });
            let _ = writeln!(writer, "{reply}");
            let _ = writer.flush();
        }
    }
}

/// Run a tool call, converting both errors and panics into an `isError` result.
///
/// Without this a panic anywhere in a tool unwinds out of the serve loop and
/// kills the process, which the client sees as the stdio pipe closing with no
/// explanation — every later call in the session fails too. One bad query
/// should cost one bad answer, not the server.
///
/// `AssertUnwindSafe` is a real assertion, not a formality: the handler owns an
/// index and a SQLite connection, and a panic mid-mutation can leave them
/// inconsistent. That is still strictly better than terminating, because the
/// next call rebuilds what it needs and the client stays connected.
fn call_tool_caught(handler: &mut dyn ToolHandler, name: &str, args: &Value) -> Value {
    install_panic_capture();
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handler.call_tool(name, args)
    }));
    match attempt {
        Ok(Ok(result)) => json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string_pretty(&result).unwrap_or_default()
            }],
            "isError": false
        }),
        Ok(Err(message)) => json!({
            "content": [{ "type": "text", "text": message }],
            "isError": true
        }),
        Err(payload) => {
            let detail = take_captured_panic()
                .or_else(|| downcast_panic_detail(&payload))
                .unwrap_or_else(|| "unknown panic".to_owned());
            eprintln!("[engram] tool '{name}' panicked: {detail}");
            json!({
                "content": [{
                    "type": "text",
                    "text": format!("internal error: tool '{name}' panicked: {detail}")
                }],
                "isError": true
            })
        }
    }
}

thread_local! {
    /// Message of the most recent panic on this thread, recorded by the hook
    /// below and consumed by [`call_tool_caught`].
    static LAST_PANIC: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

static PANIC_CAPTURE: std::sync::Once = std::sync::Once::new();

/// Record every panic's message where the catch site can read it.
///
/// Downcasting the `catch_unwind` payload is not stable across Rust versions:
/// on 1.97 `panic!("literal")` no longer hands back a `&'static str`, so the
/// obvious `downcast_ref::<&str>()` silently yields nothing and the operator
/// loses the one detail that explains the failure. `payload_as_str` on the
/// hook side is the supported accessor and keeps working.
///
/// The previous hook still runs, so normal panic output is unchanged. Storage
/// is thread-local, so concurrent tool calls cannot read each other's message.
fn install_panic_capture() {
    PANIC_CAPTURE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let message = info
                .payload_as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| info.to_string());
            LAST_PANIC.with(|slot| *slot.borrow_mut() = Some(message));
            previous(info);
        }));
    });
}

fn take_captured_panic() -> Option<String> {
    LAST_PANIC.with(|slot| slot.borrow_mut().take())
}

/// Fallback for payloads that predate or bypass the hook.
fn downcast_panic_detail(payload: &(dyn std::any::Any + Send)) -> Option<String> {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeTools;

    impl ToolHandler for FakeTools {
        fn list_tools(&self) -> Value {
            json!({ "tools": [{ "name": "echo" }] })
        }

        fn call_tool(&mut self, name: &str, args: &Value) -> Result<Value, String> {
            match name {
                "echo" => Ok(json!({ "echoed": args })),
                "boom" => panic!("tool exploded"),
                other => Err(format!("unknown tool: {other}")),
            }
        }
    }

    /// Drive the real protocol loop over in-memory streams and parse every
    /// line it wrote back into JSON.
    fn exchange(input: &str) -> Vec<Value> {
        let mut handler = FakeTools;
        let mut out: Vec<u8> = Vec::new();
        serve_io(
            &mut handler,
            std::io::Cursor::new(input.to_owned()),
            &mut out,
        );
        String::from_utf8(out)
            .expect("responses are utf8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("each response is one JSON object"))
            .collect()
    }

    fn request(id: u32, method: &str, params: Value) -> String {
        json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }).to_string()
    }

    /// The text payload of a tools/call result, parsed back into JSON.
    fn content_json(response: &Value) -> Value {
        let text = response["result"]["content"][0]["text"]
            .as_str()
            .expect("content text");
        serde_json::from_str(text).expect("content text is JSON")
    }

    #[test]
    fn initialize_echoes_the_request_id_and_announces_tools() {
        let responses = exchange(&request(1, "initialize", json!({})));
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["jsonrpc"], "2.0");
        assert_eq!(responses[0]["id"], 1);
        assert_eq!(responses[0]["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(responses[0]["result"]["serverInfo"]["name"], "engram");
        assert!(responses[0]["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_returns_the_handler_payload() {
        let responses = exchange(&request(2, "tools/list", json!({})));
        assert_eq!(responses[0]["result"]["tools"][0]["name"], "echo");
    }

    #[test]
    fn tools_call_wraps_success_as_non_error_content() {
        let responses = exchange(&request(
            3,
            "tools/call",
            json!({ "name": "echo", "arguments": { "hello": "world" } }),
        ));
        assert_eq!(responses[0]["result"]["isError"], false);
        assert_eq!(content_json(&responses[0])["echoed"]["hello"], "world");
    }

    #[test]
    fn tools_call_reports_handler_errors_as_is_error() {
        let responses = exchange(&request(4, "tools/call", json!({ "name": "nope" })));
        assert_eq!(responses[0]["result"]["isError"], true);
        assert!(responses[0]["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("unknown tool: nope"));
    }

    #[test]
    fn missing_arguments_default_to_an_empty_object() {
        let responses = exchange(&request(5, "tools/call", json!({ "name": "echo" })));
        assert_eq!(responses[0]["result"]["isError"], false);
        assert_eq!(content_json(&responses[0])["echoed"], json!({}));
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let responses = exchange(&request(6, "resources/list", json!({})));
        assert_eq!(responses[0]["id"], 6);
        assert_eq!(responses[0]["error"]["code"], -32601);
        assert!(responses[0].get("result").is_none());
    }

    #[test]
    fn notifications_get_no_reply() {
        // No `id` means a notification. Replying to one is a protocol
        // violation, so these must produce nothing at all.
        let input = format!(
            "{}\n{}\n",
            json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
            json!({ "jsonrpc": "2.0", "method": "resources/list" })
        );
        assert!(exchange(&input).is_empty());
    }

    #[test]
    fn malformed_and_blank_lines_are_skipped_without_dropping_the_session() {
        let input = format!("\n{{ not json\n\n{}\n", request(7, "tools/list", json!({})));
        let responses = exchange(&input);
        assert_eq!(responses.len(), 1, "only the valid request is answered");
        assert_eq!(responses[0]["id"], 7);
    }

    #[test]
    fn a_panicking_tool_fails_one_call_not_the_server() {
        // Expect a panic backtrace on stderr while this runs; it is caught.
        let input = format!(
            "{}\n{}\n",
            request(8, "tools/call", json!({ "name": "boom" })),
            request(9, "tools/call", json!({ "name": "echo", "arguments": {} }))
        );
        let responses = exchange(&input);
        assert_eq!(responses.len(), 2, "the server survived the panic");
        assert_eq!(responses[0]["id"], 8);
        assert_eq!(responses[0]["result"]["isError"], true);
        let text = responses[0]["result"]["content"][0]["text"]
            .as_str()
            .expect("content text");
        assert!(text.contains("boom"), "should name the tool, got: {text}");
        assert!(
            text.contains("tool exploded"),
            "panic payload was not recovered, got: {text}"
        );
        // The whole point: the next call still works.
        assert_eq!(responses[1]["id"], 9);
        assert_eq!(responses[1]["result"]["isError"], false);
    }
}
