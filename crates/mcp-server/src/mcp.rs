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
    for line in stdin.lock().lines() {
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
                match handler.call_tool(name, args) {
                    Ok(result) => Some(json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&result).unwrap_or_default()
                        }],
                        "isError": false
                    })),
                    Err(e) => Some(json!({
                        "content": [{ "type": "text", "text": e }],
                        "isError": true
                    })),
                }
            }
            "ping" => Some(json!({})),
            _ => {
                if id.is_some() {
                    // unknown request → JSON-RPC error
                    let err = json!({
                        "jsonrpc": "2.0", "id": id,
                        "error": { "code": -32601, "message": format!("method not found: {method}") }
                    });
                    let mut out = stdout.lock();
                    let _ = writeln!(out, "{err}");
                    let _ = out.flush();
                }
                continue;
            }
        };

        // Only requests (with id) get responses; notifications don't.
        if let (Some(id), Some(result)) = (id, response) {
            let reply = json!({ "jsonrpc": "2.0", "id": id, "result": result });
            let mut out = stdout.lock();
            let _ = writeln!(out, "{reply}");
            let _ = out.flush();
        }
    }
}
