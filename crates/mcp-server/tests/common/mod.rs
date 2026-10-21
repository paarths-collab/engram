//! Shared helpers for the end-to-end tests.
//!
//! - [`McpClient`]: a real MCP client over stdio. It spawns the compiled
//!   `engram` binary and speaks newline-delimited JSON-RPC exactly as
//!   `src/mcp.rs` frames it. Every read has a deadline (a reader thread feeds a
//!   channel), and the child is killed on drop, so a broken server can fail a
//!   test but never hang CI.
//! - [`Fixture`]: a tiny REAL git repository built with the `git` CLI, with
//!   fixed identities and dates so the history (and therefore the commit ids and
//!   the co-change counts) is deterministic.
//!
//! No dev-dependencies: the repo builds with `--locked`, so only std and
//! `serde_json` (already a dependency of this crate) are used.
#![allow(dead_code)]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Longest a single JSON-RPC round trip may take. Generous on purpose: the
/// first successful call also builds the in-memory index, and CI machines are
/// shared.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Longest the server may keep answering "still indexing".
pub const READY_BOUND: Duration = Duration::from_secs(90);

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// One request/response pair, kept for transcripts.
#[derive(Clone)]
pub struct Exchange {
    pub label: String,
    /// The JSON-RPC method, or `tools/call:<tool>`.
    pub method: String,
    pub request: Value,
    /// For tool calls the tool's own payload (parsed from the text content, or
    /// the raw text when it is not JSON); otherwise the JSON-RPC `result`.
    pub response: Value,
    pub is_error: bool,
    pub elapsed: Duration,
}

/// The decoded result of one `tools/call`.
#[derive(Clone)]
pub struct ToolOutput {
    pub is_error: bool,
    /// `result.content[0].text`, verbatim.
    pub text: String,
    /// `text` parsed as JSON, or `Null` when it is not JSON (error messages).
    pub json: Value,
    pub elapsed: Duration,
}

impl ToolOutput {
    /// The JSON payload of a call that must have succeeded.
    pub fn payload(&self) -> &Value {
        assert!(!self.is_error, "tool returned isError: {}", self.text);
        &self.json
    }
}

pub struct McpClient {
    child: Child,
    stdin: Option<ChildStdin>,
    rx: Receiver<String>,
    stderr: Arc<Mutex<String>>,
    next_id: u64,
    pub spawned_at: Instant,
    pub log: Vec<Exchange>,
}

impl McpClient {
    /// Spawn `engram --repo <repo>`. The working directory is deliberately not
    /// the repository, so a server that ignored `--repo` could not pass by
    /// accident.
    pub fn spawn(repo: &Path) -> McpClient {
        let mut child = Command::new(env!("CARGO_BIN_EXE_engram"))
            .arg("--repo")
            .arg(repo)
            .current_dir(std::env::temp_dir())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn the engram binary");
        let spawned_at = Instant::now();
        let stdin = child.stdin.take();

        let stdout = child.stdout.take().expect("piped stdout");
        let (tx, rx) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        // Drain stderr so a chatty server can never block on a full pipe, and
        // keep it for failure messages.
        let stderr = Arc::new(Mutex::new(String::new()));
        let err_pipe = child.stderr.take().expect("piped stderr");
        let sink = stderr.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err_pipe).lines() {
                let Ok(line) = line else { break };
                if let Ok(mut s) = sink.lock() {
                    s.push_str(&line);
                    s.push('\n');
                }
            }
        });

        McpClient {
            child,
            stdin,
            rx,
            stderr,
            next_id: 1,
            spawned_at,
            log: Vec::new(),
        }
    }

    /// Everything the server has written to stderr so far.
    pub fn stderr_text(&self) -> String {
        self.stderr.lock().map(|s| s.clone()).unwrap_or_default()
    }

    fn send(&mut self, msg: &Value) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("stdin already closed")?;
        writeln!(stdin, "{msg}")
            .and_then(|()| stdin.flush())
            .map_err(|e| format!("could not write to the server: {e}"))
    }

    /// Send a notification (no id, no reply expected).
    pub fn notify(&mut self, method: &str) -> Result<(), String> {
        self.send(&json!({ "jsonrpc": "2.0", "method": method }))
    }

    /// Send a request and wait for the response carrying the same id.
    pub fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        loop {
            let left = deadline
                .checked_duration_since(Instant::now())
                .ok_or_else(|| self.timeout_error(method))?;
            match self.rx.recv_timeout(left) {
                Ok(line) => {
                    let v: Value = serde_json::from_str(&line)
                        .map_err(|e| format!("server wrote a non-JSON line ({e}): {line}"))?;
                    if v.get("id") == Some(&json!(id)) {
                        return Ok(v);
                    }
                }
                Err(RecvTimeoutError::Timeout) => return Err(self.timeout_error(method)),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(format!(
                        "server closed stdout while waiting for {method}; stderr:\n{}",
                        self.stderr_text()
                    ))
                }
            }
        }
    }

    fn timeout_error(&self, method: &str) -> String {
        format!(
            "no response to {method} within {REQUEST_TIMEOUT:?}; stderr:\n{}",
            self.stderr_text()
        )
    }

    /// `initialize` followed by `notifications/initialized`, as a client must.
    pub fn initialize(&mut self) -> Result<Value, String> {
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "engram-e2e", "version": "0.0.0" }
        });
        let start = Instant::now();
        let response = self.request("initialize", params.clone())?;
        let result = response
            .get("result")
            .cloned()
            .ok_or_else(|| format!("initialize returned no result: {response}"))?;
        self.log.push(Exchange {
            label: "initialize".into(),
            method: "initialize".into(),
            request: params,
            response: result.clone(),
            is_error: false,
            elapsed: start.elapsed(),
        });
        self.notify("notifications/initialized")?;
        Ok(result)
    }

    /// `tools/list`; returns the `result` object.
    pub fn list_tools(&mut self) -> Result<Value, String> {
        let start = Instant::now();
        let response = self.request("tools/list", json!({}))?;
        let result = response
            .get("result")
            .cloned()
            .ok_or_else(|| format!("tools/list returned no result: {response}"))?;
        self.log.push(Exchange {
            label: "tools/list".into(),
            method: "tools/list".into(),
            request: json!({}),
            response: result.clone(),
            is_error: false,
            elapsed: start.elapsed(),
        });
        Ok(result)
    }

    /// One `tools/call` without recording it in the transcript log (used for
    /// polling).
    pub fn call_tool_quiet(&mut self, name: &str, args: Value) -> Result<ToolOutput, String> {
        let start = Instant::now();
        let response = self.request("tools/call", json!({ "name": name, "arguments": args }))?;
        let elapsed = start.elapsed();
        if let Some(err) = response.get("error") {
            return Err(format!("JSON-RPC error from tools/call {name}: {err}"));
        }
        let result = response
            .get("result")
            .ok_or_else(|| format!("tools/call {name} returned no result: {response}"))?;
        let text = result
            .pointer("/content/0/text")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("tools/call {name}: no text content in {result}"))?
            .to_owned();
        let is_error = result
            .get("isError")
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("tools/call {name}: result has no isError flag: {result}"))?;
        let json = serde_json::from_str(&text).unwrap_or(Value::Null);
        Ok(ToolOutput {
            is_error,
            text,
            json,
            elapsed,
        })
    }

    /// One `tools/call`, recorded in the log and timed on stderr.
    pub fn call_tool(&mut self, name: &str, args: Value) -> Result<ToolOutput, String> {
        self.call_labeled(name, name, args)
    }

    pub fn call_labeled(
        &mut self,
        label: &str,
        name: &str,
        args: Value,
    ) -> Result<ToolOutput, String> {
        let out = self.call_tool_quiet(name, args.clone())?;
        eprintln!("[e2e] {label}: {} ms", out.elapsed.as_millis());
        self.record_tool(label, name, args, &out);
        Ok(out)
    }

    pub fn record_tool(&mut self, label: &str, name: &str, args: Value, out: &ToolOutput) {
        self.log.push(Exchange {
            label: label.to_owned(),
            method: format!("tools/call:{name}"),
            request: json!({ "name": name, "arguments": args }),
            response: if out.json.is_null() {
                Value::String(out.text.clone())
            } else {
                out.json.clone()
            },
            is_error: out.is_error,
            elapsed: out.elapsed,
        });
    }

    /// A call that must succeed; returns its JSON payload.
    pub fn ok(&mut self, name: &str, args: Value) -> Value {
        let out = self
            .call_tool(name, args)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!out.is_error, "{name} returned isError: {}", out.text);
        out.json
    }

    /// Close stdin (the clean way to stop an MCP stdio server), wait for the
    /// process to exit, and kill it if it does not.
    pub fn stop(mut self) {
        drop(self.stdin.take());
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        eprintln!("[e2e] server did not exit after stdin closed; killing it");
        // Drop kills.
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// How a poll for readiness ended.
pub struct Settled {
    /// The first answer that was not "still indexing". An error here means the
    /// server settled into a failed state.
    pub output: ToolOutput,
    pub polls: u32,
    /// Time from spawning the process to that answer.
    pub since_spawn: Duration,
}

/// Poll a cheap canonical call until the server stops saying it is still
/// indexing, for at most `bound`.
pub fn settle(client: &mut McpClient, bound: Duration) -> Result<Settled, String> {
    let deadline = Instant::now() + bound;
    let args = json!({ "query": "execute" });
    let mut polls = 0;
    loop {
        polls += 1;
        let out = client.call_tool_quiet("search_context", args.clone())?;
        if !(out.is_error && out.text.contains("still indexing")) {
            client.record_tool("readiness probe", "search_context", args, &out);
            return Ok(Settled {
                output: out,
                polls,
                since_spawn: client.spawned_at.elapsed(),
            });
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "server was still indexing after {bound:?}: {}",
                out.text
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Wait until the index is ready. Fails the test with the server's own message
/// if indexing failed or never finished.
pub fn wait_ready(client: &mut McpClient) -> Settled {
    match settle(client, READY_BOUND) {
        Ok(s) if !s.output.is_error => {
            eprintln!(
                "[e2e] cold start to first answer: {} ms ({} polls)",
                s.since_spawn.as_millis(),
                s.polls
            );
            s
        }
        Ok(s) => panic!(
            "Engram did not become ready; server said: {}\nstderr:\n{}",
            s.output.text,
            client.stderr_text()
        ),
        Err(e) => panic!("{e}\nstderr:\n{}", client.stderr_text()),
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// `value[key][*].path` as strings.
pub fn paths_in(value: &Value, key: &str) -> Vec<String> {
    value[key]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|i| i["path"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// The reasons of an `explain_connection` payload with the given `type`.
pub fn reasons_of<'a>(explain: &'a Value, ty: &str) -> Vec<&'a Value> {
    explain["reasons"]
        .as_array()
        .map(|rs| rs.iter().filter(|r| r["type"] == ty).collect())
        .unwrap_or_default()
}

/// The `snapshot` id carried by an indexed-tool response.
pub fn snapshot_of(payload: &Value) -> String {
    payload["snapshot"]
        .as_str()
        .unwrap_or_else(|| panic!("response has no snapshot field: {payload}"))
        .to_owned()
}

// ---------------------------------------------------------------------------
// Fixture repositories
// ---------------------------------------------------------------------------

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// True (after saying so) when the end-to-end tests cannot run.
pub fn skip_without_git() -> bool {
    let present = Command::new("git")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !present {
        eprintln!(
            "SKIPPED: `git` is not available on PATH; the end-to-end tests build a real git \
             repository as their fixture"
        );
    }
    !present
}

/// A throwaway git repository under the system temp dir, removed on drop.
pub struct Fixture {
    pub root: PathBuf,
}

impl Fixture {
    /// An empty repository (`git init`), nothing committed yet.
    pub fn init(tag: &str) -> Fixture {
        let root = std::env::temp_dir().join(format!(
            "engram-e2e-{}-{}-{tag}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create fixture dir");
        let fx = Fixture { root };
        fx.git(&["init", "--quiet"], &[]);
        fx
    }

    /// Run git in the fixture with a pinned identity and no host configuration
    /// leaking in. `extra_env` carries per-commit dates.
    pub fn git(&self, args: &[&str], extra_env: &[(&str, &str)]) -> String {
        let mut cmd = Command::new("git");
        cmd.args([
            "-c",
            "user.name=Anonymous",
            "-c",
            "user.email=anonymous@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
            "-c",
            "core.autocrlf=false",
        ])
        .args(args)
        .current_dir(&self.root)
        .env("GIT_AUTHOR_NAME", "Anonymous")
        .env("GIT_AUTHOR_EMAIL", "anonymous@example.invalid")
        .env("GIT_COMMITTER_NAME", "Anonymous")
        .env("GIT_COMMITTER_EMAIL", "anonymous@example.invalid")
        .env("GIT_CONFIG_NOSYSTEM", "1");
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        let out = cmd.output().expect("run git");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_owned()
    }

    pub fn write(&self, rel: &str, content: &str) {
        let full = self.root.join(rel);
        std::fs::create_dir_all(full.parent().expect("parent")).expect("mkdir");
        std::fs::write(full, content).expect("write fixture file");
    }

    pub fn append(&self, rel: &str, content: &str) {
        let mut text = std::fs::read_to_string(self.root.join(rel)).expect("read fixture file");
        text.push_str(content);
        self.write(rel, &text);
    }

    pub fn read(&self, rel: &str) -> String {
        std::fs::read_to_string(self.root.join(rel)).expect("read fixture file")
    }

    pub fn remove(&self, rel: &str) {
        std::fs::remove_file(self.root.join(rel)).expect("remove fixture file");
    }

    pub fn rename(&self, from: &str, to: &str) {
        let dest = self.root.join(to);
        std::fs::create_dir_all(dest.parent().expect("parent")).expect("mkdir");
        std::fs::rename(self.root.join(from), dest).expect("rename fixture file");
    }

    /// Stage everything and commit with the fixed identity at a fixed date
    /// (`day` of January 2024, noon UTC).
    pub fn commit_all(&self, message: &str, day: u32) {
        self.git(&["add", "-A"], &[]);
        let date = format!("2024-01-{day:02}T12:00:00+0000");
        self.git(
            &["commit", "--quiet", "-m", message],
            &[("GIT_AUTHOR_DATE", &date), ("GIT_COMMITTER_DATE", &date)],
        );
    }

    /// A one-commit repository holding exactly `files`. Same files, same
    /// message, same date: byte-identical history wherever it is built.
    pub fn mini(tag: &str, files: &[(&str, &str)]) -> Fixture {
        let fx = Fixture::init(tag);
        for (path, content) in files {
            fx.write(path, content);
        }
        fx.commit_all("Add modules", 1);
        fx
    }

    /// Replace this fixture's path in `text` with `<repo>`.
    pub fn sanitize(&self, text: &str) -> String {
        let mut out = text.replace(&*self.root.to_string_lossy(), "<repo>");
        if let Ok(canon) = std::fs::canonicalize(&self.root) {
            out = out.replace(&*canon.to_string_lossy(), "<repo>");
        }
        out
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

pub const API: &str = "app/api.py";
pub const SERVICE: &str = "app/service.py";
pub const REPO: &str = "app/repo.py";
pub const TEST_SERVICE: &str = "tests/test_service.py";
pub const AUDIT: &str = "app/audit.py";
pub const METRICS: &str = "app/metrics.py";
pub const CONFIG: &str = "app/config.py";
pub const CACHE: &str = "app/cache.py";
pub const SCHEMAS: &str = "app/schemas.py";

/// The standard fixture's history, oldest first: message and the files each
/// commit touches. `service.py` changes with `api.py` in eight commits and with
/// each of audit/metrics/config/cache in four, but with `repo.py` in only three
/// (and with `schemas.py` in three), so `repo.py` ranks 6th or 7th among the
/// co-change partners of `service.py` and falls outside the top five that
/// `expand_connections` returns for it.
pub const HISTORY: &[(&str, &[&str])] = &[
    (
        "Add order service skeleton",
        &[
            API,
            SERVICE,
            REPO,
            TEST_SERVICE,
            AUDIT,
            METRICS,
            CONFIG,
            CACHE,
            SCHEMAS,
        ],
    ),
    (
        "Audit and meter order runs",
        &[API, SERVICE, AUDIT, METRICS],
    ),
    ("Cache configured lookups", &[API, SERVICE, CONFIG, CACHE]),
    ("Validate order payloads", &[API, SERVICE, SCHEMAS, AUDIT]),
    ("Meter configured runs", &[API, SERVICE, METRICS, CONFIG]),
    ("Cache validated payloads", &[API, SERVICE, CACHE, SCHEMAS]),
    ("Audit configured runs", &[API, SERVICE, AUDIT, CONFIG]),
    ("Move order lookup into the repo layer", &[SERVICE, REPO]),
    ("Handle missing orders in the repo layer", &[SERVICE, REPO]),
    (
        "Cover metered cached runs",
        &[API, SERVICE, METRICS, CACHE, TEST_SERVICE],
    ),
    ("Tune the repo layer", &[REPO]),
];

fn initial_content(path: &str) -> &'static str {
    match path {
        API => {
            "\"\"\"HTTP entrypoint: routes incoming requests to the service layer.\"\"\"\n\
             from app.service import execute\n\
             \n\
             ROUTES = {\"/orders/run\": \"handle_run\"}\n\
             \n\
             \n\
             def handle_run(payload):\n\
             \x20   \"\"\"Route handler for POST /orders/run; delegates to the service.\"\"\"\n\
             \x20   return execute(payload)\n"
        }
        SERVICE => {
            "\"\"\"Order service: orchestrates persistence for an order run.\"\"\"\n\
             from app.repo import load\n\
             \n\
             \n\
             def execute(order):\n\
             \x20   \"\"\"Execute an order: load its record and return a status envelope.\"\"\"\n\
             \x20   record = load(order[\"id\"])\n\
             \x20   return {\"status\": \"ok\", \"record\": record}\n"
        }
        REPO => {
            "\"\"\"Persistence layer for orders.\"\"\"\n\
             \n\
             _STORE = {}\n\
             \n\
             \n\
             def load(order_id):\n\
             \x20   \"\"\"Load an order record by id.\"\"\"\n\
             \x20   return _STORE.get(order_id)\n"
        }
        TEST_SERVICE => {
            "from app.service import execute\n\
             \n\
             \n\
             def test_execute_returns_ok():\n\
             \x20   assert execute({\"id\": 1})[\"status\"] == \"ok\"\n"
        }
        AUDIT => "def record_audit(entry):\n    return {\"audited\": entry}\n",
        METRICS => "def count_run(name):\n    return {\"counted\": name}\n",
        CONFIG => "def read_setting(key):\n    return {\"setting\": key}\n",
        CACHE => "def cached_lookup(key):\n    return {\"cached\": key}\n",
        SCHEMAS => "def validate_payload(payload):\n    return bool(payload)\n",
        other => panic!("no initial content for {other}"),
    }
}

impl Fixture {
    /// The standard Python fixture: `HISTORY` replayed as real commits, each
    /// later commit appending a comment line to the files it touches.
    ///
    /// Import edges the Engram resolver should find (module-name needles):
    /// api -> service -> repo, and test_service -> service. `api.py` does not
    /// import `repo.py`.
    pub fn standard(tag: &str) -> Fixture {
        let fx = Fixture::init(tag);
        for (i, (message, files)) in HISTORY.iter().enumerate() {
            for file in *files {
                if i == 0 {
                    fx.write(file, initial_content(file));
                } else {
                    fx.append(file, &format!("# change {i}: {message}\n"));
                }
            }
            fx.commit_all(message, u32::try_from(i).expect("small") + 1);
        }
        fx
    }
}

/// A running server over a fixture. Field order matters: the client (and its
/// child process) is dropped before the fixture directory is removed.
pub struct Session {
    pub client: McpClient,
    pub fx: Fixture,
    pub ready: Settled,
}

impl Session {
    /// Build the standard fixture, start the server, handshake, and wait until
    /// it is ready.
    pub fn start(tag: &str) -> Session {
        let fx = Fixture::standard(tag);
        let mut client = McpClient::spawn(&fx.root);
        client.initialize().expect("initialize");
        let ready = wait_ready(&mut client);
        Session { client, fx, ready }
    }
}
