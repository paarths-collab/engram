"""B2 corrected: measure the ACTUAL bytes engram returns, not a self-reported
field that main does not populate. Token proxy = chars / 4 on both sides."""
import json
import subprocess
import sys
import threading
import time
import os

BIN, REPO = sys.argv[1], sys.argv[2]
proc = subprocess.Popen(
    [BIN, "--repo", REPO],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, encoding="utf-8", errors="replace", bufsize=1,
)
_id = 0


def call(method, params=None):
    global _id
    _id += 1
    m = {"jsonrpc": "2.0", "id": _id, "method": method}
    if params:
        m["params"] = params
    proc.stdin.write(json.dumps(m) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def tool_raw(name, args, retries=90):
    """Return the raw text payload the agent receives, and the parsed object."""
    for _ in range(retries):
        r = call("tools/call", {"name": name, "arguments": args})
        res = r.get("result", {})
        txt = res.get("content", [{}])[0].get("text", "")
        if not res.get("isError"):
            return txt, json.loads(txt)
        time.sleep(4)
    raise RuntimeError("never ready")


TASKS = [
    ("add retry logic to a chat model call", ["retry", "max_retries", "with_retry"]),
    ("fix streaming token callback for anthropic", ["stream", "callback", "anthropic"]),
    ("implement a new output parser for JSON", ["output_parser", "JsonOutputParser", "parse"]),
    ("cache embeddings to avoid recomputation", ["CacheBackedEmbeddings", "embed", "cache"]),
    ("add a rate limiter to the base language model", ["rate_limiter", "InMemoryRateLimiter", "acquire"]),
]

RG = "rg"


def rg_tokens(keywords):
    matched = set()
    for kw in keywords:
        try:
            out = subprocess.run(
                [RG, "-l", "-i", "--type", "py", kw, REPO],
                capture_output=True, text=True, timeout=60,
            ).stdout
        except Exception:
            continue
        for line in out.splitlines():
            if line.strip():
                matched.add(line.strip())
    chars = sum(os.path.getsize(f) for f in matched if os.path.exists(f))
    return len(matched), chars // 4


call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
tool_raw("get_task_context", {"task": "warmup"})  # wait out indexing

print("=" * 82)
print(f"{'task':40} {'engram tok':>11} {'rg files':>9} {'rg tok':>10} {'saving':>8}")
print("=" * 82)
e_tot = r_tot = 0
for task, kw in TASKS:
    raw, obj = tool_raw("get_task_context", {"task": task})
    e_tok = len(raw) // 4  # exactly the bytes the agent must ingest
    n_ev = len(obj.get("evidence", []))
    rgf, r_tok = rg_tokens(kw)
    e_tot += e_tok
    r_tot += r_tok
    saving = 1 - (e_tok / r_tok) if r_tok else 0
    print(f"{task[:40]:40} {e_tok:>11} {rgf:>9} {r_tok:>10} {saving:>7.1%}")
print("=" * 82)
saving = 1 - (e_tot / r_tot) if r_tot else 0
print(f"{'TOTAL':40} {e_tot:>11} {'':>9} {r_tot:>10} {saving:>7.1%}")
print()
print(f"engram payload total: {e_tot:,} tokens")
print(f"grep-and-read total : {r_tot:,} tokens")
print(f"engram is {r_tot / e_tot:.0f}x smaller than reading every grep match.")

proc.stdin.close()
proc.terminate()
