"""Isolate the cause: does engram find _merge.py with a distilled query but not
the raw issue? Feeds progressively cleaner queries, reports the rank of the
ground-truth file each time."""
import json
import subprocess
import sys
import time

BIN, REPO, ISSUE = sys.argv[1], sys.argv[2], sys.argv[3]
GT = "libs/core/langchain_core/utils/_merge.py"
raw = open(ISSUE, encoding="utf-8", errors="replace").read()
title = raw.splitlines()[0]

QUERIES = [
    ("raw issue (1500 chars)", raw[:1500]),
    ("issue title only", title),
    ("title minus the [Bug] prefix",
     title.replace("[Bug]", "").strip()),
    ("one-line problem statement",
     "merge_dicts concatenates identical string metadata across streaming chunks"),
    ("just the function + symptom",
     "merge_dicts doubles model_name and finish_reason"),
]

proc = subprocess.Popen(
    [BIN, "--repo", REPO], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1,
)
_id = 0


def call(m, p=None):
    global _id
    _id += 1
    msg = {"jsonrpc": "2.0", "id": _id, "method": m}
    if p:
        msg["params"] = p
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def tool(name, args, retries=90):
    for _ in range(retries):
        r = call("tools/call", {"name": name, "arguments": args})
        res = r.get("result", {})
        txt = res.get("content", [{}])[0].get("text", "")
        if not res.get("isError"):
            return json.loads(txt)
        time.sleep(4)
    raise RuntimeError("never ready")


def rank_of_gt(evidence):
    for i, e in enumerate(evidence, 1):
        if e.get("path") == GT:
            return i
    return None


call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
tool("get_task_context", {"task": "warmup"})

print("=" * 74)
print(f"{'query':44} {'GT rank':>8} {'n':>4}")
print("=" * 74)
for label, q in QUERIES:
    p = tool("get_task_context", {"task": q})
    ev = p.get("evidence", [])
    r = rank_of_gt(ev)
    print(f"{label[:44]:44} {(str(r) if r else 'MISS'):>8} {len(ev):>4}")
print("=" * 74)
print("\nIf clean queries hit and the raw issue misses, the fix is query")
print("distillation (agent or server), not the retrieval core.")

proc.stdin.close()
proc.terminate()
