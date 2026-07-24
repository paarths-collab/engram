"""Objective: give engram ONLY the real issue text. Does it rank the file the
real fix touched (_merge.py) at the top? Ground truth is the merged PR."""
import json
import subprocess
import sys
import time

BIN, REPO, ISSUE = sys.argv[1], sys.argv[2], sys.argv[3]
GROUND_TRUTH = "libs/core/langchain_core/utils/_merge.py"

issue_text = open(ISSUE, encoding="utf-8", errors="replace").read()
# Use the title + the human-written prose, trimmed to what an agent would paste.
task = issue_text[:1500]

proc = subprocess.Popen(
    [BIN, "--repo", REPO],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, encoding="utf-8", errors="replace", bufsize=1,
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


call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
tool("get_task_context", {"task": "warmup"})

print("=" * 78)
print("BLIND RETRIEVAL on real issue #38366")
print("Agent pastes the issue; engram gets nothing else. Ground truth:")
print(f"  {GROUND_TRUTH}")
print("=" * 78)

p = tool("get_task_context", {"task": task})
ev = p.get("evidence", [])
print(f"\nengram returned {len(ev)} evidence items:\n")
rank = None
for i, e in enumerate(ev, 1):
    path = e.get("path", "?")
    sym = e.get("symbol") or ""
    hit = "  <== GROUND TRUTH" if path == GROUND_TRUTH else ""
    if path == GROUND_TRUTH and rank is None:
        rank = i
    print(f"  {i:2}. [{e.get('score', 0):5.2f}] {sym:34} {path}{hit}")

print()
print("=" * 78)
if rank:
    print(f"RESULT: engram ranked the correct file #{rank} of {len(ev)}, "
          f"from the issue text alone.")
else:
    print("RESULT: correct file NOT in the returned set.")
print("=" * 78)

# Also try find_existing_implementation, the anti-duplication tool, on the concept.
print("\nfind_existing_implementation('merge streaming metadata dicts'):")
fe = tool("find_existing_implementation", {"concept": "merge streaming metadata dicts"})
for c in fe.get("existing_candidates", [])[:5]:
    path = c.get("path", "?")
    hit = "  <== GROUND TRUTH" if path == GROUND_TRUTH else ""
    print(f"   [{c.get('score', 0):5.2f}] {c.get('symbol') or '':30} {path}{hit}")

proc.stdin.close()
proc.terminate()
