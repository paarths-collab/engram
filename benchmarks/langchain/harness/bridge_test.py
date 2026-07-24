"""Does engram's connection graph bridge from the streaming-metadata code
(where the symptom lives) to _merge.py (where the bug lives)?"""
import json
import subprocess
import sys
import time

BIN, REPO = sys.argv[1], sys.argv[2]
GT = "libs/core/langchain_core/utils/_merge.py"

ANCHORS = [
    "libs/core/langchain_core/messages/ai.py",
    "libs/core/langchain_core/outputs/chat_generation.py",
    "libs/core/langchain_core/messages/base.py",
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


call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
tool("find_connected_files", {"files": ["libs/core/langchain_core/utils/_merge.py"]})

print("=" * 76)
print("BRIDGE TEST: anchor on the streaming/metadata code, can engram reach")
print(f"the bug at {GT} via recorded connections?")
print("=" * 76)
for anchor in ANCHORS:
    c = tool("find_connected_files", {"files": [anchor]})
    imports = [x["path"] for x in c.get("import_expansions", [])]
    cochange = [x["path"] for x in c.get("cochange_expansions", [])]
    all_conn = set(imports) | set(cochange)
    reached = GT in all_conn
    print(f"\nanchor: {anchor}")
    print(f"   import edges : {len(imports)}   co-change: {len(cochange)}")
    print(f"   reaches _merge.py? {'YES' if reached else 'NO'}")
    if imports:
        for p in imports[:6]:
            mark = "  <== BUG" if p == GT else ""
            print(f"      import: {p}{mark}")

# Reverse: what does _merge.py connect TO (its dependents)?
print("\n" + "=" * 76)
print("Reverse: dependents of _merge.py (who would engram flag if you edit it?)")
print("=" * 76)
c = tool("find_connected_files", {"files": [GT]})
for p in [x["path"] for x in c.get("import_expansions", [])][:10]:
    print(f"   {p}")

proc.stdin.close()
proc.terminate()
