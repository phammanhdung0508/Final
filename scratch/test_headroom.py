import json
import subprocess
import os

payload = "This is a repetitive synthetic payload. " * 200

# Start MCP server
p = subprocess.Popen(["/home/sunf/.local/bin/headroom", "mcp", "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def send(req):
    p.stdin.write(json.dumps(req) + "\n")
    p.stdin.flush()
    res = p.stdout.readline()
    return json.loads(res)

send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}})

p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
p.stdin.flush()

res_compress = send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "headroom_compress", "arguments": {"content": payload}}})

content_text = res_compress['result']['content'][0]['text']

import re
match = re.search(r"hash=([a-f0-9]+)", content_text)
hash_val = match.group(1)

res_retrieve = send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "headroom_retrieve", "arguments": {"hash": hash_val}}})

retrieved_text = res_retrieve['result']['content'][0]['text']

try:
    retrieved_json = json.loads(retrieved_text)
    actual_retrieved = retrieved_json.get("original_content", retrieved_text)
except json.JSONDecodeError:
    actual_retrieved = retrieved_text

print(f"Original length: {len(payload)}")
print(f"Retrieved length: {len(actual_retrieved)}")
print(f"Match: {payload == actual_retrieved}")

try:
    data = json.loads(content_text)
    print(f"Hash: {data.get('hash')}")
    print(f"Original tokens: {data.get('original_tokens')}")
    print(f"Compressed tokens: {data.get('compressed_tokens')}")
    print(f"Tokens saved: {data.get('tokens_saved')}")
    print(f"Savings %: {data.get('savings_percent')}")
except Exception as e:
    print("Failed to parse json:", e)

