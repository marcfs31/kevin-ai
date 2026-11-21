#!/usr/bin/env python3
"""A tiny MCP server over stdio for tools/test_chat_tools.py. Tools: echo (read-only), write_file,
big, slow, env, crash, image, structured, add_tool (announces a new tool), replies. Not for real use."""

import json
import os
import sys
import time

PAGES = "--pages" in sys.argv
TOOLS = [
    {"name": "echo", "description": "Returns its text.", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
     "annotations": {"readOnlyHint": True}},
    {"name": "write_file", "description": "Pretends to write a file.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "big", "description": "A very long answer.", "inputSchema": {"type": "object"}},
    {"name": "slow", "description": "Takes five seconds.", "inputSchema": {"type": "object"}},
    {"name": "env", "description": "The names of its environment variables.", "inputSchema": {"type": "object"}},
    {"name": "crash", "description": "Exits.", "inputSchema": {"type": "object"}},
    {"name": "image", "description": "An image.", "inputSchema": {"type": "object"}},
    {"name": "structured", "description": "Structured output only.", "inputSchema": {"type": "object"}},
    {"name": "add_tool", "description": "Adds a tool and says so.", "inputSchema": {"type": "object"}},
    {"name": "replies", "description": "What the client answered to the server's own request.", "inputSchema": {"type": "object"}},
    {"name": "bad name!", "description": "Dropped: not a usable name.", "inputSchema": {"type": "object"}},
]
seen = []


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def result(rid, res):
    send({"jsonrpc": "2.0", "id": rid, "result": res})


for line in sys.stdin:
    msg = json.loads(line)
    if "id" in msg and "method" not in msg:        # the client's answer to a request of ours
        seen.append(msg)
        continue
    method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if method == "initialize":
        result(rid, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {"listChanged": True}},
                     "serverInfo": {"name": "fake", "version": "1"}})
        send({"jsonrpc": "2.0", "id": "s1", "method": "roots/list"})      # a request the client does not support
    elif method == "tools/list":
        if PAGES and not params.get("cursor"):
            result(rid, {"tools": TOOLS[:3], "nextCursor": "p2"})
        else:
            result(rid, {"tools": TOOLS[3:] if PAGES else TOOLS})
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name == "echo":
            result(rid, {"content": [{"type": "text", "text": str(args.get("text", ""))}]})
        elif name == "write_file":
            result(rid, {"content": [{"type": "text", "text": "wrote " + str(args.get("path"))}]})
        elif name == "big":
            result(rid, {"content": [{"type": "text", "text": "x" * 20000}]})
        elif name == "slow":
            time.sleep(5)
            result(rid, {"content": [{"type": "text", "text": "late"}]})
        elif name == "env":
            result(rid, {"content": [{"type": "text", "text": " ".join(sorted(os.environ))}]})
        elif name == "crash":
            sys.stderr.write("crashing on purpose\n")
            sys.stderr.flush()
            os._exit(3)
        elif name == "image":
            result(rid, {"content": [{"type": "image", "mimeType": "image/png", "data": "AAAA" * 10},
                                     {"type": "resource", "resource": {"uri": "file:///a", "text": "from a resource"}},
                                     {"type": "resource_link", "uri": "file:///b", "name": "b"}]})
        elif name == "structured":
            result(rid, {"content": [], "structuredContent": {"answer": 42}})
        elif name == "add_tool":
            TOOLS.append({"name": "added", "description": "Appeared later.", "inputSchema": {"type": "object"}})
            result(rid, {"content": [{"type": "text", "text": "added"}]})
            send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
        elif name == "replies":
            result(rid, {"content": [{"type": "text", "text": json.dumps(seen)}]})
        else:
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": "unknown tool " + str(name)}})
    elif method == "ping":
        result(rid, {})
