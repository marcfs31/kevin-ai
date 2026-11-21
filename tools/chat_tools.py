#!/usr/bin/env python3
"""Local tool service for the chat web app: MCP servers and an optional web search.

    chat_tools.py [--host 127.0.0.1] [--port 8081] [--config ~/.config/lm-chat/mcp.json]

The browser reaches it through the same-origin /tools/* paths (the web dev proxy or
chat_server.py forwards them), so the page's CSP stays 'self' only.

- MCP servers are listed in a config file on THIS machine, in the usual shape
  {"mcpServers": {"name": {"command": ..., "args": [...], "env": {...}}}} or, for a remote
  server, {"url": "https://...", "bearerEnv": "TOKEN_VARIABLE"}. The browser never sends a
  command to run: it can only start, call and stop servers by name, and add an entry from the
  catalog next to this file (tools/chat_tools_catalog.json), whose commands are pinned.
- A server starts on its first use and is stopped after it has been idle for a while; only a few
  run at once and none is started when memory is low, because this runs beside the model.
- A child process gets a minimal environment (PATH, HOME, LANG and what its config lists), never
  this process's secrets.
- Servers and tools that reach the network only run when the request says the internet is on.
  The first start of a catalog package also needs it, because the package is downloaded then.
- POST /tools/search asks a SearXNG instance (its URL comes with the request) and returns text.
- It listens on 127.0.0.1. A --host that is not loopback needs CHAT_PASSWORD in the environment,
  as chat_server.py does; without a password requests must name a loopback Host.

Endpoints (JSON; every POST needs Content-Type: application/json and a loopback Origin):
  GET  /tools/status                          the servers, the catalog and the limits
  GET  /tools/servers/<id>/tools[?internet=1] the tools of a configured server (starts it)
  POST /tools/servers/<id>/call               {"name", "arguments", "internet"}
  POST /tools/servers/<id>/stop
  POST /tools/catalog/<id>/add                {"internet"}: writes the catalog entry into the config
  POST /tools/search                          {"query", "searchUrl", "count", "internet"}

Python 3 standard library only.
"""

import argparse
import base64
import binascii
import collections
import hmac
import http.client
import http.server
import ipaddress
import json
import math
import os
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chat_server  # noqa: E402  (loopback and password checks shared with the proxy)

PROTOCOL_VERSION = "2025-06-18"
MAX_BODY = 256 * 1024
MAX_RESULT_CHARS = 8000
MAX_LINE = 8 * 1024 * 1024
MAX_HTTP_REPLY = 4 * 1024 * 1024
MAX_SEARCH_REPLY = 512 * 1024
START_TIMEOUT_S = 60
CALL_TIMEOUT_S = 60
SEARCH_TIMEOUT_S = 10
IDLE_SECONDS = 300
MAX_RUNNING = 3
MEMORY_FLOOR_MB = 700
SERVER_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
SAFE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")


class ToolError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class ConfigError(ValueError):
    pass


# ---- configuration -------------------------------------------------------------------------

def config_path(env):
    return env.get("LM_CHAT_TOOLS_CONFIG") or os.path.join(os.path.expanduser("~"), ".config", "lm-chat", "mcp.json")


def _str_list(v, what):
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ConfigError(f"{what} must be a list of strings")
    return v


def _str_map(v, what):
    if v is None:
        return {}
    if not isinstance(v, dict) or not all(isinstance(k, str) and isinstance(x, str) for k, x in v.items()):
        raise ConfigError(f"{what} must be an object of strings")
    return v


def parse_spec(sid, spec):
    """One server entry, validated and filled in. `network` says whether it reaches the internet."""
    if not SERVER_ID.match(sid):
        raise ConfigError(f"bad server name {sid!r}: lowercase letters, digits, - and _, at most 32")
    if not isinstance(spec, dict):
        raise ConfigError(f"{sid}: the entry must be an object")
    title = spec.get("title") if isinstance(spec.get("title"), str) else sid
    out = {"id": sid, "title": title, "catalog": spec.get("catalog") if isinstance(spec.get("catalog"), str) else None}
    if "command" in spec:
        if not isinstance(spec["command"], str) or not spec["command"]:
            raise ConfigError(f"{sid}: command must be a string")
        out.update(kind="stdio", command=spec["command"], args=_str_list(spec.get("args"), f"{sid}: args"),
                   env=_str_map(spec.get("env"), f"{sid}: env"), network=bool(spec.get("network", False)),
                   install=bool(spec.get("install", False)))
    elif "url" in spec:
        url = spec["url"]
        parts = urllib.parse.urlsplit(url) if isinstance(url, str) else None
        if not parts or parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            raise ConfigError(f"{sid}: url must be http(s) without a user name or password")
        out.update(kind="http", url=url, bearer_env=spec.get("bearerEnv") if isinstance(spec.get("bearerEnv"), str) else None,
                   allow_private=bool(spec.get("allowPrivate", False)), network=bool(spec.get("network", True)), install=False)
    else:
        raise ConfigError(f"{sid}: needs a command or a url")
    return out


def load_config(path):
    """{id: spec} from the config file; {} when there is none. Raises ConfigError when it is unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise ConfigError(f"{path}: {e}") from None
    servers = doc.get("mcpServers") if isinstance(doc, dict) else None
    if not isinstance(servers, dict):
        raise ConfigError(f'{path}: expected {{"mcpServers": {{...}}}}')
    return {sid: parse_spec(sid, spec) for sid, spec in servers.items()}


def load_catalog(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    out = {}
    for entry in doc["servers"]:
        spec = parse_spec(entry["id"], {k: v for k, v in entry.items() if k not in ("id", "description", "verified", "runtime")})
        spec["description"] = entry.get("description", "")
        spec["runtime"] = entry.get("runtime", "")
        spec["verified"] = bool(entry.get("verified", False))
        spec["catalog"] = entry["id"]
        out[entry["id"]] = spec
    return out


def memory_available_mb():
    """MemAvailable in MiB on Linux; None elsewhere or when it cannot be read."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


# ---- outbound requests: address checks ------------------------------------------------------

def check_address(host, port, allow_private):
    """The address to connect to for host:port, refusing the ones a tool must never reach: link-local
    (cloud metadata), unspecified and multicast always, private and loopback unless allowed.
    Resolving once and connecting to that address keeps a DNS answer from changing in between."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise ToolError("unreachable", f"cannot resolve {host}: {e}", 502) from None
    if not infos:
        raise ToolError("unreachable", f"cannot resolve {host}", 502)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_link_local or ip.is_unspecified or ip.is_multicast or ip.is_reserved:
            raise ToolError("blocked_address", f"{host} resolves to an address that is never allowed ({ip})", 403)
        if not allow_private and (ip.is_private or ip.is_loopback):
            raise ToolError("blocked_address", f"{host} is a private or loopback address and this server is not marked allowPrivate", 403)
    return infos[0][4][0].split("%")[0]


def open_connection(url, allow_private, timeout):
    parts = urllib.parse.urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    ip = check_address(parts.hostname, port, allow_private)
    try:
        sock = socket.create_connection((ip, port), timeout=timeout)
    except OSError as e:
        raise ToolError("unreachable", f"cannot connect to {parts.hostname}: {e}", 502) from None
    if parts.scheme == "https":
        try:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parts.hostname)
        except (ssl.SSLError, OSError) as e:
            sock.close()
            raise ToolError("unreachable", f"TLS to {parts.hostname} failed: {e}", 502) from None
    conn = http.client.HTTPConnection(parts.hostname, port, timeout=timeout)
    conn.sock = sock
    return conn


# ---- MCP over stdio ---------------------------------------------------------------------------

class StdioClient:
    """One child process speaking JSON-RPC 2.0, one message per line."""

    def __init__(self, sid, spec, env):
        self.sid = sid
        self.spec = spec
        child_env = {k: env[k] for k in SAFE_ENV if k in env}
        child_env.update(spec["env"])
        if spec["command"].rsplit("/", 1)[-1] in ("npx", "npm", "node"):
            child_env.setdefault("npm_config_ignore_scripts", "true")   # a package may not run its install scripts
        try:
            self.proc = subprocess.Popen([spec["command"]] + spec["args"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, env=child_env, start_new_session=True)
        except OSError as e:
            raise ToolError("start_failed", f"cannot run {spec['command']}: {e}", 502) from None
        self.lock = threading.Lock()
        self.next_id = 1
        self.pending = {}
        self.dead = None
        self.stale = False
        self.stderr = collections.deque(maxlen=40)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        reason = "the server closed its output"
        try:
            while True:
                line = self.proc.stdout.readline(MAX_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_LINE:
                    reason = "the server sent a line over 8 MiB"
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue    # not JSON-RPC: a stray log line on stdout
                if isinstance(msg, dict):
                    self._handle(msg)
        except (OSError, ValueError):
            pass
        self._fail(reason)

    def _read_stderr(self):
        try:
            for line in self.proc.stderr:
                self.stderr.append(line.decode("utf-8", "replace").rstrip()[:300])
        except (OSError, ValueError):
            pass

    def _fail(self, reason):
        with self.lock:
            self.dead = self.dead or reason
            waiting = list(self.pending.values())
            self.pending.clear()
        for slot in waiting:
            slot[0].set()

    def _handle(self, msg):
        if "id" in msg and ("result" in msg or "error" in msg):
            with self.lock:
                slot = self.pending.pop(msg["id"], None)
            if slot:
                slot[1] = msg
                slot[0].set()
        elif "method" in msg and "id" in msg:        # a request from the server: only ping is answered
            if msg["method"] == "ping":
                self._send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
            else:
                self._send({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "not supported"}})
        elif msg.get("method") == "notifications/tools/list_changed":
            self.stale = True

    def _send(self, msg):
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        try:
            with self.lock:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
        except (OSError, ValueError):
            self._fail("the server is not reading its input")

    def request(self, method, params=None, timeout=CALL_TIMEOUT_S):
        if self.dead:
            raise ToolError("server_exited", self.describe_exit(), 502)
        slot = [threading.Event(), None]
        with self.lock:
            rid = self.next_id
            self.next_id += 1
            self.pending[rid] = slot
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        if not slot[0].wait(timeout):
            with self.lock:
                self.pending.pop(rid, None)
            raise ToolError("timeout", f"{method} took longer than {timeout} s", 504)
        if slot[1] is None:
            raise ToolError("server_exited", self.describe_exit(), 502)
        if "error" in slot[1]:
            err = slot[1]["error"]
            raise ToolError("server_error", str(err.get("message", err))[:500] if isinstance(err, dict) else str(err)[:500], 502)
        return slot[1].get("result") or {}

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def describe_exit(self):
        tail = " | ".join(list(self.stderr)[-3:])
        return f"{self.dead or 'the server stopped'}" + (f" ({tail})" if tail else "")

    def close(self):
        self._fail("stopped")
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.proc.kill()
            except OSError:
                pass
        for pipe in (self.proc.stdout, self.proc.stderr):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass


# ---- MCP over Streamable HTTP -----------------------------------------------------------------

class HttpClient:
    """A remote server: one POST per message, the answer as JSON or as an event stream."""

    def __init__(self, sid, spec, env):
        self.sid = sid
        self.spec = spec
        self.session = None
        self.dead = None
        self.stale = False
        self.next_id = 1
        self.lock = threading.Lock()
        self.stderr = collections.deque(maxlen=1)
        token = env.get(spec["bearer_env"]) if spec["bearer_env"] else None
        if spec["bearer_env"] and not token:
            raise ToolError("start_failed", f"the environment variable {spec['bearer_env']} is not set", 502)
        self.token = token

    def _post(self, payload, timeout, want_id):
        parts = urllib.parse.urlsplit(self.spec["url"])
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "User-Agent": "lm-chat-tools", "MCP-Protocol-Version": PROTOCOL_VERSION}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        conn = open_connection(self.spec["url"], self.spec["allow_private"], timeout)
        try:
            conn.request("POST", (parts.path or "/") + (f"?{parts.query}" if parts.query else ""), json.dumps(payload).encode(), headers)
            res = conn.getresponse()
            session = res.getheader("Mcp-Session-Id")
            if session:
                self.session = session
            if res.status in (202, 204):
                return None
            if res.status >= 400:
                raise ToolError("server_error", f"the server answered {res.status}", 502)
            ctype = (res.getheader("Content-Type") or "").split(";")[0].strip().lower()
            if ctype == "text/event-stream":
                return self._read_events(res, want_id)
            body = res.read(MAX_HTTP_REPLY + 1)
            if len(body) > MAX_HTTP_REPLY:
                raise ToolError("too_large", "the server's answer is over 4 MiB", 502)
            try:
                return json.loads(body)
            except ValueError:
                raise ToolError("bad_reply", "the server did not answer with JSON", 502) from None
        except (OSError, http.client.HTTPException) as e:
            raise ToolError("unreachable", f"{type(e).__name__}: {e}", 502) from None
        finally:
            conn.close()

    @staticmethod
    def _read_events(res, want_id):
        total = 0
        data = []
        while True:
            line = res.readline(MAX_HTTP_REPLY + 1)
            total += len(line)
            if total > MAX_HTTP_REPLY:
                raise ToolError("too_large", "the server's event stream is over 4 MiB", 502)
            if not line:
                break
            line = line.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and data:
                try:
                    msg = json.loads("\n".join(data))
                except ValueError:
                    msg = None
                data = []
                if isinstance(msg, dict) and msg.get("id") == want_id and ("result" in msg or "error" in msg):
                    return msg
        raise ToolError("bad_reply", "the event stream ended without an answer", 502)

    def request(self, method, params=None, timeout=CALL_TIMEOUT_S):
        with self.lock:
            rid = self.next_id
            self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        reply = self._post(msg, timeout, rid)
        if not isinstance(reply, dict):
            raise ToolError("bad_reply", "the server sent no answer", 502)
        if "error" in reply:
            err = reply["error"]
            raise ToolError("server_error", str(err.get("message", err))[:500] if isinstance(err, dict) else str(err)[:500], 502)
        return reply.get("result") or {}

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        try:
            self._post(msg, CALL_TIMEOUT_S, None)
        except ToolError:
            pass

    def close(self):
        pass


# ---- tools ------------------------------------------------------------------------------------

def normalize_tools(listing):
    """The tools of a tools/list result, in the shape the chat uses; entries that are not usable are dropped."""
    out = []
    for t in listing:
        if not isinstance(t, dict) or not isinstance(t.get("name"), str) or not TOOL_NAME.match(t["name"]):
            continue
        schema = t.get("inputSchema") if isinstance(t.get("inputSchema"), dict) else {"type": "object", "properties": {}}
        ann = t.get("annotations") if isinstance(t.get("annotations"), dict) else {}
        desc = (t.get("description") if isinstance(t.get("description"), str) else "")[:600]
        out.append({
            "name": t["name"],
            "title": t["title"] if isinstance(t.get("title"), str) else t["name"],
            "description": desc,
            "inputSchema": schema,
            "readOnly": ann.get("readOnlyHint") is True,
            "tokens": math.ceil(len(json.dumps({"name": t["name"], "description": desc, "parameters": schema})) / 3.5),
        })
    return out


def normalize_result(res):
    """A tools/call result as text the model can read, cut to MAX_RESULT_CHARS."""
    parts = []
    for c in res.get("content") or []:
        kind = c.get("type") if isinstance(c, dict) else None
        if kind == "text":
            parts.append(str(c.get("text", "")))
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} {c.get('mimeType', '')} left out, {len(str(c.get('data', ''))) * 3 // 4} bytes]")
        elif kind == "resource":
            r = c.get("resource") if isinstance(c.get("resource"), dict) else {}
            parts.append(str(r["text"]) if isinstance(r.get("text"), str) else f"[resource {r.get('uri', '')}]")
        elif kind == "resource_link":
            parts.append(f"[resource {c.get('uri', '')}]")
        else:
            parts.append(f"[{kind or 'unknown'} content left out]")
    if not parts and res.get("structuredContent") is not None:
        parts.append(json.dumps(res["structuredContent"], ensure_ascii=False))
    text = "\n".join(parts)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + f"\n[... {len(text) - MAX_RESULT_CHARS} more characters left out]"
    return {"content": text, "isError": bool(res.get("isError"))}


class Manager:
    """The configured servers: start on first use, stop when idle, never more than a few at once."""

    def __init__(self, cfg_path, catalog, env, state_dir, max_running=MAX_RUNNING, idle=IDLE_SECONDS,
                 memory_floor=MEMORY_FLOOR_MB, memory_fn=memory_available_mb):
        self.cfg_path = cfg_path
        self.catalog = catalog
        self.env = env
        self.state_dir = state_dir
        self.max_running = max_running
        self.idle = idle
        self.memory_floor = memory_floor
        self.memory_fn = memory_fn
        self.lock = threading.Lock()
        self.running = {}        # id -> {"client", "last", "busy", "tools"}
        self.failures = {}       # id -> last start error text
        self.start_locks = {}

    # -- configuration

    def servers(self):
        return load_config(self.cfg_path)

    def spec(self, sid):
        try:
            servers = self.servers()
        except ConfigError as e:
            raise ToolError("bad_config", str(e), 500) from None
        if sid not in servers:
            raise ToolError("unknown_server", f"no server named {sid!r}", 404)
        return servers[sid]

    # -- state

    def status(self):
        try:
            servers = self.servers()
            error = None
        except ConfigError as e:
            servers, error = {}, str(e)
        listed = []
        for sid, spec in sorted(servers.items()):
            with self.lock:
                run = self.running.get(sid)
            listed.append({
                "id": sid, "title": spec["title"], "kind": spec["kind"], "network": spec["network"], "catalog": spec["catalog"],
                "state": "ready" if run and not run["client"].dead else ("failed" if sid in self.failures else "stopped"),
                "error": self.failures.get(sid), "tools": len(run["tools"]) if run and run["tools"] is not None else None,
            })
        catalog = [{"id": c["id"], "title": c["title"], "description": c["description"], "runtime": c["runtime"],
                    "network": c["network"], "verified": c["verified"], "added": c["id"] in servers}
                   for c in self.catalog.values()]
        return {"ok": True, "version": 1, "configError": error, "configPath": self.cfg_path, "servers": listed, "catalog": catalog,
                "limits": {"maxRunning": self.max_running, "idleSeconds": self.idle, "memoryFloorMb": self.memory_floor,
                           "memoryAvailableMb": self.memory_fn(), "maxResultChars": MAX_RESULT_CHARS}}

    def _marker(self, sid):
        return os.path.join(self.state_dir, "installed", sid)

    def _gate(self, spec, internet):
        if spec["network"] and not internet:
            raise ToolError("internet_off", f"{spec['id']} reaches the internet and the internet is off", 403)
        if spec["install"] and not os.path.exists(self._marker(spec["id"])) and not internet:
            raise ToolError("internet_off", f"{spec['id']} downloads its package on the first start: turn the internet on once", 403)

    # -- lifecycle

    def _ensure(self, sid, internet):
        spec = self.spec(sid)
        self._gate(spec, internet)
        with self.lock:
            lock = self.start_locks.setdefault(sid, threading.Lock())
        with lock:
            with self.lock:
                run = self.running.get(sid)
                if run and not run["client"].dead:
                    return run
                self.running.pop(sid, None)
            self._make_room(sid)
            avail = self.memory_fn()
            if avail is not None and avail < self.memory_floor:
                raise ToolError("low_memory", f"{avail} MiB available, under the {self.memory_floor} MiB floor: not starting {sid}", 503)
            client = (StdioClient if spec["kind"] == "stdio" else HttpClient)(sid, spec, self.env)
            try:
                client.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                              "clientInfo": {"name": "lm-chat-tools", "version": "1"}}, START_TIMEOUT_S)
                client.notify("notifications/initialized")
            except ToolError as e:
                client.close()
                self.failures[sid] = e.message
                raise
            self.failures.pop(sid, None)
            if spec["install"]:
                os.makedirs(os.path.dirname(self._marker(sid)), exist_ok=True)
                with open(self._marker(sid), "w", encoding="ascii") as f:
                    f.write("1\n")
            run = {"client": client, "last": time.monotonic(), "busy": 0, "tools": None}
            with self.lock:
                self.running[sid] = run
            return run

    def _make_room(self, sid):
        with self.lock:
            live = {k: v for k, v in self.running.items() if not v["client"].dead and k != sid}
            if len(live) < self.max_running:
                return
            idle = [(v["last"], k) for k, v in live.items() if v["busy"] == 0]
            if not idle:
                raise ToolError("too_many_servers", f"{len(live)} servers are busy: wait for one to finish", 503)
            victim = min(idle)[1]
            run = self.running.pop(victim)
        run["client"].close()

    def stop(self, sid):
        with self.lock:
            run = self.running.pop(sid, None)
        if run:
            run["client"].close()
        return {"ok": True}

    def reap(self):
        """Stops the servers that were idle for longer than the limit and the ones whose process died."""
        now = time.monotonic()
        with self.lock:
            gone = [k for k, v in self.running.items() if v["client"].dead or (v["busy"] == 0 and now - v["last"] > self.idle)]
            runs = [self.running.pop(k) for k in gone]
        for run in runs:
            run["client"].close()
        return gone

    def shutdown(self):
        with self.lock:
            runs = list(self.running.values())
            self.running.clear()
        for run in runs:
            run["client"].close()

    # -- tools

    def tools(self, sid, internet):
        run = self._ensure(sid, internet)
        client = run["client"]
        if run["tools"] is None or client.stale:
            listing, cursor = [], None
            for _ in range(20):
                res = client.request("tools/list", {"cursor": cursor} if cursor else {}, CALL_TIMEOUT_S)
                listing.extend(res.get("tools") or [])
                cursor = res.get("nextCursor")
                if not cursor:
                    break
            client.stale = False
            run["tools"] = normalize_tools(listing)
        run["last"] = time.monotonic()
        return {"ok": True, "tools": run["tools"]}

    def call(self, sid, name, arguments, internet):
        if not isinstance(name, str) or not TOOL_NAME.match(name):
            raise ToolError("bad_request", "name must be a tool name", 400)
        if not isinstance(arguments, dict):
            raise ToolError("bad_request", "arguments must be an object", 400)
        run = self._ensure(sid, internet)
        run["busy"] += 1
        try:
            res = run["client"].request("tools/call", {"name": name, "arguments": arguments}, CALL_TIMEOUT_S)
        finally:
            run["busy"] -= 1
            run["last"] = time.monotonic()
        out = normalize_result(res)
        out["ok"] = True
        return out

    # -- catalog

    def add_from_catalog(self, cid, internet, scratch):
        entry = self.catalog.get(cid)
        if not entry:
            raise ToolError("unknown_catalog_entry", f"no catalog entry {cid!r}", 404)
        if entry["network"] and not internet:
            raise ToolError("internet_off", f"{cid} reaches the internet and the internet is off", 403)
        try:
            with open(self.cfg_path, encoding="utf-8") as f:
                doc = json.load(f)
        except FileNotFoundError:
            doc = {"mcpServers": {}}
        except (OSError, ValueError) as e:
            raise ToolError("bad_config", f"{self.cfg_path}: {e}", 500) from None
        if not isinstance(doc, dict) or not isinstance(doc.get("mcpServers"), dict):
            raise ToolError("bad_config", f'{self.cfg_path}: expected {{"mcpServers": {{...}}}}', 500)
        if cid in doc["mcpServers"]:
            return {"ok": True, "added": False}
        os.makedirs(scratch, exist_ok=True)
        item = {"title": entry["title"], "catalog": cid, "command": entry["command"],
                "args": [a.replace("{ROOT}", scratch) for a in entry["args"]], "network": entry["network"], "install": entry["install"]}
        if entry["env"]:
            item["env"] = {k: v.replace("{ROOT}", scratch) for k, v in entry["env"].items()}
        doc["mcpServers"][cid] = item
        os.makedirs(os.path.dirname(self.cfg_path) or ".", exist_ok=True)
        tmp = self.cfg_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        os.replace(tmp, self.cfg_path)
        return {"ok": True, "added": True}


# ---- web search -------------------------------------------------------------------------------

def web_search(search_url, query, count):
    """Asks a SearXNG instance (its JSON API) and returns the results as text. The instance is the
    user's own choice, so a private or loopback address is fine; link-local and the like never are."""
    parts = urllib.parse.urlsplit(search_url) if isinstance(search_url, str) else None
    if not parts or parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ToolError("bad_request", "searchUrl must be an http(s) address without a user name or password", 400)
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ToolError("bad_request", "query must be a non-empty string of at most 500 characters", 400)
    count = count if isinstance(count, int) and 1 <= count <= 10 else 5
    path = parts.path.rstrip("/") + "/search"
    qs = urllib.parse.urlencode({"q": query, "format": "json"})
    conn = open_connection(search_url, True, SEARCH_TIMEOUT_S)
    try:
        conn.request("GET", f"{path}?{qs}", headers={"Accept": "application/json", "User-Agent": "lm-chat-tools", "Host": parts.netloc})
        res = conn.getresponse()
        if res.status != 200:
            raise ToolError("search_failed", f"the search server answered {res.status}", 502)
        body = res.read(MAX_SEARCH_REPLY + 1)
    except (OSError, http.client.HTTPException) as e:
        raise ToolError("search_failed", f"{type(e).__name__}: {e}", 502) from None
    finally:
        conn.close()
    if len(body) > MAX_SEARCH_REPLY:
        raise ToolError("search_failed", "the search answer is over 512 KiB", 502)
    try:
        results = json.loads(body).get("results")
    except (ValueError, AttributeError):
        raise ToolError("search_failed", "the search server did not answer with JSON (is the json format enabled?)", 502) from None
    lines = []
    for i, r in enumerate((results if isinstance(results, list) else [])[:count], 1):
        if isinstance(r, dict):
            lines.append(f"{i}. {str(r.get('title', ''))[:200]}\n   {str(r.get('url', ''))[:300]}\n   {str(r.get('content', ''))[:300]}".rstrip())
    return {"ok": True, "content": "\n".join(lines) if lines else "no results", "isError": False}


# ---- HTTP front -------------------------------------------------------------------------------

ROUTES = (
    ("GET", re.compile(r"^/tools/status$"), "status"),
    ("GET", re.compile(r"^/tools/servers/([a-z0-9_-]+)/tools$"), "tools"),
    ("POST", re.compile(r"^/tools/servers/([a-z0-9_-]+)/call$"), "call"),
    ("POST", re.compile(r"^/tools/servers/([a-z0-9_-]+)/stop$"), "stop"),
    ("POST", re.compile(r"^/tools/catalog/([a-z0-9_-]+)/add$"), "add"),
    ("POST", re.compile(r"^/tools/search$"), "search"),
)


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "chat_tools"
    protocol_version = "HTTP/1.1"
    timeout = 120

    def version_string(self):
        return self.server_version

    def log_message(self, fmt, *args):
        sys.stderr.write("chat_tools: %s %s\n" % (self.address_string(), fmt % args))

    def reply(self, status, obj, extra=()):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def fail(self, status, code, message, extra=()):
        # The request body may be unread (too large, wrong type): close, so its bytes are not read as the next request.
        self.close_connection = True
        self.reply(status, {"ok": False, "error": {"code": code, "message": message}}, [*extra, ("Connection", "close")])

    def authorized(self):
        password = self.server.password
        if password is None:
            host = urllib.parse.urlsplit("//" + (self.headers.get("Host") or "")).hostname or ""
            if host in chat_server.LOOPBACK_NAMES:
                return True
            self.fail(403, "bad_host", "requests must be addressed to a loopback host")
            return False
        header = self.headers.get("Authorization") or ""
        given = b""
        if header[:6].lower() == "basic ":
            try:
                given = base64.b64decode(header[6:].strip(), validate=True).partition(b":")[2]
            except (binascii.Error, ValueError):
                given = b""
        if hmac.compare_digest(given, password.encode()):
            return True
        self.fail(401, "unauthorized", "password required", [("WWW-Authenticate", 'Basic realm="lm-chat-tools"')])
        return False

    def same_origin(self):
        """A page from another site must not drive this service: a POST must come from a loopback
        origin (or none, for a command-line client) and not be marked cross-site."""
        site = (self.headers.get("Sec-Fetch-Site") or "").lower()
        if site and site not in ("same-origin", "none"):
            return False
        origin = self.headers.get("Origin")
        if origin:
            return chat_server.is_loopback(urllib.parse.urlsplit(origin).hostname or "")
        return True

    def body(self):
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise ToolError("bad_request", "send JSON with Content-Type: application/json", 415)
        try:
            n = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            raise ToolError("too_large", f"the body must be at most {MAX_BODY // 1024} KiB", 413)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            doc = json.loads(raw)
        except ValueError:
            raise ToolError("bad_request", "the body is not valid JSON", 400) from None
        if not isinstance(doc, dict):
            raise ToolError("bad_request", "the body must be a JSON object", 400)
        return doc

    def route(self):
        if not self.authorized():
            return
        url = urllib.parse.urlsplit(self.path)
        for method, pattern, name in ROUTES:
            m = pattern.match(url.path)
            if not m:
                continue
            if method != self.command:
                return self.fail(405, "method_not_allowed", f"use {method}", [("Allow", method)])
            try:
                if method == "POST" and not self.same_origin():
                    return self.fail(403, "forbidden", "cross-site request")
                return self.reply(200, self.dispatch(name, m.groups(), url.query))
            except ToolError as e:
                return self.fail(e.status, e.code, e.message)
            except Exception as e:  # keep the service up whatever a server does
                sys.stderr.write(f"chat_tools: {type(e).__name__}: {e}\n")
                return self.fail(500, "internal_error", "internal error")
        self.fail(404, "not_found", "no such path")

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = route

    def dispatch(self, name, groups, query):
        mgr = self.server.manager
        if name == "status":
            return mgr.status()
        if name == "tools":
            flag = urllib.parse.parse_qs(query).get("internet", ["0"])[0]
            return mgr.tools(groups[0], flag == "1")
        doc = self.body()
        internet = doc.get("internet") is True
        if name == "call":
            return mgr.call(groups[0], doc.get("name"), doc.get("arguments", {}), internet)
        if name == "stop":
            return mgr.stop(groups[0])
        if name == "add":
            return mgr.add_from_catalog(groups[0], internet, self.server.scratch)
        if not internet:
            raise ToolError("internet_off", "web search needs the internet and it is off", 403)
        return web_search(doc.get("searchUrl"), doc.get("query"), doc.get("count"))


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, manager, password, scratch):
        self.address_family = socket.AF_INET6 if ":" in addr[0] else socket.AF_INET
        self.manager = manager
        self.password = password
        self.scratch = scratch
        super().__init__(addr, Handler)


def main(argv=None, env=None):
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--config", default=config_path(env))
    ap.add_argument("--catalog", default=os.path.join(HERE, "chat_tools_catalog.json"))
    ap.add_argument("--state-dir", default=os.path.join(os.path.expanduser("~"), ".local", "share", "lm-chat"))
    ap.add_argument("--scratch", default=os.path.join(os.path.expanduser("~"), "chat-files"),
                    help="the one directory the filesystem server may touch (default ~/chat-files)")
    ap.add_argument("--max-running", type=int, default=MAX_RUNNING)
    ap.add_argument("--idle-seconds", type=float, default=IDLE_SECONDS)
    args = ap.parse_args(argv)
    password = chat_server.check_config(args.host, env)
    try:
        catalog = load_catalog(args.catalog)
    except (OSError, ValueError, KeyError, ConfigError) as e:
        sys.exit(f"chat_tools: catalog {args.catalog}: {e}")
    manager = Manager(args.config, catalog, dict(env), args.state_dir, args.max_running, args.idle_seconds)
    try:
        server = Server((args.host, args.port), manager, password, args.scratch)
    except OSError as e:
        sys.exit(f"chat_tools: {e}")

    def reaper():
        while True:
            time.sleep(15)
            manager.reap()

    threading.Thread(target=reaper, daemon=True).start()
    host = f"[{args.host}]" if ":" in args.host else args.host
    print(f"chat_tools: http://{host}:{server.server_address[1]}/tools/status (config {args.config})", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
