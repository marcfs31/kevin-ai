#!/usr/bin/env python3
"""Serve the chat web app (web/dist) and forward its API calls to lm-serve.

    chat_server.py [--host 127.0.0.1] [--port 8000] [--dist web/dist]
                   [--upstream http://127.0.0.1:8080] [--upstream-timeout 3600]

- GET of a file under --dist is served with its MIME type and a strict Content-Security-Policy;
  "/" is index.html. No directory listings; a path that resolves outside --dist is a 404.
- /v1/* and /health go to --upstream unchanged, and the answer comes back as it arrives
  (server-sent events are not buffered). When the browser goes away, the upstream connection is
  closed, which stops lm-serve's reply, even while it is still reading the prompt.
- LM_SERVE_API_KEY in the environment is sent upstream as "Authorization: Bearer <key>"; it is
  never sent to the browser, and the browser's own Authorization and Cookie headers are dropped.
- It listens on 127.0.0.1. A --host that is not loopback needs CHAT_PASSWORD in the
  environment: every request then needs HTTP Basic auth with that password (any user name).
  Without a password, requests must name a loopback Host (a DNS-rebinding guard).
- Request bodies are limited to 24 MiB, like lm-serve's image limit.

Python 3 standard library only.
"""

import argparse
import base64
import binascii
import hmac
import http.client
import http.server
import ipaddress
import json
import os
import select
import socket
import sys
import urllib.parse

MAX_BODY = 24 * 1024 * 1024
CLIENT_TIMEOUT_S = 60
CONNECT_TIMEOUT_S = 10

CSP = "; ".join([
    "default-src 'none'",
    "script-src 'self'",
    "style-src 'self'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "worker-src 'self'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "object-src 'none'",
])

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".wasm": "application/wasm",
    ".txt": "text/plain; charset=utf-8",
}

LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_config(host, env):
    """The password for a non-loopback host; exits with a message when it is missing."""
    password = env.get("CHAT_PASSWORD", "")
    if not is_loopback(host) and not password:
        sys.exit(f"chat_server: --host {host} is reachable from the network: set CHAT_PASSWORD first")
    return password or None


def index_dist(root):
    """URL path -> file for every regular file under root that stays inside it."""
    files = {}
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            full = os.path.join(dirpath, name)
            real = os.path.realpath(full)
            if os.path.commonpath([root, real]) == root and os.path.isfile(real):
                files["/" + os.path.relpath(full, root).replace(os.sep, "/")] = real
    return files


def response_headers(upstream):
    """Headers for the browser, chosen by the proxy: nothing upstream sends is copied verbatim,
    so a folded or hostile upstream header cannot split the response."""
    ctype = (upstream.getheader("Content-Type") or "").lower()
    if ctype.startswith("text/event-stream"):
        ctype = "text/event-stream"
    elif ctype.startswith("application/json"):
        ctype = "application/json"
    else:
        ctype = "text/plain; charset=utf-8"
    out = [("Content-Type", ctype), ("Cache-Control", "no-store")]
    retry = (upstream.getheader("Retry-After") or "").strip()
    if retry.isascii() and retry.isdigit():
        out.append(("Retry-After", str(int(retry))))
    return out


def is_api(path):
    return path == "/health" or path.startswith("/v1/")


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "chat_server"
    protocol_version = "HTTP/1.1"
    timeout = CLIENT_TIMEOUT_S

    def version_string(self):
        return self.server_version

    # ---- shared --------------------------------------------------------------------------

    def common_headers(self):
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")

    def send_json_error(self, status, code, message, extra=()):
        body = json.dumps({"error": {"message": message, "type": "proxy_error", "code": code}}).encode()
        self.send_response(status)
        self.common_headers()
        for k, v in extra:
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True

    def authorized(self):
        password = self.server.password
        if password is None:
            host = urllib.parse.urlsplit("//" + (self.headers.get("Host") or "")).hostname or ""
            if host in LOOPBACK_NAMES:
                return True
            self.send_json_error(403, "bad_host", "requests must be addressed to a loopback host")
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
        self.send_json_error(401, "unauthorized", "password required",
                             [("WWW-Authenticate", 'Basic realm="lm-chat", charset="UTF-8"')])
        return False

    def route(self):
        if not self.authorized():
            return
        path = urllib.parse.urlsplit(self.path).path
        if is_api(path):
            self.proxy()
        elif self.command in ("GET", "HEAD"):
            self.static(path)
        else:
            self.send_json_error(405, "method_not_allowed", "method not allowed", [("Allow", "GET, HEAD")])

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = route

    def log_message(self, fmt, *args):
        sys.stderr.write("chat_server: %s %s\n" % (self.address_string(), fmt % args))

    # ---- static files --------------------------------------------------------------------

    def static(self, path):
        rel = urllib.parse.unquote(path)
        if "\x00" in rel:
            self.send_json_error(400, "bad_path", "bad path")
            return
        if rel == "/":
            rel = "/index.html"
        # The URL is only a key into the files found under dist, never part of a path itself.
        target = self.server.find_file(rel)
        if target is None:
            self.send_json_error(404, "not_found", "not found")
            return
        ext = os.path.splitext(target)[1].lower()
        with open(target, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.common_headers()
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        immutable = rel.startswith("/assets/")
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if immutable else "no-cache")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    # ---- API proxy -----------------------------------------------------------------------

    def client_gone(self):
        """True when the browser has closed its side of the connection."""
        try:
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (BlockingIOError, InterruptedError, socket.timeout):
            return False
        except OSError:
            return True

    def proxy(self):
        length = self.headers.get("Content-Length")
        body = None
        if length is not None:
            try:
                n = int(length)
            except ValueError:
                n = -1
            if n < 0:
                self.send_json_error(400, "bad_length", "bad Content-Length")
                return
            if n > MAX_BODY:
                self.send_json_error(413, "too_large", "request body over 24 MiB")
                return
            body = self.rfile.read(n)
            if len(body) != n:
                self.close_connection = True
                return
        elif self.headers.get("Transfer-Encoding"):
            self.send_json_error(411, "length_required", "Content-Length required")
            return

        up = self.server.upstream
        headers = {"Host": up.netloc}
        for name in ("Content-Type", "Accept"):
            if self.headers.get(name):
                headers[name] = self.headers[name]
        if body is not None:
            headers["Content-Length"] = str(len(body))
        if self.server.api_key:
            headers["Authorization"] = "Bearer " + self.server.api_key
        conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
        # A short connect timeout, so an upstream host that is off fails fast; then the long
        # one, since lm-serve may read a prompt for minutes before it answers.
        conn = conn_cls(up.hostname, up.port, timeout=CONNECT_TIMEOUT_S)
        try:
            try:
                conn.connect()
                conn.sock.settimeout(self.server.upstream_timeout)
                conn.request(self.command, self.path, body=body, headers=headers)
            except OSError:
                self.send_json_error(502, "upstream_unreachable", "lm-serve is not reachable")
                return
            # lm-serve sends nothing until the first token, which can take minutes: watch the
            # browser meanwhile and hang up on lm-serve if it leaves.
            watch_client = True
            while True:
                watched = [conn.sock] + ([self.connection] if watch_client else [])
                ready, _, _ = select.select(watched, [], [], 1.0)
                if conn.sock in ready:
                    break
                if self.connection in ready:
                    if self.client_gone():
                        self.close_connection = True
                        return
                    watch_client = False  # it sent more bytes; stop peeking at them
            try:
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException):
                self.send_json_error(502, "upstream_failed", "lm-serve closed the connection")
                return
            self.send_response(resp.status)
            self.common_headers()
            for name, value in response_headers(resp):
                self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            if self.command == "HEAD":
                return
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (OSError, http.client.HTTPException):
            # The headers are out already: the browser sees the stream end without a finish.
            self.close_connection = True
        finally:
            conn.close()


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, dist, upstream, password, api_key, upstream_timeout=3600):
        family = socket.AF_INET6 if ":" in addr[0] else socket.AF_INET
        self.address_family = family
        self.dist = os.path.realpath(dist)
        self.upstream = urllib.parse.urlsplit(upstream)
        if self.upstream.scheme not in ("http", "https") or not self.upstream.hostname:
            raise ValueError(f"bad --upstream {upstream!r}")
        self.password = password
        self.api_key = api_key or None
        self.upstream_timeout = upstream_timeout
        self.files = index_dist(self.dist)
        super().__init__(addr, Handler)

    def find_file(self, url_path):
        """The file for a URL path, re-reading dist once on a miss (a new build was deployed)."""
        found = self.files.get(url_path)
        if found is None:
            self.files = index_dist(self.dist)
            found = self.files.get(url_path)
        return found if found is not None and os.path.isfile(found) else None


def main(argv=None, env=None):
    env = os.environ if env is None else env
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dist", default=os.path.join(os.path.dirname(here), "web", "dist"))
    ap.add_argument("--upstream", default="http://127.0.0.1:8080")
    ap.add_argument("--upstream-timeout", type=float, default=3600.0,
                    help="seconds lm-serve may stay silent before the request fails (default 3600)")
    args = ap.parse_args(argv)
    password = check_config(args.host, env)
    if not os.path.isfile(os.path.join(args.dist, "index.html")):
        sys.exit(f"chat_server: {args.dist}/index.html not found: build the app first (npm run build in web/)")
    try:
        server = Server((args.host, args.port), args.dist, args.upstream, password, env.get("LM_SERVE_API_KEY"),
                        args.upstream_timeout)
    except (OSError, ValueError) as e:
        sys.exit(f"chat_server: {e}")
    host = f"[{args.host}]" if ":" in args.host else args.host
    print(f"chat_server: http://{host}:{server.server_address[1]}/ -> {args.upstream}"
          f"{' (password required)' if password else ''}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
