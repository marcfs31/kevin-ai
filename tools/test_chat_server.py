#!/usr/bin/env python3
"""Tests for tools/chat_server.py against a fake lm-serve. Run from anywhere:

    python3 -m unittest discover -s tools -p 'test_chat_server.py' -v
"""

import base64
import gzip
import http.client
import http.server
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import chat_server  # noqa: E402

chat_server.Handler.log_message = lambda *args: None  # keep the test output readable

EVENTS = [b'data: {"choices":[{"index":0,"delta":{"content":"%d"}}]}\n\n' % i for i in range(5)] + [b"data: [DONE]\n\n"]


class FakeUpstream(http.server.ThreadingHTTPServer):
    """A stand-in for lm-serve that records requests and streams slow SSE."""

    daemon_threads = True

    def __init__(self):
        self.requests = []
        self.gone = threading.Event()
        self.prefill_s = 0.0
        self.events = EVENTS
        super().__init__(("127.0.0.1", 0), FakeHandler)


class FakeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.requests.append(("GET", self.path, dict(self.headers), b""))
        body = b'{"object":"list","data":[{"id":"fake"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append(("POST", self.path, dict(self.headers), body))
        if self.path == "/v1/folded":
            self.wfile.write(b"HTTP/1.1 503 Busy\r\nContent-Type: application/json\r\nRetry-After: 1\r\n"
                             b"\tSet-Cookie: injected=1\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            self.close_connection = True
            return
        if self.path == "/v1/bad":
            err = b'{"error":{"message":"no","type":"invalid_request_error","code":"image_invalid"}}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return
        # Like lm-serve: silence while reading the prompt, watching for a vanished client.
        deadline = time.monotonic() + self.server.prefill_s
        while time.monotonic() < deadline:
            self.connection.settimeout(0.05)
            try:
                if self.connection.recv(1, socket.MSG_PEEK) == b"":
                    self.server.gone.set()
                    return
            except (socket.timeout, BlockingIOError):
                pass
            except OSError:
                self.server.gone.set()
                return
        self.connection.settimeout(None)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for ev in self.server.events:
                self.wfile.write(ev)
                self.wfile.flush()
                time.sleep(0.05)
        except OSError:
            self.server.gone.set()
        self.close_connection = True


def start(server):
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    return server


class ProxyTest(unittest.TestCase):
    password = None
    api_key = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dist = os.path.join(self.tmp.name, "dist")
        os.makedirs(os.path.join(self.dist, "assets"))
        with open(os.path.join(self.dist, "index.html"), "w") as f:
            f.write("<!doctype html><title>t</title>")
        with open(os.path.join(self.dist, "assets", "app.js"), "w") as f:
            f.write("console.log(1)")
        with open(os.path.join(self.tmp.name, "secret.txt"), "w") as f:
            f.write("outside")
        self.upstream = start(FakeUpstream())
        self.tools_upstream = start(FakeUpstream())
        url = "http://127.0.0.1:%d" % self.upstream.server_address[1]
        tools_url = "http://127.0.0.1:%d" % self.tools_upstream.server_address[1]
        self.proxy = start(chat_server.Server(("127.0.0.1", 0), self.dist, url, self.password, self.api_key,
                                              tools_upstream=tools_url))
        self.port = self.proxy.server_address[1]

    def tearDown(self):
        for s in (self.proxy, self.upstream, self.tools_upstream):
            s.shutdown()
            s.server_close()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp, data


class StaticTest(ProxyTest):
    def test_index_with_csp_and_mime(self):
        resp, data = self.request("GET", "/")
        self.assertEqual(resp.status, 200)
        self.assertIn(b"<title>t</title>", data)
        self.assertEqual(resp.getheader("Content-Type"), "text/html; charset=utf-8")
        csp = resp.getheader("Content-Security-Policy")
        self.assertIn("script-src 'self'", csp)
        self.assertNotIn("unsafe", csp)
        self.assertEqual(resp.getheader("X-Content-Type-Options"), "nosniff")

    def test_asset_mime_and_cache(self):
        resp, _ = self.request("GET", "/assets/app.js")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "text/javascript; charset=utf-8")
        self.assertIn("immutable", resp.getheader("Cache-Control"))

    def test_no_directory_listing(self):
        for path in ("/assets/", "/assets", "/missing.js"):
            resp, data = self.request("GET", path)
            self.assertEqual(resp.status, 404, path)
            self.assertNotIn(b"app.js", data)

    def test_path_traversal_refused(self):
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/%2e%2e/%2e%2e/secret.txt", "/..%2fsecret.txt"):
            resp, data = self.request("GET", path)
            self.assertEqual(resp.status, 404, path)
            self.assertNotIn(b"outside", data)

    def test_symlink_out_of_dist_refused(self):
        os.symlink(os.path.join(self.tmp.name, "secret.txt"), os.path.join(self.dist, "link.txt"))
        resp, data = self.request("GET", "/link.txt")
        self.assertEqual(resp.status, 404)
        self.assertNotIn(b"outside", data)

    def test_file_added_after_start_is_served(self):
        with open(os.path.join(self.dist, "assets", "new.css"), "w") as f:
            f.write("a{}")
        resp, data = self.request("GET", "/assets/new.css")
        self.assertEqual((resp.status, data), (200, b"a{}"))
        self.assertEqual(resp.getheader("Content-Type"), "text/css; charset=utf-8")

    def test_post_to_static_refused(self):
        resp, _ = self.request("POST", "/index.html", body=b"x")
        self.assertEqual(resp.status, 405)

    def test_foreign_host_refused_without_password(self):
        resp, _ = self.request("GET", "/", headers={"Host": "evil.example:8000"})
        self.assertEqual(resp.status, 403)


class ApiTest(ProxyTest):
    def test_health_and_models_forwarded(self):
        resp, data = self.request("GET", "/v1/models")
        self.assertEqual(resp.status, 200)
        self.assertEqual(json.loads(data)["data"][0]["id"], "fake")
        self.assertEqual(self.upstream.requests[-1][1], "/v1/models")

    def test_sse_passes_through_in_order_without_buffering(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/v1/chat/completions", body=b'{"stream":true}', headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
        first = time.monotonic()
        got = resp.read1(65536)
        early = time.monotonic() - first
        rest = resp.read()
        conn.close()
        # The first event arrives long before the last is sent (5 x 50 ms).
        self.assertLess(early, 0.2)
        self.assertTrue(got.startswith(EVENTS[0]))
        self.assertEqual(got + rest, b"".join(EVENTS))
        self.assertEqual(self.upstream.requests[-1][3], b'{"stream":true}')

    def test_error_status_and_body_forwarded(self):
        resp, data = self.request("POST", "/v1/bad", body=b"{}", headers={"Content-Type": "application/json"})
        self.assertEqual(resp.status, 400)
        self.assertEqual(json.loads(data)["error"]["code"], "image_invalid")

    def test_folded_upstream_header_not_passed_on(self):
        resp, _ = self.request("POST", "/v1/folded", body=b"{}")
        self.assertEqual(resp.status, 503)
        self.assertIsNone(resp.getheader("Set-Cookie"))
        self.assertIsNone(resp.getheader("Retry-After"))
        self.assertEqual(resp.getheader("Content-Type"), "application/json")

    def test_response_headers_are_the_proxys_own(self):
        class Up:
            def __init__(self, **h):
                self.h = h

            def getheader(self, name):
                return self.h.get(name.replace("-", "_"))

        h = dict(chat_server.response_headers(Up(Content_Type="text/event-stream; charset=utf-8", Retry_After=" 5 ")))
        self.assertEqual(h, {"Content-Type": "text/event-stream", "Cache-Control": "no-store", "Retry-After": "5"})
        h = dict(chat_server.response_headers(Up(Content_Type="text/html\r\nX: y", Retry_After="Wed, 21 Oct")))
        self.assertEqual(h, {"Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store"})

    def test_disconnect_while_reading_prompt_closes_upstream(self):
        self.upstream.prefill_s = 5.0
        s = socket.create_connection(("127.0.0.1", self.port))
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  b"Content-Length: 2\r\n\r\n{}")
        time.sleep(0.3)
        s.close()
        self.assertTrue(self.upstream.gone.wait(3.0), "upstream connection was not closed")

    def test_disconnect_while_streaming_closes_upstream(self):
        self.upstream.events = EVENTS[:1] * 60
        s = socket.create_connection(("127.0.0.1", self.port))
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 2\r\n\r\n{}")
        self.assertIn(b"data: ", s.recv(65536) + s.recv(65536))
        s.close()
        self.assertTrue(self.upstream.gone.wait(3.0), "upstream connection was not closed")

    def test_body_limit(self):
        s = socket.create_connection(("127.0.0.1", self.port))
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: %d\r\n\r\n" % (24 * 1024 * 1024 + 1))
        head = s.recv(4096)
        s.close()
        self.assertTrue(head.startswith(b"HTTP/1.1 413"), head)
        self.assertEqual(self.upstream.requests, [])

    def test_no_key_and_no_browser_credentials_upstream(self):
        self.request("GET", "/health", headers={"Authorization": "Bearer from-browser", "Cookie": "a=b"})
        headers = self.upstream.requests[-1][2]
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("Cookie", headers)

    def test_upstream_down_is_502(self):
        self.upstream.shutdown()
        self.upstream.server_close()
        resp, data = self.request("GET", "/health")
        self.assertEqual(resp.status, 502)
        self.assertEqual(json.loads(data)["error"]["code"], "upstream_unreachable")
        self.upstream = start(FakeUpstream())  # for tearDown

    def test_connect_is_bounded_then_waits_long(self):
        upstream_port = self.upstream.server_address[1]
        seen = {}
        real = http.client.HTTPConnection

        class Recording(real):
            def connect(self):
                if self.port == upstream_port:
                    seen["connect"] = self.timeout
                super().connect()

            def request(self, *args, **kwargs):
                if self.port == upstream_port:
                    seen["request"] = self.sock.gettimeout()
                super().request(*args, **kwargs)

        with mock.patch.object(chat_server.http.client, "HTTPConnection", Recording):
            resp, _ = self.request("GET", "/health")
        self.assertEqual(resp.status, 200)
        self.assertEqual(seen, {"connect": chat_server.CONNECT_TIMEOUT_S, "request": self.proxy.upstream_timeout})


class GzipTest(ProxyTest):
    def get(self, path, encoding=None, method="GET"):
        hdr = {"Accept-Encoding": encoding} if encoding else {}
        return self.request(method, path, headers=hdr)

    def big_asset(self):
        with open(os.path.join(self.dist, "assets", "big.js"), "w") as f:
            f.write("console.log('hello');\n" * 400)
        return "/assets/big.js"

    def test_a_text_file_is_gzipped_when_asked_and_decodes_to_the_same_bytes(self):
        path = self.big_asset()
        raw = self.get(path)[1]
        resp, packed = self.get(path, "gzip, deflate, br")
        self.assertEqual(resp.getheader("Content-Encoding"), "gzip")
        self.assertEqual(resp.getheader("Vary"), "Accept-Encoding")
        self.assertEqual(int(resp.getheader("Content-Length")), len(packed))
        self.assertLess(len(packed), len(raw) // 10)
        self.assertEqual(gzip.decompress(packed), raw)
        self.assertIsNone(self.get(path)[0].getheader("Content-Encoding"))

    def test_not_gzipped_when_refused_unknown_or_too_small_and_never_for_images(self):
        path = self.big_asset()
        for enc in ("identity", "gzip;q=0", "br", "gzip;q=junk"):
            resp, body = self.get(path, enc)
            self.assertIsNone(resp.getheader("Content-Encoding"), enc)
            self.assertEqual(resp.getheader("Vary"), "Accept-Encoding")
        self.assertIsNone(self.get("/assets/app.js", "gzip")[0].getheader("Content-Encoding"))       # 14 bytes: not worth it
        with open(os.path.join(self.dist, "assets", "logo.png"), "wb") as f:
            f.write(b"\x89PNG" + b"0" * 2000)
        resp, _ = self.get("/assets/logo.png", "gzip")
        self.assertIsNone(resp.getheader("Content-Encoding"))
        self.assertIsNone(resp.getheader("Vary"))

    def test_head_reports_the_compressed_length_and_the_cache_follows_a_changed_file(self):
        path = self.big_asset()
        resp, body = self.get(path, "gzip", method="HEAD")
        self.assertEqual(body, b"")
        packed_len = int(resp.getheader("Content-Length"))
        self.assertEqual(packed_len, len(self.get(path, "gzip")[1]))
        with open(os.path.join(self.dist, "assets", "big.js"), "w") as f:
            f.write("console.log('changed');\n" * 900)
        os.utime(os.path.join(self.dist, "assets", "big.js"), None)
        again = self.get(path, "gzip")[1]
        self.assertEqual(gzip.decompress(again), ("console.log('changed');\n" * 900).encode())


class ToolsTest(ProxyTest):
    def test_tools_paths_go_to_the_tool_service_and_not_to_lm_serve(self):
        resp, data = self.request("GET", "/tools/status?x=1")
        self.assertEqual(resp.status, 200)
        self.assertEqual(self.tools_upstream.requests[-1][:2], ("GET", "/tools/status?x=1"))
        self.assertEqual(self.upstream.requests, [])
        self.request("GET", "/health")
        self.assertEqual(self.upstream.requests[-1][1], "/health")
        self.assertEqual(len(self.tools_upstream.requests), 1)

    def test_post_body_and_fetch_site_reach_the_tool_service_but_not_its_credentials_or_origin(self):
        self.request("POST", "/tools/search", body=b'{"query":"x"}', headers={
            "Content-Type": "application/json", "Origin": "http://127.0.0.1:%d" % self.port, "Sec-Fetch-Site": "same-origin",
            "Authorization": "Bearer from-browser", "Cookie": "a=b"})
        method, path, headers, body = self.tools_upstream.requests[-1]
        self.assertEqual((method, path, body), ("POST", "/tools/search", b'{"query":"x"}'))
        self.assertEqual(headers["Sec-Fetch-Site"], "same-origin")
        self.assertNotIn("Origin", headers)
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("Cookie", headers)

    def test_an_origin_equal_to_the_host_the_page_was_served_on_passes_and_another_site_does_not(self):
        # The browser names the address it used in both Host and Origin (behind a password that can be a LAN
        # address; here a name the proxy accepts without one), whatever the port.
        hdr = {"Content-Type": "application/json", "Host": "localhost:8000", "Origin": "http://localhost:8000"}
        resp, _ = self.request("POST", "/tools/search", body=b"{}", headers=hdr)
        self.assertEqual(resp.status, 200)
        before = len(self.tools_upstream.requests)
        resp, data = self.request("POST", "/tools/search", body=b"{}", headers={**hdr, "Origin": "https://evil.example"})
        self.assertEqual((resp.status, json.loads(data)["error"]["code"]), (403, "forbidden"))
        self.assertEqual(len(self.tools_upstream.requests), before)

    def test_the_origin_is_not_passed_to_lm_serve(self):
        self.request("GET", "/health", headers={"Origin": "http://127.0.0.1:5173", "Sec-Fetch-Site": "same-origin"})
        headers = self.upstream.requests[-1][2]
        self.assertNotIn("Origin", headers)
        self.assertNotIn("Sec-Fetch-Site", headers)

    def test_a_tool_service_that_is_down_is_a_502_that_names_it(self):
        self.tools_upstream.shutdown()
        self.tools_upstream.server_close()
        resp, data = self.request("GET", "/tools/status")
        self.assertEqual(resp.status, 502)
        err = json.loads(data)["error"]
        self.assertEqual(err["code"], "upstream_unreachable")
        self.assertIn("tool service", err["message"])
        self.tools_upstream = start(FakeUpstream())  # for tearDown

    def test_without_a_tool_service_configured_the_paths_are_404(self):
        proxy = start(chat_server.Server(("127.0.0.1", 0), self.dist, "http://127.0.0.1:1", None, None))
        try:
            conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
            conn.request("GET", "/tools/status")
            resp = conn.getresponse()
            self.assertEqual((resp.status, json.loads(resp.read())["error"]["code"]), (404, "tools_not_configured"))
            conn.close()
        finally:
            proxy.shutdown()
            proxy.server_close()


class KeyTest(ProxyTest):
    api_key = "sk-test"

    def test_the_key_is_for_lm_serve_only_and_is_not_sent_to_the_tool_service(self):
        self.request("GET", "/tools/status")
        self.assertNotIn("Authorization", self.tools_upstream.requests[-1][2])
        self.request("GET", "/health")
        self.assertIn("Authorization", self.upstream.requests[-1][2])

    def test_key_added_upstream_never_to_browser(self):
        resp, data = self.request("GET", "/v1/models", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(self.upstream.requests[-1][2].get("Authorization"), "Bearer sk-test")
        self.assertNotIn(b"sk-test", data)
        self.assertNotIn("sk-test", str(resp.getheaders()))


class PasswordTest(ProxyTest):
    password = "correct horse"

    def auth(self, password):
        return {"Authorization": "Basic " + base64.b64encode(b"any:" + password.encode()).decode()}

    def test_password_required(self):
        resp, _ = self.request("GET", "/")
        self.assertEqual(resp.status, 401)
        self.assertIn("Basic", resp.getheader("WWW-Authenticate"))
        resp, _ = self.request("GET", "/v1/models", headers=self.auth("wrong"))
        self.assertEqual(resp.status, 401)
        self.assertEqual(self.upstream.requests, [])
        resp, _ = self.request("GET", "/", headers={"Authorization": "Basic !!notbase64"})
        self.assertEqual(resp.status, 401)

    def test_password_accepted(self):
        resp, _ = self.request("GET", "/", headers=self.auth("correct horse"))
        self.assertEqual(resp.status, 200)
        resp, _ = self.request("GET", "/v1/models", headers=self.auth("correct horse"))
        self.assertEqual(resp.status, 200)
        self.assertNotIn("Authorization", self.upstream.requests[-1][2])


class ConfigTest(unittest.TestCase):
    def test_off_loopback_needs_password(self):
        with self.assertRaises(SystemExit) as cm:
            chat_server.check_config("0.0.0.0", {})
        self.assertIn("CHAT_PASSWORD", str(cm.exception.code))
        with self.assertRaises(SystemExit):
            chat_server.check_config("192.168.1.96", {"CHAT_PASSWORD": ""})
        self.assertEqual(chat_server.check_config("0.0.0.0", {"CHAT_PASSWORD": "pw"}), "pw")

    def test_loopback_needs_no_password(self):
        for host in ("127.0.0.1", "localhost", "::1", "127.0.0.2"):
            self.assertIsNone(chat_server.check_config(host, {}), host)

    def test_main_refuses_network_host_without_password(self):
        with self.assertRaises(SystemExit):
            chat_server.main(["--host", "0.0.0.0", "--port", "0"], env={})


if __name__ == "__main__":
    unittest.main()
