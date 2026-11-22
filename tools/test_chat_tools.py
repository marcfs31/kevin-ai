#!/usr/bin/env python3
"""Tests for tools/chat_tools.py against a fake MCP server (tools/fake_mcp_server.py), a fake remote
MCP server and a fake SearXNG. Run from anywhere:

    python3 -m unittest discover -s tools -p 'test_chat_tools.py' -v
"""

import base64
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

import chat_tools  # noqa: E402

chat_tools.Handler.log_message = lambda *args: None
FAKE = os.path.join(TOOLS, "fake_mcp_server.py")


def stdio(*extra, **kw):
    spec = {"command": sys.executable, "args": [FAKE, *extra]}
    spec.update(kw)
    return spec


class Base(unittest.TestCase):
    """A service on a free port with a config file the test fills in."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = os.path.join(self.tmp, "mcp.json")
        self.scratch = os.path.join(self.tmp, "scratch")
        self.env = {"PATH": os.environ.get("PATH", ""), "HOME": self.tmp, "SECRET_TOKEN": "hunter2"}
        self.catalog = chat_tools.load_catalog(os.path.join(TOOLS, "chat_tools_catalog.json"))
        self.managers = []

    def tearDown(self):
        for m in self.managers:
            m.shutdown()

    def write_config(self, servers):
        with open(self.cfg, "w", encoding="utf-8") as f:
            json.dump({"mcpServers": servers}, f)

    def manager(self, **kw):
        kw.setdefault("memory_fn", lambda: None)
        m = chat_tools.Manager(self.cfg, self.catalog, self.env, os.path.join(self.tmp, "state"), **kw)
        self.managers.append(m)
        return m

    def serve(self, manager=None, password=None):
        manager = manager or self.manager()
        server = chat_tools.Server(("127.0.0.1", 0), manager, password, self.scratch)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.port = server.server_address[1]
        return server

    def http(self, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        hdr = {"Content-Type": "application/json"} if method == "POST" else {}
        hdr.update(headers or {})
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        conn.request(method, path, data, hdr)
        res = conn.getresponse()
        payload = res.read()
        conn.close()
        try:
            return res.status, json.loads(payload)
        except ValueError:
            return res.status, payload


class Configuration(Base):
    def test_valid_config_is_filled_in(self):
        self.write_config({"fs": {"command": "npx", "args": ["-y", "x"], "env": {"A": "b"}}, "remote": {"url": "https://example.com/mcp", "bearerEnv": "TOK"}})
        servers = chat_tools.load_config(self.cfg)
        self.assertEqual(servers["fs"]["kind"], "stdio")
        self.assertFalse(servers["fs"]["network"])
        self.assertEqual(servers["remote"]["kind"], "http")
        self.assertTrue(servers["remote"]["network"])
        self.assertEqual(servers["remote"]["bearer_env"], "TOK")

    def test_missing_file_means_no_servers(self):
        self.assertEqual(chat_tools.load_config(self.cfg), {})

    def test_bad_entries_are_refused_with_the_reason(self):
        for bad, text in [
            ({"Bad Name": {"command": "x"}}, "bad server name"),
            ({"a": {}}, "needs a command or a url"),
            ({"a": {"command": "x", "args": "no"}}, "args must be a list"),
            ({"a": {"command": "x", "env": {"k": 1}}}, "env must be an object"),
            ({"a": {"url": "ftp://x"}}, "http(s)"),
            ({"a": {"url": "https://user:pw@x/"}}, "without a user name"),
            ({"a": "x"}, "must be an object"),
        ]:
            self.write_config(bad)
            with self.assertRaises(chat_tools.ConfigError, msg=str(bad)) as cm:
                chat_tools.load_config(self.cfg)
            self.assertIn(text, str(cm.exception))

    def test_unreadable_config_shows_in_status_instead_of_crashing(self):
        with open(self.cfg, "w") as f:
            f.write("{not json")
        self.serve()
        status, doc = self.http("GET", "/tools/status")
        self.assertEqual(status, 200)
        self.assertIn("mcp.json", doc["configError"])
        self.assertEqual(doc["servers"], [])

    def test_the_catalog_is_pinned_and_parses(self):
        self.assertGreaterEqual(len(self.catalog), 5)
        for cid, entry in self.catalog.items():
            joined = " ".join(entry["args"])
            pinned = "==" in joined or any("@" in a[1:] and a.rsplit("@", 1)[1][0].isdigit() for a in entry["args"])
            self.assertTrue(pinned, cid + " is not pinned")
            self.assertNotIn("latest", joined, cid)


class Basics(Base):
    def setUp(self):
        super().setUp()
        self.write_config({"fake": stdio()})
        self.serve()

    def tools(self):
        status, doc = self.http("GET", "/tools/servers/fake/tools")
        self.assertEqual(status, 200, doc)
        return {t["name"]: t for t in doc["tools"]}

    def call(self, name, args=None, **extra):
        return self.http("POST", "/tools/servers/fake/call", {"name": name, "arguments": args or {}, **extra})

    def test_lists_normalized_tools_and_drops_unusable_ones(self):
        tools = self.tools()
        self.assertIn("echo", tools)
        self.assertNotIn("bad name!", tools)
        self.assertTrue(tools["echo"]["readOnly"])
        self.assertFalse(tools["write_file"]["readOnly"])
        self.assertGreater(tools["echo"]["tokens"], 5)
        self.assertEqual(tools["echo"]["inputSchema"]["type"], "object")

    def test_calls_a_tool_and_returns_its_text(self):
        status, doc = self.call("echo", {"text": "héllo"})
        self.assertEqual((status, doc["content"], doc["isError"]), (200, "héllo", False))

    def test_a_server_error_is_a_502_with_its_message(self):
        status, doc = self.call("nope")
        self.assertEqual(status, 502)
        self.assertEqual(doc["error"]["code"], "server_error")
        self.assertIn("unknown tool", doc["error"]["message"])

    def test_long_results_are_cut_and_say_so(self):
        status, doc = self.call("big")
        self.assertEqual(status, 200)
        self.assertLess(len(doc["content"]), chat_tools.MAX_RESULT_CHARS + 100)
        self.assertIn("12000 more characters left out", doc["content"])

    def test_other_content_types_become_text(self):
        status, doc = self.call("image")
        self.assertIn("[image image/png left out", doc["content"])
        self.assertIn("from a resource", doc["content"])
        self.assertIn("[resource file:///b]", doc["content"])
        self.assertEqual(self.call("structured")[1]["content"], '{"answer": 42}')

    def test_the_server_is_started_once_and_listed_as_ready(self):
        self.tools()
        status, doc = self.http("GET", "/tools/status")
        row = doc["servers"][0]
        self.assertEqual((row["id"], row["state"], row["tools"]), ("fake", "ready", 10))

    def test_a_request_the_client_does_not_support_gets_an_error_answer(self):
        self.tools()
        time.sleep(0.3)
        seen = json.loads(self.call("replies")[1]["content"])
        self.assertEqual(seen[0]["id"], "s1")
        self.assertEqual(seen[0]["error"]["code"], -32601)

    def test_a_tool_announced_later_appears_after_list_changed(self):
        self.assertNotIn("added", self.tools())
        self.call("add_tool")
        time.sleep(0.3)
        self.assertIn("added", self.tools())

    def test_unknown_server_and_bad_tool_name_and_bad_arguments(self):
        self.assertEqual(self.http("GET", "/tools/servers/nobody/tools")[0], 404)
        self.assertEqual(self.http("POST", "/tools/servers/fake/call", {"name": "../x", "arguments": {}})[0], 400)
        self.assertEqual(self.http("POST", "/tools/servers/fake/call", {"name": "echo", "arguments": [1]})[0], 400)

    def test_stop_ends_the_process_and_the_next_call_starts_a_new_one(self):
        self.tools()
        mgr = self.managers[0]
        pid = mgr.running["fake"]["client"].proc.pid
        self.assertEqual(self.http("POST", "/tools/servers/fake/stop", {})[0], 200)
        self.assertNotIn("fake", mgr.running)
        self.assertEqual(self.call("echo", {"text": "again"})[1]["content"], "again")
        self.assertNotEqual(mgr.running["fake"]["client"].proc.pid, pid)

    def test_a_crash_is_reported_and_the_server_restarts_on_the_next_call(self):
        status, doc = self.call("crash")
        self.assertEqual(status, 502)
        self.assertEqual(doc["error"]["code"], "server_exited")
        self.assertIn("crashing on purpose", doc["error"]["message"])
        self.assertEqual(self.call("echo", {"text": "back"})[1]["content"], "back")

    def test_a_call_that_takes_too_long_times_out(self):
        with mock.patch.object(chat_tools, "CALL_TIMEOUT_S", 1):
            status, doc = self.call("slow")
        self.assertEqual((status, doc["error"]["code"]), (504, "timeout"))

    def test_the_child_gets_a_minimal_environment_without_the_services_secrets(self):
        names = self.call("env")[1]["content"].split()
        self.assertNotIn("SECRET_TOKEN", names)
        self.assertIn("PATH", names)

    def test_config_env_reaches_only_the_server_that_lists_it(self):
        self.write_config({"fake": stdio(env={"MY_SETTING": "1"}), "other": stdio()})
        self.assertIn("MY_SETTING", self.call("env")[1]["content"].split())
        status, doc = self.http("POST", "/tools/servers/other/call", {"name": "env", "arguments": {}})
        self.assertNotIn("MY_SETTING", doc["content"].split())


class StdinWrites(Base):
    def test_a_write_stuck_on_a_full_pipe_does_not_stop_the_reading_of_answers(self):
        self.write_config({"fake": stdio()})
        m = self.manager()
        client = m._ensure("fake", False)["client"]
        release = threading.Event()

        class Stuck:
            def write(self, data):
                release.wait(10)

            def flush(self):
                pass

            def close(self):
                pass

        real = client.proc.stdin
        client.proc.stdin = Stuck()
        self.addCleanup(release.set)
        writer = threading.Thread(target=client._send, args=({"jsonrpc": "2.0", "method": "x"},), daemon=True)
        writer.start()
        time.sleep(0.2)
        slot = [threading.Event(), None]
        client.pending[99] = slot
        reader = threading.Thread(target=client._handle, args=({"jsonrpc": "2.0", "id": 99, "result": {}},), daemon=True)
        reader.start()
        reader.join(2)
        self.assertFalse(reader.is_alive(), "the reader waited for the blocked writer")
        self.assertTrue(slot[0].is_set())
        release.set()
        client.proc.stdin = real


class Reaper(Base):
    def test_an_error_in_one_pass_is_logged_and_does_not_end_the_reaper(self):
        class Broken:
            calls = 0

            def reap(self):
                Broken.calls += 1
                raise OSError("odd child")

        with mock.patch.object(sys, "stderr") as err:
            self.assertEqual(chat_tools.reap_safely(Broken()), [])
            self.assertEqual(chat_tools.reap_safely(Broken()), [])
        self.assertEqual(Broken.calls, 2)
        self.assertIn("odd child", "".join(c.args[0] for c in err.write.call_args_list))
        self.write_config({"fake": stdio()})
        m = self.manager(idle=0)
        m._ensure("fake", False)
        time.sleep(0.05)
        self.assertEqual(chat_tools.reap_safely(m), ["fake"])


class ListingCache(Base):
    """The tool list is kept from a server's last start, so opening the chat does not start every enabled server."""

    def setUp(self):
        super().setUp()
        self.write_config({"fake": stdio()})
        self.serve()

    def listing(self, query=""):
        return self.http("GET", "/tools/servers/fake/tools" + query)[1]

    def state(self):
        return self.http("GET", "/tools/status")[1]["servers"][0]["state"]

    def test_a_stopped_server_is_answered_from_the_kept_list_without_starting(self):
        first = self.listing()
        self.assertFalse(first["cached"])
        self.http("POST", "/tools/servers/fake/stop", {})
        self.assertEqual(self.state(), "stopped")
        again = self.listing()
        self.assertTrue(again["cached"])
        self.assertEqual(again["tools"], first["tools"])
        self.assertEqual(self.state(), "stopped")     # still not running: nothing was started
        # A call starts it, as before.
        self.assertEqual(self.http("POST", "/tools/servers/fake/call", {"name": "echo", "arguments": {"text": "x"}})[0], 200)
        self.assertEqual(self.state(), "ready")

    def test_fresh_starts_the_server_and_asks_again(self):
        self.listing()
        self.http("POST", "/tools/servers/fake/stop", {})
        doc = self.listing("?fresh=1")
        self.assertFalse(doc["cached"])
        self.assertEqual(self.state(), "ready")

    def test_a_changed_start_command_makes_the_kept_list_stale(self):
        self.listing()
        self.http("POST", "/tools/servers/fake/stop", {})
        self.write_config({"fake": stdio("--other-flag-that-changes-the-fingerprint")})
        self.assertEqual(self.state(), "stopped")
        with mock.patch.object(chat_tools.Manager, "_ensure", side_effect=chat_tools.ToolError("start_failed", "would start", 502)):
            status, doc = self.http("GET", "/tools/servers/fake/tools")
        self.assertEqual((status, doc["error"]["code"]), (502, "start_failed"))    # not answered from the old list

    def test_a_network_server_still_needs_the_internet_for_a_kept_list(self):
        self.write_config({"fake": stdio(network=True)})
        self.assertEqual(self.http("GET", "/tools/servers/fake/tools")[1]["error"]["code"], "internet_off")
        self.listing("?internet=1")
        self.http("POST", "/tools/servers/fake/stop", {})
        self.assertEqual(self.http("GET", "/tools/servers/fake/tools")[1]["error"]["code"], "internet_off")


class Pagination(Base):
    def test_every_page_of_tools_is_collected(self):
        self.write_config({"fake": stdio("--pages")})
        self.serve()
        status, doc = self.http("GET", "/tools/servers/fake/tools")
        self.assertEqual(len(doc["tools"]), 10)


class Lifecycle(Base):
    def test_idle_servers_are_stopped_and_their_process_ends(self):
        self.write_config({"a": stdio()})
        mgr = self.manager(idle=0)
        mgr.tools("a", False)
        proc = mgr.running["a"]["client"].proc
        time.sleep(0.05)
        self.assertEqual(mgr.reap(), ["a"])
        proc.wait(timeout=5)
        self.assertEqual(mgr.status()["servers"][0]["state"], "stopped")

    def test_a_busy_server_is_not_reaped(self):
        self.write_config({"a": stdio()})
        mgr = self.manager(idle=0)
        mgr.tools("a", False)
        mgr.running["a"]["busy"] = 1
        time.sleep(0.05)
        self.assertEqual(mgr.reap(), [])

    def test_at_most_max_running_and_the_least_recently_used_idle_one_makes_room(self):
        self.write_config({"a": stdio(), "b": stdio(), "c": stdio()})
        mgr = self.manager(max_running=2)
        for sid in ("a", "b"):
            mgr.tools(sid, False)
            time.sleep(0.02)
        mgr.tools("c", False)
        self.assertEqual(sorted(mgr.running), ["b", "c"])

    def test_when_every_running_server_is_busy_a_new_one_is_refused(self):
        self.write_config({"a": stdio(), "b": stdio()})
        mgr = self.manager(max_running=1)
        mgr.tools("a", False)
        mgr.running["a"]["busy"] = 1
        with self.assertRaises(chat_tools.ToolError) as cm:
            mgr.tools("b", False)
        self.assertEqual((cm.exception.code, cm.exception.status), ("too_many_servers", 503))

    def test_no_server_starts_when_memory_is_low(self):
        self.write_config({"a": stdio()})
        mgr = self.manager(memory_floor=700, memory_fn=lambda: 300)
        with self.assertRaises(chat_tools.ToolError) as cm:
            mgr.tools("a", False)
        self.assertEqual((cm.exception.code, cm.exception.status), ("low_memory", 503))
        self.assertEqual(mgr.running, {})

    def test_unknown_memory_does_not_block_a_start(self):
        self.write_config({"a": stdio()})
        self.assertEqual(len(self.manager().tools("a", False)["tools"]), 10)

    def test_a_command_that_cannot_run_is_a_clear_start_failure(self):
        self.write_config({"broken": {"command": "/nonexistent/binary"}})
        with self.assertRaises(chat_tools.ToolError) as cm:
            self.manager().tools("broken", False)
        self.assertEqual(cm.exception.code, "start_failed")

    def test_a_server_that_never_answers_initialize_fails_after_the_start_timeout(self):
        self.write_config({"mute": {"command": sys.executable, "args": ["-c", "import time; time.sleep(30)"]}})
        mgr = self.manager()
        with mock.patch.object(chat_tools, "START_TIMEOUT_S", 1):
            with self.assertRaises(chat_tools.ToolError) as cm:
                mgr.tools("mute", False)
        self.assertEqual(cm.exception.code, "timeout")
        self.assertEqual(mgr.status()["servers"][0]["state"], "failed")


class InternetGate(Base):
    def test_a_network_server_needs_the_internet_flag(self):
        self.write_config({"net": stdio(network=True)})
        mgr = self.manager()
        with self.assertRaises(chat_tools.ToolError) as cm:
            mgr.tools("net", False)
        self.assertEqual((cm.exception.code, cm.exception.status), ("internet_off", 403))
        self.assertEqual(len(mgr.tools("net", True)["tools"]), 10)

    def test_a_package_is_downloaded_once_so_the_first_start_needs_the_internet_and_later_ones_do_not(self):
        self.write_config({"pkg": stdio(install=True)})
        mgr = self.manager()
        with self.assertRaises(chat_tools.ToolError) as cm:
            mgr.tools("pkg", False)
        self.assertIn("first start", cm.exception.message)
        mgr.tools("pkg", True)
        mgr.stop("pkg")
        self.assertEqual(len(mgr.tools("pkg", False)["tools"]), 10)      # marker written: no download needed now

    def test_a_failed_first_start_does_not_count_as_installed(self):
        self.write_config({"pkg": {"command": "/nonexistent/x", "install": True}})
        mgr = self.manager()
        with self.assertRaises(chat_tools.ToolError):
            mgr.tools("pkg", True)
        self.assertFalse(os.path.exists(mgr._marker("pkg")))

    def test_a_local_server_runs_with_the_internet_off(self):
        self.write_config({"local": stdio()})
        self.assertEqual(len(self.manager().tools("local", False)["tools"]), 10)

    def test_over_http_the_flag_must_be_exactly_one_or_true(self):
        self.write_config({"net": stdio(network=True)})
        self.serve()
        self.assertEqual(self.http("GET", "/tools/servers/net/tools?internet=true")[0], 403)
        self.assertEqual(self.http("POST", "/tools/servers/net/call", {"name": "echo", "arguments": {}, "internet": "yes"})[0], 403)
        self.assertEqual(self.http("POST", "/tools/servers/net/call", {"name": "echo", "arguments": {}, "internet": True})[0], 200)


class CatalogAdd(Base):
    def setUp(self):
        super().setUp()
        self.serve()

    def test_adds_a_pinned_entry_with_the_scratch_folder_filled_in(self):
        status, doc = self.http("POST", "/tools/catalog/filesystem/add", {})
        self.assertEqual((status, doc["added"]), (200, True))
        with open(self.cfg) as f:
            entry = json.load(f)["mcpServers"]["filesystem"]
        self.assertEqual(entry["command"], "npx")
        self.assertIn(self.scratch, entry["args"])
        self.assertIn("@modelcontextprotocol/server-filesystem@2.4.0", entry["args"])
        self.assertTrue(os.path.isdir(self.scratch))
        self.assertTrue(entry["install"])
        self.assertEqual(self.http("POST", "/tools/catalog/filesystem/add", {})[1]["added"], False)

    def test_env_values_get_the_scratch_folder_too_and_other_entries_are_kept(self):
        self.write_config({"mine": stdio()})
        self.http("POST", "/tools/catalog/memory/add", {})
        with open(self.cfg) as f:
            servers = json.load(f)["mcpServers"]
        self.assertIn("mine", servers)
        self.assertEqual(servers["memory"]["env"]["MEMORY_FILE_PATH"], os.path.join(self.scratch, "memory.jsonl"))

    def test_a_network_entry_needs_the_internet_flag(self):
        status, doc = self.http("POST", "/tools/catalog/fetch/add", {})
        self.assertEqual((status, doc["error"]["code"]), (403, "internet_off"))
        self.assertEqual(self.http("POST", "/tools/catalog/fetch/add", {"internet": True})[0], 200)

    def test_the_browser_cannot_supply_a_command(self):
        status, doc = self.http("POST", "/tools/catalog/filesystem/add", {"command": "/bin/sh", "args": ["-c", "id"], "title": "x"})
        self.assertEqual(status, 200)
        with open(self.cfg) as f:
            entry = json.load(f)["mcpServers"]["filesystem"]
        self.assertEqual(entry["command"], "npx")
        self.assertNotIn("/bin/sh", json.dumps(entry))
        self.assertEqual(self.http("POST", "/tools/catalog/evil/add", {"command": "/bin/sh"})[0], 404)
        self.assertEqual(self.http("POST", "/tools/servers/evil/call", {"command": "/bin/sh", "name": "x"})[0], 404)

    def test_an_unreadable_config_is_not_overwritten(self):
        with open(self.cfg, "w") as f:
            f.write("{broken")
        status, doc = self.http("POST", "/tools/catalog/filesystem/add", {})
        self.assertEqual((status, doc["error"]["code"]), (500, "bad_config"))
        with open(self.cfg) as f:
            self.assertEqual(f.read(), "{broken")

    def test_status_lists_the_catalog_and_marks_the_added_ones(self):
        self.http("POST", "/tools/catalog/time/add", {})
        status, doc = self.http("GET", "/tools/status")
        added = {c["id"]: c["added"] for c in doc["catalog"]}
        self.assertTrue(added["time"])
        self.assertFalse(added["git"])
        self.assertEqual([s["id"] for s in doc["servers"]], ["time"])


class Front(Base):
    def setUp(self):
        super().setUp()
        self.write_config({"fake": stdio()})
        self.serve()

    def test_status_shape_and_limits(self):
        status, doc = self.http("GET", "/tools/status")
        self.assertEqual(status, 200)
        self.assertEqual(doc["limits"]["maxRunning"], chat_tools.MAX_RUNNING)
        self.assertEqual(doc["limits"]["maxResultChars"], chat_tools.MAX_RESULT_CHARS)

    def test_wrong_method_unknown_path_and_non_json_bodies(self):
        self.assertEqual(self.http("POST", "/tools/status", {})[0], 405)
        self.assertEqual(self.http("GET", "/tools/servers/fake/call")[0], 405)
        self.assertEqual(self.http("GET", "/tools/nothing")[0], 404)
        self.assertEqual(self.http("GET", "/tools/servers/BAD%20NAME/tools")[0], 404)
        self.assertEqual(self.http("POST", "/tools/search", raw=b"x=1", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.http("POST", "/tools/search", raw=b"{nope")[0], 400)
        self.assertEqual(self.http("POST", "/tools/search", raw=b"[1]")[0], 400)

    def test_an_oversized_body_is_refused(self):
        # Only the header is sent: the answer comes before any body is read, so there is no race with the
        # server closing the connection while a large body is still being written.
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
            sock.sendall(b"POST /tools/search HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                         b"Content-Length: " + str(chat_tools.MAX_BODY + 10).encode() + b"\r\n\r\n")
            head = sock.recv(4096)
        self.assertTrue(head.startswith(b"HTTP/1.1 413"), head)

    def test_a_cross_site_post_is_refused_and_a_loopback_origin_is_fine(self):
        self.assertEqual(self.http("POST", "/tools/servers/fake/stop", {}, {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.http("POST", "/tools/servers/fake/stop", {}, {"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.http("POST", "/tools/servers/fake/stop", {}, {"Origin": "http://127.0.0.1:5173", "Sec-Fetch-Site": "same-origin"})[0], 200)
        self.assertEqual(self.http("POST", "/tools/servers/fake/stop", {}, {"Origin": "http://localhost:5173"})[0], 200)

    def test_a_cross_site_get_that_would_start_a_server_is_refused_but_status_stays_open(self):
        for headers in ({"Origin": "https://evil.example"}, {"Sec-Fetch-Site": "cross-site"}):
            self.assertEqual(self.http("GET", "/tools/servers/fake/tools", headers=headers)[0], 403)
        self.assertEqual(self.http("GET", "/tools/status", headers={"Sec-Fetch-Site": "cross-site"})[0], 200)
        self.assertEqual(self.http("GET", "/tools/servers/fake/tools", headers={"Origin": "http://localhost:5173", "Sec-Fetch-Site": "same-origin"})[0], 200)
        self.assertEqual(self.http("GET", "/tools/servers/fake/tools")[0], 200)

    def test_a_request_that_names_another_host_is_refused(self):
        self.assertEqual(self.http("GET", "/tools/status", headers={"Host": "evil.example"})[0], 403)

    def test_a_password_is_required_when_one_is_set(self):
        self.serve(password="s3cret")
        self.assertEqual(self.http("GET", "/tools/status")[0], 401)
        wrong = "Basic " + base64.b64encode(b"u:nope").decode()
        self.assertEqual(self.http("GET", "/tools/status", headers={"Authorization": wrong})[0], 401)
        right = "Basic " + base64.b64encode(b"u:s3cret").decode()
        self.assertEqual(self.http("GET", "/tools/status", headers={"Authorization": right})[0], 200)


class FakeSearx(http.server.ThreadingHTTPServer):
    def __init__(self, reply, status=200):
        owner = self
        self.seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(h):
                owner.seen.append((h.path, dict(h.headers)))
                body = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
                h.send_response(status)
                h.send_header("Content-Type", "application/json")
                h.send_header("Content-Length", str(len(body)))
                h.end_headers()
                h.wfile.write(body)

            def log_message(h, *a):
                pass
        super().__init__(("127.0.0.1", 0), H)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class Search(Base):
    def setUp(self):
        super().setUp()
        self.serve()

    def searx(self, reply, status=200):
        s = FakeSearx(reply, status)
        self.addCleanup(s.server_close)
        self.addCleanup(s.shutdown)
        return s

    def ask(self, **body):
        return self.http("POST", "/tools/search", body)

    def test_returns_numbered_results_as_text_with_no_cookies_or_identity(self):
        s = self.searx({"results": [{"title": "One", "url": "https://one.example", "content": "first"}, {"title": "Two", "url": "https://two.example", "content": "second"}]})
        status, doc = self.ask(query="llama", searchUrl=s.url, internet=True)
        self.assertEqual(status, 200)
        self.assertEqual(doc["content"], "1. One\n   https://one.example\n   first\n2. Two\n   https://two.example\n   second")
        path, headers = s.seen[0]
        self.assertEqual(path, "/search?q=llama&format=json")
        self.assertEqual(headers["User-Agent"], "lm-chat-tools")
        self.assertNotIn("Cookie", headers)

    def test_off_means_off_and_nothing_is_contacted(self):
        s = self.searx({"results": []})
        status, doc = self.ask(query="x", searchUrl=s.url)
        self.assertEqual((status, doc["error"]["code"]), (403, "internet_off"))
        self.assertEqual(s.seen, [])
        self.assertEqual(self.ask(query="x", searchUrl=s.url, internet="true")[0], 403)

    def test_the_count_is_capped_and_defaults_to_five(self):
        s = self.searx({"results": [{"title": str(i), "url": "u", "content": "c"} for i in range(20)]})
        self.assertEqual(self.ask(query="x", searchUrl=s.url, internet=True)[1]["content"].count("\n   u\n"), 5)
        self.assertEqual(self.ask(query="x", searchUrl=s.url, internet=True, count=2)[1]["content"].count("\n   u\n"), 2)
        self.assertEqual(self.ask(query="x", searchUrl=s.url, internet=True, count=999)[1]["content"].count("\n   u\n"), 5)

    def test_no_results_says_so(self):
        s = self.searx({"results": []})
        self.assertEqual(self.ask(query="x", searchUrl=s.url, internet=True)[1]["content"], "no results")

    def test_bad_requests(self):
        s = self.searx({"results": []})
        for body, text in [
            ({"query": "x", "searchUrl": "ftp://x"}, "searchUrl"),
            ({"query": "x", "searchUrl": "http://user:pw@127.0.0.1:1"}, "user name"),
            ({"query": "x"}, "searchUrl"),
            ({"query": "", "searchUrl": s.url}, "query"),
            ({"query": "x" * 501, "searchUrl": s.url}, "query"),
            ({"query": 5, "searchUrl": s.url}, "query"),
        ]:
            status, doc = self.ask(internet=True, **body)
            self.assertEqual(status, 400, body)
            self.assertIn(text, doc["error"]["message"])

    def test_a_failing_or_non_json_search_server_is_a_502(self):
        self.assertEqual(self.ask(query="x", searchUrl=self.searx({}, 500).url, internet=True)[1]["error"]["code"], "search_failed")
        status, doc = self.ask(query="x", searchUrl=self.searx(b"<html>no json</html>").url, internet=True)
        self.assertEqual(status, 502)
        self.assertIn("json format", doc["error"]["message"])

    def test_an_unreachable_search_server_is_a_502_not_a_crash(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        status, doc = self.ask(query="x", searchUrl=f"http://127.0.0.1:{port}", internet=True)
        self.assertEqual((status, doc["error"]["code"]), (502, "unreachable"))

    def test_a_link_local_search_address_is_never_allowed(self):
        status, doc = self.ask(query="x", searchUrl="http://169.254.169.254", internet=True)
        self.assertEqual((status, doc["error"]["code"]), (403, "blocked_address"))

    def test_a_private_address_needs_to_be_listed_and_this_service_is_never_a_search_instance(self):
        real_check, real_open = chat_tools.check_address, chat_tools.open_connection

        def check(host, port, allow_private):
            return "10.0.0.5" if host == "lan-box" else real_check(host, port, allow_private)

        def opened(url, allow_private, timeout, ip=None):
            if ip == "10.0.0.5":
                raise chat_tools.ToolError("unreachable", "refused", 502)
            return real_open(url, allow_private, timeout, ip)

        with mock.patch.object(chat_tools, "check_address", check), mock.patch.object(chat_tools, "open_connection", opened):
            status, doc = self.ask(query="x", searchUrl="http://lan-box:9", internet=True)
            self.assertEqual((status, doc["error"]["code"]), (403, "blocked_address"))
            self.assertIn("--search-allow lan-box:9", doc["error"]["message"])
            # Listed, it is tried (and nothing listens there).
            server = self.serve()
            server.search_allow = frozenset({"lan-box:9"})
            status, doc = self.ask(query="x", searchUrl="http://lan-box:9", internet=True)
            self.assertEqual((status, doc["error"]["code"]), (502, "unreachable"))
            status, doc = self.ask(query="x", searchUrl=f"http://127.0.0.1:{self.port}", internet=True)
            self.assertEqual((status, doc["error"]["code"]), (403, "blocked_address"))
            self.assertIn("itself", doc["error"]["message"])

    def test_a_sub_path_instance_is_searched_under_its_path(self):
        s = self.searx({"results": []})
        self.ask(query="x", searchUrl=s.url + "/searx/", internet=True)
        self.assertTrue(s.seen[0][0].startswith("/searx/search?"))


class Tls(unittest.TestCase):
    def test_the_context_refuses_anything_older_than_tls_1_2_and_still_verifies_certificates(self):
        import ssl
        ctx = chat_tools.tls_context()
        self.assertGreaterEqual(ctx.minimum_version, ssl.TLSVersion.TLSv1_2)
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(ctx.check_hostname)


class Addresses(unittest.TestCase):
    def test_never_allowed_addresses(self):
        for host in ("169.254.169.254", "0.0.0.0", "224.0.0.1", "169.254.1.1"):
            for allow in (False, True):
                with self.assertRaises(chat_tools.ToolError, msg=f"{host} {allow}") as cm:
                    chat_tools.check_address(host, 80, allow)
                self.assertEqual(cm.exception.code, "blocked_address")

    def test_private_and_loopback_only_when_allowed(self):
        for host in ("127.0.0.1", "10.1.2.3", "192.168.1.96", "172.16.0.5"):
            with self.assertRaises(chat_tools.ToolError):
                chat_tools.check_address(host, 80, False)
            self.assertEqual(chat_tools.check_address(host, 80, True), host)

    def test_a_public_address_passes(self):
        self.assertEqual(chat_tools.check_address("93.184.216.34", 443, False), "93.184.216.34")

    def test_a_name_that_resolves_to_a_private_address_is_refused(self):
        with self.assertRaises(chat_tools.ToolError) as cm:
            chat_tools.check_address("localhost", 80, False)
        self.assertEqual(cm.exception.code, "blocked_address")


class FakeRemote(http.server.ThreadingHTTPServer):
    """A Streamable HTTP MCP server: JSON for most replies, an event stream for tools/list."""

    def __init__(self, token="tok"):
        owner = self
        self.seen = []

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(h):
                msg = json.loads(h.rfile.read(int(h.headers["Content-Length"])))
                owner.seen.append((msg.get("method"), dict(h.headers)))
                if token and h.headers.get("Authorization") != "Bearer " + token:
                    h.send_response(401)
                    h.send_header("Content-Length", "0")
                    h.end_headers()
                    return
                method = msg.get("method")
                if "id" not in msg:
                    h.send_response(202)
                    h.send_header("Content-Length", "0")
                    h.end_headers()
                    return
                extra, sse = [], False
                if method == "initialize":
                    body, extra = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "r", "version": "1"}}, [("Mcp-Session-Id", "sess-1")]
                elif method == "tools/list":
                    body, sse = {"tools": [{"name": "remote_echo", "description": "d", "inputSchema": {"type": "object"}}]}, True
                else:
                    body = {"content": [{"type": "text", "text": "remote:" + json.dumps(msg["params"]["arguments"])}]}
                rpc = {"jsonrpc": "2.0", "id": msg["id"], "result": body}
                if sse:
                    data = ("event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {}}) + "\n\n"
                            + "event: message\ndata: " + json.dumps(rpc) + "\n\n").encode()
                    mime = "text/event-stream"
                else:
                    data, mime = json.dumps(rpc).encode(), "application/json"
                h.send_response(200)
                h.send_header("Content-Type", mime)
                h.send_header("Content-Length", str(len(data)))
                for k, v in extra:
                    h.send_header(k, v)
                h.end_headers()
                h.wfile.write(data)

            def log_message(h, *a):
                pass
        super().__init__(("127.0.0.1", 0), H)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}/mcp"


class RemoteServers(Base):
    def remote(self, **kw):
        r = FakeRemote(**kw)
        self.addCleanup(r.server_close)
        self.addCleanup(r.shutdown)
        return r

    def test_lists_and_calls_a_remote_server_with_its_session_and_bearer_token(self):
        r = self.remote()
        self.env["REMOTE_TOKEN"] = "tok"
        self.write_config({"remote": {"url": r.url, "bearerEnv": "REMOTE_TOKEN", "allowPrivate": True}})
        mgr = self.manager()
        self.assertEqual([t["name"] for t in mgr.tools("remote", True)["tools"]], ["remote_echo"])
        out = mgr.call("remote", "remote_echo", {"a": 1}, True)
        self.assertEqual(out["content"], 'remote:{"a": 1}')
        later = [h for m, h in r.seen if m == "tools/call"][0]
        self.assertEqual(later["Mcp-Session-Id"], "sess-1")
        self.assertEqual(later["Authorization"], "Bearer tok")
        self.assertIn("notifications/initialized", [m for m, _ in r.seen])

    def test_a_private_address_is_blocked_unless_the_entry_says_allowPrivate(self):
        r = self.remote(token=None)
        self.write_config({"remote": {"url": r.url}})
        with self.assertRaises(chat_tools.ToolError) as cm:
            self.manager().tools("remote", True)
        self.assertEqual(cm.exception.code, "blocked_address")
        self.assertEqual(r.seen, [])

    def test_the_internet_flag_gates_a_remote_server_and_a_missing_token_variable_is_named(self):
        r = self.remote()
        self.write_config({"remote": {"url": r.url, "bearerEnv": "NOT_SET_ANYWHERE", "allowPrivate": True}})
        mgr = self.manager()
        with self.assertRaises(chat_tools.ToolError) as cm:
            mgr.tools("remote", False)
        self.assertEqual(cm.exception.code, "internet_off")
        with self.assertRaises(chat_tools.ToolError) as cm:
            mgr.tools("remote", True)
        self.assertIn("NOT_SET_ANYWHERE", cm.exception.message)

    def test_a_rejected_token_is_a_clear_error(self):
        r = self.remote(token="right")
        self.env["REMOTE_TOKEN"] = "wrong"
        self.write_config({"remote": {"url": r.url, "bearerEnv": "REMOTE_TOKEN", "allowPrivate": True}})
        with self.assertRaises(chat_tools.ToolError) as cm:
            self.manager().tools("remote", True)
        self.assertIn("401", cm.exception.message)


class Normalizers(unittest.TestCase):
    def test_tool_listing_defaults_and_drops(self):
        out = chat_tools.normalize_tools([
            {"name": "a", "description": "d" * 900, "inputSchema": "not a dict", "annotations": {"readOnlyHint": "yes"}},
            {"name": "b", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}, "title": "B"},
            {"name": "has space"}, {"description": "no name"}, "junk", {"name": 5},
        ])
        self.assertEqual([t["name"] for t in out], ["a", "b"])
        self.assertEqual(len(out[0]["description"]), 600)
        self.assertEqual(out[0]["inputSchema"], {"type": "object", "properties": {}})
        self.assertFalse(out[0]["readOnly"])             # only a real True counts
        self.assertTrue(out[1]["readOnly"])
        self.assertEqual(out[1]["title"], "B")

    def test_result_text_error_flag_and_empty(self):
        self.assertEqual(chat_tools.normalize_result({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "isError": True}), {"content": "a\nb", "isError": True})
        self.assertEqual(chat_tools.normalize_result({}), {"content": "", "isError": False})
        self.assertIn("[video content left out]", chat_tools.normalize_result({"content": [{"type": "video"}]})["content"])


if __name__ == "__main__":
    unittest.main()
