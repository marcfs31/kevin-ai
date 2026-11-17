#!/usr/bin/env python3
"""Tests for tools/bench_serve.py: the stream parsing, the memory arithmetic and the budget on
synthetic data, and whole runs against a fake lm-serve (a small HTTP server written to a
temporary directory) including the failure paths. Run from anywhere:

    python3 -m unittest discover -s tools -p 'test_bench_serve.py' -v
"""

import contextlib
import io
import json
import os
import socket
import stat
import sys
import tempfile
import textwrap
import unittest

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import bench_serve as bs  # noqa: E402

FAKE = textwrap.dedent('''\
    #!/usr/bin/env python3
    """A fake lm-serve: same flags, same log line, SSE replies with known token counts."""
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    mode = os.environ.get("FAKE_MODE", "")
    if mode == "exit":
        print("lm-serve: no such model", file=sys.stderr, flush=True)
        sys.exit(3)
    seen = {}
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            msgs = req["messages"]
            if mode == "http400":
                self.send_response(400); self.end_headers()
                self.wfile.write(b'{"error":{"message":"context_length_exceeded"}}'); return
            ptok = sum(len(m["content"].split()) for m in msgs) + 12
            cached = 0
            if msgs[0]["role"] == "system":
                key = msgs[0]["content"]
                seen[key] = seen.get(key, 0) + 1
                if seen[key] >= 3:
                    cached = len(key.split()) + 6
            fed = ptok - cached
            time.sleep(fed * 0.0004)
            ctok = min(req["max_tokens"], 5)
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            def send(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\\n\\n"); self.wfile.flush()
            send({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
            for _ in range(ctok):
                time.sleep(0.02)
                send({"choices": [{"delta": {"content": "a"}}]})
            send({"choices": [{"delta": {}, "finish_reason": "length"}]})
            if mode != "nousage":
                send({"choices": [], "usage": {"prompt_tokens": ptok, "completion_tokens": ctok,
                                               "total_tokens": ptok + ctok}})
            if mode != "nodone":
                self.wfile.write(b"data: [DONE]\\n\\n"); self.wfile.flush()
            print("lm-serve: POST /v1/chat/completions 200 | prompt %d, %d from cache, fed %d at 4.50 tok/s | "
                  "generated %d at 2.30 tok/s | 1.0 s" % (ptok, cached, fed, ctok), file=sys.stderr, flush=True)
    srv = HTTPServer(("127.0.0.1", port), H)
    print("lm-serve: listening on http://127.0.0.1:%d (model fake, context 4096, prefix cache 0 MiB)" % port,
          file=sys.stderr, flush=True)
    srv.serve_forever()
''')


def sse(*objs, done=True):
    ev = [b"data: " + json.dumps(o).encode() + b"\n" for o in objs]
    if done:
        ev.append(b"data: [DONE]\n")
    return ev


def stamped(lines, times):
    return list(zip(times, lines))


TEXT = {"choices": [{"delta": {"content": "a"}}]}
USAGE = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}}
ROLE = {"choices": [{"delta": {"role": "assistant", "content": ""}}]}


class ParseStreamTests(unittest.TestCase):
    def test_numbers(self):
        # sent at t=10; role chunk at 10.2; first text at 14, last at 16, usage after
        ev = stamped(sse(ROLE, TEXT, TEXT, TEXT, TEXT, TEXT, USAGE), [10.2, 14.0, 14.5, 15.0, 15.5, 16.0, 16.1, 16.2])
        r = bs.parse_stream(ev, 10.0)
        self.assertEqual((r["prompt_tokens"], r["completion_tokens"]), (100, 5))
        self.assertAlmostEqual(r["ttft_s"], 4.0)
        self.assertAlmostEqual(r["prompt_tps"], 25.0)
        self.assertAlmostEqual(r["gen_tps"], 2.0)           # 4 tokens after the first, over 2.0 s
        self.assertAlmostEqual(r["total_s"], 6.0)

    def test_ttft_skips_the_empty_role_chunk(self):
        ev = stamped(sse(ROLE, TEXT, TEXT, USAGE), [1.0, 9.0, 10.0, 10.1, 10.2])
        self.assertAlmostEqual(bs.parse_stream(ev, 0.0)["ttft_s"], 9.0)

    def test_one_token_has_no_generation_speed(self):
        u = {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51}}
        r = bs.parse_stream(stamped(sse(ROLE, TEXT, u), [1, 2, 3, 4]), 0.0)
        self.assertIsNone(r["gen_tps"])

    def test_failures(self):
        ok = [ROLE, TEXT, TEXT, USAGE]
        t = [1, 2, 3, 4, 5]
        cases = {
            "no [DONE]": stamped(sse(*ok, done=False), t),
            "no usage": stamped(sse(ROLE, TEXT, TEXT), t),
            "no text": stamped(sse(ROLE, USAGE), t),
            "malformed": stamped([b"data: {not json}\n"] + sse(*ok), t),
            "error chunk": stamped(sse({"error": {"message": "boom"}}), t),
            "zero tokens": stamped(sse(ROLE, TEXT, {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}), t),
        }
        for name, ev in cases.items():
            with self.subTest(name), self.assertRaises(bs.Failure):
                bs.parse_stream(ev, 0.0)

    def test_error_message_is_kept(self):
        with self.assertRaisesRegex(bs.Failure, "boom"):
            bs.parse_stream(stamped(sse({"error": {"message": "boom"}}), [1, 2]), 0.0)

    def test_non_text_lines_are_ignored(self):
        ev = stamped([b": keep-alive\n", b"\n"] + sse(ROLE, TEXT, TEXT, USAGE), [0.5, 0.6, 1, 2, 3, 4, 5])
        self.assertEqual(bs.parse_stream(ev, 0.0)["prompt_tokens"], 100)


class LogAndProcTests(unittest.TestCase):
    def test_log_line(self):
        line = ("lm-serve: POST /v1/chat/completions 200 | prompt 40, 30 from cache, fed 10 at 4.28 tok/s | "
                "generated 1 at 284.87 tok/s | 2.4 s")
        self.assertEqual(bs.parse_log_line(line), {"prompt": 40, "cached": 30, "fed": 10})

    def test_other_lines_are_not_request_logs(self):
        self.assertIsNone(bs.parse_log_line("lm-serve: GET /health 200 | 0.000 s"))
        self.assertIsNone(bs.parse_log_line("lm-serve: listening on http://127.0.0.1:8099 (model q, context 81408)"))

    def test_status(self):
        text = "Name:\tlm-serve\nVmHWM:\t  5300000 kB\nVmRSS:\t  5100000 kB\n"
        self.assertEqual(bs.parse_status(text), {"rss": 5100000 * 1024, "hwm": 5300000 * 1024})
        self.assertIsNone(bs.parse_status("VmRSS:\t 1 kB\n"))      # one field missing: unknown, not zero
        self.assertIsNone(bs.parse_status(None))

    def test_meminfo(self):
        text = "MemTotal:       11599772 kB\nMemFree: 1 kB\nMemAvailable:   10992208 kB\n"
        self.assertEqual(bs.parse_meminfo(text), {"available": 10992208 * 1024, "total": 11599772 * 1024})
        self.assertIsNone(bs.parse_meminfo("MemTotal: 5 kB\n"))
        self.assertIsNone(bs.parse_meminfo("MemTotal: 5 MB\nMemAvailable: 4 MB\n"))   # only kB is understood

    def test_cpu_clock(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "scaling_max_freq"), "w") as f:
                f.write("800000\n")
            got = bs.read_cpu_mhz(d)
        self.assertEqual(got, {"scaling_max_mhz": 800, "cpuinfo_max_mhz": None})    # missing file: unknown


GIB = bs.GIB


class MemoryTests(unittest.TestCase):
    def runs(self, pairs):
        return [{"prompt_tokens": p, "completion_tokens": 0, "hwm": h} for p, h in pairs]

    def test_linfit_exact(self):
        slope, icpt = bs.linfit([0, 1000, 2000], [5 * GIB, 5 * GIB + 64e6, 5 * GIB + 128e6])
        self.assertAlmostEqual(slope, 64e3)
        self.assertAlmostEqual(icpt, 5 * GIB)

    def test_linfit_needs_two_different_points(self):
        self.assertIsNone(bs.linfit([5], [1]))
        self.assertIsNone(bs.linfit([7, 7, 7], [1, 2, 3]))

    def test_extrapolation_fits(self):
        runs = self.runs([(1000, 5 * GIB + 1000 * 64e3), (2000, 5 * GIB + 2000 * 64e3)])
        ex = bs.extrapolate(runs, 10000, 11 * GIB)
        self.assertAlmostEqual(ex["kib_per_token"], 64e3 / 1024)
        self.assertAlmostEqual(ex["rss_at_ctx_gib"], (5 * GIB + 10000 * 64e3) / GIB)
        self.assertTrue(ex["fits"])
        self.assertTrue(ex["estimate"])

    def test_extrapolation_does_not_fit(self):
        runs = self.runs([(1000, 5 * GIB + 1000 * 64e3), (2000, 5 * GIB + 2000 * 64e3)])
        ex = bs.extrapolate(runs, 200000, 11 * GIB)       # 5 GiB + 12.8 GB
        self.assertFalse(ex["fits"])

    def test_the_margin_counts(self):
        runs = self.runs([(0, 10.0 * GIB), (1000, 10.0 * GIB)])  # flat: the estimate is 10 GiB
        self.assertTrue(bs.extrapolate(runs, 5000, 10.5 * GIB + 1)["fits"])
        self.assertFalse(bs.extrapolate(runs, 5000, 10.5 * GIB - 1)["fits"])   # 0.5 GiB stays free

    def test_unknown_stays_unknown(self):
        self.assertIsNone(bs.extrapolate(self.runs([(1000, None), (2000, None)]), 5000, 11 * GIB)["rss_at_ctx_gib"])
        self.assertIsNone(bs.extrapolate(self.runs([(1000, 6 * GIB)]), 5000, 11 * GIB)["rss_at_ctx_gib"])
        ok = self.runs([(1000, 5 * GIB), (2000, 6 * GIB)])
        self.assertIsNone(bs.extrapolate(ok, None, 11 * GIB)["rss_at_ctx_gib"])           # window unknown
        self.assertIsNone(bs.extrapolate(ok, 5000, None)["fits"])                          # RAM unknown
        self.assertIsNotNone(bs.extrapolate(ok, 5000, None)["rss_at_ctx_gib"])

    def test_runs_without_a_reading_are_ignored(self):
        runs = self.runs([(1000, 5 * GIB), (1500, None), (2000, 6 * GIB)])
        ex = bs.extrapolate(runs, 3000, 11 * GIB)
        self.assertAlmostEqual(ex["rss_at_ctx_gib"], 7.0)


class BudgetAndPromptTests(unittest.TestCase):
    def test_estimate(self):
        prev = {"prompt_tps": 4.0, "gen_tps": 2.0}
        self.assertAlmostEqual(bs.next_estimate(1000, 32, prev), 250 + 16)
        self.assertIsNone(bs.next_estimate(1000, 32, None))                    # nothing to go on: run it
        self.assertAlmostEqual(bs.next_estimate(400, 10, {"prompt_tps": 4.0, "gen_tps": None}), 110)

    def test_budget(self):
        self.assertTrue(bs.over_budget(3000, 700, 3600))
        self.assertFalse(bs.over_budget(3000, 600, 3600))
        self.assertFalse(bs.over_budget(99999, None, 3600))                    # unknown estimate never skips

    def test_text_is_deterministic_and_salted(self):
        a, b = bs.make_text(50, 1), bs.make_text(50, 1)
        self.assertEqual(a, b)
        self.assertEqual(len(a.split()), 50)
        self.assertNotEqual(a, bs.make_text(50, 2))

    def test_words_for_and_ratio(self):
        self.assertEqual(bs.words_for(112, 1.0), 100)
        self.assertEqual(bs.words_for(5, 1.0), 1)                              # never zero words
        self.assertAlmostEqual(bs.next_ratio(112, 100), 1.0)
        self.assertEqual(bs.next_ratio(10 ** 6, 10), 3.0)                      # clamped
        self.assertEqual(bs.next_ratio(1, 100), 0.5)

    def test_table_shows_unknown(self):
        row = {"prompt_tokens": 100, "ttft_s": 2.0, "prompt_tps": 50.0, "gen_tps": None, "hwm": None, "available": None}
        text = bs.format_table([row])
        self.assertIn("n/a", text)
        self.assertNotIn("None", text)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = os.path.join(self.tmp.name, "fake-lm-serve")
        with open(self.fake, "w") as f:
            f.write(FAKE)
        os.chmod(self.fake, os.stat(self.fake).st_mode | stat.S_IXUSR)
        self.out = os.path.join(self.tmp.name, "out.json")

    def load(self):
        with open(self.out) as f:
            return json.load(f)

    def go(self, *extra, mode=""):
        argv = ["--model", "/nonexistent", "--serve-bin", self.fake, "--port", str(free_port()),
                "--out", self.out, "--start-timeout", "20", "--read-timeout", "20"] + list(extra)
        old = os.environ.get("FAKE_MODE")
        os.environ["FAKE_MODE"] = mode
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = bs.main(argv)
        finally:
            if old is None:
                os.environ.pop("FAKE_MODE", None)
            else:
                os.environ["FAKE_MODE"] = old
        return rc, out.getvalue(), err.getvalue()

    def test_whole_run(self):
        rc, out, err = self.go("--lens", "64,200,400", "--gen", "5")
        self.assertEqual(rc, 0, err)
        res = self.load()
        self.assertEqual(res["server"]["context"], 4096)
        self.assertEqual([r["completion_tokens"] for r in res["runs"]], [5, 5, 5])
        ptoks = [r["prompt_tokens"] for r in res["runs"]]
        self.assertEqual(ptoks, sorted(ptoks))
        self.assertTrue(all(abs(r["prompt_tokens"] - r["target"]) < 0.5 * r["target"] for r in res["runs"][1:]))
        for r in res["runs"]:
            self.assertGreater(r["prompt_tps"], 0)
            self.assertGreater(r["gen_tps"], 0)
            self.assertEqual(r["cached"], 0)
            self.assertEqual(r["fed"], r["prompt_tokens"])
        self.assertIn("prompt tok/s", out)
        if os.path.exists("/proc/self/status"):     # Linux: the fake server's RSS is read
            self.assertTrue(all(r["hwm"] and r["hwm"] >= r["rss"] > 0 for r in res["runs"]))
            self.assertIsNotNone(res["memory_at_context"]["kib_per_token"])
        else:                                       # elsewhere: unknown, never a made-up number
            self.assertTrue(all(r["hwm"] is None for r in res["runs"]))
            self.assertIsNone(res["memory_at_context"]["rss_at_ctx_gib"])
            self.assertIn("unknown", out)

    def test_cache_test_shows_the_restore(self):
        rc, out, err = self.go("--lens", "64", "--gen", "5", "--cache-test", "--cache-len", "300")
        self.assertEqual(rc, 0, err)
        c = self.load()["cache_test"]
        self.assertEqual([r["cached"] for r in c][:2], [0, 0])
        self.assertGreater(c[2]["cached"], 250)
        self.assertLess(c[2]["ttft_s"], c[0]["ttft_s"])
        self.assertIn("from the cache", out)

    def test_budget_skips_long_prompts(self):
        rc, out, err = self.go("--lens", "64,100000", "--gen", "5", "--time-budget-min", "0.01")
        self.assertEqual(rc, 0, err)
        res = self.load()
        self.assertEqual(res["skipped_lens"], [100000])
        self.assertEqual(len(res["runs"]), 1)
        self.assertIn("skipped", out)

    def test_server_that_exits_fails(self):
        rc, out, err = self.go("--lens", "64", mode="exit")
        self.assertEqual(rc, 1)
        self.assertIn("bench_serve FAILED", err)
        self.assertIn("exited with code 3", err)
        self.assertFalse(os.path.exists(self.out))

    def test_http_error_fails_with_its_message(self):
        rc, out, err = self.go("--lens", "64", mode="http400")
        self.assertEqual(rc, 1)
        self.assertIn("HTTP 400", err)
        self.assertIn("context_length_exceeded", err)

    def test_stream_without_usage_fails(self):
        rc, out, err = self.go("--lens", "64", mode="nousage")
        self.assertEqual(rc, 1)
        self.assertIn("no usage", err)

    def test_stream_without_done_fails(self):
        rc, out, err = self.go("--lens", "64", mode="nodone")
        self.assertEqual(rc, 1)
        self.assertIn("[DONE]", err)

    def test_missing_binary_fails(self):
        argv = ["--model", "/x", "--serve-bin", "/nonexistent/lm-serve", "--out", self.out]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(bs.main(argv), 1)
        self.assertIn("could not start", err.getvalue())

    def test_bad_arguments(self):
        for extra, needle in ((["--lens", "a,b"], "integers"), (["--lens", "4"], "at least 16"),
                              (["--gen", "1"], "at least 16"), (["--cache-test", "--cache-mb", "0"], "cache-mb")):
            with self.subTest(extra):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    self.assertEqual(bs.main(["--model", "/x"] + extra), 1)
                self.assertIn(needle, err.getvalue())


if __name__ == "__main__":
    unittest.main()
