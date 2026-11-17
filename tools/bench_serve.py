#!/usr/bin/env python3
"""Measure the speed and memory of bin/lm-serve through its OpenAI-compatible API.

    bench_serve.py --model ~/models/qwen3.5-4b \\
                   --weights ~/models/qwen3.5-4b/model.q8.safetensors \\
                   [--lens 128,512,1024,2048] [--gen 32] [--cache-test] [--out bench_serve.json]

It starts lm-serve itself, waits for /health, sends one streamed request per prompt length and
stops the server. Without --weights the bf16 checkpoint in --model is used (the mode that
needs the most RAM).

Per request: prompt tokens (the server's own count), time to first token, prompt tokens/s
(prompt tokens over the time to the first token, which includes decoding that one token) and
generation tokens/s (the tokens after the first, over the time between the first and the last
text chunk). After each request the server process's VmRSS and VmHWM and the machine's
MemAvailable are read from /proc (Linux only; elsewhere they are reported as unknown).

Memory at the full context window is an ESTIMATE: a line through the peak RSS of the runs
(against tokens fed), extended to the window lm-serve printed at startup. KV memory is only
touched as tokens arrive, so RSS grows with the tokens actually fed, not with the window.
Prompts are slow to feed (about 4 to 5 tokens/s on the target), so --time-budget-min skips
the lengths that would not finish in time, and the window is never filled.

--cache-test sends three requests that share one long system prompt: the third is restored
from the prefix cache (the server's log line says how many tokens came from it).

Fails closed: a server that exits or never becomes healthy, an HTTP error, a stream without
[DONE], without usage or without text stops with `bench_serve FAILED: <what>` and exit 1.
Pure Python 3, no dependencies; talks only to the server it starts on 127.0.0.1."""

import argparse
import http.client
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
GIB = 1024.0 ** 3
MARGIN = 512 << 20          # what lm-serve keeps free (src/model/lm_ctx.h)
TEMPLATE_OVERHEAD = 12      # chat-template tokens around the user text, roughly
PROMPT_PREFIX = "Summarize this in one sentence: "

WORDS = ("table river window garden silver pencil mountain engine paper bridge cloud morning "
         "carpet lantern market forest copper ladder island harbor violin meadow anchor basket "
         "candle desert feather glacier hammer jungle kettle lemon marble needle orange pillow "
         "quartz ribbon saddle tunnel umbrella valley whistle yellow zipper blanket compass "
         "dragon ember fabric gravel helmet iron jacket kernel lagoon magnet nectar oyster "
         "puzzle rocket sponge timber velvet walnut").split()


class Failure(Exception):
    pass


# ---------------------------------------------------------------- pure helpers

def make_text(n_words, salt):
    rng = random.Random(salt)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def words_for(target_tokens, ratio):
    """How many words give about target_tokens, at `ratio` tokens per word."""
    return max(1, round((target_tokens - TEMPLATE_OVERHEAD) / ratio))


def next_ratio(prompt_tokens, n_words):
    return min(3.0, max(0.5, (prompt_tokens - TEMPLATE_OVERHEAD) / max(1, n_words)))


def parse_stream(events, t_send):
    """events: [(time, raw line)] of a streamed reply. Returns the measured numbers."""
    first = last = usage = None
    done = False
    for t, line in events:
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            done = True
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            raise Failure("malformed stream chunk: %r" % payload[:80])
        if not isinstance(obj, dict):
            raise Failure("malformed stream chunk: %r" % payload[:80])
        if "error" in obj:
            err = obj["error"]
            raise Failure("server error in the stream: %s" % (err.get("message", err) if isinstance(err, dict) else err))
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            if (ch.get("delta") or {}).get("content"):
                if first is None:
                    first = t
                last = t
    if not done:
        raise Failure("the stream ended without [DONE]")
    if first is None:
        raise Failure("the reply had no text")
    try:
        ptok, ctok = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
    except (TypeError, KeyError, ValueError):
        raise Failure("the stream had no usage (prompt_tokens, completion_tokens)")
    if ptok < 1 or ctok < 1:
        raise Failure("implausible usage: prompt_tokens %d, completion_tokens %d" % (ptok, ctok))
    ttft = first - t_send
    if ttft <= 0:
        raise Failure("non-positive time to first token")
    gen_tps = (ctok - 1) / (last - first) if ctok >= 2 and last > first else None
    return {"prompt_tokens": ptok, "completion_tokens": ctok, "ttft_s": ttft,
            "prompt_tps": ptok / ttft, "gen_tps": gen_tps, "total_s": last - t_send}


LOG_RE = re.compile(r"POST /v1/chat/completions .*?prompt (\d+), (\d+) from cache, fed (\d+)")


def parse_log_line(line):
    m = LOG_RE.search(line)
    return {"prompt": int(m.group(1)), "cached": int(m.group(2)), "fed": int(m.group(3))} if m else None


def _kb_field(text, field):
    m = re.search(r"^%s:\s+(\d+) kB" % re.escape(field), text or "", re.M)
    return int(m.group(1)) * 1024 if m else None


def parse_status(text):
    """VmRSS and VmHWM in bytes from /proc/<pid>/status text, or None when either is missing."""
    rss, hwm = _kb_field(text, "VmRSS"), _kb_field(text, "VmHWM")
    return {"rss": rss, "hwm": hwm} if rss is not None and hwm is not None else None


def parse_meminfo(text):
    avail, total = _kb_field(text, "MemAvailable"), _kb_field(text, "MemTotal")
    return {"available": avail, "total": total} if avail is not None and total is not None else None


def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def read_cpu_mhz(cpufreq_dir="/sys/devices/system/cpu/cpu0/cpufreq"):
    out = {}
    for key, name in (("scaling_max_mhz", "scaling_max_freq"), ("cpuinfo_max_mhz", "cpuinfo_max_freq")):
        text = read_file(os.path.join(cpufreq_dir, name))
        try:
            out[key] = int(text.strip()) // 1000
        except (AttributeError, ValueError):
            out[key] = None
    return out


def linfit(xs, ys):
    """Least-squares (slope, intercept), or None when there are fewer than two distinct x."""
    n = len(xs)
    if n < 2 or max(xs) == min(xs):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def extrapolate(runs, ctx, mem_total):
    """Peak RSS at `ctx` tokens from the runs' peak RSS, an estimate; unknown without the data."""
    pts = [(r["prompt_tokens"] + r["completion_tokens"], r["hwm"]) for r in runs if r.get("hwm") is not None]
    fit = linfit([p[0] for p in pts], [p[1] for p in pts])
    res = {"kib_per_token": None, "rss_at_ctx_gib": None, "fits": None, "ctx": ctx, "estimate": True}
    if fit is None or not ctx:
        return res
    slope, icpt = fit
    est = icpt + slope * ctx
    res["kib_per_token"] = slope / 1024.0
    res["rss_at_ctx_gib"] = est / GIB
    if mem_total:
        res["fits"] = est <= mem_total - MARGIN
    return res


def next_estimate(target, gen, prev):
    """Seconds the next run should take, from the previous run's speeds; None if unknown."""
    if not prev or not prev.get("prompt_tps"):
        return None
    return target / prev["prompt_tps"] + gen / (prev.get("gen_tps") or 1.0)


def over_budget(elapsed_s, est_s, budget_s):
    return est_s is not None and elapsed_s + est_s > budget_s


def fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def format_table(runs):
    head = "%8s %8s %12s %11s %10s %12s" % ("prompt", "ttft s", "prompt tok/s", "gen tok/s", "RSS GiB", "avail GiB")
    rows = [head]
    for r in runs:
        rows.append("%8d %8s %12s %11s %10s %12s" % (
            r["prompt_tokens"], fmt(r["ttft_s"], ".1f"), fmt(r["prompt_tps"], ".2f"), fmt(r["gen_tps"], ".2f"),
            fmt(None if r.get("hwm") is None else r["hwm"] / GIB, ".2f"),
            fmt(None if r.get("available") is None else r["available"] / GIB, ".2f")))
    return "\n".join(rows)


# ------------------------------------------------------------------- the server

class Server:
    def __init__(self, cmd):
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except OSError as e:
            raise Failure("could not start %s: %s" % (cmd[0], e))
        self.lines = []
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()

    def _drain(self):
        for raw in self.proc.stderr:
            with self.lock:
                self.lines.append(raw.decode(errors="replace").rstrip())

    def snapshot(self):
        with self.lock:
            return list(self.lines)

    def wait_ready(self, port, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                raise Failure("lm-serve exited with code %d before it was ready: %s" % (rc, " | ".join(self.snapshot()[-3:])))
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    c.close()
                    return self.context()
                c.close()
            except OSError:
                pass
            time.sleep(0.5)
        raise Failure("lm-serve was not healthy after %ds" % timeout)

    def context(self):
        for line in self.snapshot():
            m = re.search(r"listening on .*context (\d+)", line)
            if m:
                return int(m.group(1))
        return None

    def log_after(self, index, wait_s=5.0):
        """The first request log line past `index` (stderr lines already seen), or None."""
        deadline = time.monotonic() + wait_s
        while True:
            lines = self.snapshot()
            for i in range(index, len(lines)):
                got = parse_log_line(lines[i])
                if got:
                    return got
            if time.monotonic() > deadline:
                return None
            time.sleep(0.1)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def post_chat(port, messages, gen, timeout):
    body = json.dumps({"messages": messages, "max_tokens": gen, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        t_send = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise Failure("HTTP %d: %s" % (resp.status, resp.read(300).decode(errors="replace").strip()))
        events = []
        while True:
            line = resp.readline()
            if not line:
                break
            events.append((time.perf_counter(), line))
    except OSError as e:
        raise Failure("request failed (no data for %ss?): %s" % (timeout, e))
    finally:
        conn.close()
    return parse_stream(events, t_send)


def measure(srv, port, messages, gen, timeout):
    """One request, then the server's memory and the machine's."""
    seen = len(srv.snapshot())
    r = post_chat(port, messages, gen, timeout)
    log = srv.log_after(seen)
    r["cached"] = None if log is None else log["cached"]
    r["fed"] = None if log is None else log["fed"]
    st = parse_status(read_file("/proc/%d/status" % srv.proc.pid))
    r["rss"], r["hwm"] = (st["rss"], st["hwm"]) if st else (None, None)
    mi = parse_meminfo(read_file("/proc/meminfo"))
    r["available"] = mi["available"] if mi else None
    return r


# ------------------------------------------------------------------------- main

def parse_args(argv):
    p = argparse.ArgumentParser(description="Measure lm-serve speed and memory through its API.")
    p.add_argument("--model", required=True, help="model directory")
    p.add_argument("--weights", help="a q8 file; omit for the bf16 checkpoint")
    p.add_argument("--serve-bin", default=os.path.join(ROOT, "bin", "lm-serve"))
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--ctx", type=int, help="pass --ctx to lm-serve (default: automatic)")
    p.add_argument("--cache-mb", type=int, help="lm-serve --cache-mb (default 0, or 512 with --cache-test)")
    p.add_argument("--lens", default="128,512,1024,2048", help="target prompt tokens, comma separated")
    p.add_argument("--gen", type=int, default=32, help="reply tokens per request")
    p.add_argument("--cache-test", action="store_true", help="also show the prefix cache")
    p.add_argument("--cache-len", type=int, default=256, help="system prompt tokens for --cache-test")
    p.add_argument("--time-budget-min", type=float, default=60)
    p.add_argument("--start-timeout", type=int, default=600, help="seconds to wait for /health")
    p.add_argument("--read-timeout", type=int, default=3600, help="seconds without data from the server")
    p.add_argument("--out", default="bench_serve.json")
    a = p.parse_args(argv)
    try:
        a.lens = [int(x) for x in a.lens.split(",")]
    except ValueError:
        raise Failure("--lens must be comma separated integers")
    if not a.lens or any(n < 16 for n in a.lens) or a.gen < 2:
        raise Failure("--lens needs values of at least 16 and --gen at least 2")
    if a.cache_mb is None:
        a.cache_mb = 512 if a.cache_test else 0
    if a.cache_test and a.cache_mb < 1:
        raise Failure("--cache-test needs --cache-mb of at least 1")
    return a


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def run(argv):
    a = parse_args(argv)
    cmd = [a.serve_bin, "--model", a.model, "--port", str(a.port), "--cache-mb", str(a.cache_mb)]
    if a.weights:
        cmd += ["--weights", a.weights]
    if a.ctx:
        cmd += ["--ctx", str(a.ctx)]
    t0 = time.monotonic()
    srv = Server(cmd)
    runs, cache_runs, skipped = [], [], []
    try:
        ctx = srv.wait_ready(a.port, a.start_timeout)
        log("lm-serve ready after %.0f s, context %s" % (time.monotonic() - t0, ctx))
        ratio, prev = 1.3, None
        start = time.monotonic()
        for i, target in enumerate(a.lens):
            if over_budget(time.monotonic() - start, next_estimate(target, a.gen, prev), a.time_budget_min * 60):
                skipped.append(target)
                log("skipping %d tokens: it would not finish within %g min" % (target, a.time_budget_min))
                continue
            n_words = words_for(target, ratio)
            text = PROMPT_PREFIX + make_text(n_words, 1000 + i)
            r = measure(srv, a.port, [{"role": "user", "content": text}], a.gen, a.read_timeout)
            r["target"] = target
            ratio = next_ratio(r["prompt_tokens"], n_words)
            runs.append(r)
            prev = r
            log("  %5d prompt tokens: ttft %.1f s, %s tok/s prompt, %s tok/s generating" % (
                r["prompt_tokens"], r["ttft_s"], fmt(r["prompt_tps"], ".2f"), fmt(r["gen_tps"], ".2f")))
        if a.cache_test:
            system = make_text(words_for(a.cache_len, ratio), 77)
            for k, user in enumerate(("Say hi.", "Say yes.", "Say ok.")):
                r = measure(srv, a.port, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                            a.gen, a.read_timeout)
                cache_runs.append(r)
                log("  cache request %d: %d prompt tokens, %s from cache, ttft %.1f s" % (
                    k + 1, r["prompt_tokens"], r["cached"], r["ttft_s"]))
        mem = parse_meminfo(read_file("/proc/meminfo"))
        ex = extrapolate(runs, ctx, mem["total"] if mem else None)
    finally:
        srv.stop()
    result = {"machine": dict(read_cpu_mhz(), mem_total_gib=None if not mem else mem["total"] / GIB),
              "server": {"cmd": cmd, "context": ctx}, "runs": runs, "cache_test": cache_runs,
              "skipped_lens": skipped, "memory_at_context": ex}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    return result


def report(res):
    out = [format_table(res["runs"])]
    cpu = res["machine"]
    if cpu.get("scaling_max_mhz") is not None:
        out.append("cpu max clock: %d MHz (rated %s MHz)" % (cpu["scaling_max_mhz"], cpu["cpuinfo_max_mhz"]))
    ex = res["memory_at_context"]
    if ex["rss_at_ctx_gib"] is None:
        out.append("memory at the full context: unknown (needs /proc and at least two different prompt lengths)")
    else:
        out.append("ESTIMATE: %.2f KiB of RSS per token fed; peak RSS at the full %d-token context about %.2f GiB (%s)" % (
            ex["kib_per_token"], ex["ctx"], ex["rss_at_ctx_gib"],
            {True: "fits in RAM", False: "does NOT fit in RAM", None: "fit unknown"}[ex["fits"]]))
    if res["cache_test"]:
        for k, r in enumerate(res["cache_test"]):
            out.append("cache request %d: %d prompt tokens, %s from the cache, ttft %.1f s" % (
                k + 1, r["prompt_tokens"], fmt(r["cached"], "d"), r["ttft_s"]))
    if res["skipped_lens"]:
        out.append("skipped (time budget): %s" % ", ".join(str(n) for n in res["skipped_lens"]))
    return "\n".join(out)


def main(argv=None):
    try:
        res = run(sys.argv[1:] if argv is None else argv)
    except Failure as e:
        print("bench_serve FAILED: %s" % e, file=sys.stderr)
        return 1
    print(report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
