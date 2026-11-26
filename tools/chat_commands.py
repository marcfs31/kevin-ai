"""Allowlisted commands for the project tools (tools/chat_projects.py): tests, builds, linters.

The owner lists them per project in the registry, which the agent cannot edit:

    "commands": {"test": {"argv": ["make", "test"], "timeout": 120, "safe": false, "cwd": "web", "env": {"CI": "1"}}}

Each command becomes its own tool, run_<project>_<name>, so the approval card can say what will run and a
command marked "safe" can run without asking while the others ask. The agent cannot add arguments: the argv is
fixed, run without a shell, in the project folder (or a folder inside it), with a minimal environment (no secrets
of the service), a private HOME that is removed afterwards, no stdin, a time limit that kills the whole process
group, and only the tail of the output kept. A command still has the network and the user's files its own
program can reach: this is an allowlist, not a sandbox. Python 3 standard library only.
"""

import collections
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading

from chat_jail import ToolFail, resolve

COMMAND_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,15}$")
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
FORBIDDEN_ENV = ("PATH", "HOME", "TMPDIR", "LD_", "DYLD_", "GIT_", "PYTHONPATH", "NODE_OPTIONS")
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 900
MAX_ARGV = 32
MAX_ARG = 500
TAIL_BYTES = 64 * 1024
MAX_OUT = 6000


class CommandError(ValueError):
    pass


def parse_commands(pid, raw):
    """{name: {"name", "argv", "timeout", "safe", "cwd", "env"}} from a project's registry entry. Raises CommandError."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise CommandError(f"{pid}: commands must be an object")
    out = {}
    for name, spec in raw.items():
        if not COMMAND_NAME.match(name):
            raise CommandError(f"{pid}: bad command name {name!r}: lowercase letters, digits, - and _, at most 16")
        if not isinstance(spec, dict):
            raise CommandError(f"{pid}/{name}: the entry must be an object")
        argv = spec.get("argv")
        if not isinstance(argv, list) or not 1 <= len(argv) <= MAX_ARGV or not all(
                isinstance(a, str) and a and len(a) <= MAX_ARG and "\x00" not in a for a in argv):
            raise CommandError(f"{pid}/{name}: argv must be a list of 1 to {MAX_ARGV} non-empty strings")
        timeout = spec.get("timeout", DEFAULT_TIMEOUT)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT:
            raise CommandError(f"{pid}/{name}: timeout must be a whole number of seconds from 1 to {MAX_TIMEOUT}")
        if not isinstance(spec.get("safe", False), bool):
            raise CommandError(f"{pid}/{name}: safe must be true or false")
        cwd = spec.get("cwd")
        if cwd is not None and (not isinstance(cwd, str) or not cwd or os.path.isabs(cwd) or ".." in cwd.split("/")):
            raise CommandError(f"{pid}/{name}: cwd must be a folder inside the project")
        env = spec.get("env", {})
        if not isinstance(env, dict) or len(env) > 20:
            raise CommandError(f"{pid}/{name}: env must be an object of at most 20 names")
        for k, v in env.items():
            if not ENV_NAME.match(k) or k.startswith(FORBIDDEN_ENV) or not isinstance(v, str) or len(v) > MAX_ARG or "\x00" in v:
                raise CommandError(f"{pid}/{name}: environment variable {k!r} is not allowed")
        out[name] = {"name": name, "argv": argv, "timeout": timeout, "safe": spec.get("safe", False), "cwd": cwd, "env": env}
    return out


def tool_name(pid, name):
    return f"run_{pid}_{name}"


def tool_def(pid, cmd):
    where = f" in {cmd['cwd']}" if cmd["cwd"] else ""
    return {
        "name": tool_name(pid, cmd["name"]),
        "description": f"Run `{shlex.join(cmd['argv'])}`{where} in the project {pid}; stops after {cmd['timeout']} s.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": cmd["safe"]},
    }


def _tail(text_bytes, total, max_out=MAX_OUT):
    text = text_bytes.decode("utf-8", "replace")
    cut = ""
    if total > len(text_bytes):
        cut = f"[{total - len(text_bytes)} earlier bytes left out]\n"
    if len(text) > max_out:
        cut = f"[output cut to the last {max_out} characters]\n"
        text = text[-max_out:]
    return cut + text


def run(project, cmd, env_in=None, max_out=MAX_OUT):
    """Runs one allowlisted command. Returns (text, ok): the exit status and the tail of its output."""
    root = project["root"]
    cwd = resolve(root, cmd["cwd"]) if cmd["cwd"] else root
    if not os.path.isdir(cwd):
        raise ToolFail(f"the folder {cmd['cwd']} of this command does not exist")
    home = tempfile.mkdtemp(prefix="kevin-run-")
    env = {"PATH": (env_in or os.environ).get("PATH", "/usr/bin:/bin"), "HOME": home, "TMPDIR": home, "LANG": "C.UTF-8", "CI": "1"}
    env.update(cmd["env"])
    ring, total = collections.deque(), [0]
    try:
        try:
            proc = subprocess.Popen(cmd["argv"], cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except FileNotFoundError:
            raise ToolFail(f"command not found: {cmd['argv'][0]}") from None
        except PermissionError:
            raise ToolFail(f"cannot run {cmd['argv'][0]}: permission denied") from None

        def drain():
            held = 0
            for chunk in iter(lambda: proc.stdout.read(8192), b""):
                total[0] += len(chunk)
                ring.append(chunk)
                held += len(chunk)
                while held - len(ring[0]) >= TAIL_BYTES:
                    held -= len(ring.popleft())

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        timed_out = False
        try:
            rc = proc.wait(timeout=cmd["timeout"])
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            rc = proc.wait()
        try:
            os.killpg(proc.pid, signal.SIGKILL)         # anything it left running in the group
        except (ProcessLookupError, PermissionError):
            pass
        reader.join(5)
        proc.stdout.close()
        output = _tail(b"".join(ring), total[0], max_out)
        if timed_out:
            return f"timed out after {cmd['timeout']} s and was stopped\n{output}".rstrip(), False
        return f"exit code {rc}\n{output}".rstrip(), rc == 0
    finally:
        shutil.rmtree(home, ignore_errors=True)
