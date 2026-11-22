"""An append-only record of what the agent changed in the projects of this host (tools/chat_projects.py).

One JSON object per line in a file only its owner can read: when, which project, which tool, a short summary of
the arguments (never file contents or tokens), whether it worked and the first words of the result. Read-only
calls are not logged. The file is renamed to <name>.1 when it passes max_bytes, so it cannot fill the disk.
Python 3 standard library only.
"""

import json
import os
import threading
import time

MAX_FIELD = 300


class Audit:
    def __init__(self, path, max_bytes=5 * 1024 * 1024):
        self.path = path
        self.max_bytes = max_bytes
        self.lock = threading.Lock()

    def log(self, project, tool, summary, ok, result):
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "project": project,
            "tool": tool,
            "summary": str(summary)[:MAX_FIELD],
            "ok": bool(ok),
            "result": str(result)[:MAX_FIELD],
        }
        line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
        with self.lock:
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                if os.path.exists(self.path) and os.path.getsize(self.path) + len(line) > self.max_bytes:
                    os.replace(self.path, self.path + ".1")
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    os.write(fd, line)
                finally:
                    os.close(fd)
            except OSError:
                pass    # a log that cannot be written must not stop the work; the call's own result still reports it


def summarize(tool, args):
    """What to record about a call: paths, names and sizes, never contents."""
    if tool == "fs_write":
        content = args.get("content")
        return f"{args.get('path')} ({len(content.encode('utf-8')) if isinstance(content, str) else 0} bytes)"
    if tool == "fs_edit":
        return f"{args.get('path')}"
    if tool == "git_commit":
        message = str(args.get("message") or "")
        return message.splitlines()[0][:100] if message.strip() else ""
    if tool in ("git_add", "git_restore"):
        return "all" if args.get("all") is True else ", ".join(str(p) for p in (args.get("paths") or []))[:200]
    if tool in ("git_branch_create", "git_switch"):
        return str(args.get("name", ""))
    return ""
