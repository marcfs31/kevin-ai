"""The path jail and the small helpers every project tool shares (tools/chat_projects.py, tools/chat_git.py).

Every path an agent gives is relative to its project root and is resolved with realpath: no absolute paths,
no "..", no symlink that leaves the root, nothing under .git. Python 3 standard library only.
"""

import os
import re

GB = 1024 ** 3


class PathError(Exception):
    pass


class ToolFail(Exception):
    """A tool could not do what was asked; the message goes back to the model as the result."""


def inside(root, path):
    return path == root or path.startswith(root + os.sep)


def resolve(root, rel):
    """The real path of `rel` inside `root`, or PathError. `root` is already a real path."""
    if rel is None or rel == "":
        rel = "."
    if not isinstance(rel, str) or "\x00" in rel:
        raise PathError("the path must be a text path")
    if os.path.isabs(rel) or rel.startswith("~"):
        raise PathError("use a path relative to the project root")
    parts = [p for p in re.split(r"[\\/]+", rel) if p not in ("", ".")]
    if ".." in parts:
        raise PathError(".. is not allowed")
    if ".git" in parts:
        raise PathError(".git is not available to file tools")
    real = os.path.realpath(os.path.join(root, *parts))
    if not inside(root, real):
        raise PathError("the path leaves the project (a symlink?)")
    if ".git" in os.path.relpath(real, root).split(os.sep):
        raise PathError(".git is not available to file tools")
    return real


def rel_of(root, real):
    r = os.path.relpath(real, root)
    return "." if r == "." else r.replace(os.sep, "/")


def looks_binary(chunk):
    return b"\x00" in chunk


def int_arg(v, default, low, high):
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int):
        raise ToolFail("expected a whole number")
    return max(low, min(high, v))
