"""Projects for the chat agent: a native tool server (no subprocess) that works on folders under one base.

It is used by tools/chat_tools.py, which lists it beside the MCP servers as the server "project". One copy
of the service runs per host (the model server with its second SSD, the Mac), so a project always lives on
the host whose service is asked.

- The registry (--projects-registry, default ~/.config/lm-chat/projects.json) is edited by the owner on that
  host, never by the agent: {"projects": {"<id>": {"title": "...", "root": "<folder under the base>"}}}.
  A root must resolve inside the base, so the agent can only ever reach folders the owner put there.
- Every path an agent gives is relative to its project root and is resolved with realpath: no absolute
  paths, no "..", no symlink that leaves the root, nothing under .git.
- Tools: project_list, fs_read, fs_list, fs_search, fs_write, fs_edit. Writes are capped in size and refused
  when the volume is short of free space (the SSD also holds the models).
- doc_parse, only when the owner has installed SoMark OXR on this host (KEVIN_OXR_BIN and
  KEVIN_OXR_MODEL). It runs that fixed binary on one jailed image. The model cannot pass a
  program, a URL, or extra arguments. The model server does not set those variables, so the
  tool stays off beside the loaded chat model.

Python 3 standard library only.
"""

import collections
import fnmatch
import hashlib
import json
import os
import re
import shutil
import tempfile
import time

import chat_audit
import chat_commands
import chat_git
import chat_remote
from chat_jail import GB, PathError, ToolFail, inside, int_arg, looks_binary, rel_of, resolve  # noqa: F401  (re-exported)

PROTOCOL_VERSION = "2025-06-18"
PROJECT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_READ_BYTES = 200 * 1024
MAX_READ_FILE = 20 * 1024 * 1024
DEFAULT_READ_LINES = 400
MAX_READ_LINES = 2000
MAX_WRITE_BYTES = 2 * 1024 * 1024
MAX_LIST = 200
MAX_MATCHES = 50
MAX_SEARCH_FILE = 1024 * 1024
MAX_PATTERN = 200
SEARCH_SECONDS = 10
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", "build"}


# ---- registry ----------------------------------------------------------------------------------

class RegistryError(ValueError):
    pass


DEFAULT_IDENTITY = {"name": "Kevin AI", "email": "developer@marcfors.com"}


def _identity(doc, path):
    ident = doc.get("identity", DEFAULT_IDENTITY)
    if not isinstance(ident, dict):
        raise RegistryError(f"{path}: identity must be an object with a name and an email")
    name, email = ident.get("name", DEFAULT_IDENTITY["name"]), ident.get("email", DEFAULT_IDENTITY["email"])
    for v in (name, email):
        if not isinstance(v, str) or not v.strip() or len(v) > 100 or any(c in v for c in "\n\r\x00<>"):
            raise RegistryError(f"{path}: the identity needs a plain name and email")
    if "@" not in email:
        raise RegistryError(f"{path}: the identity email must contain @")
    return {"name": name.strip(), "email": email.strip()}


def load_registry_doc(path, base):
    """(identity, {id: project}) from the registry file; the default identity and {} when there is none. Raises RegistryError."""
    base_real = os.path.realpath(base)
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        return dict(DEFAULT_IDENTITY), {}
    except (OSError, ValueError) as e:
        raise RegistryError(f"{path}: {e}") from None
    projects = doc.get("projects") if isinstance(doc, dict) else None
    if not isinstance(projects, dict):
        raise RegistryError(f'{path}: expected {{"projects": {{...}}}}')
    identity = _identity(doc, path)
    out = {}
    for pid, p in projects.items():
        if not PROJECT_ID.match(pid):
            raise RegistryError(f"bad project name {pid!r}: lowercase letters, digits, - and _, at most 32")
        if not isinstance(p, dict) or not isinstance(p.get("root"), str) or not p["root"]:
            raise RegistryError(f"{pid}: needs a root folder")
        root = os.path.realpath(os.path.join(base_real, p["root"]))
        if not inside(base_real, root) or root == base_real:
            raise RegistryError(f"{pid}: the root must be a folder inside {base_real}")
        if not os.path.isdir(root):
            raise RegistryError(f"{pid}: {root} is not a folder")
        title = p.get("title") if isinstance(p.get("title"), str) and p["title"] else pid
        branch = p.get("defaultBranch", "main")
        if not isinstance(branch, str) or not chat_git.BRANCH_NAME.match(branch):
            raise RegistryError(f"{pid}: defaultBranch must be a branch name")
        if not isinstance(p.get("allowMain", False), bool):
            raise RegistryError(f"{pid}: allowMain must be true or false")
        try:
            commands = chat_commands.parse_commands(pid, p.get("commands"))
        except chat_commands.CommandError as e:
            raise RegistryError(str(e)) from None
        remote = p.get("remote")
        if remote is not None and not chat_remote.valid_remote(remote):
            raise RegistryError(f"{pid}: remote must be an https URL without credentials, for example https://github.com/owner/repo.git")
        token_env = p.get("tokenEnv")
        if token_env is not None and not chat_remote.valid_token_env(token_env):
            raise RegistryError(f"{pid}: tokenEnv must be the name of an environment variable, for example KEVIN_GITHUB_TOKEN")
        out[pid] = {"id": pid, "title": title, "root": root, "defaultBranch": branch, "allowMain": p.get("allowMain", False),
                    "commands": commands, "remote": remote, "tokenEnv": token_env}
    return identity, out


def load_registry(path, base):
    """{id: project} from the registry file; {} when there is none. Raises RegistryError."""
    return load_registry_doc(path, base)[1]


# ---- the path jail -----------------------------------------------------------------------------

# ---- tools -------------------------------------------------------------------------------------

PROJECT_ARG = {"type": "string", "description": "The project id; optional when there is only one."}


def _tool(name, description, props, required=(), read_only=False):
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": {"project": PROJECT_ARG, **props}, "required": list(required)},
        "annotations": {"readOnlyHint": read_only},
    }


TOOLS = [
    _tool("project_list", "List the projects you can work on, with the free disk space.", {}, read_only=True),
    _tool("fs_read", "Read a text file of the project, from a line; long files come in parts.",
          {"path": {"type": "string"}, "offset": {"type": "integer", "description": "First line, from 1."},
           "limit": {"type": "integer", "description": "Lines to read, at most 2000."}}, ["path"], True),
    _tool("fs_list", "List a folder of the project, one to three levels deep.",
          {"path": {"type": "string", "description": "Folder, default the project root."},
           "depth": {"type": "integer", "description": "Levels, 1 to 3."}}, read_only=True),
    _tool("fs_search", "Find a literal text in the files of the project; returns path:line: text.",
          {"pattern": {"type": "string"}, "path": {"type": "string", "description": "Folder to search, default the root."},
           "glob": {"type": "string", "description": "Only file names matching, for example *.py."},
           "ignore_case": {"type": "boolean"}}, ["pattern"], True),
    _tool("fs_write", "Create a file or replace it entirely with the given text.",
          {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _tool("fs_edit", "Replace an exact piece of text in a file; the text must appear once unless replace_all is set.",
          {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"},
           "replace_all": {"type": "boolean"}}, ["path", "old", "new"]),
]
TOOLS += chat_git.GIT_DEFS + chat_remote.REMOTE_DEFS

# SoMark OXR (document images -> markdown). Present only when this host has the runtime.
# The paths are the owner's environment, never arguments from the model.
DOC_PARSE = _tool(
    "doc_parse",
    "Read one document image in this project with SoMark OXR and return its text as markdown. "
    "Use it when the chat model cannot see the picture. path is a png, jpeg, or bmp file inside the project. "
    "SoMark OXR runs on this machine; it does not replace the chat model.",
    {"path": {"type": "string", "description": "Project-relative image path (png, jpeg, or bmp)."}},
    ["path"], True)
OXR_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
OXR_MAX_BYTES = 15 * 1024 * 1024
OXR_MAX_OUT = 64 * 1024
OXR_TIMEOUT = 300


def _oxr_file(path, exe=False):
    if not isinstance(path, str) or not path or path.startswith("-") or "\x00" in path:
        return False
    if not os.path.isfile(path) or os.path.islink(path):
        return False
    return not exe or os.access(path, os.X_OK)


def oxr_runtime():
    """(binary, model, layout or None) from the environment, or None when this host has no OXR."""
    binary = os.environ.get("KEVIN_OXR_BIN", "")
    model = os.environ.get("KEVIN_OXR_MODEL", "")
    layout = os.environ.get("KEVIN_OXR_LAYOUT", "")
    if not _oxr_file(binary, exe=True) or not _oxr_file(model):
        return None
    if layout and not _oxr_file(layout):
        layout = ""
    return binary, model, layout or None


def tool_defs(extra=()):
    tools = list(TOOLS)
    if oxr_runtime():
        tools.append(DOC_PARSE)
    return tools + list(extra)


def doc_parse(root, args):
    runtime = oxr_runtime()
    if not runtime:
        raise ToolFail("SoMark OXR is not installed on this machine")
    binary, model, _layout = runtime
    real = resolve(root, args.get("path"))
    rel = rel_of(root, real)
    if not os.path.isfile(real):
        raise ToolFail(f"{rel} is not a file")
    if os.path.splitext(real)[1].lower() not in OXR_EXTS:
        raise ToolFail(f"{rel} must be a png, jpeg, or bmp image")
    if os.path.getsize(real) > OXR_MAX_BYTES:
        raise ToolFail(f"{rel} is too large to parse")
    # CrispEmbed's OXR recognition stage. Its --ocr-engine path ignores a separate layout file.
    argv = [binary, "--ocr-pipeline", real, "--ocr-engine", "oxr", "--ocr-rec", model, "-t", "4"]
    text, ok = chat_commands.run(
        {"root": root},
        {"argv": argv, "timeout": OXR_TIMEOUT, "cwd": None, "env": {
            "CRISPEMBED_OXR": "1",
            "CRISPEMBED_ACCEPT_OXR_LICENSE": "1",
        }},
        max_out=OXR_MAX_OUT)
    if not ok:
        raise ToolFail(text)
    return f"{rel} parsed with SoMark OXR\n{text}"
# The tools that change something: these are recorded in the audit log.
CHANGING = {t["name"] for t in TOOLS if not t["annotations"]["readOnlyHint"]}
# Changes whenever a definition changes, so a kept tool list is never older than the code.
TOOLS_VERSION = hashlib.sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest()[:12]


def _text_file(real, rel):
    if not os.path.exists(real):
        raise ToolFail(f"{rel} does not exist")
    if not os.path.isfile(real):
        raise ToolFail(f"{rel} is not a file")
    size = os.path.getsize(real)
    if size > MAX_READ_FILE:
        raise ToolFail(f"{rel} is {size // (1024 * 1024)} MB: too large to read")
    with open(real, "rb") as f:
        head = f.read(8192)
        if looks_binary(head):
            raise ToolFail(f"{rel} is a binary file")
        raw = head + f.read()
    return raw.decode("utf-8", "replace")


def fs_read(root, args, host):
    real = resolve(root, args.get("path"))
    rel = rel_of(root, real)
    lines = _text_file(real, rel).splitlines()
    if not lines:
        return f"{rel} is empty"
    first = int_arg(args.get("offset"), 1, 1, 10 ** 9)
    if first > len(lines):
        return f"{rel} has {len(lines)} lines; offset {first} is past the end"
    count = int_arg(args.get("limit"), DEFAULT_READ_LINES, 1, MAX_READ_LINES)
    chosen, used = [], 0
    for line in lines[first - 1:first - 1 + count]:
        used += len(line.encode("utf-8")) + 1
        if used > MAX_READ_BYTES:
            break
        chosen.append(line)
    last = first + len(chosen) - 1
    more = f"\n[more lines follow: call again with offset {last + 1}]" if last < len(lines) else ""
    return f"{rel} lines {first}-{last} of {len(lines)}\n" + "\n".join(chosen) + more


def fs_list(root, args, host):
    real = resolve(root, args.get("path"))
    if not os.path.isdir(real):
        raise ToolFail(f"{rel_of(root, real)} is not a folder")
    depth = int_arg(args.get("depth"), 1, 1, 3)
    out, more = [], 0

    def walk(folder, level):
        nonlocal more
        try:
            entries = sorted(os.scandir(folder), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError as e:
            out.append(f"{rel_of(root, folder)}: {e.strerror}")
            return
        for e in entries:
            if e.name in IGNORED_DIRS and e.is_dir(follow_symlinks=False):
                continue
            if len(out) >= MAX_LIST:
                more += 1
                continue
            rel = rel_of(root, e.path)
            if e.is_symlink():
                out.append(f"{rel} -> link")
            elif e.is_dir(follow_symlinks=False):
                out.append(rel + "/")
                if level < depth:
                    walk(e.path, level + 1)
            else:
                out.append(f"{rel} ({e.stat(follow_symlinks=False).st_size} bytes)")

    walk(real, 1)
    if not out:
        return f"{rel_of(root, real)} is empty"
    return "\n".join(out) + (f"\n[{more} more entries not shown]" if more else "")


def fs_search(root, args, host):
    # Literal text only: a model-written regular expression can backtrack for ever and Python cannot stop it.
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_PATTERN:
        raise ToolFail(f"pattern must be a text of 1 to {MAX_PATTERN} characters")
    base = resolve(root, args.get("path"))
    glob = args.get("glob") if isinstance(args.get("glob"), str) else None
    fold = args.get("ignore_case") is True
    needle = pattern.lower() if fold else pattern
    deadline = time.monotonic() + SEARCH_SECONDS
    hits, more = [], 0
    for folder, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not os.path.islink(os.path.join(folder, d)))
        for name in sorted(files):
            if glob and not fnmatch.fnmatch(name, glob):
                continue
            path = os.path.join(folder, name)
            if os.path.islink(path):
                continue
            try:
                if os.path.getsize(path) > MAX_SEARCH_FILE:
                    continue
                with open(path, "rb") as f:
                    raw = f.read()
            except OSError:
                continue
            if looks_binary(raw[:4096]):
                continue
            for n, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
                if needle in (line.lower() if fold else line):
                    if len(hits) < MAX_MATCHES:
                        hits.append(f"{rel_of(root, path)}:{n}: {line.strip()[:200]}")
                    else:
                        more += 1
            if time.monotonic() > deadline:
                hits.append("[the search ran out of time; narrow it with path or glob]")
                return "\n".join(hits)
    if not hits:
        return "no matches"
    return "\n".join(hits) + (f"\n[{more} more matches; narrow the search with path or glob]" if more else "")


def _room(root, host, size):
    if size > MAX_WRITE_BYTES:
        raise ToolFail(f"{size} bytes is over the {MAX_WRITE_BYTES // (1024 * 1024)} MB limit for one file")
    free = shutil.disk_usage(root).free
    if free - size < host.min_free_bytes:
        raise ToolFail(f"the disk has {free // GB} GB free, under the {host.min_free_bytes // GB} GB kept for the models: nothing was written")


def _store(real, data):
    """Writes the bytes next to the file and moves them into place, keeping the file's mode."""
    folder = os.path.dirname(real)
    os.makedirs(folder, exist_ok=True)
    mode = os.stat(real).st_mode & 0o7777 if os.path.exists(real) else 0o644
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".kevin-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, real)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def fs_write(root, args, host):
    real = resolve(root, args.get("path"))
    rel = rel_of(root, real)
    content = args.get("content")
    if not isinstance(content, str):
        raise ToolFail("content must be text")
    if real == root or os.path.isdir(real):
        raise ToolFail(f"{rel} is a folder")
    data = content.encode("utf-8")
    _room(root, host, len(data))
    existed = os.path.exists(real)
    _store(real, data)
    return f"{'replaced' if existed else 'created'} {rel} ({len(data)} bytes)"


def fs_edit(root, args, host):
    real = resolve(root, args.get("path"))
    rel = rel_of(root, real)
    old, new = args.get("old"), args.get("new")
    if not isinstance(old, str) or not old or not isinstance(new, str):
        raise ToolFail("old must be the text to replace (not empty) and new the text to put there")
    text = _text_file(real, rel)
    found = text.count(old)
    if found == 0:
        raise ToolFail(f"the text to replace was not found in {rel}; read the file again and copy it exactly")
    if found > 1 and args.get("replace_all") is not True:
        raise ToolFail(f"the text appears {found} times in {rel}; add more around it so it appears once, or set replace_all")
    data = text.replace(old, new).encode("utf-8")
    _room(root, host, len(data))
    _store(real, data)
    return f"replaced {found} occurrence{'s' if found != 1 else ''} in {rel}"


FILE_TOOLS = {"fs_read": fs_read, "fs_list": fs_list, "fs_search": fs_search, "fs_write": fs_write, "fs_edit": fs_edit}


# ---- the host and its client -------------------------------------------------------------------

def _result(text, is_error=False):
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class ProjectHost:
    def __init__(self, base, registry_path, min_free_gb=5, audit=None):
        self.base = os.path.realpath(base)
        self.registry_path = registry_path
        self.min_free_bytes = int(min_free_gb * GB)
        self.audit = audit                       # a chat_audit.Audit, or None: nothing is recorded

    def projects(self):
        return load_registry(self.registry_path, self.base)

    def identity(self):
        return load_registry_doc(self.registry_path, self.base)[0]

    def command_tools(self):
        """One tool per allowlisted command of every project: run_<project>_<name>."""
        return [chat_commands.tool_def(pid, cmd) for pid, p in self.projects().items() for cmd in p["commands"].values()]

    def tools_version(self):
        """Changes when a tool definition or an allowlisted command changes, so a kept tool list is never stale."""
        try:
            commands = {pid: p["commands"] for pid, p in self.projects().items()}
        except RegistryError:
            return TOOLS_VERSION + "-bad"
        return hashlib.sha256((TOOLS_VERSION + json.dumps(commands, sort_keys=True)).encode()).hexdigest()[:12]

    def _command(self, name):
        for pid, p in self.projects().items():
            for cmd in p["commands"].values():
                if chat_commands.tool_name(pid, cmd["name"]) == name:
                    return p, cmd
        return None, None

    def _record_change(self, project, tool, args, result):
        """Audit a call that changes something (read-only tools are not recorded)."""
        if self.audit is not None and tool in CHANGING:
            self._record(project, tool, args, not result["isError"], result["content"][0]["text"])

    def _record_failure(self, tool, args, error):
        if self.audit is not None and (tool in CHANGING or tool.startswith("run_")):
            pid = args.get("project") if isinstance(args.get("project"), str) else "?"
            self._record(pid, tool, args, False, f"error: {error}")

    def _record(self, project, tool, args, ok, text):
        if self.audit is not None:
            self.audit.log(project, tool, chat_audit.summarize(tool, args), ok, text)

    def summary(self):
        """What the chat page may show about the projects: ids, titles, rules and command names, never paths or tokens."""
        projects = self.projects()
        return {"projects": [
            {"id": p["id"], "title": p["title"], "defaultBranch": p["defaultBranch"], "allowMain": p["allowMain"],
             "hasRemote": bool(p.get("remote")), "commands": sorted(p["commands"])}
            for p in projects.values()], "freeGb": shutil.disk_usage(self.base).free // GB}

    def problem(self):
        """The reason the registry cannot be used, or None."""
        try:
            self.projects()
            return None
        except RegistryError as e:
            return str(e)

    def _project(self, args):
        projects = self.projects()
        pid = args.get("project")
        if pid is None or pid == "":
            if len(projects) == 1:
                return next(iter(projects.values()))
            raise ToolFail(f"say which project (project argument); available: {', '.join(sorted(projects)) or 'none yet'}")
        if not isinstance(pid, str) or pid not in projects:
            raise ToolFail(f"no project named {pid!r}; available: {', '.join(sorted(projects)) or 'none yet'}")
        return projects[pid]

    def call(self, name, args):
        """A tools/call result: text content, with isError set when the tool could not do it."""
        try:
            if name == "project_list":
                return _result(self._list())
            if name == "doc_parse":
                project = self._project(args)
                return _result(doc_parse(project["root"], args))
            if name in FILE_TOOLS:
                project = self._project(args)
                result = _result(FILE_TOOLS[name](project["root"], args, self))
                self._record_change(project["id"], name, args, result)
                return result
            if name in chat_git.GIT_HANDLERS:
                project = self._project(args)
                result = _result(chat_git.GIT_HANDLERS[name](project, args, self))
                self._record_change(project["id"], name, args, result)
                return result
            if name in chat_remote.REMOTE_HANDLERS:
                project = self._project(args)
                result = _result(chat_remote.REMOTE_HANDLERS[name](project, args, self))
                self._record_change(project["id"], name, args, result)
                return result
            if name.startswith("run_"):
                project, cmd = self._command(name)
                if cmd is None:
                    raise ToolFail(f"no tool named {name!r}")
                text, ok = chat_commands.run(project, cmd)
                result = _result(text, not ok)
                self._record(project["id"], name, args, ok, text)
                return result
            raise ToolFail(f"no tool named {name!r}")
        except (PathError, ToolFail) as e:
            self._record_failure(name, args, e)
            return _result(f"error: {e}", True)
        except RegistryError as e:
            return _result(f"error: the project registry cannot be used: {e}", True)
        except OSError as e:
            return _result(f"error: {e.strerror or e}", True)

    def _list(self):
        projects = self.projects()
        free = shutil.disk_usage(self.base).free // GB
        if not projects:
            return f"no projects yet under {self.base} ({free} GB free)"
        lines = [f"{p['id']}: {p['title']} ({os.path.relpath(p['root'], self.base)})" for p in projects.values()]
        return "\n".join(lines) + f"\n{free} GB free on the disk"


class NativeProjectClient:
    """Speaks just enough MCP for the Manager: initialize, tools/list and tools/call, in this process."""

    def __init__(self, sid, spec, env, host):
        self.sid = sid
        self.spec = spec
        self.host = host
        self.dead = None
        self.stale = False
        self.stderr = collections.deque(maxlen=1)

    def request(self, method, params=None, timeout=None):
        if method == "initialize":
            return {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "serverInfo": {"name": "kevin-projects", "version": TOOLS_VERSION}}
        if method == "tools/list":
            try:
                return {"tools": tool_defs(self.host.command_tools())}
            except RegistryError:
                return {"tools": tool_defs()}
        if method == "tools/call":
            params = params or {}
            return self.host.call(params.get("name"), params.get("arguments") or {})
        return {}

    def notify(self, method, params=None):
        pass

    def close(self):
        pass
