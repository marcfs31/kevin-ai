"""Local git for the project tools (tools/chat_projects.py): status, diff, log, show, branches, add, commit, restore.

Every call runs `git` on the project's folder with an argument list (never a shell) and a hardened setup, so
a repository cannot run code through its own configuration or hooks:

- GIT_CONFIG_NOSYSTEM and GIT_CONFIG_GLOBAL=/dev/null: no system or user configuration;
- -c core.hooksPath=/dev/null, core.fsmonitor=false, diff.external=, credential.helper=, core.editor=true,
  protocol.ext.allow=never and protocol.file.allow=never on every call;
- --no-ext-diff and --no-textconv where a diff is made, no terminal prompts, a 30 second limit.

Rules for the agent: commits never go to the project's default branch unless the owner allowed it (allowMain),
branches it creates are named kevin/<name>, and a path it names goes through the same jail as the file tools.
The committer is the identity in the registry. Python 3 standard library only.
"""

import os
import re
import subprocess

from chat_jail import ToolFail, int_arg, rel_of, resolve

GIT_TIMEOUT = 30
MAX_OUT = 6000
MAX_PATHS = 50
MAX_MESSAGE = 2000
AGENT_PREFIX = "kevin/"
BRANCH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,79}$")
REV = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/~^@{}-]{0,99}$")

GIT_PREFIX = [
    "git",
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.fsmonitor=false",
    "-c", "core.pager=cat",
    "-c", "core.editor=true",
    "-c", "core.quotepath=false",
    "-c", "diff.external=",
    "-c", "credential.helper=",
    "-c", "color.ui=false",
    "-c", "protocol.ext.allow=never",
    "-c", "protocol.file.allow=never",
]


def git_env(identity=None):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.devnull,
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_EDITOR": "true",
    }
    if identity:
        env.update({"GIT_AUTHOR_NAME": identity["name"], "GIT_AUTHOR_EMAIL": identity["email"],
                    "GIT_COMMITTER_NAME": identity["name"], "GIT_COMMITTER_EMAIL": identity["email"]})
    return env


def run_git(root, args, identity=None, timeout=GIT_TIMEOUT):
    """(exit code, stdout, stderr) of `git <args>` in the project folder."""
    try:
        p = subprocess.run(GIT_PREFIX + list(args), cwd=root, env=git_env(identity), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ToolFail(f"git took longer than {timeout} seconds") from None
    except FileNotFoundError:
        raise ToolFail("git is not installed on this host") from None
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def _repo(root):
    if not os.path.exists(os.path.join(root, ".git")):
        raise ToolFail("this project is not a git repository yet: call git_init first")


def _fail(err, what):
    raise ToolFail((err.strip() or f"git {what} failed")[:800])


def _ok(root, args, identity=None):
    rc, out, err = run_git(root, args, identity)
    if rc != 0:
        _fail(err or out, args[0])
    return out


def _cap(text, hint):
    text = text.rstrip("\n")
    if len(text) <= MAX_OUT:
        return text
    return text[:MAX_OUT] + f"\n[cut at {MAX_OUT} characters: {hint}]"


def _paths(root, value, what="paths"):
    if not isinstance(value, list) or not value or not all(isinstance(p, str) and p for p in value):
        raise ToolFail(f"{what} must be a list of paths relative to the project")
    if len(value) > MAX_PATHS:
        raise ToolFail(f"at most {MAX_PATHS} paths at a time")
    return [rel_of(root, resolve(root, p)) for p in value]


def _branch(args, key="name"):
    name = args.get(key)
    if not isinstance(name, str) or not BRANCH_NAME.match(name) or ".." in name or "//" in name \
            or name.endswith((".lock", "/", ".")) or "@{" in name:
        raise ToolFail("the branch name may use letters, digits, . _ - and /, and no '..' or trailing '/' or '.lock'")
    return name


def _rev(value, what="rev"):
    if not isinstance(value, str) or not REV.match(value) or ".." in value:
        raise ToolFail(f"{what} must be a branch, tag or commit name")
    return value


def _current(root):
    """The branch HEAD is on, also before the first commit (rev-parse cannot say); None when HEAD is detached."""
    rc, out, _ = run_git(root, ["symbolic-ref", "--short", "-q", "HEAD"])
    return out.strip() if rc == 0 and out.strip() else None


# ---- tools -------------------------------------------------------------------------------------

def git_status(project, args, host):
    root = project["root"]
    _repo(root)
    lines = _ok(root, ["status", "--porcelain=v1", "-b", "--untracked-files=normal"]).splitlines()
    head, entries = lines[0], lines[1:]
    if not entries:
        return f"{head}\nno changes"
    more = f"\n[{len(entries) - 200} more entries]" if len(entries) > 200 else ""
    return head + "\n" + "\n".join(entries[:200]) + more


def git_diff(project, args, host):
    root = project["root"]
    _repo(root)
    cmd = ["diff", "--no-ext-diff", "--no-textconv", "--no-color"]
    if args.get("staged") is True:
        cmd.append("--staged")
    if args.get("stat") is True:
        cmd.append("--stat")
    if args.get("path") not in (None, ""):
        cmd += ["--", rel_of(root, resolve(root, args["path"]))]
    out = _ok(root, cmd)
    return _cap(out, "give a path, or stat for a summary") if out.strip() else "no differences"


def git_log(project, args, host):
    root = project["root"]
    _repo(root)
    cmd = ["log", f"-n{int_arg(args.get('limit'), 10, 1, 30)}", "--format=%h %ad %an: %s", "--date=short"]
    if args.get("path") not in (None, ""):
        cmd += ["--", rel_of(root, resolve(root, args["path"]))]
    rc, out, err = run_git(root, cmd)
    if rc != 0:
        if "does not have any commits" in err:
            return "no commits yet"
        _fail(err, "log")
    return out.rstrip("\n") or "no commits"


def git_show(project, args, host):
    root = project["root"]
    _repo(root)
    cmd = ["show", "--no-ext-diff", "--no-textconv", "--no-color", "--format=%h %an %ad%n%s%n%n%b", "--date=short"]
    if args.get("patch") is not True:
        cmd.append("--stat")
    out = _ok(root, cmd + [_rev(args.get("rev", "HEAD")), "--"])
    return _cap(out, "show with patch off, or look at one file with git_diff")


def git_branches(project, args, host):
    root = project["root"]
    _repo(root)
    return _ok(root, ["branch", "--list", "-vv", "--no-color"]).rstrip("\n") or "no branches yet (make a first commit)"


def git_branch_create(project, args, host):
    root = project["root"]
    _repo(root)
    name = _branch(args)
    if not name.startswith(AGENT_PREFIX):
        name = AGENT_PREFIX + name
    name = _branch({"name": name})
    cmd = ["switch", "-c", name]
    if args.get("from") not in (None, ""):
        cmd.append(_rev(args["from"], "from"))
    _ok(root, cmd)
    return f"created and switched to {name}"


def git_switch(project, args, host):
    root = project["root"]
    _repo(root)
    name = _branch(args)
    _ok(root, ["switch", name])
    return f"switched to {name}"


def git_add(project, args, host):
    root = project["root"]
    _repo(root)
    if args.get("all") is True:
        _ok(root, ["add", "-A"])
    else:
        _ok(root, ["add", "--"] + _paths(root, args.get("paths")))
    return _cap(_ok(root, ["status", "--short"]), "stage fewer files at a time") or "nothing staged"


def git_commit(project, args, host):
    root = project["root"]
    _repo(root)
    message = args.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE or "\x00" in message:
        raise ToolFail(f"message must be text of 1 to {MAX_MESSAGE} characters")
    branch = _current(root)
    if branch is None or branch == "HEAD":
        raise ToolFail("HEAD is not on a branch: create one with git_branch_create first")
    if branch == project["defaultBranch"] and not project["allowMain"]:
        raise ToolFail(f"{branch} is the default branch: create a branch with git_branch_create and commit there")
    rc, _, _ = run_git(root, ["diff", "--cached", "--quiet"])
    if rc == 0:
        raise ToolFail("nothing is staged: stage files with git_add first")
    _ok(root, ["commit", "-m", message], identity=host.identity())
    return _ok(root, ["log", "-1", "--stat", "--format=%h %s"]).rstrip("\n")


def git_restore(project, args, host):
    root = project["root"]
    _repo(root)
    paths = _paths(root, args.get("paths"))
    cmd = ["restore"] + (["--staged"] if args.get("staged") is True else [])
    _ok(root, cmd + ["--"] + paths)
    return "restored " + ", ".join(paths)


def git_init(project, args, host):
    root = project["root"]
    if os.path.exists(os.path.join(root, ".git")):
        raise ToolFail("this project is already a git repository")
    _ok(root, ["init", "-b", project["defaultBranch"]])
    return f"initialised a repository on {project['defaultBranch']}"


# ---- definitions -------------------------------------------------------------------------------

def _def(name, description, props, required=(), read_only=False, destructive=False):
    ann = {"readOnlyHint": read_only}
    if destructive:
        ann["destructiveHint"] = True
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string", "description": "The project id; optional when there is only one."}, **props},
            "required": list(required)},
        "annotations": ann,
    }


PATHS = {"type": "array", "items": {"type": "string"}, "description": "Paths relative to the project."}

GIT_DEFS = [
    _def("git_status", "Show the branch and the changed files of the project.", {}, read_only=True),
    _def("git_diff", "Show the changes in the files (unstaged by default).",
         {"path": {"type": "string"}, "staged": {"type": "boolean"}, "stat": {"type": "boolean", "description": "Only a summary."}}, read_only=True),
    _def("git_log", "Show recent commits.", {"limit": {"type": "integer"}, "path": {"type": "string"}}, read_only=True),
    _def("git_show", "Show one commit: its message and files.",
         {"rev": {"type": "string", "description": "Default HEAD."}, "patch": {"type": "boolean"}}, read_only=True),
    _def("git_branches", "List the branches.", {}, read_only=True),
    _def("git_branch_create", "Create a branch named kevin/<name> and switch to it.",
         {"name": {"type": "string"}, "from": {"type": "string", "description": "Start point, default the current commit."}}, ["name"]),
    _def("git_switch", "Switch to an existing branch.", {"name": {"type": "string"}}, ["name"]),
    _def("git_add", "Stage files for the next commit, or everything with all.", {"paths": PATHS, "all": {"type": "boolean"}}),
    _def("git_commit", "Commit what is staged. Not allowed on the default branch.", {"message": {"type": "string"}}, ["message"]),
    _def("git_restore", "Throw away changes in files (or unstage them with staged). Cannot be undone.",
         {"paths": PATHS, "staged": {"type": "boolean"}}, ["paths"], destructive=True),
    _def("git_init", "Make the project a git repository (only when it is not one yet).", {}),
]

GIT_HANDLERS = {
    "git_status": git_status, "git_diff": git_diff, "git_log": git_log, "git_show": git_show, "git_branches": git_branches,
    "git_branch_create": git_branch_create, "git_switch": git_switch, "git_add": git_add, "git_commit": git_commit,
    "git_restore": git_restore, "git_init": git_init,
}
