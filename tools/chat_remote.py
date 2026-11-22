"""Remote git and pull requests for the project tools (tools/chat_projects.py): clone, fetch, pull, push, pr_create.

The owner decides where a project's code lives: the registry entry carries `remote` (an https URL) and optionally
`tokenEnv` (the name of an environment variable of this service that holds a GitHub token). The agent never names
a URL, a remote or a token:

- clone, fetch, pull and push talk only to the project's own `remote`, by URL, never to another one;
- push sends the current branch to the same branch name, never forced, never a deletion, never the default
  branch unless the registry allows it; pull only fast-forwards;
- the token reaches git only through a GIT_ASKPASS helper that reads it from the child's environment (it is
  not in the command line, the remote URL or the repository configuration) and is scrubbed from every output;
- the repository's own configuration cannot redirect the traffic: url rewrites make the call fail, and the
  proxy, header, TLS and helper settings are overridden on the command line;
- pr_create calls the GitHub REST API over https with the same token.

Python 3 standard library only.
"""

import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.error
import urllib.request

import chat_git
from chat_jail import ToolFail

NET_TIMEOUT = 120
API_TIMEOUT = 30
MAX_OUT = 6000
REMOTE_URL = re.compile(r"^https://[A-Za-z0-9][A-Za-z0-9.-]*/[A-Za-z0-9._~/-]+$")
GITHUB_REPO = re.compile(r"^https://github\.com/([A-Za-z0-9][A-Za-z0-9-]*)/([A-Za-z0-9._-]+?)(?:\.git)?/?$")
TOKEN_ENV = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
DEFAULT_TOKEN_ENV = "KEVIN_GITHUB_TOKEN"
API_BASE = "https://api.github.com"
# Tests only: let a local bare repository and a fake API server stand in for GitHub.
TEST_LOCAL = False

NET_FLAGS = [
    "-c", "http.proxy=", "-c", "http.sslVerify=true", "-c", "http.extraHeader=", "-c", "http.followRedirects=false",
    "-c", "core.gitProxy=", "-c", "core.sshCommand=false", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always",
    "-c", "transfer.fsckObjects=true",
]


def valid_remote(url):
    if TEST_LOCAL and isinstance(url, str) and url.startswith("file:///"):
        return True
    return isinstance(url, str) and len(url) <= 200 and bool(REMOTE_URL.match(url)) and ".." not in url


def valid_token_env(name):
    return isinstance(name, str) and bool(TOKEN_ENV.match(name))


def _remote(project):
    url = project.get("remote")
    if not url:
        raise ToolFail("the owner has not set a remote for this project in the registry")
    return url


def _scrub(text, token):
    return text.replace(token, "***") if token else text


def _token(project, required):
    name = project.get("tokenEnv") or DEFAULT_TOKEN_ENV
    token = os.environ.get(name, "").strip()
    if required and not token:
        raise ToolFail(f"no token: the service has no {name} in its environment (the owner sets it, never the agent)")
    return token


def _guard_config(root):
    """Refuses to go online when the repository rewrites URLs: that could send the token somewhere else."""
    rc, out, _ = chat_git.run_git(root, ["config", "--local", "--name-only", "--get-regexp", r"^url\."])
    if rc == 0 and out.strip():
        raise ToolFail("the repository configuration rewrites URLs (url.*): the owner must remove that before going online")


def run_net(root, args, token, identity=None):
    """git with the network on: (exit code, output), the token scrubbed. The token goes only through the askpass helper."""
    helper_dir = tempfile.mkdtemp(prefix="kevin-askpass-")
    try:
        os.chmod(helper_dir, 0o700)
        helper = os.path.join(helper_dir, "askpass.sh")
        with open(helper, "w", encoding="utf-8") as f:
            f.write('#!/bin/sh\ncase "$1" in Username*) echo x-access-token;; *) printf %s "$KEVIN_GIT_TOKEN";; esac\n')
        os.chmod(helper, stat.S_IRWXU)
        env = chat_git.git_env(identity)
        env.update({"GIT_ASKPASS": helper, "KEVIN_GIT_TOKEN": token, "GIT_TERMINAL_PROMPT": "0", "HOME": helper_dir})
        prefix = list(chat_git.GIT_PREFIX)
        if TEST_LOCAL:
            prefix += ["-c", "protocol.file.allow=always"]
            flags = []
        else:
            flags = NET_FLAGS
        try:
            p = subprocess.run(prefix + flags + list(args), cwd=root, env=env, capture_output=True, timeout=NET_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise ToolFail(f"git took longer than {NET_TIMEOUT} seconds") from None
        except FileNotFoundError:
            raise ToolFail("git is not installed on this host") from None
        out = _scrub((p.stdout + p.stderr).decode("utf-8", "replace"), token).strip()
        return p.returncode, out[-MAX_OUT:]
    finally:
        shutil.rmtree(helper_dir, ignore_errors=True)


def _net_ok(root, args, token, what, identity=None):
    rc, out = run_net(root, args, token, identity)
    if rc != 0:
        raise ToolFail((out or f"git {what} failed")[:800])
    return out


def _branch_now(root):
    branch = chat_git._current(root)
    if branch is None or branch == "HEAD":
        raise ToolFail("HEAD is not on a branch")
    return branch


# ---- tools -------------------------------------------------------------------------------------

def git_clone(project, args, host):
    root = project["root"]
    url = _remote(project)
    if os.path.exists(os.path.join(root, ".git")):
        raise ToolFail("this project is already a git repository")
    if os.listdir(root):
        raise ToolFail("the project folder is not empty: clone needs an empty folder")
    out = _net_ok(root, ["clone", "--no-tags", "--", url, "."], _token(project, False), "clone")
    return out or f"cloned {url}"


def git_fetch(project, args, host):
    root = project["root"]
    chat_git._repo(root)
    _guard_config(root)
    out = _net_ok(root, ["fetch", "--no-tags", "--prune", "--no-recurse-submodules", "--", _remote(project),
                         "+refs/heads/*:refs/remotes/origin/*"], _token(project, False), "fetch")
    return out or "up to date"


def git_pull(project, args, host):
    root = project["root"]
    chat_git._repo(root)
    _guard_config(root)
    branch = _branch_now(root)
    if chat_git.run_git(root, ["status", "--porcelain=v1", "--untracked-files=no"])[1].strip():
        raise ToolFail("there are uncommitted changes: commit or restore them before pulling")
    out = _net_ok(root, ["pull", "--ff-only", "--no-rebase", "--no-recurse-submodules", "--no-tags", "--", _remote(project),
                         f"refs/heads/{branch}"], _token(project, False), "pull")
    return out or "already up to date"


def git_push(project, args, host):
    root = project["root"]
    chat_git._repo(root)
    _guard_config(root)
    branch = _branch_now(root)
    if branch == project["defaultBranch"] and not project["allowMain"]:
        raise ToolFail(f"{branch} is the default branch: push a branch made with git_branch_create instead")
    out = _net_ok(root, ["push", "--no-verify", "--no-force-with-lease", "--no-recurse-submodules", "--", _remote(project),
                         f"refs/heads/{branch}:refs/heads/{branch}"], _token(project, True), "push")
    return out or f"pushed {branch}"


def _api(method, path, token, payload=None):
    url = API_BASE + path
    if not TEST_LOCAL and not url.startswith("https://"):
        raise ToolFail("the API address must be https")
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "kevin-ai",
        "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as r:        # noqa: S310 (https checked above)
            return r.status, json.loads(r.read(1_000_000) or b"{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read(100_000) or b"{}")
        except ValueError:
            body = {}
        return e.code, body
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise ToolFail(_scrub(f"cannot reach the GitHub API: {e}", token)) from None


def pr_create(project, args, host):
    root = project["root"]
    chat_git._repo(root)
    m = GITHUB_REPO.match(_remote(project))
    if not m:
        raise ToolFail("pull requests need a github.com remote in the registry")
    owner, repo = m.group(1), m.group(2)
    title, body = args.get("title"), args.get("body", "")
    if not isinstance(title, str) or not title.strip() or len(title) > 200 or "\x00" in title:
        raise ToolFail("title must be text of 1 to 200 characters")
    if not isinstance(body, str) or len(body) > 20000 or "\x00" in body:
        raise ToolFail("body must be text of at most 20000 characters")
    head = _branch_now(root)
    if head == project["defaultBranch"]:
        raise ToolFail(f"{head} is the default branch: open a pull request from a branch made with git_branch_create")
    token = _token(project, True)
    status, doc = _api("POST", f"/repos/{owner}/{repo}/pulls", token,
                       {"title": title.strip(), "body": body, "head": head, "base": project["defaultBranch"],
                        "draft": args.get("draft") is True})
    if status == 201 and isinstance(doc.get("html_url"), str):
        return f"opened pull request #{doc.get('number')}: {doc['html_url']}"
    message = _scrub(str(doc.get("message") or "no reason given"), token)[:300]
    errors = "; ".join(str(e.get("message", e))[:150] for e in doc.get("errors", []) if isinstance(e, dict))[:300]
    raise ToolFail(f"GitHub answered {status}: {message}" + (f" ({errors})" if errors else "") + " (is the branch pushed?)")


# ---- definitions -------------------------------------------------------------------------------

def _def(name, description, props=None, required=()):
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string", "description": "The project id; optional when there is only one."}, **(props or {})},
            "required": list(required)},
        "annotations": {"readOnlyHint": False, "openWorldHint": True},
    }


REMOTE_DEFS = [
    _def("git_clone", "Download the project's repository from its remote into the empty project folder."),
    _def("git_fetch", "Download new commits of the project's remote without changing your files."),
    _def("git_pull", "Bring the current branch up to date with its remote (fast-forward only, no uncommitted changes)."),
    _def("git_push", "Upload the current branch to the project's remote. Never forced; not the default branch."),
    _def("pr_create", "Open a GitHub pull request from the current (pushed) branch into the default branch.",
         {"title": {"type": "string"}, "body": {"type": "string"}, "draft": {"type": "boolean"}}, ["title"]),
]

REMOTE_HANDLERS = {"git_clone": git_clone, "git_fetch": git_fetch, "git_pull": git_pull, "git_push": git_push, "pr_create": pr_create}
