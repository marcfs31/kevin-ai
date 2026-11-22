#!/usr/bin/env python3
"""Tests for tools/chat_remote.py: clone, fetch, pull, push and pull requests, with a local bare repository
standing in for the remote (file protocol, enabled only by the test flag) and a fake GitHub API server.

    python3 -m unittest discover -s tools -p 'test_chat_remote.py' -v
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import chat_projects as cp  # noqa: E402
import chat_remote  # noqa: E402

TOKEN = "ghp_SECRET_TOKEN_1234567890"


def git(cwd, *args):
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(["git", "-c", "user.name=Setup", "-c", "user.email=setup@example.org", "-c", "protocol.file.allow=always",
                           *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout


class FakeGitHub(BaseHTTPRequestHandler):
    seen = []
    status = 201

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeGitHub.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if FakeGitHub.status == 201:
            doc = {"number": 7, "html_url": "https://github.com/acme/demo/pull/7"}
        else:
            doc = {"message": "Validation Failed", "errors": [{"message": "No commits between main and kevin/x"}]}
        data = json.dumps(doc).encode()
        self.send_response(FakeGitHub.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class Base(unittest.TestCase):
    def setUp(self):
        chat_remote.TEST_LOCAL = True
        self.tmp = tempfile.mkdtemp()
        self.origin = os.path.join(self.tmp, "origin.git")
        seed = os.path.join(self.tmp, "seed")
        os.makedirs(seed)
        git(seed, "init", "-b", "main")
        with open(os.path.join(seed, "README.md"), "w") as f:
            f.write("hello\n")
        git(seed, "add", ".")
        git(seed, "commit", "-m", "first")
        git(self.tmp, "clone", "--bare", seed, self.origin)
        self.seed = seed
        self.base = os.path.join(self.tmp, "base")
        self.root = os.path.realpath(os.path.join(self.base, "demo"))
        os.makedirs(self.root)
        self.registry = os.path.join(self.tmp, "projects.json")
        self.settings = {"root": "demo", "remote": "file://" + self.origin}
        self.save()
        self.host = cp.ProjectHost(self.base, self.registry, min_free_gb=0)
        self.env = mock.patch.dict(os.environ, {"KEVIN_GITHUB_TOKEN": TOKEN})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        chat_remote.TEST_LOCAL = False
        shutil.rmtree(self.tmp, ignore_errors=True)

    def save(self):
        with open(self.registry, "w") as f:
            json.dump({"projects": {"demo": self.settings}}, f)

    def call(self, tool, /, **args):
        res = self.host.call(tool, args)
        return res["isError"], res["content"][0]["text"]

    def ok(self, tool, /, **args):
        err, text = self.call(tool, **args)
        self.assertFalse(err, text)
        return text

    def fails(self, tool, /, **args):
        err, text = self.call(tool, **args)
        self.assertTrue(err, text)
        return text

    def cloned(self):
        self.ok("git_clone")
        self.ok("git_branch_create", name="work")

    def commit_here(self, name="a.txt"):
        with open(os.path.join(self.root, name), "w") as f:
            f.write("x\n")
        self.ok("git_add", paths=[name])
        self.ok("git_commit", message=f"add {name}")

    def remote_branches(self):
        return subprocess.run(["git", "--git-dir", self.origin, "branch", "--list"], capture_output=True, text=True).stdout


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class RemoteTests(Base):
    # ---- clone / fetch / pull ----

    def test_clone_into_empty_folder(self):
        self.ok("git_clone")
        self.assertTrue(os.path.exists(os.path.join(self.root, "README.md")))
        self.assertIn("main", self.ok("git_status"))

    def test_clone_refuses_a_folder_with_files_or_a_repository(self):
        with open(os.path.join(self.root, "x.txt"), "w") as f:
            f.write("x")
        self.assertIn("not empty", self.fails("git_clone"))
        os.remove(os.path.join(self.root, "x.txt"))
        self.ok("git_clone")
        self.assertIn("already a git repository", self.fails("git_clone"))

    def test_clone_needs_a_remote_in_the_registry(self):
        self.settings.pop("remote")
        self.save()
        self.assertIn("remote", self.fails("git_clone"))

    def test_fetch_and_pull_fast_forward(self):
        self.ok("git_clone")
        with open(os.path.join(self.seed, "new.txt"), "w") as f:
            f.write("n\n")
        git(self.seed, "add", ".")
        git(self.seed, "commit", "-m", "second")
        git(self.seed, "push", self.origin, "main")
        self.ok("git_fetch")
        self.ok("git_pull")
        self.assertTrue(os.path.exists(os.path.join(self.root, "new.txt")))

    def test_pull_refuses_uncommitted_changes(self):
        self.ok("git_clone")
        with open(os.path.join(self.root, "README.md"), "a") as f:
            f.write("local edit\n")
        self.assertIn("uncommitted", self.fails("git_pull"))

    def test_pull_never_makes_a_merge_commit(self):
        self.ok("git_clone")
        self.settings["allowMain"] = True
        self.save()
        self.commit_here("local.txt")
        with open(os.path.join(self.seed, "far.txt"), "w") as f:
            f.write("f\n")
        git(self.seed, "add", ".")
        git(self.seed, "commit", "-m", "diverge")
        git(self.seed, "push", self.origin, "main")
        self.fails("git_pull")
        self.assertEqual(git(self.root, "rev-list", "--merges", "--count", "HEAD").strip(), "0")

    # ---- push ----

    def test_push_sends_the_branch_and_nothing_else(self):
        self.cloned()
        self.commit_here()
        self.ok("git_push")
        self.assertIn("kevin/work", self.remote_branches())

    def test_push_refuses_the_default_branch(self):
        self.ok("git_clone")
        self.assertIn("default branch", self.fails("git_push"))
        self.assertNotIn("kevin", self.remote_branches())

    def test_push_needs_a_token(self):
        self.cloned()
        self.commit_here()
        os.environ.pop("KEVIN_GITHUB_TOKEN", None)
        self.assertIn("no token", self.fails("git_push"))

    def test_push_is_never_forced_and_takes_no_target(self):
        for tool in chat_remote.REMOTE_DEFS:
            props = tool["inputSchema"]["properties"]
            for bad in ("force", "url", "remote", "branch", "delete", "token"):
                self.assertNotIn(bad, props, tool["name"])
        self.cloned()
        self.commit_here()
        self.ok("git_push")
        git(self.root, "reset", "--hard", "HEAD~1")
        self.commit_here("other.txt")
        self.fails("git_push")            # a rewritten branch is rejected as non-fast-forward

    def test_push_uses_the_token_only_through_the_environment(self):
        self.cloned()
        self.commit_here()
        seen = {}
        real = subprocess.run

        def spy(cmd, **kw):
            if "push" in cmd:
                seen["cmd"], seen["env"] = cmd, kw["env"]
            return real(cmd, **kw)

        with mock.patch.object(chat_remote.subprocess, "run", spy):
            self.ok("git_push")
        self.assertNotIn(TOKEN, " ".join(seen["cmd"]))
        self.assertEqual(seen["env"]["KEVIN_GIT_TOKEN"], TOKEN)
        self.assertIn("GIT_ASKPASS", seen["env"])
        with open(os.path.join(self.root, ".git", "config")) as f:
            self.assertNotIn(TOKEN, f.read())

    def test_the_token_is_scrubbed_from_output(self):
        self.assertEqual(chat_remote._scrub(f"fatal: bad {TOKEN} here", TOKEN), "fatal: bad *** here")
        with mock.patch.object(chat_remote.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=1, stdout=b"", stderr=f"remote: {TOKEN}".encode())
            rc, out = chat_remote.run_net(self.root, ["status"], TOKEN)
        self.assertNotIn(TOKEN, out)

    def test_askpass_helper_is_removed_afterwards(self):
        paths = []
        real = subprocess.run

        def spy(cmd, **kw):
            if "GIT_ASKPASS" in kw["env"]:
                paths.append(kw["env"]["GIT_ASKPASS"])
            return real(cmd, **kw)

        self.ok("git_clone")
        with mock.patch.object(chat_remote.subprocess, "run", spy):
            self.ok("git_fetch")
        self.assertTrue(paths)
        self.assertFalse(os.path.exists(paths[0]))

    def test_a_url_rewrite_in_the_repository_stops_the_call(self):
        self.ok("git_clone")
        git(self.root, "config", "url.https://evil.example/.insteadOf", "https://github.com/")
        for tool in ("git_fetch", "git_pull", "git_push"):
            self.assertIn("rewrites URLs", self.fails(tool), tool)

    def test_outside_tests_only_https_is_allowed_whatever_the_repository_says(self):
        self.ok("git_clone")
        git(self.root, "config", "protocol.file.allow", "always")
        git(self.root, "config", "http.proxy", "http://127.0.0.1:9")
        chat_remote.TEST_LOCAL = False
        rc, out = chat_remote.run_net(self.root, ["ls-remote", "--", "file://" + self.origin], "")
        self.assertNotEqual(rc, 0)
        self.assertIn("not allowed", out)

    # ---- registry ----

    def test_registry_accepts_only_plain_https_remotes_and_env_names(self):
        chat_remote.TEST_LOCAL = False
        for good in ("https://github.com/acme/demo.git", "https://git.example.org/team/app"):
            self.assertTrue(chat_remote.valid_remote(good), good)
        for bad in ("http://github.com/a/b", "git@github.com:a/b.git", "ssh://github.com/a/b", "file:///tmp/x",
                    "https://user:pw@github.com/a/b", "https://github.com/a/b?x=1", "https://github.com/../b", "https://github.com", ""):
            self.assertFalse(chat_remote.valid_remote(bad), bad)
        self.assertTrue(chat_remote.valid_token_env("KEVIN_GITHUB_TOKEN"))
        for bad in ("lower", "A B", "1X", "", "X;rm"):
            self.assertFalse(chat_remote.valid_token_env(bad), bad)

    def test_a_bad_remote_in_the_registry_is_reported(self):
        chat_remote.TEST_LOCAL = False
        self.settings["remote"] = "http://github.com/a/b"
        self.save()
        self.assertIn("remote must be an https URL", self.fails("git_fetch"))

    def test_the_remote_tools_ask_and_are_recorded(self):
        defs = {t["name"]: t for t in chat_remote.REMOTE_DEFS}
        self.assertEqual(set(defs), {"git_clone", "git_fetch", "git_pull", "git_push", "pr_create"})
        for t in defs.values():
            self.assertFalse(t["annotations"]["readOnlyHint"])
            self.assertIn(t["name"], cp.CHANGING)


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class PullRequestTests(Base):
    def setUp(self):
        super().setUp()
        self.server = HTTPServer(("127.0.0.1", 0), FakeGitHub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.old_base = chat_remote.API_BASE
        chat_remote.API_BASE = f"http://127.0.0.1:{self.server.server_port}"
        FakeGitHub.seen, FakeGitHub.status = [], 201
        self.cloned()
        self.settings["remote"] = "https://github.com/acme/demo.git"      # for the API path; git is not used here
        self.save()

    def tearDown(self):
        chat_remote.API_BASE = self.old_base
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def test_opens_a_pull_request_from_the_current_branch(self):
        text = self.ok("pr_create", title="Add a", body="why")
        self.assertIn("https://github.com/acme/demo/pull/7", text)
        sent = FakeGitHub.seen[0]
        self.assertEqual(sent["path"], "/repos/acme/demo/pulls")
        self.assertEqual(sent["auth"], f"Bearer {TOKEN}")
        self.assertEqual((sent["body"]["head"], sent["body"]["base"], sent["body"]["title"]), ("kevin/work", "main", "Add a"))
        self.assertNotIn(TOKEN, text)

    def test_refuses_a_bad_title_a_missing_token_and_the_default_branch(self):
        self.assertIn("title", self.fails("pr_create", title=" "))
        os.environ.pop("KEVIN_GITHUB_TOKEN", None)
        self.assertIn("no token", self.fails("pr_create", title="x"))
        os.environ["KEVIN_GITHUB_TOKEN"] = TOKEN
        git(self.root, "switch", "main")
        self.assertIn("default branch", self.fails("pr_create", title="x"))
        self.assertEqual(FakeGitHub.seen, [])

    def test_reports_what_github_said(self):
        FakeGitHub.status = 422
        text = self.fails("pr_create", title="x")
        self.assertIn("422", text)
        self.assertIn("No commits between", text)
        self.assertNotIn(TOKEN, text)

    def test_needs_a_github_remote(self):
        self.settings["remote"] = "https://gitlab.com/acme/demo.git"
        self.save()
        self.assertIn("github.com", self.fails("pr_create", title="x"))


if __name__ == "__main__":
    unittest.main()
