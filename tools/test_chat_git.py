#!/usr/bin/env python3
"""Tests for tools/chat_git.py with real git on temp repositories: the tools, the branch rules, the jail on
paths, and the hardening (hooks, fsmonitor and an external diff in a repository's own config must not run).

    python3 -m unittest discover -s tools -p 'test_chat_git.py' -v
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import chat_git  # noqa: E402
import chat_projects as cp  # noqa: E402


def git(cwd, *args):
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(["git", "-c", "user.name=Setup", "-c", "user.email=setup@example.org", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, check=True).stdout


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = os.path.join(self.tmp, "base")
        self.root = os.path.realpath(os.path.join(self.base, "demo"))
        os.makedirs(self.root)
        self.registry = os.path.join(self.tmp, "projects.json")
        self.settings = {"root": "demo"}
        self.identity = {"name": "Kevin Test", "email": "kevin@example.org"}
        self.save()
        self.host = cp.ProjectHost(self.base, self.registry, min_free_gb=0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def save(self):
        with open(self.registry, "w") as f:
            json.dump({"identity": self.identity, "projects": {"demo": self.settings}}, f)

    def put(self, rel, text="x\n"):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    def repo(self, branch="main", commit=True):
        git(self.root, "init", "-b", branch)
        if commit:
            self.put("README.md", "hello\n")
            git(self.root, "add", ".")
            git(self.root, "commit", "-m", "first")

    def call(self, tool, /, **args):
        res = self.host.call(tool, args)
        return res["isError"], res["content"][0]["text"]

    def ok(self, tool, /, **args):
        err, text = self.call(tool, **args)
        self.assertFalse(err, text)
        return text

    def fails(self, tool, contains, /, **args):
        err, text = self.call(tool, **args)
        self.assertTrue(err, text)
        self.assertIn(contains, text)


class Reading(Base):
    def test_a_folder_that_is_not_a_repository_says_so_and_git_init_fixes_it(self):
        self.fails("git_status", "git_init")
        self.assertEqual(self.ok("git_init"), "initialised a repository on main")
        self.fails("git_init", "already")
        self.assertIn("## No commits yet on main", self.ok("git_status"))
        self.assertEqual(self.ok("git_log"), "no commits yet")

    def test_status_diff_log_show_and_branches_on_a_repository(self):
        self.repo()
        self.assertEqual(self.ok("git_status"), "## main\nno changes")
        self.put("README.md", "hello\nworld\n")
        self.put("new.txt", "n\n")
        status = self.ok("git_status")
        self.assertIn(" M README.md", status)
        self.assertIn("?? new.txt", status)
        self.assertIn("+world", self.ok("git_diff"))
        self.assertIn("README.md", self.ok("git_diff", stat=True))
        self.assertEqual(self.ok("git_diff", path="new.txt"), "no differences")
        self.assertEqual(self.ok("git_diff", staged=True), "no differences")
        self.assertRegex(self.ok("git_log"), r"^[0-9a-f]{7,} \d{4}-\d\d-\d\d Setup: first$")
        self.assertIn("first", self.ok("git_show"))
        self.assertIn("README.md", self.ok("git_show"))
        self.assertIn("+hello", self.ok("git_show", patch=True))
        self.assertRegex(self.ok("git_branches"), r"\* main\s+[0-9a-f]{7,} first")

    def test_a_long_diff_is_cut_with_a_hint(self):
        self.repo()
        self.put("big.txt", "".join(f"line {i}\n" for i in range(3000)))
        git(self.root, "add", "big.txt")
        out = self.ok("git_diff", staged=True)
        self.assertLess(len(out), chat_git.MAX_OUT + 200)
        self.assertIn("give a path, or stat", out)

    def test_refs_and_paths_that_look_like_options_or_leave_the_project_are_refused(self):
        self.repo()
        for rev in ["--output=/tmp/kevin-x", "-p", "a..b", "HEAD;ls", "", 5]:
            self.fails("git_show", "must be a branch, tag or commit", rev=rev)
        self.assertFalse(os.path.exists("/tmp/kevin-x"))
        self.fails("git_diff", ".. is not", path="../x")
        self.fails("git_diff", "relative", path="/etc/passwd")
        self.fails("git_log", ".git", path=".git/config")


class Branches(Base):
    def test_a_created_branch_is_prefixed_switched_to_and_listed(self):
        self.repo()
        self.assertEqual(self.ok("git_branch_create", name="fix-typo"), "created and switched to kevin/fix-typo")
        self.assertEqual(self.ok("git_branch_create", name="kevin/other", **{"from": "main"}), "created and switched to kevin/other")
        branches = self.ok("git_branches")
        self.assertIn("kevin/fix-typo", branches)
        self.assertRegex(branches, r"\* kevin/other")
        self.assertEqual(self.ok("git_switch", name="main"), "switched to main")
        self.assertIn("## main", self.ok("git_status"))

    def test_bad_branch_names_and_start_points_are_refused_before_git_runs(self):
        self.repo()
        for name in ["-D", "--force", "a..b", "x y", "a//b", "bad.lock", "end/", "end.", "a@{u}", "", None, "é", "x" * 100]:
            self.fails("git_branch_create", "branch name", name=name)
            self.fails("git_switch", "branch name", name=name)
        self.fails("git_branch_create", "must be a branch", name="ok", **{"from": "--detach"})
        self.ok("git_branch_create", name="twice")
        self.fails("git_branch_create", "already exists", name="twice")
        self.fails("git_switch", "invalid reference", name="nope")

    def test_switching_with_conflicting_changes_is_left_to_git_and_refused(self):
        self.repo()
        self.ok("git_branch_create", name="a")
        self.put("README.md", "on a\n")
        git(self.root, "commit", "-am", "on a")
        self.ok("git_switch", name="main")
        self.put("README.md", "dirty on main\n")
        err, text = self.call("git_switch", name="kevin/a")
        self.assertTrue(err, text)


class Committing(Base):
    def test_add_and_commit_on_a_branch_use_the_registry_identity(self):
        self.repo()
        self.ok("git_branch_create", name="feature")
        self.put("a.txt", "a\n")
        self.put("sub/b.txt", "b\n")
        self.assertIn("A  a.txt", self.ok("git_add", paths=["a.txt"]))
        self.assertIn("A  sub/b.txt", self.ok("git_add", paths=["sub"]))
        out = self.ok("git_commit", message="Add two files\n\nWith a body.")
        self.assertRegex(out, r"^[0-9a-f]{7,} Add two files")
        self.assertIn("a.txt", out)
        self.assertEqual(git(self.root, "log", "-1", "--format=%an <%ae> | %cn <%ce>").strip(),
                         "Kevin Test <kevin@example.org> | Kevin Test <kevin@example.org>")
        self.assertIn("With a body.", git(self.root, "log", "-1", "--format=%B"))
        self.assertEqual(self.ok("git_status"), "## kevin/feature\nno changes")

    def test_add_all_stages_everything_and_a_missing_paths_list_is_refused(self):
        self.repo()
        self.put("a.txt")
        self.put("b.txt")
        out = self.ok("git_add", all=True)
        self.assertIn("A  a.txt", out)
        self.assertIn("A  b.txt", out)
        self.fails("git_add", "list of paths")
        self.fails("git_add", "list of paths", paths=[])
        self.fails("git_add", "list of paths", paths="a.txt")
        self.fails("git_add", "at most", paths=["a.txt"] * (chat_git.MAX_PATHS + 1))

    def test_commits_on_the_default_branch_in_detached_head_or_with_nothing_staged_are_refused(self):
        self.repo()
        self.put("a.txt")
        self.ok("git_add", paths=["a.txt"])
        self.fails("git_commit", "default branch", message="direct")
        self.assertEqual(git(self.root, "log", "--oneline").count("\n"), 1)
        self.ok("git_branch_create", name="f")
        self.fails("git_commit", "message must be", message="   ")
        self.fails("git_commit", "message must be", message=None)
        self.fails("git_commit", "message must be", message="x" * (chat_git.MAX_MESSAGE + 1))
        self.ok("git_commit", message="ok")
        self.fails("git_commit", "nothing is staged", message="again")
        git(self.root, "checkout", "-q", "--detach")
        self.put("c.txt")
        self.ok("git_add", paths=["c.txt"])
        self.fails("git_commit", "not on a branch", message="detached")

    def test_allow_main_and_a_different_default_branch_follow_the_registry(self):
        self.settings = {"root": "demo", "allowMain": True}
        self.save()
        self.repo()
        self.put("a.txt")
        self.ok("git_add", paths=["a.txt"])
        self.ok("git_commit", message="on main, allowed")
        self.settings = {"root": "demo", "defaultBranch": "trunk"}
        self.save()
        git(self.root, "branch", "-m", "trunk")
        self.put("b.txt")
        self.ok("git_add", paths=["b.txt"])
        self.fails("git_commit", "trunk is the default branch", message="no")
        git(self.root, "branch", "-m", "main")
        self.ok("git_commit", message="main is not the default here")

    def test_restore_discards_changes_or_unstages_and_stays_in_the_jail(self):
        self.repo()
        self.put("README.md", "changed\n")
        self.assertEqual(self.ok("git_restore", paths=["README.md"]), "restored README.md")
        with open(os.path.join(self.root, "README.md")) as f:
            self.assertEqual(f.read(), "hello\n")
        self.put("a.txt")
        self.ok("git_add", paths=["a.txt"])
        self.ok("git_restore", paths=["a.txt"], staged=True)
        self.assertIn("?? a.txt", self.ok("git_status"))
        self.fails("git_restore", ".. is not", paths=["../x"])
        self.fails("git_restore", "list of paths", paths=[])
        self.fails("git_restore", "did not match", paths=["missing.txt"])

    def test_the_commit_message_cannot_become_an_option(self):
        self.repo()
        self.ok("git_branch_create", name="m")
        self.put("a.txt")
        self.ok("git_add", paths=["a.txt"])
        self.ok("git_commit", message="--amend --no-edit")
        self.assertIn("--amend --no-edit", git(self.root, "log", "-1", "--format=%s"))
        self.assertEqual(git(self.root, "log", "--oneline").count("\n"), 2)


class Hardening(Base):
    """A repository's own configuration and hooks must not be able to run code through the tools."""

    def marker(self, name):
        return os.path.join(self.tmp, name)

    def script(self, name):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            f.write(f"#!/bin/sh\ntouch '{self.marker(name + '.ran')}'\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        return path

    def test_hooks_do_not_run_on_commit(self):
        self.repo()
        for hook in ("pre-commit", "commit-msg", "post-commit", "prepare-commit-msg"):
            path = os.path.join(self.root, ".git", "hooks", hook)
            with open(path, "w") as f:
                f.write(f"#!/bin/sh\ntouch '{self.marker('hook-' + hook)}'\n")
            os.chmod(path, 0o755)
        self.ok("git_branch_create", name="h")
        self.put("a.txt")
        self.ok("git_add", paths=["a.txt"])
        self.ok("git_commit", message="hooked")
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith("hook-")], [])

    def test_fsmonitor_and_an_external_diff_in_the_repository_config_do_not_run(self):
        self.repo()
        fsm, ext = self.script("fsmonitor.sh"), self.script("extdiff.sh")
        git(self.root, "config", "core.fsmonitor", fsm)
        git(self.root, "config", "diff.external", ext)
        self.put("README.md", "changed\n")
        self.ok("git_status")
        self.ok("git_diff")
        self.ok("git_show", patch=True)
        self.assertFalse(os.path.exists(self.marker("fsmonitor.sh.ran")))
        self.assertFalse(os.path.exists(self.marker("extdiff.sh.ran")))

    def test_an_editor_pager_and_credential_helper_in_the_config_do_not_run(self):
        self.repo()
        for key in ("core.editor", "core.pager", "credential.helper"):
            git(self.root, "config", key, self.script(key.replace(".", "-") + ".sh"))
        self.ok("git_branch_create", name="e")
        self.put("a.txt")
        self.ok("git_add", paths=["a.txt"])
        self.ok("git_commit", message="x")
        self.ok("git_log")
        self.ok("git_show")
        self.assertEqual([n for n in os.listdir(self.tmp) if n.endswith(".ran")], [])

    def test_the_users_global_git_config_is_not_read(self):
        home = os.path.join(self.tmp, "home")
        os.makedirs(home)
        with open(os.path.join(home, ".gitconfig"), "w") as f:
            f.write("[user]\n\tname = Global Name\n\temail = global@example.org\n")
        old = os.environ.get("HOME")
        os.environ["HOME"] = home
        try:
            self.repo()
            self.ok("git_branch_create", name="g")
            self.put("a.txt")
            self.ok("git_add", paths=["a.txt"])
            self.ok("git_commit", message="who")
        finally:
            if old is None:
                del os.environ["HOME"]
            else:
                os.environ["HOME"] = old
        self.assertEqual(git(self.root, "log", "-1", "--format=%an").strip(), "Kevin Test")

    def test_the_environment_given_to_git_is_minimal_and_has_no_secrets(self):
        os.environ["KEVIN_TEST_SECRET"] = "hunter2"
        try:
            env = chat_git.git_env({"name": "N", "email": "e@x"})
        finally:
            del os.environ["KEVIN_TEST_SECRET"]
        self.assertNotIn("KEVIN_TEST_SECRET", env)
        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(env["GIT_CONFIG_GLOBAL"], os.devnull)
        self.assertEqual((env["GIT_AUTHOR_NAME"], env["GIT_COMMITTER_EMAIL"]), ("N", "e@x"))
        for flag in ("core.hooksPath=/dev/null", "core.fsmonitor=false", "diff.external=", "credential.helper=", "protocol.ext.allow=never"):
            self.assertIn(flag, chat_git.GIT_PREFIX)

    def test_a_missing_git_binary_or_a_slow_command_is_an_error_result(self):
        self.repo()
        old = os.environ["PATH"]
        os.environ["PATH"] = self.tmp                     # no git there
        try:
            self.fails("git_status", "not installed")
        finally:
            os.environ["PATH"] = old
        with self.assertRaises(chat_git.ToolFail):
            chat_git.run_git(self.root, ["status"], timeout=0.0001)


if __name__ == "__main__":
    unittest.main()
