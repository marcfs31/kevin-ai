#!/usr/bin/env python3
"""Tests for tools/chat_commands.py and tools/chat_audit.py through the project host: the allowlist, one tool per
command, running without a shell, the time limit, the environment, the output tail and the audit log.

    python3 -m unittest discover -s tools -p 'test_chat_commands.py' -v
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import chat_audit  # noqa: E402
import chat_commands as cc  # noqa: E402
import chat_projects as cp  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = os.path.join(self.tmp, "base")
        self.root = os.path.realpath(os.path.join(self.base, "demo"))
        os.makedirs(os.path.join(self.root, "sub"))
        self.registry = os.path.join(self.tmp, "projects.json")
        self.audit_path = os.path.join(self.tmp, "state", "audit.jsonl")
        self.commands = {"hello": {"argv": ["sh", "-c", "echo hello; echo oops >&2"]}}
        self.save()
        self.host = cp.ProjectHost(self.base, self.registry, min_free_gb=0, audit=chat_audit.Audit(self.audit_path))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def save(self, projects=None):
        with open(self.registry, "w") as f:
            json.dump({"projects": projects or {"demo": {"root": "demo", "commands": self.commands}}}, f)

    def call(self, tool, /, **args):
        res = self.host.call(tool, args)
        return res["isError"], res["content"][0]["text"]

    def entries(self):
        if not os.path.exists(self.audit_path):
            return []
        with open(self.audit_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]


class Parsing(Base):
    def test_a_valid_command_is_filled_in_with_defaults(self):
        cmds = cc.parse_commands("demo", {"test": {"argv": ["make", "test"]}})
        self.assertEqual(cmds["test"], {"name": "test", "argv": ["make", "test"], "timeout": 120, "safe": False, "cwd": None, "env": {}})
        self.assertEqual(cc.parse_commands("demo", None), {})

    def test_every_malformed_command_is_refused_with_the_reason(self):
        ok = {"argv": ["true"]}
        for raw, text in [
            ([], "must be an object"), ({"Bad Name": ok}, "bad command name"), ({"x" * 17: ok}, "bad command name"),
            ({"a": "make"}, "must be an object"), ({"a": {}}, "argv must be"), ({"a": {"argv": []}}, "argv must be"),
            ({"a": {"argv": "make test"}}, "argv must be"), ({"a": {"argv": ["ok", ""]}}, "argv must be"),
            ({"a": {"argv": ["ok", 5]}}, "argv must be"), ({"a": {"argv": ["a\x00b"]}}, "argv must be"),
            ({"a": {"argv": ["x"] * 33}}, "argv must be"),
            ({"a": {**ok, "timeout": 0}}, "timeout"), ({"a": {**ok, "timeout": 901}}, "timeout"), ({"a": {**ok, "timeout": True}}, "timeout"),
            ({"a": {**ok, "safe": "yes"}}, "safe must"),
            ({"a": {**ok, "cwd": "/etc"}}, "cwd must"), ({"a": {**ok, "cwd": "../x"}}, "cwd must"), ({"a": {**ok, "cwd": ""}}, "cwd must"),
            ({"a": {**ok, "env": {"PATH": "/x"}}}, "not allowed"), ({"a": {**ok, "env": {"HOME": "/x"}}}, "not allowed"),
            ({"a": {**ok, "env": {"LD_PRELOAD": "/x"}}}, "not allowed"), ({"a": {**ok, "env": {"GIT_SSH_COMMAND": "x"}}}, "not allowed"),
            ({"a": {**ok, "env": {"lower": "x"}}}, "not allowed"), ({"a": {**ok, "env": {"A": 1}}}, "not allowed"),
            ({"a": {**ok, "env": {f"V{i}": "x" for i in range(21)}}}, "at most 20"),
        ]:
            with self.assertRaises(cc.CommandError) as e:
                cc.parse_commands("demo", raw)
            self.assertIn(text, str(e.exception), raw)

    def test_a_bad_command_makes_the_whole_registry_unusable_with_the_reason(self):
        self.save({"demo": {"root": "demo", "commands": {"x": {"argv": []}}}})
        with self.assertRaises(cp.RegistryError) as e:
            cp.load_registry(self.registry, self.base)
        self.assertIn("argv must be", str(e.exception))


class Tools(Base):
    def test_each_command_is_a_tool_that_says_what_it_runs_and_only_a_safe_one_is_read_only(self):
        self.commands = {"test": {"argv": ["make", "test"], "timeout": 60, "safe": True},
                         "build": {"argv": ["npm", "run", "build it"], "cwd": "sub"}}
        os.makedirs(os.path.join(self.base, "demo2"))
        self.save({"demo": {"root": "demo", "commands": self.commands}, "other": {"root": "demo2"}})
        tools = {t["name"]: t for t in self.host.command_tools()}
        self.assertEqual(sorted(tools), ["run_demo_build", "run_demo_test"])
        self.assertEqual(tools["run_demo_test"]["description"], "Run `make test` in the project demo; stops after 60 s.")
        self.assertEqual(tools["run_demo_build"]["description"], "Run `npm run 'build it'` in sub in the project demo; stops after 120 s.")
        self.assertTrue(tools["run_demo_test"]["annotations"]["readOnlyHint"])
        self.assertFalse(tools["run_demo_build"]["annotations"]["readOnlyHint"])
        for t in tools.values():
            self.assertEqual(t["inputSchema"], {"type": "object", "properties": {}, "required": []})

    def test_the_client_lists_the_file_git_and_command_tools_together(self):
        client = cp.NativeProjectClient("project", {}, {}, self.host)
        names = [t["name"] for t in client.request("tools/list")["tools"]]
        self.assertIn("fs_read", names)
        self.assertIn("git_commit", names)
        self.assertEqual(names[-1], "run_demo_hello")
        self.save({"x": {"root": "../outside"}})                          # a bad registry still lists the fixed tools
        self.assertNotIn("run_demo_hello", [t["name"] for t in client.request("tools/list")["tools"]])

    def test_the_kept_tool_list_version_follows_the_commands(self):
        before = self.host.tools_version()
        self.assertEqual(self.host.tools_version(), before)
        self.commands["hello"]["argv"] = ["echo", "other"]
        self.save()
        self.assertNotEqual(self.host.tools_version(), before)
        self.save({"x": {"root": "../outside"}})
        self.assertTrue(self.host.tools_version().endswith("-bad"))


class Running(Base):
    def test_a_command_reports_its_exit_code_and_both_output_streams(self):
        err, text = self.call("run_demo_hello")
        self.assertFalse(err)
        self.assertEqual(text.splitlines()[0], "exit code 0")
        self.assertIn("hello", text)
        self.assertIn("oops", text)

    def test_a_failing_command_is_an_error_result_with_its_output(self):
        self.commands = {"fail": {"argv": ["sh", "-c", "echo bad news; exit 3"]}}
        self.save()
        err, text = self.call("run_demo_fail")
        self.assertTrue(err)
        self.assertEqual(text, "exit code 3\nbad news")

    def test_there_is_no_shell_and_the_agent_cannot_add_arguments(self):
        marker = os.path.join(self.tmp, "injected")
        self.commands = {"echo": {"argv": ["echo", "a; touch " + marker, "$(touch " + marker + ")", "`touch " + marker + "`"]}}
        self.save()
        err, text = self.call("run_demo_echo", extra="; touch " + marker, argv=["touch", marker])
        self.assertFalse(err)
        self.assertIn("a; touch", text)
        self.assertFalse(os.path.exists(marker))

    def test_it_runs_in_the_project_or_a_folder_inside_it_and_never_outside(self):
        self.commands = {"here": {"argv": ["pwd"]}, "there": {"argv": ["pwd"], "cwd": "sub"}, "gone": {"argv": ["pwd"], "cwd": "nope"}}
        self.save()
        self.assertEqual(self.call("run_demo_here")[1].splitlines()[1], self.root)
        self.assertEqual(self.call("run_demo_there")[1].splitlines()[1], os.path.join(self.root, "sub"))
        err, text = self.call("run_demo_gone")
        self.assertTrue(err)
        self.assertIn("does not exist", text)
        os.symlink(self.tmp, os.path.join(self.root, "link"))
        self.commands = {"out": {"argv": ["pwd"], "cwd": "link"}}
        self.save()
        err, text = self.call("run_demo_out")
        self.assertTrue(err)
        self.assertIn("leaves the project", text)

    def test_the_environment_is_minimal_with_a_private_home_that_is_removed(self):
        os.environ["KEVIN_TEST_SECRET"] = "hunter2"
        self.commands = {"env": {"argv": ["sh", "-c", "env | sort; echo home=$HOME; test -d \"$HOME\" && echo home-exists"], "env": {"MODE": "ci"}}}
        self.save()
        try:
            err, text = self.call("run_demo_env")
        finally:
            del os.environ["KEVIN_TEST_SECRET"]
        self.assertFalse(err)
        self.assertNotIn("KEVIN_TEST_SECRET", text)
        self.assertNotIn("hunter2", text)
        self.assertIn("MODE=ci", text)
        self.assertIn("CI=1", text)
        self.assertIn("home-exists", text)
        home = [line for line in text.splitlines() if line.startswith("home=")][0][5:]
        self.assertIn("kevin-run-", home)
        self.assertFalse(os.path.exists(home))

    def test_there_is_no_stdin(self):
        self.commands = {"read": {"argv": ["sh", "-c", "cat; echo done"]}}
        self.save()
        self.assertEqual(self.call("run_demo_read")[1], "exit code 0\ndone")

    def test_a_command_that_runs_too_long_is_killed_with_everything_it_started(self):
        marker = os.path.join(self.tmp, "child-survived")
        self.commands = {"slow": {"argv": ["sh", "-c", f"(sleep 3; touch {marker}) & echo started; sleep 30"], "timeout": 1}}
        self.save()
        start = time.monotonic()
        err, text = self.call("run_demo_slow")
        self.assertLess(time.monotonic() - start, 10)
        self.assertTrue(err)                        # it did not finish: the model is told it failed
        self.assertEqual(text, "timed out after 1 s and was stopped\nstarted")
        time.sleep(3.5)
        self.assertFalse(os.path.exists(marker), "a child of the command was left running")

    def test_only_the_tail_of_a_long_output_is_kept(self):
        self.commands = {"loud": {"argv": ["sh", "-c", "i=0; while [ $i -lt 4000 ]; do echo line-$i-xxxxxxxxxxxxxxxxxxxxxxxx; i=$((i+1)); done"]}}
        self.save()
        err, text = self.call("run_demo_loud")
        self.assertFalse(err)
        self.assertLess(len(text), cc.MAX_OUT + 200)
        self.assertIn("output cut to the last", text)
        self.assertTrue(text.rstrip().endswith("line-3999-xxxxxxxxxxxxxxxxxxxxxxxx"))
        self.assertNotIn("line-0-", text)

    def test_a_missing_program_a_missing_permission_and_an_unknown_tool_are_clear(self):
        self.commands = {"nope": {"argv": ["definitely-not-installed-xyz"]}, "noexec": {"argv": [os.path.join(self.root, "sub")]}}
        self.save()
        self.assertEqual(self.call("run_demo_nope"), (True, "error: command not found: definitely-not-installed-xyz"))
        self.assertTrue(self.call("run_demo_noexec")[0])
        self.assertEqual(self.call("run_demo_missing"), (True, "error: no tool named 'run_demo_missing'"))
        self.assertTrue(self.call("run_other_nope")[0])

    def test_two_projects_have_their_own_commands(self):
        os.makedirs(os.path.join(self.base, "two"))
        self.save({"demo": {"root": "demo", "commands": {"id": {"argv": ["echo", "one"]}}},
                   "two": {"root": "two", "commands": {"id": {"argv": ["echo", "two"]}}}})
        self.assertEqual(self.call("run_demo_id")[1], "exit code 0\none")
        self.assertEqual(self.call("run_two_id")[1], "exit code 0\ntwo")


class Auditing(Base):
    def test_changing_calls_are_recorded_and_reads_are_not(self):
        self.call("fs_write", path="a.txt", content="secret contents here\n")
        self.call("fs_read", path="a.txt")
        self.call("fs_list")
        self.call("fs_edit", path="a.txt", old="secret", new="public")
        self.call("run_demo_hello")
        e = self.entries()
        self.assertEqual([x["tool"] for x in e], ["fs_write", "fs_edit", "run_demo_hello"])
        self.assertEqual(e[0]["summary"], "a.txt (21 bytes)")
        self.assertEqual((e[0]["project"], e[0]["ok"]), ("demo", True))
        self.assertRegex(e[0]["time"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertIn("exit code 0", e[2]["result"])

    def test_contents_and_messages_beyond_the_first_line_are_never_logged(self):
        self.call("fs_write", path="a.txt", content="top secret token abc123\n")
        self.call("fs_edit", path="a.txt", old="abc123", new="zzz")
        with open(self.audit_path, encoding="utf-8") as f:
            raw = f.read()
        self.assertNotIn("abc123", raw)
        self.assertNotIn("top secret", raw)
        self.assertEqual(chat_audit.summarize("git_commit", {"message": "Title\n\nbody with detail"}), "Title")
        self.assertEqual(chat_audit.summarize("git_add", {"all": True}), "all")
        self.assertEqual(chat_audit.summarize("git_add", {"paths": ["a", "b"]}), "a, b")
        self.assertEqual(chat_audit.summarize("git_switch", {"name": "kevin/x"}), "kevin/x")
        self.assertEqual(chat_audit.summarize("fs_read", {"path": "x"}), "")

    def test_a_refused_change_is_recorded_as_failed(self):
        self.call("fs_write", path="../x", content="y")
        self.call("run_demo_nothere")
        e = self.entries()
        self.assertEqual([(x["tool"], x["ok"]) for x in e], [("fs_write", False), ("run_demo_nothere", False)])
        self.assertIn(".. is not allowed", e[0]["result"])

    def test_the_log_is_private_rotates_and_cannot_stop_the_work(self):
        self.call("fs_write", path="a.txt", content="x")
        self.assertEqual(os.stat(self.audit_path).st_mode & 0o777, 0o600)
        small = chat_audit.Audit(os.path.join(self.tmp, "small.jsonl"), max_bytes=300)
        for i in range(20):
            small.log("demo", "fs_write", f"file{i}", True, "created")
        self.assertTrue(os.path.exists(small.path + ".1"))
        self.assertLessEqual(os.path.getsize(small.path), 300)
        blocked = chat_audit.Audit(os.path.join(self.audit_path, "inside-a-file.jsonl"))       # the parent is a file
        blocked.log("demo", "fs_write", "x", True, "y")                                        # no exception
        host = cp.ProjectHost(self.base, self.registry, min_free_gb=0, audit=blocked)
        self.assertFalse(host.call("fs_write", {"path": "b.txt", "content": "ok"})["isError"])

    def test_without_an_audit_log_nothing_is_written(self):
        host = cp.ProjectHost(self.base, self.registry, min_free_gb=0)
        self.assertFalse(host.call("fs_write", {"path": "c.txt", "content": "ok"})["isError"])
        self.assertFalse(os.path.exists(self.audit_path))


if __name__ == "__main__":
    unittest.main()
