#!/usr/bin/env python3
"""Tests for tools/chat_projects.py: the registry, the path jail and the file tools, on temp folders.

    python3 -m unittest discover -s tools -p 'test_chat_projects.py' -v
"""

import json
import os
import sys
import tempfile
import unittest

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import chat_projects as cp  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = os.path.join(self.tmp, "base")
        os.makedirs(os.path.join(self.base, "demo", "src"))
        os.makedirs(os.path.join(self.base, "other"))
        self.registry = os.path.join(self.tmp, "projects.json")
        self.outside = os.path.join(self.tmp, "outside")
        os.makedirs(self.outside)
        with open(os.path.join(self.outside, "secret.txt"), "w") as f:
            f.write("do not read\n")
        self.write_registry({"demo": {"title": "Demo", "root": "demo"}})
        self.host = cp.ProjectHost(self.base, self.registry, min_free_gb=0)
        self.root = os.path.realpath(os.path.join(self.base, "demo"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_registry(self, projects):
        with open(self.registry, "w") as f:
            json.dump({"projects": projects}, f)

    def put(self, rel, text="", binary=False):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb" if binary else "w") as f:
            f.write(text)
        return path

    def call(self, name, **args):
        res = self.host.call(name, args)
        self.assertEqual(res["content"][0]["type"], "text")
        return res["isError"], res["content"][0]["text"]

    def ok(self, name, **args):
        err, text = self.call(name, **args)
        self.assertFalse(err, text)
        return text

    def fails(self, name, contains, **args):
        err, text = self.call(name, **args)
        self.assertTrue(err, text)
        self.assertIn(contains, text)


class Registry(Base):
    def test_a_valid_registry_is_filled_in_and_a_missing_one_means_no_projects(self):
        p = cp.load_registry(self.registry, self.base)
        self.assertEqual(p["demo"], {"id": "demo", "title": "Demo", "root": self.root})
        self.assertEqual(cp.load_registry(os.path.join(self.tmp, "none.json"), self.base), {})

    def test_a_root_outside_the_base_a_bad_id_or_a_missing_folder_is_refused(self):
        for projects, text in [
            ({"x": {"root": "../outside"}}, "inside"),
            ({"x": {"root": self.outside}}, "inside"),
            ({"x": {"root": "."}}, "inside"),
            ({"Bad Name": {"root": "demo"}}, "bad project name"),
            ({"x": {"root": "nope"}}, "not a folder"),
            ({"x": {}}, "needs a root"),
        ]:
            self.write_registry(projects)
            with self.assertRaises(cp.RegistryError) as e:
                cp.load_registry(self.registry, self.base)
            self.assertIn(text, str(e.exception), projects)
        with open(self.registry, "w") as f:
            f.write("{nope")
        with self.assertRaises(cp.RegistryError):
            cp.load_registry(self.registry, self.base)

    def test_a_symlinked_root_that_leaves_the_base_is_refused(self):
        os.symlink(self.outside, os.path.join(self.base, "link"))
        self.write_registry({"x": {"root": "link"}})
        with self.assertRaises(cp.RegistryError):
            cp.load_registry(self.registry, self.base)

    def test_the_host_reports_why_the_registry_is_unusable(self):
        self.assertIsNone(self.host.problem())
        self.write_registry({"x": {"root": "../outside"}})
        self.assertIn("inside", self.host.problem())
        self.fails("fs_list", "registry cannot be used")


class Jail(Base):
    def test_paths_that_leave_the_project_are_refused_for_every_tool(self):
        os.symlink(self.outside, os.path.join(self.root, "escape"))
        os.symlink(os.path.join(self.outside, "secret.txt"), os.path.join(self.root, "peek.txt"))
        for bad, text in [("/etc/passwd", "relative"), ("~/x", "relative"), ("../outside/secret.txt", ".. is not"),
                          ("src/../../x", ".. is not"), ("escape/secret.txt", "leaves the project"), ("peek.txt", "leaves the project"),
                          ("a\x00b", "text path"), (".git/config", ".git"), ("src/.git/x", ".git")]:
            for tool, extra in [("fs_read", {}), ("fs_list", {}), ("fs_write", {"content": "x"}), ("fs_edit", {"old": "a", "new": "b"})]:
                self.fails(tool, text, path=bad, **extra)
        self.assertFalse(os.path.exists(os.path.join(self.outside, "x")))
        with open(os.path.join(self.outside, "secret.txt")) as f:
            self.assertEqual(f.read(), "do not read\n")

    def test_a_new_file_through_a_symlinked_folder_out_of_the_root_is_refused(self):
        os.symlink(self.outside, os.path.join(self.root, "escape"))
        self.fails("fs_write", "leaves the project", path="escape/new.txt", content="x")
        self.assertFalse(os.path.exists(os.path.join(self.outside, "new.txt")))

    def test_backslashes_and_dot_segments_stay_inside(self):
        self.put("src/a.txt", "hello\n")
        self.assertIn("hello", self.ok("fs_read", path="./src//a.txt"))
        self.assertIn("hello", self.ok("fs_read", path="src\\a.txt"))
        self.fails("fs_read", ".. is not", path="src\\..\\..\\x")

    def test_the_search_and_list_do_not_follow_links_or_enter_git(self):
        self.put("a.txt", "needle\n")
        self.put(".git/config", "needle in git\n")
        os.symlink(self.outside, os.path.join(self.root, "escape"))
        os.makedirs(os.path.join(self.root, "node_modules"))
        self.put("node_modules/lib.js", "needle\n")
        out = self.ok("fs_search", pattern="needle")
        self.assertIn("a.txt:1", out)
        self.assertNotIn(".git", out)
        self.assertNotIn("secret", out)
        self.assertNotIn("node_modules", out)
        listing = self.ok("fs_list")
        self.assertIn("escape -> link", listing)
        self.assertNotIn(".git", listing)


class Files(Base):
    def test_read_gives_a_header_a_range_and_a_way_to_continue(self):
        self.put("f.txt", "".join(f"line {i}\n" for i in range(1, 11)))
        out = self.ok("fs_read", path="f.txt", offset=3, limit=2)
        self.assertEqual(out, "f.txt lines 3-4 of 10\nline 3\nline 4\n[more lines follow: call again with offset 5]")
        self.assertTrue(self.ok("fs_read", path="f.txt").endswith("line 10"))
        self.assertIn("past the end", self.ok("fs_read", path="f.txt", offset=99))
        self.put("empty.txt", "")
        self.assertIn("is empty", self.ok("fs_read", path="empty.txt"))

    def test_read_refuses_binary_folders_missing_files_and_bad_numbers(self):
        self.put("b.bin", b"\x00\x01\x02", binary=True)
        self.fails("fs_read", "binary", path="b.bin")
        self.fails("fs_read", "not a file", path="src")
        self.fails("fs_read", "does not exist", path="nope.txt")
        self.put("f.txt", "x\n")
        self.fails("fs_read", "whole number", path="f.txt", limit="many")
        self.fails("fs_read", "whole number", path="f.txt", offset=True)

    def test_read_cuts_a_very_long_part_by_bytes_and_says_where_to_go_on(self):
        self.put("big.txt", "".join("x" * 1000 + "\n" for _ in range(400)))
        out = self.ok("fs_read", path="big.txt", limit=2000)
        self.assertLess(len(out), cp.MAX_READ_BYTES + 500)
        self.assertRegex(out, r"\[more lines follow: call again with offset \d+\]$")

    def test_list_orders_folders_first_limits_depth_and_the_count(self):
        self.put("src/deep/x.py", "1")
        self.put("b.txt", "12345")
        self.put("a.txt", "1")
        self.assertEqual(self.ok("fs_list"), "src/\na.txt (1 bytes)\nb.txt (5 bytes)")
        two = self.ok("fs_list", depth=2)
        self.assertIn("src/deep/", two)
        self.assertNotIn("x.py", two)
        self.assertIn("src/deep/x.py (1 bytes)", self.ok("fs_list", depth=3))
        for i in range(cp.MAX_LIST + 5):
            self.put(f"many/f{i:04}.txt", "")
        out = self.ok("fs_list", path="many")
        self.assertEqual(out.count("\n"), cp.MAX_LIST)
        self.assertTrue(out.endswith("[5 more entries not shown]"))
        self.fails("fs_list", "not a folder", path="a.txt")

    def test_search_is_literal_case_aware_filtered_by_glob_and_capped(self):
        self.put("a.py", "Alpha = 1\nbeta = 2\n")
        self.put("b.txt", "alpha here\n")
        self.put("c.bin", b"alpha\x00", binary=True)
        self.assertEqual(self.ok("fs_search", pattern="Alpha"), "a.py:1: Alpha = 1")
        both = self.ok("fs_search", pattern="alpha", ignore_case=True)
        self.assertIn("a.py:1: Alpha = 1", both)
        self.assertIn("b.txt:1: alpha here", both)
        self.assertNotIn("c.bin", both)
        self.assertEqual(self.ok("fs_search", pattern="alpha", ignore_case=True, glob="*.py"), "a.py:1: Alpha = 1")
        self.assertEqual(self.ok("fs_search", pattern="zzz"), "no matches")
        self.assertEqual(self.ok("fs_search", pattern=".*"), "no matches")            # a literal, not a regular expression
        self.put("many.txt", "hit\n" * 80)
        out = self.ok("fs_search", pattern="hit", path=".")
        self.assertEqual(out.count("many.txt:"), cp.MAX_MATCHES)
        self.assertIn("30 more matches", out)
        self.fails("fs_search", "pattern", pattern="")
        self.fails("fs_search", "pattern", pattern="x" * (cp.MAX_PATTERN + 1))

    def test_write_creates_folders_replaces_keeps_the_mode_and_leaves_no_temp_files(self):
        self.assertEqual(self.ok("fs_write", path="new/dir/a.txt", content="héllo\n"), "created new/dir/a.txt (7 bytes)")
        with open(os.path.join(self.root, "new/dir/a.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "héllo\n")
        path = os.path.join(self.root, "run.sh")
        self.put("run.sh", "old")
        os.chmod(path, 0o755)
        self.assertEqual(self.ok("fs_write", path="run.sh", content="new"), "replaced run.sh (3 bytes)")
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o755)
        self.assertEqual([n for n in os.listdir(self.root) if n.startswith(".kevin-")], [])
        self.fails("fs_write", "is a folder", path="src", content="x")
        self.fails("fs_write", "is a folder", path=".", content="x")
        self.fails("fs_write", "must be text", path="a.txt", content=5)

    def test_write_and_edit_refuse_big_files_and_a_nearly_full_disk(self):
        self.fails("fs_write", "limit", path="big.txt", content="x" * (cp.MAX_WRITE_BYTES + 1))
        self.assertFalse(os.path.exists(os.path.join(self.root, "big.txt")))
        short = cp.ProjectHost(self.base, self.registry, min_free_gb=10 ** 6)
        res = short.call("fs_write", {"path": "a.txt", "content": "x"})
        self.assertTrue(res["isError"])
        self.assertIn("kept for the models", res["content"][0]["text"])
        self.assertFalse(os.path.exists(os.path.join(self.root, "a.txt")))
        self.put("e.txt", "one\n")
        res = short.call("fs_edit", {"path": "e.txt", "old": "one", "new": "two"})
        self.assertTrue(res["isError"])
        with open(os.path.join(self.root, "e.txt")) as f:
            self.assertEqual(f.read(), "one\n")

    def test_edit_replaces_one_exact_piece_and_explains_every_refusal(self):
        self.put("e.txt", "a = 1\nb = 2\nb = 2\n")
        self.assertEqual(self.ok("fs_edit", path="e.txt", old="a = 1", new="a = 10"), "replaced 1 occurrence in e.txt")
        self.fails("fs_edit", "appears 2 times", path="e.txt", old="b = 2", new="b = 3")
        self.assertEqual(self.ok("fs_edit", path="e.txt", old="b = 2", new="b = 3", replace_all=True), "replaced 2 occurrences in e.txt")
        with open(os.path.join(self.root, "e.txt")) as f:
            self.assertEqual(f.read(), "a = 10\nb = 3\nb = 3\n")
        self.fails("fs_edit", "not found", path="e.txt", old="zzz", new="y")
        self.fails("fs_edit", "not empty", path="e.txt", old="", new="y")
        self.fails("fs_edit", "does not exist", path="gone.txt", old="a", new="b")
        self.put("b.bin", b"\x00", binary=True)
        self.fails("fs_edit", "binary", path="b.bin", old="a", new="b")


class Selection(Base):
    def test_one_project_is_the_default_and_several_need_a_name(self):
        self.put("a.txt", "hi\n")
        self.assertIn("hi", self.ok("fs_read", path="a.txt"))
        self.write_registry({"demo": {"title": "Demo", "root": "demo"}, "other": {"root": "other"}})
        self.fails("fs_read", "say which project", path="a.txt")
        self.fails("fs_read", "no project named 'nope'", path="a.txt", project="nope")
        self.assertIn("hi", self.ok("fs_read", path="a.txt", project="demo"))
        self.fails("fs_read", "no project named", path="a.txt", project=5)
        self.write_registry({})
        self.fails("fs_read", "none yet", path="a.txt")

    def test_two_projects_do_not_see_each_other(self):
        self.write_registry({"demo": {"root": "demo"}, "other": {"root": "other"}})
        self.put("only-demo.txt", "x\n")
        self.fails("fs_read", "does not exist", path="only-demo.txt", project="other")
        self.fails("fs_read", ".. is not", path="../other/x", project="demo")

    def test_project_list_names_projects_and_free_space(self):
        out = self.ok("project_list")
        self.assertTrue(out.startswith("demo: Demo (demo)\n"))
        self.assertRegex(out, r"\d+ GB free on the disk$")
        self.write_registry({})
        self.assertIn("no projects yet", self.ok("project_list"))

    def test_an_unknown_tool_is_an_error_result_not_a_crash(self):
        self.fails("rm_rf", "no tool named", path="x")


class Client(Base):
    def test_it_speaks_the_three_calls_the_manager_makes(self):
        client = cp.NativeProjectClient("project", {}, {}, self.host)
        self.assertEqual(client.request("initialize")["serverInfo"]["name"], "kevin-projects")
        names = [t["name"] for t in client.request("tools/list")["tools"]]
        self.assertEqual(names, ["project_list", "fs_read", "fs_list", "fs_search", "fs_write", "fs_edit"])
        res = client.request("tools/call", {"name": "project_list", "arguments": {}})
        self.assertFalse(res["isError"])
        self.assertIsNone(client.dead)
        client.notify("notifications/initialized")
        client.close()

    def test_reads_are_marked_read_only_and_writes_are_not(self):
        hints = {t["name"]: t["annotations"]["readOnlyHint"] for t in cp.TOOLS}
        self.assertEqual(hints, {"project_list": True, "fs_read": True, "fs_list": True, "fs_search": True, "fs_write": False, "fs_edit": False})
        for t in cp.TOOLS:
            self.assertEqual(t["inputSchema"]["type"], "object")
            self.assertIn("project", t["inputSchema"]["properties"])
            self.assertNotIn("project", t["inputSchema"]["required"])

    def test_the_tool_list_version_follows_the_definitions(self):
        self.assertRegex(cp.TOOLS_VERSION, r"^[0-9a-f]{12}$")
        import copy
        import hashlib
        changed = copy.deepcopy(cp.TOOLS)
        changed[1]["description"] += "!"
        self.assertNotEqual(hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).hexdigest()[:12], cp.TOOLS_VERSION)


if __name__ == "__main__":
    unittest.main()
