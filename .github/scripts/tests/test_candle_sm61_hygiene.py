import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("hygiene", Path(__file__).resolve().parents[1] / "candle-sm61-hygiene.py")
hygiene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene)


class HygieneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cwd = os.getcwd()
        self.addCleanup(os.chdir, self.cwd)
        root = Path(self.temporary.name)
        subprocess.run(["git", "init", "--bare", "-q", str(root / "remote")], check=True)
        subprocess.run(["git", "init", "-q", str(root / "work")], check=True)
        os.chdir(root / "work")
        hygiene.command("git", "config", "user.name", "Hygiene test")
        hygiene.command("git", "config", "user.email", "hygiene@example.invalid")
        hygiene.command("git", "remote", "add", "origin", str(root / "remote"))
        self.old = self.commit("one")
        self.new = self.commit("two")
        self.target = "prepare/extended/extended-40969957630-1/fp8_candle"
        self.other = "prepare/extended/extended-41033583724-1/fp8_candle"
        hygiene.command("git", "push", "origin", self.old + ":refs/heads/" + self.target,
                        self.old + ":refs/heads/" + self.other, self.new + ":refs/heads/extended")
        self.plan = {"schema_version": 1, "repository": "haricot/candle", "scope": "sm61-hygiene-20261030",
                     "phase": "preview", "completed_runs": [40969957630, 41033583724],
                     "targets": [{"ref": self.target, "sha": self.old}, {"ref": self.other, "sha": self.old}],
                     "protected_heads": {"extended": self.new}, "retired_workflows": []}
        self.refs = hygiene.Refs()

    def commit(self, text):
        Path("data.txt").write_text(text)
        hygiene.command("git", "add", "data.txt")
        hygiene.command("git", "commit", "-qm", text)
        return hygiene.command("git", "rev-parse", "HEAD")

    def test_archive_then_atomic_delete_keeps_exact_commits_and_protected_heads(self):
        hygiene.validate(self.plan)
        self.refs.check(self.plan)
        self.refs.archive(self.plan)
        self.refs.delete(self.plan)
        self.refs.check(self.plan, deleted=True)
        for target in self.plan["targets"]:
            actual = hygiene.command("git", "ls-remote", "origin", "refs/tags/" + hygiene.TAG_PREFIX + target["ref"] + "^{}")
            self.assertEqual(actual.split()[0], self.old)

    def test_moved_target_rejects_all_deletions(self):
        self.refs.archive(self.plan)
        hygiene.command("git", "push", "origin", self.new + ":refs/heads/" + self.other)
        with self.assertRaises(subprocess.CalledProcessError):
            self.refs.delete(self.plan)
        self.assertEqual(self.refs.heads()[self.target], self.old)
        self.assertEqual(self.refs.heads()[self.other], self.new)

    def test_conflicting_archive_prevents_deletion(self):
        tag = hygiene.TAG_PREFIX + self.target
        hygiene.command("git", "tag", "-a", tag, self.new, "-m", "collision")
        hygiene.command("git", "push", "origin", "refs/tags/" + tag)
        with self.assertRaisesRegex(RuntimeError, "collision"):
            self.refs.archive(self.plan)
        self.refs.check(self.plan)

    def test_lightweight_archive_is_not_accepted(self):
        tag = hygiene.TAG_PREFIX + self.target
        hygiene.command("git", "tag", tag, self.old)
        hygiene.command("git", "push", "origin", "refs/tags/" + tag)
        with self.assertRaisesRegex(RuntimeError, "collision"):
            self.refs.archive(self.plan)

    def test_protected_or_immutable_candidate_is_never_a_target(self):
        self.plan["targets"][0]["ref"] = "integration/extended/candidate/extended-40969957630-1"
        with self.assertRaises(RuntimeError):
            hygiene.validate(self.plan)

    def test_current_cuda_workflow_cannot_be_retired(self):
        self.plan["retired_workflows"] = [".github/workflows/candle-extended-composite.yml"]
        with self.assertRaisesRegex(RuntimeError, "current workflow"):
            hygiene.validate(self.plan)

    def test_preview_fingerprint_binds_every_target(self):
        apply = copy.deepcopy(self.plan)
        apply.update(phase="apply", preview_run_id=123456789)
        self.assertEqual(hygiene.fingerprint(apply), hygiene.fingerprint(self.plan))
        apply["targets"][0]["sha"] = self.new
        self.assertNotEqual(hygiene.fingerprint(apply), hygiene.fingerprint(self.plan))

    def test_repeat_archive_accepts_only_matching_annotated_tags(self):
        self.refs.archive(self.plan)
        self.refs.archive(self.plan)
        self.refs.check(self.plan)


if __name__ == "__main__":
    unittest.main()
