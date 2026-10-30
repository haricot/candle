import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

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

    def followup(self, ref):
        self.plan.update(schema_version=2, scope="sm61-hygiene-20261030-followup",
                         promoted_runs={"40969957630": {"candidate": self.old, "gpu_job_id": 1,
                                                       "promotion_job_id": 2}},
                         orphan_workflows=[], validation={"run_id": 40969957630,
                                                        "gpu_job_id": 1, "promotion_job_id": 2})
        self.plan["targets"] = [{"ref": ref, "sha": self.old}]
        self.proof_ref = "archive/extended/sm61-validated-40969957630-1"
        hygiene.command("git", "push", "origin", self.old + ":refs/heads/" + ref,
                        self.old + ":refs/heads/" + self.proof_ref)

    @staticmethod
    def successful_api(path):
        if "/jobs?" in path:
            return {"jobs": [{"id": 1, "conclusion": "success"}, {"id": 2, "conclusion": "success"}]}
        if path.startswith("/actions/runs/"):
            return {"id": 40969957630, "status": "completed", "conclusion": "success"}
        if path.startswith("/pulls?"):
            return []
        return {"workflow_runs": []}

    def test_completed_promotion_backup_is_archived_before_branch_removal(self):
        ref = "archive/promotion/extended-40969957630-1/before/extended"
        self.followup(ref)
        hygiene.validate(self.plan)
        with mock.patch.object(hygiene, "api", side_effect=self.successful_api):
            hygiene.preflight(self.plan, self.refs)
        self.refs.archive(self.plan)
        self.refs.delete(self.plan)
        self.refs.check(self.plan, deleted=True)
        self.assertEqual(self.refs.heads()[self.proof_ref], self.old)
        tag = hygiene.TAG_PREFIX + ref
        self.assertEqual(hygiene.command("git", "ls-remote", "origin", "refs/tags/" + tag + "^{}").split()[0], self.old)

    def test_unattested_promotion_backup_is_rejected(self):
        self.followup("archive/promotion/extended-12345-1/before/extended")
        with self.assertRaises(RuntimeError):
            hygiene.validate(self.plan)

    def test_failed_promotion_rejects_cleanup_even_after_gpu_success(self):
        self.followup("archive/promotion/extended-40969957630-1/before/extended")
        def failed(path):
            value = self.successful_api(path)
            if path.startswith("/actions/runs/40969957630/jobs"):
                value["jobs"][1]["conclusion"] = "failure"
            return value
        with mock.patch.object(hygiene, "api", side_effect=failed), self.assertRaises(RuntimeError):
            hygiene.preflight(self.plan, self.refs)

    def test_merged_ci_ref_requires_actual_ancestry_of_protected_head(self):
        ref = "ci/extended-examples-legacy-alias-validation"
        self.followup(ref)
        self.plan["targets"][0]["merged_into"] = "extended"
        hygiene.validate(self.plan)
        with mock.patch.object(hygiene, "api", side_effect=self.successful_api):
            hygiene.preflight(self.plan, self.refs)
        self.plan["protected_heads"] = {"extended": self.old}
        self.plan["targets"][0]["sha"] = self.new
        hygiene.command("git", "push", "origin", self.new + ":refs/heads/" + ref)
        hygiene.command("git", "push", "--force", "origin", self.old + ":refs/heads/extended")
        with mock.patch.object(hygiene, "api", side_effect=self.successful_api), self.assertRaises(subprocess.CalledProcessError):
            hygiene.preflight(self.plan, self.refs)

    def test_orphan_workflow_is_rejected_if_present_on_any_current_branch(self):
        self.followup("archive/promotion/extended-40969957630-1/before/extended")
        self.plan.update(orphan_workflows=["data.txt"], retired_workflows=["data.txt"])
        with mock.patch.object(hygiene, "api", side_effect=self.successful_api), self.assertRaisesRegex(RuntimeError, "still exists"):
            hygiene.preflight(self.plan, self.refs)

    def test_open_pr_preserves_target_branch(self):
        self.followup("archive/promotion/extended-40969957630-1/before/extended")
        def open_pr(path):
            if path.startswith("/pulls?"):
                return [{"base": {"ref": "extended"}, "head": {"ref": self.plan["targets"][0]["ref"]}}]
            return self.successful_api(path)
        with mock.patch.object(hygiene, "api", side_effect=open_pr), self.assertRaisesRegex(RuntimeError, "open PR"):
            hygiene.preflight(self.plan, self.refs)


if __name__ == "__main__":
    unittest.main()
