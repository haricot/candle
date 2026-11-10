#!/usr/bin/env python3
"""R4.3 safety contract and atomic Git remote smoke tests."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("candle-historical-reachability.py")
spec = importlib.util.spec_from_file_location("candle_historical_reachability", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def heads_fixture():
    sha = lambda letter: letter * 40
    return {
        "main": sha("a"), "extended": sha("b"), "cuda_legacy": sha("c"),
        "archive/extended/sm61-validated-1-1": sha("b"),
        "archive/legacy/sm61-validated-1-1": sha("c"),
        "archive/unified/previous/extended": sha("a"),
        "bf16_candle": sha("d"),
        "prepare/extended/old/bf16_candle": sha("e"),
        "integration/extended/candidate/old": sha("a"),
        "integration/legacy-cuda/candidate/protected": sha("f"),
        "archive/promotion/old/before/extended": sha("e"),
    }


class Planning(unittest.TestCase):
    def test_unique_commits_are_tagged_before_removal(self):
        heads = heads_fixture()
        p = audit.plan(heads, {"integration/legacy-cuda/candidate/protected"},
                       {"a" * 40, "b" * 40, "c" * 40}, [])
        self.assertEqual(len(p["deletions"]), 3)
        self.assertEqual(len(p["preservation_tags"]), 1)
        self.assertEqual(len(p["preservation_tags"]["e" * 40]), 2)
        self.assertEqual(p["deletions"]["integration/extended/candidate/old"]
                         ["preservation"], "REACHABLE")
        self.assertEqual(p["skipped"][0]["reason"], "OPEN_PR")
        self.assertIn("archive/extended/sm61-validated-1-1", p["survivor_roots"])
        self.assertIn("archive/unified/previous/extended", p["survivor_roots"])

    def test_other_running_campaign_blocks_every_delete(self):
        p = audit.plan(heads_fixture(), set(), set(), ["42"])
        self.assertFalse(p["deletions"])
        self.assertEqual(len(p["skipped"]), 4)

    def test_unrecognized_refs_and_missing_proof_are_preserved(self):
        heads = heads_fixture()
        heads["archive/promotion/unknown"] = "e" * 40
        p = audit.plan(heads, set(), set(), [])
        self.assertEqual(p["skipped"][0]["reason"], "UNKNOWN_SHAPE")
        heads.pop("archive/extended/sm61-validated-1-1")
        with self.assertRaises(ValueError):
            audit.plan(heads, set(), set(), [])

    def test_no_canonical_ref_is_prunable(self):
        for name in ("extended", "cuda_legacy",
                     "archive/unified/previous/extended",
                     "archive/extended/sm61-validated-1-1",
                     "integration/proposal-sm61-all-six"):
            self.assertFalse(audit.is_prunable(name))

    def test_confirmation_binds_exact_refs_and_shas(self):
        heads = heads_fixture()
        p = audit.plan(heads, set(), set(), [])
        token = audit.confirmation_token(p)
        self.assertTrue(token.startswith("PRUNE_HISTORICAL_"))
        other = heads_fixture()
        other["prepare/extended/old/bf16_candle"] = "1" * 40
        self.assertNotEqual(token, audit.confirmation_token(
            audit.plan(other, set(), set(), [])))
        other = heads_fixture()
        other["extended"] = "2" * 40
        self.assertNotEqual(token, audit.confirmation_token(
            audit.plan(other, set(), set(), [])))


class AtomicRemote(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        root = Path(self.tmp.name)
        self.local = root / "local"
        self.remote = root / "remote.git"
        subprocess.check_call(["git", "init", "--bare", "-q", str(self.remote)])
        subprocess.check_call(["git", "init", "-q", str(self.local)])
        os.chdir(self.local)
        git("config", "user.name", "R4.3 Test")
        git("config", "user.email", "r43@example.test")
        git("remote", "add", "origin", str(self.remote))
        git("commit", "--allow-empty", "-qm", "base")
        self.base = git("rev-parse", "HEAD")
        git("commit", "--allow-empty", "-qm", "unique")
        self.unique = git("rev-parse", "HEAD")
        git("branch", "-f", "main", self.base)
        git("branch", "-f", "extended", self.base)
        git("branch", "-f", "cuda_legacy", self.base)
        git("branch", "archive/extended/sm61-validated-1-1", self.base)
        git("branch", "archive/legacy/sm61-validated-1-1", self.base)
        git("branch", "prepare/extended/old/feature", self.unique)
        git("branch", "integration/extended/candidate/old", self.base)
        git("push", "-q", "origin", "--all")
        self.old_pr = audit.retention.open_pr_refs
        self.old_active = audit.retention.other_active_runs
        audit.retention.open_pr_refs = lambda repo: set()
        audit.retention.other_active_runs = lambda repo, run_id: []

    def tearDown(self):
        audit.retention.open_pr_refs = self.old_pr
        audit.retention.other_active_runs = self.old_active
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_atomic_archive_and_cleanup_retains_unique_commit(self):
        before = audit.retention.remote_heads()
        reachable = audit.retention.retained_graph([self.base])
        p = audit.plan(before, set(), reachable, [])
        self.assertEqual(len(p["deletions"]), 2)
        self.assertEqual(len(p["preservation_tags"]), 1)
        self.assertEqual(audit.apply(p, before, "haricot/candle", "12345"), 2)
        after = audit.retention.remote_heads()
        self.assertNotIn("prepare/extended/old/feature", after)
        self.assertNotIn("integration/extended/candidate/old", after)
        tag = audit.ARCHIVE_TAG_PREFIX + "12345/" + self.unique
        self.assertEqual(git("ls-remote", "--tags", "origin",
                             "refs/tags/" + tag + "^{}").split("\t")[0], self.unique)
        self.assertTrue(audit.retention.remote_tag_exists(
            audit.HISTORY_TAG_PREFIX + "12345"))
        self.assertEqual(after["extended"], self.base)

    def test_changed_branch_fails_before_any_delete(self):
        before = audit.retention.remote_heads()
        p = audit.plan(before, set(), {self.base}, [])
        git("push", "-q", "origin", self.unique + ":refs/heads/extended")
        with self.assertRaises(ValueError):
            audit.apply(p, before, "haricot/candle", "67890")
        self.assertIn("prepare/extended/old/feature", audit.retention.remote_heads())
        self.assertFalse(audit.retention.remote_tag_exists(
            audit.HISTORY_TAG_PREFIX + "67890"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
