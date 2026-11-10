#!/usr/bin/env python3
"""Offline safeguards for reachability-qualified Unified branch retention."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("candle-unified-retention.py")
spec = importlib.util.spec_from_file_location("unified_retention", SCRIPT)
retention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retention)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c, self.d, self.e, self.f = [x * 40 for x in "abcdef"]
        self.campaign = "extended-fixture-1"
        self.env = {"GITHUB_REPOSITORY": "haricot/candle", "CAMPAIGN": self.campaign,
                    "CANDIDATE_SHA": self.e, "CANDIDATE_REF": f"integration/extended/candidate/{self.campaign}",
                    "LEGACY_CANDIDATE_SHA": self.f,
                    "LEGACY_CANDIDATE_REF": f"integration/legacy-cuda/candidate/{self.campaign}",
                    "BASE_SHA": self.a, "RUN_ID": "42", "RUN_ATTEMPT": "1"}
        feature = lambda target, source, prepared, kind: {
            "feature": target, "target_ref": target, "source_sha": source,
            "prepared_sha": prepared, "preparation_ref": f"prepare/{kind}/{self.campaign}/{target}",
        }
        self.extended = {"campaign": self.campaign, "main_sha": self.a,
                         "promotion_target": "extended", "update_canonical_feature_refs": True,
                         "old_integration_sha": self.c,
                         "features": [feature("alpha", self.a, self.b, "extended"),
                                      feature("beta", self.c, self.d, "extended")]}
        self.legacy = {"campaign": self.campaign, "main_sha": self.a,
                       "promotion_target": "cuda_legacy", "old_integration_sha": self.d,
                       "features": [feature("alpha", self.a, self.b, "legacy")]}
        self.heads = {"alpha": self.b, "beta": self.d, "extended": self.e,
                      "cuda_legacy": self.f,
                      self.env["CANDIDATE_REF"]: self.e,
                      self.env["LEGACY_CANDIDATE_REF"]: self.f,
                      f"prepare/extended/{self.campaign}/alpha": self.b,
                      f"prepare/extended/{self.campaign}/beta": self.d,
                      f"archive/promotion/{self.campaign}/before/alpha": self.a,
                      f"archive/promotion/{self.campaign}/before/beta": self.c,
                      f"archive/promotion/{self.campaign}/before/extended": self.c,
                      f"archive/promotion/{self.campaign}/before/cuda_legacy": self.d,
                      "archive/extended/sm61-validated-42-1": self.e,
                      "archive/legacy/sm61-validated-42-1": self.f,
                      "prepare/extended/older/alpha": self.a,
                      "integration/extended/candidate/unique-older": "0" * 40,
                      "archive/promotion/historical/before/alpha": self.a}

    def plan(self, protected=(), active=(), graph=None):
        plan = retention.build_plan(self.heads, set(protected), self.extended, self.legacy,
                                    self.env, graph, active)
        plan.update(campaign=self.env["CAMPAIGN"], run_id=self.env["RUN_ID"],
                    run_attempt=self.env["RUN_ATTEMPT"])
        return plan

    def test_only_one_rolling_previous_per_canonical_ref(self):
        result = self.plan()
        self.assertEqual(result["updates"], {
            "archive/unified/previous/alpha": self.a,
            "archive/unified/previous/beta": self.c,
            "archive/unified/previous/extended": self.c,
            "archive/unified/previous/cuda_legacy": self.d})
        self.assertIn(self.env["CANDIDATE_REF"], result["deletions"])
        self.assertIn(f"archive/promotion/{self.campaign}/before/beta", result["deletions"])
        self.assertIn("prepare/extended/older/alpha", result["deletions"])
        self.assertNotIn("integration/extended/candidate/unique-older", result["deletions"])
        self.assertIn("archive/promotion/historical/before/alpha", result["deletions"])
        self.assertNotIn("archive/extended/sm61-validated-42-1", result["deletions"])
        self.assertNotIn("archive/legacy/sm61-validated-42-1", result["deletions"])

    def test_missing_sm61_qualification_blocks_pruning(self):
        self.heads.pop("archive/extended/sm61-validated-42-1")
        with self.assertRaisesRegex(ValueError, "missing preserved SM61 qualification"):
            self.plan()

    def test_another_active_campaign_blocks_all_temporary_deletions(self):
        result = self.plan(active=("321",))
        self.assertFalse(result["deletions"])
        self.assertTrue(result["skipped"])
        self.assertIn("OTHER_ACTIVE_CAMPAIGN", {item["reason"] for item in result["skipped"]})

    def test_reachable_ancestor_can_be_pruned_but_independent_work_cannot(self):
        self.heads["prepare/extended/older/ancestor"] = "9" * 40
        self.heads["prepare/extended/older/independent"] = "8" * 40
        result = self.plan(graph={"9" * 40})
        self.assertIn("prepare/extended/older/ancestor", result["deletions"])
        self.assertNotIn("prepare/extended/older/independent", result["deletions"])

    def test_unproved_history_survives(self):
        self.heads["prepare/extended/older/unmerged-work"] = "0" * 40
        result = self.plan()
        self.assertNotIn("prepare/extended/older/unmerged-work", result["deletions"])
        self.assertIn("prepare/extended/older/unmerged-work", result["unproved_remaining"])

    def test_stale_or_diverged_canonical_ref_rejects_entire_cleanup(self):
        self.heads["alpha"] = self.f
        with self.assertRaisesRegex(ValueError, "canonical target moved"):
            self.plan()

    def test_published_temporary_ref_moving_rejects_entire_cleanup(self):
        self.heads[self.env["CANDIDATE_REF"]] = self.a
        with self.assertRaisesRegex(ValueError, "temporary ref changed"):
            self.plan()

    def test_open_pr_head_and_base_are_protected(self):
        protected = (f"prepare/extended/{self.campaign}/alpha", self.env["CANDIDATE_REF"])
        result = self.plan(protected)
        self.assertEqual({item["ref"] for item in result["skipped"]}, set(protected))
        for ref in protected:
            self.assertNotIn(ref, result["deletions"])

    def test_a_second_successful_run_is_idempotent(self):
        result = self.plan()
        for ref in result["deletions"]:
            self.heads.pop(ref)
        self.heads.update(result["updates"])
        again = self.plan()
        self.assertEqual(again["updates"], {})
        self.assertEqual(again["deletions"], {})

    def test_preserved_sm61_ref_moving_before_push_blocks_cleanup(self):
        plan = self.plan()
        changed = dict(self.heads)
        changed["archive/extended/sm61-validated-42-1"] = self.a
        with patch.object(retention, "remote_heads", return_value=changed):
            with self.assertRaisesRegex(ValueError, "preserved ancestry proof moved"):
                retention.apply_plan(plan, self.heads, "haricot/candle")

    def test_new_active_run_before_push_blocks_cleanup(self):
        plan = self.plan()
        with (patch.object(retention, "remote_heads", return_value=self.heads),
              patch.object(retention, "open_pr_refs", return_value=set()),
              patch.object(retention, "other_active_runs", return_value=["777"])):
            with self.assertRaisesRegex(ValueError, "another active campaign"):
                retention.apply_plan(plan, self.heads, "haricot/candle")

    def test_next_campaign_rotates_existing_previous_without_new_history(self):
        first = self.plan()
        for ref in first["deletions"]:
            self.heads.pop(ref)
        self.heads.update(first["updates"])
        original = dict(self.heads)
        new_campaign = "extended-fixture-2"
        next_shas = {"alpha": "1" * 40, "beta": "2" * 40,
                     "extended": "3" * 40, "cuda_legacy": "4" * 40}
        self.extended["campaign"] = new_campaign
        self.legacy["campaign"] = new_campaign
        self.extended["old_integration_sha"] = self.e
        self.legacy["old_integration_sha"] = self.f
        for manifest in (self.extended, self.legacy):
            for feat in manifest["features"]:
                feat["source_sha"] = original[feat["target_ref"]]
                feat["prepared_sha"] = next_shas[feat["target_ref"]]
                feat["preparation_ref"] = feat["preparation_ref"].replace(self.campaign, new_campaign)
        self.env.update(CAMPAIGN=new_campaign, CANDIDATE_SHA=next_shas["extended"],
                        LEGACY_CANDIDATE_SHA=next_shas["cuda_legacy"], RUN_ID="43",
                        CANDIDATE_REF=f"integration/extended/candidate/{new_campaign}",
                        LEGACY_CANDIDATE_REF=f"integration/legacy-cuda/candidate/{new_campaign}")
        self.heads.update(next_shas)
        self.heads["archive/extended/sm61-validated-43-1"] = next_shas["extended"]
        self.heads["archive/legacy/sm61-validated-43-1"] = next_shas["cuda_legacy"]
        for target in next_shas:
            self.heads[f"archive/promotion/{new_campaign}/before/{target}"] = original[target]
        next_plan = self.plan()
        for target in next_shas:
            self.assertEqual(next_plan["updates"][f"archive/unified/previous/{target}"], original[target])
        self.assertEqual(sum(x.startswith("archive/unified/previous/") for x in
                             (set(original) | set(next_plan["updates"]))), 4)

    def test_open_pr_prevents_rotation_of_rollback_ref(self):
        with self.assertRaisesRegex(ValueError, "open PR protects a rolling"):
            self.plan(protected=("archive/unified/previous/alpha",))

    def test_promotion_proof_is_required_before_any_plan(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "manifest.json").write_text(json.dumps(self.extended))
            (root / "legacy-manifest.json").write_text(json.dumps(self.legacy))
            (root / "summary.txt").write_text("PROMOTION=FAIL\n")
            with self.assertRaisesRegex(ValueError, "promotion proof"):
                retention.validated_inputs(root, self.env)
            (root / "summary.txt").write_text("\n".join((
                "PROMOTION=PASS", f"CAMPAIGN={self.campaign}",
                f"CANDIDATE_SHA={self.e}", "TARGET_BRANCH=extended")))
            retention.validated_inputs(root, self.env)


class AtomicRemoteTests(RetentionTests):
    """Exercise the final atomic update against a disposable local Git remote."""

    def test_atomic_update_and_pruning_against_bare_git_remote(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            repo, remote = root / "repo", root / "remote.git"
            repo.mkdir()
            subprocess.check_call(["git", "init", "-q", "-b", "main", str(repo)])
            subprocess.check_call(["git", "init", "-q", "--bare", str(remote)])

            def command(*args):
                return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

            command("config", "user.email", "test@example.invalid")
            command("config", "user.name", "Retention fixture")
            commits = []
            for i in range(6):
                (repo / "state").write_text(str(i))
                command("add", "state")
                command("commit", "-qm", f"fixture {i}")
                commits.append(command("rev-parse", "HEAD"))
            # Construct an unmerged descendant without moving any stable ref.
            unproved_sha = command("commit-tree", command("rev-parse", "HEAD^{tree}"),
                                   "-p", command("rev-parse", "HEAD"),
                                   "-m", "unmerged historical candidate")
            self.a, self.b, self.c, self.d, self.e, self.f = commits
            self.setUp()  # Recreate manifests/heads from the real SHA identities below.
            # `setUp` assigns placeholder identities; replace in all nested data.
            mapping = dict(zip([x * 40 for x in "abcdef"], commits))

            def replace(obj):
                if isinstance(obj, dict): return {k: replace(v) for k, v in obj.items()}
                if isinstance(obj, list): return [replace(v) for v in obj]
                return mapping.get(obj, obj)

            self.heads = replace(self.heads)
            self.heads["integration/extended/candidate/unique-older"] = unproved_sha
            self.extended = replace(self.extended)
            self.legacy = replace(self.legacy)
            self.env = replace(self.env)
            command("remote", "add", "origin", str(remote))
            for ref, commit in self.heads.items():
                command("update-ref", f"refs/heads/{ref}", commit)
            command("push", "-q", "--all", "origin")
            cwd = Path.cwd()
            os.chdir(repo)
            try:
                graph = retention.retained_graph(
                    commit for ref, commit in self.heads.items()
                    if not ref.startswith(retention.PRUNABLE_PREFIXES))
                self.assertIn(commits[0], graph)
                self.assertNotIn(unproved_sha, graph)
                result = self.plan(graph=graph)
                with (patch.object(retention, "open_pr_refs", return_value=set()),
                      patch.object(retention, "other_active_runs", return_value=[])):
                    self.assertGreater(retention.apply_plan(result, self.heads, "haricot/candle"), 0)
                actual = retention.remote_heads()
                for ref, commit in result["updates"].items():
                    self.assertEqual(actual[ref], commit)
                for ref in result["deletions"]:
                    self.assertNotIn(ref, actual)
                self.assertIn("integration/extended/candidate/unique-older", actual)
                self.assertNotIn("archive/promotion/historical/before/alpha", actual)
                self.assertEqual(actual["archive/extended/sm61-validated-42-1"], self.env["CANDIDATE_SHA"])
                self.assertEqual(actual["archive/legacy/sm61-validated-42-1"], self.env["LEGACY_CANDIDATE_SHA"])
                tag = retention.history_tag(self.campaign)
                tagged = command("for-each-ref", "--format=%(contents)", f"refs/tags/{tag}")
                history = json.loads(tagged)
                self.assertEqual(history["campaign"], self.campaign)
                self.assertIn({"ref": self.env["CANDIDATE_REF"], "sha": self.env["CANDIDATE_SHA"]},
                              history["deleted"])
                self.assertTrue(command("ls-remote", "--tags", "origin", f"refs/tags/{tag}"))
                self.heads = actual
                again = self.plan()
                self.assertEqual(again["updates"], {})
                self.assertEqual(again["deletions"], {})
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
