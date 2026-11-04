#!/usr/bin/env python3
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).with_name("candle-publish-qualified-standalones.py")
loader = importlib.util.spec_from_file_location("publication", SCRIPT)
p = importlib.util.module_from_spec(loader)
loader.loader.exec_module(p)


class PublicationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((p.ROOT / "standalone-publication.json").read_text())
        self.run = {"id": self.spec["source_run"], "head_sha": self.spec["orchestrator"],
                    "head_branch": "orch/extended-v1", "run_attempt": 1,
                    "path": ".github/workflows/candle-extended-composite.yml",
                    "status": "completed", "conclusion": "success"}
        self.jobs = {"total_count": len(p.JOBS), "jobs": [{"name": n, "run_id": self.spec["source_run"],
            "run_attempt": 1, "status": "completed", "conclusion": "success", "labels":
            ["self-hosted", "cuda", "sm61", "candle-extended", "extended-" + str(self.spec["source_run"])]}
            for n in p.JOBS]}
        self.artifacts = {"artifacts": [{**v, "expired": False, "digest": "sha256:" + v["sha256"],
                         "workflow_run": {"id": self.spec["source_run"]}} for v in self.spec["artifacts"].values()]}

    def test_exact_successful_physical_qualification_is_accepted(self):
        p.check_run(self.spec, self.run, self.jobs, self.artifacts)

    def test_changed_attempt_source_or_failed_job_is_rejected(self):
        for changed in ({"run_attempt": 2}, {"head_sha": "f" * 40}, {"conclusion": "failure"}, {"status": "in_progress"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                p.check_run(self.spec, {**self.run, **changed}, self.jobs, self.artifacts)
        self.jobs["jobs"][0]["conclusion"] = "failure"
        with self.assertRaises(ValueError): p.check_run(self.spec, self.run, self.jobs, self.artifacts)

    def test_expired_or_different_artifact_is_rejected(self):
        for field, value in (("expired", True), ("digest", "sha256:" + "f" * 64)):
            bad = copy.deepcopy(self.artifacts)
            bad["artifacts"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.check_run(self.spec, self.run, self.jobs, bad)

    def test_gpu_label_from_another_campaign_is_rejected(self):
        for job in self.jobs["jobs"]:
            if job["name"] == "SM61 Extended runtime": job["labels"] = ["self-hosted", "cuda", "sm61"]
        with self.assertRaises(ValueError): p.check_run(self.spec, self.run, self.jobs, self.artifacts)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = json.loads((p.ROOT / "standalone-publication.json").read_text())
        self.manifest = {"campaign": self.spec["campaign"], "main_sha": self.spec["main"],
                         "integration_target": "extended", "feature_count": 13, "features": [{}] * 13,
                         "update_canonical_feature_refs": False}
        self.source = json.dumps(self.manifest).encode()
        binary, lock = b"qualified executable fixture", b"qualified lockfile fixture"
        self.build = {"candidate_sha": self.spec["candidate"], "campaign": self.spec["campaign"],
                      "run_id": str(self.spec["source_run"]), "run_attempt": "1",
                      "source_manifest_sha256": p.sha(self.source), "lock_sha256": p.sha(lock),
                      "cpu_profile": "avx2", "rustflags": "-C target-cpu=x86-64-v3",
                      "environment": {"compiler_image": self.spec["compiler_image"]},
                      "gates": {"bf16": {"binaries": [{"path": "bin/" + p.sha(binary), "sha256": p.sha(binary)}]}}}
        common = {"candidate_sha": self.spec["candidate"], "campaign": self.spec["campaign"],
                  "lock_sha256": p.sha(lock), "gpu_executed": False}
        cpu = {**common, "status": "CPU_CUDA_BUILT_PASSED", "cargo_invoked": False,
               "manifest_sha256": p.sha(json.dumps(self.build).encode())}
        compile_proof = {**common, "status": "CUDA_COMPILE_PASSED", "exit_code": 0, "target": "sm_61"}
        summary = "\n".join(["EXTENDED_SM61_VALIDATION=PASS", "EXTENDED_SM61_CANDIDATE=" + self.spec["candidate"],
                             "EXTENDED_SM61_CAMPAIGN=" + self.spec["campaign"], "GATE_bf16=PASS"])
        self.data = {"build": {"manifest.json": json.dumps(self.build).encode(), "Cargo.lock": lock, "bin/" + p.sha(binary): binary},
                     "cpu": {"report.json": json.dumps(cpu).encode()}, "compile": {"report.json": json.dumps(compile_proof).encode()},
                     "gpu": {"summary.txt": summary.encode(), "gpu.txt": b"NVIDIA Pascal, 6.1, driver, GPU-fixture\n"},
                     "promotion": {"manifest.json": self.source, "summary.txt": b"PROMOTION=PASS\n"}}

    def verify(self):
        for kind, entries in self.data.items():
            record = self.spec["artifacts"][kind]
            path = self.root / (str(record["id"]) + ".zip")
            with zipfile.ZipFile(path, "w") as z:
                for name, data in entries.items(): z.writestr(name, data)
            record["sha256"] = p.sha(path.read_bytes())
        return p.verify_files(self.spec, self.root, self.source)

    def test_all_original_proofs_are_reused_without_executing_the_payload(self):
        self.assertEqual(self.verify(), self.manifest)

    def test_another_cpu_candidate_is_rejected_even_with_valid_zip_hash(self):
        proof = json.loads(self.data["cpu"]["report.json"])
        proof["candidate_sha"] = "f" * 40
        self.data["cpu"]["report.json"] = json.dumps(proof).encode()
        with self.assertRaisesRegex(ValueError, "proof identity"): self.verify()

    def test_missing_gpu_gate_is_rejected(self):
        self.data["gpu"]["summary.txt"] = b"EXTENDED_SM61_VALIDATION=PASS\n"
        with self.assertRaisesRegex(ValueError, "physical SM61"): self.verify()

    def test_original_immutable_source_policy_cannot_be_rewritten(self):
        self.data["promotion"]["manifest.json"] = json.dumps({**self.manifest, "update_canonical_feature_refs": True}).encode()
        with self.assertRaisesRegex(ValueError, "immutable candidate"): self.verify()

    def test_tampered_executable_is_rejected(self):
        name = next(n for n in self.data["build"] if n.startswith("bin/"))
        self.data["build"][name] += b"tampered"
        with self.assertRaisesRegex(ValueError, "executable SHA256"): self.verify()


class AtomicSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo, self.remote = self.root / "repo", self.root / "remote.git"
        self.repo.mkdir()
        self.execute("git", "init", "-q", "-b", "main", str(self.repo))
        self.git("config", "user.name", "Publication test")
        self.git("config", "user.email", "publication@example.invalid")
        (self.repo / "base.txt").write_text("base\n")
        base = self.commit("base")
        (self.repo / "upstream.txt").write_text("current main\n")
        main = self.commit("current main")
        features = []
        for name in ("alpha", "beta"):
            self.git("checkout", "-q", "--detach", base)
            (self.repo / (name + ".txt")).write_text(name + "\n")
            source = self.commit("source " + name)
            self.git("update-ref", "refs/heads/" + name, source)
            self.git("checkout", "-q", "--detach", main)
            (self.repo / (name + ".txt")).write_text(name + "\n")
            prepared = self.commit("prepare " + name)
            prep_ref = "prepare/test/" + name
            self.git("update-ref", "refs/heads/" + prep_ref, prepared)
            features.append({"feature": name, "target_ref": name, "source_sha": source, "prepared_sha": prepared,
                             "preparation_ref": prep_ref, "prepared_patch_sha256": p.sha(p.git(self.repo, "diff", "--binary", "--full-index", main, prepared))})
        self.git("checkout", "-q", "--detach", main)
        for item in features: self.git("merge", "--no-ff", "--no-edit", item["prepared_sha"])
        self.candidate = self.git("rev-parse", "HEAD").decode().strip()
        self.git("update-ref", "refs/heads/extended", self.candidate)
        self.git("update-ref", "refs/heads/unrelated", base)
        self.execute("git", "init", "-q", "--bare", str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "--all", "origin")
        self.spec = {"main": main, "candidate": self.candidate, "campaign": "test"}
        self.manifest = {"features": features}
        self.before = p.remote_heads(self.repo)

    def execute(self, *args):
        return subprocess.check_output(args, stderr=subprocess.DEVNULL)

    def git(self, *args):
        return self.execute("git", "-C", str(self.repo), *args)

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD").decode().strip()

    def test_atomic_publication_archives_old_sources_and_preserves_extended(self):
        targets = p.promote(self.repo, self.spec, self.manifest)
        after = p.remote_heads(self.repo)
        for name, expected in targets.items():
            self.assertEqual(after[name], expected)
            self.assertEqual(after["archive/promotion/test/before/" + name], self.before[name])
        self.assertEqual(after["extended"], self.before["extended"])
        self.assertEqual(after["unrelated"], self.before["unrelated"])
        self.assertEqual(p.promote(self.repo, self.spec, self.manifest), targets)

    def test_one_moved_lease_blocks_all_standalone_updates(self):
        self.git("push", "--force", "origin", self.spec["main"] + ":refs/heads/beta")
        with self.assertRaisesRegex(ValueError, "lease changed"):
            p.promote(self.repo, self.spec, self.manifest)
        self.assertEqual(p.remote_heads(self.repo)["alpha"], self.before["alpha"])

    def test_server_rejection_cannot_half_publish_standalones(self):
        hook = self.remote / "hooks/pre-receive"
        hook.write_text('#!/bin/sh\nwhile read old new ref; do\n  if [ "$ref" = refs/heads/beta ]; then exit 1; fi\ndone\n')
        hook.chmod(0o755)
        with self.assertRaises(subprocess.CalledProcessError):
            p.promote(self.repo, self.spec, self.manifest)
        after = p.remote_heads(self.repo)
        self.assertEqual(after["alpha"], self.before["alpha"])
        self.assertEqual(after["beta"], self.before["beta"])

    def normal_promoter(self, publish_sources):
        manifest = {**self.manifest, "schema_version": 1, "main_sha": self.spec["main"],
                    "campaign": "test", "integration_target": "extended", "old_integration_sha": self.spec["main"],
                    "update_canonical_feature_refs": publish_sources}
        path = self.repo / "candle-integration/extended.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(manifest))
        candidate = self.commit("immutable normal promotion candidate")
        ref = "integration/extended/candidate/test"
        self.git("push", "--force", "origin", candidate + ":refs/heads/" + ref,
                 self.spec["main"] + ":refs/heads/extended")
        tools = self.root / "tools"
        tools.mkdir()
        gh = tools / "gh"
        gh.write_text('#!/bin/sh\ntest "$*" = "auth setup-git"\n')
        gh.chmod(0o755)
        env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
               "GH_TOKEN": "offline-fixture", "GITHUB_REPOSITORY": "fixture/candle",
               "CANDIDATE_SHA": candidate, "CANDIDATE_REF": ref, "CAMPAIGN": "test",
               "BASE_SHA": self.spec["main"], "TARGET_BRANCH": "extended",
               "MANIFEST_PATH": "candle-integration/extended.json", "PROMOTION_KIND": "extended",
               "RUN_ID": "123", "RUN_ATTEMPT": "1", "RUNNER_TEMP": str(self.root / "reports")}
        script = SCRIPT.with_name("candle-promote-validated.sh")
        for _ in range(2):
            subprocess.run(["bash", str(script)], cwd=self.repo, env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return candidate

    def test_normal_workflow_publishes_standalones_and_is_idempotent(self):
        candidate = self.normal_promoter(True)
        after = p.remote_heads(self.repo)
        for item in self.manifest["features"]:
            self.assertEqual(after[item["target_ref"]], item["prepared_sha"])
            self.assertEqual(after["archive/promotion/test/before/" + item["target_ref"]], item["source_sha"])
        self.assertEqual(after["extended"], candidate)

    def test_historical_false_policy_is_still_respected_without_rewriting_manifest(self):
        candidate = self.normal_promoter(False)
        after = p.remote_heads(self.repo)
        for item in self.manifest["features"]:
            self.assertEqual(after[item["target_ref"]], item["source_sha"])
        self.assertEqual(after["extended"], candidate)


if __name__ == "__main__":
    unittest.main()
