#!/usr/bin/env python3
"""Offline, dependency-free tests for the immutable SM61 build archive."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_once", HERE / "candle-extended-build-once.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SHA = "a" * 40
ENVIRONMENT = {
    "rustc": "rustc test",
    "nvcc_release": "12.9",
    "gcc14": "14.0",
    "cudnn_package": "9.10.2.21-1",
    "container_recipe_sha256": "b" * 64,
}
FEATURES = [
    "bf16_candle", "fp8_candle", "fp4_candle", "cudnn_fallback_candle",
    "moe_simt_f16_candle", "asd_core_v3_standalone", "embeddinggemma2",
    "lean4_standalone", "fireredvad_standalone", "smart_turn_standalone",
    "transcribepsy_standalone", "nemotron_asr_streaming_standalone",
    "pockettts_sm61_standalone",
]


class BuildOnceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        (self.repo / "candle-integration").mkdir()
        self.source = self.repo / "candle-integration/extended.json"
        self.source.write_text(json.dumps({
            "schema_version": 1,
            "kind": "extended-multi",
            "campaign": "selftest-1",
            "feature_count": len(FEATURES),
            "features": [
                {"feature": feature, "cuda_features": ["cuda"]}
                for feature in FEATURES
            ],
        }))
        (self.repo / "Cargo.lock").write_text("selftest lock\n")
        self.environ = {
            "CANDIDATE_SHA": SHA,
            "CHECK_CAMPAIGN": "selftest-1",
            "LOCK_SHA256": module.sha256(self.repo / "Cargo.lock"),
            "CUDA_COMPUTE_CAP": "61",
            "GITHUB_RUN_ID": "40897509915",
            "GITHUB_RUN_ATTEMPT": "1",
        }
        self.oldcwd = Path.cwd()
        os.chdir(self.repo)
        self.addCleanup(os.chdir, self.oldcwd)
        self.mock_env = mock.patch.dict(os.environ, self.environ, clear=True)
        self.mock_env.start()
        self.addCleanup(self.mock_env.stop)
        self.mock_git = mock.patch.object(module, "output", return_value=SHA)
        self.mock_git.start()
        self.addCleanup(self.mock_git.stop)
        self.mock_environment = mock.patch.object(module, "environment", return_value=ENVIRONMENT)
        self.mock_environment.start()
        self.addCleanup(self.mock_environment.stop)

    def make_bundle(self):
        root = self.repo / "bundle"
        root.mkdir()
        shutil.copyfile(self.repo / "Cargo.lock", root / "Cargo.lock")
        (root / "bin").mkdir()
        shutil.copyfile("/bin/true", root / "bin/test")
        digest = module.sha256(root / "bin/test")
        (root / "bin/test").rename(root / "bin" / digest)
        manifest = {
            "schema": 1,
            "candidate_sha": SHA,
            "campaign": "selftest-1",
            "run_id": self.environ["GITHUB_RUN_ID"],
            "run_attempt": "1",
            "lock_sha256": module.sha256(self.repo / "Cargo.lock"),
            "source_manifest_sha256": module.sha256(self.source),
            "cuda_features": "cuda",
            "rustflags": "-C target-cpu=x86-64",
            "environment": ENVIRONMENT,
            "gates": {
                "bf16": {
                    "kind": "test",
                    "binaries": [{
                        "path": "bin/" + digest,
                        "sha256": digest,
                        "bytes": (root / "bin" / digest).stat().st_size,
                    }],
                    "cwd": ".",
                    "argv": [],
                },
            },
        }
        self.write_manifest(root, manifest)
        os.environ["EXTENDED_BUILD_ONCE_MANIFEST_SHA256"] = module.sha256(root / "manifest.json")
        return root, manifest

    @staticmethod
    def write_manifest(root, manifest):
        data = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        (root / "manifest.json").write_text(data)
        (root / "manifest.sha256").write_text(hashlib.sha256(data.encode()).hexdigest() + "\n")

    def test_valid_bundle_and_forward_rerun(self):
        root, _ = self.make_bundle()
        self.assertEqual(module.load_bundle(self.repo, root)["schema"], 1)
        os.environ["GITHUB_RUN_ATTEMPT"] = "2"
        self.assertEqual(module.load_bundle(self.repo, root)["schema"], 1)

    def test_binary_tampering_is_rejected(self):
        root, manifest = self.make_bundle()
        binary = root / manifest["gates"]["bf16"]["binaries"][0]["path"]
        with binary.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(RuntimeError, "executable digest mismatch"):
            module.load_bundle(self.repo, root)

    def test_untracked_lock_from_archive_and_tampering(self):
        root, _ = self.make_bundle()
        (self.repo / "Cargo.lock").unlink()
        self.assertEqual(module.load_bundle(self.repo, root)["schema"], 1)
        with (root / "Cargo.lock").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(RuntimeError, "archived Cargo.lock mismatch"):
            module.load_bundle(self.repo, root)

    def test_manifest_rewrite_fails_independent_hosted_attestation(self):
        root, manifest = self.make_bundle()
        manifest["gates"]["bf16"]["cwd"] = "candle-core"
        self.write_manifest(root, manifest)
        with self.assertRaisesRegex(RuntimeError, "hosted build attestation digest mismatch"):
            module.load_bundle(self.repo, root)

    def test_prebuilt_executable_runs_without_cargo(self):
        root, _ = self.make_bundle()
        args = type("Args", (), {
            "bundle": str(root), "action": "run-gate", "name": "bf16",
            "report": str(self.repo / "report"),
        })()
        module.execute(args)

    def test_source_lock_and_run_identity_are_bound(self):
        root, _ = self.make_bundle()
        os.environ["GITHUB_RUN_ID"] = "another"
        with self.assertRaisesRegex(RuntimeError, "run ID mismatch"):
            module.load_bundle(self.repo, root)
        os.environ["GITHUB_RUN_ID"] = self.environ["GITHUB_RUN_ID"]
        (self.repo / "Cargo.lock").write_text("wrong lock\n")
        with self.assertRaisesRegex(RuntimeError, "Cargo.lock mismatch"):
            module.load_bundle(self.repo, root)

    def test_all_original_gpu_gates_have_hosted_builds(self):
        # Simulate compiler JSON without invoking CUDA on the CPU-only runner.
        def fake_compile(_repo, _out, _name, _argv, _env, wanted=()):
            return {name: "/bin/true" for name in wanted}

        with mock.patch.object(module, "log_cargo", side_effect=fake_compile):
            module.build(type("Args", (), {"out": str(self.repo / "built")})())
        built = json.loads((self.repo / "built/manifest.json").read_text())
        expected = {
            "combined_compile", "downstream_transformers_compile",
            "downstream_qwen3_cuda_compile", "embeddinggemma2_cuda_compile",
            "embeddinggemma2_cpu_contracts", "bf16", "fp8", "fp4_mxfp4",
            "cudnn_fallback", "moe_simt_f16", "asd_module_provider",
            "asd_runtime_profile_swap", "fireredvad_cuda_contracts",
            "smart_turn_cuda_contracts", "transcribepsy_cuda_contracts",
            "transcribepsy_cuda_cli", "nemotron_asr_cuda_contracts",
            "nemotron_asr_cuda_cli", "nemotron_asr_cuda_real_parity",
            "pockettts_cuda_contracts", "pockettts_flowlm_mimi_cuda",
        }
        self.assertEqual(set(built["gates"]), expected)
        shell = (HERE / "candle-extended-sm61.sh").read_text()
        legacy_gates = set(re.findall(r"\brun_gate\s+([a-z][a-z0-9_]+)\b", shell))
        self.assertEqual(legacy_gates, expected - {"embeddinggemma2_cpu_contracts"})
        self.assertEqual(built["environment"], ENVIRONMENT)
        self.assertEqual(built["rustflags"], "-C target-cpu=x86-64")

    def test_cpu_qualify_uses_same_immutable_cuda_binaries(self):
        root, manifest = self.make_bundle()
        exemplar = manifest["gates"]["bf16"]
        for name in module.cpu_plan(set(FEATURES)):
            manifest["gates"][name] = {
                **exemplar, "cwd": "candle-transformers", "argv": ["models::test"]
            }
        manifest["gates"]["pockettts_flowlm_mimi_cuda"] = {
            **exemplar, "kind": "example", "argv": ["--device", "cuda"]
        }
        self.write_manifest(root, manifest)
        os.environ["EXTENDED_BUILD_ONCE_MANIFEST_SHA256"] = module.sha256(root / "manifest.json")

        def fake_exec(argv, _cwd, log=None):
            if log:
                Path(log).write_text(
                    "POCKETTTS_SM61_A1=PASS backend=cpu\n" if argv[-2:] == ["--device", "cpu"]
                    else "model contracts PASS\n"
                )

        args = type("Args", (), {
            "bundle": str(root), "action": "cpu-qualify", "name": None,
            "report": str(self.repo / "cpu-evidence"),
        })()
        with mock.patch.object(module, "exec_cmd", side_effect=fake_exec):
            module.execute(args)
        proof = json.loads((self.repo / "cpu-evidence/report.json").read_text())
        self.assertEqual(proof["status"], "CPU_CUDA_BUILT_PASSED")
        self.assertIn("pockettts_cpu_preflight", proof["gates"])
        self.assertFalse(proof["cargo_invoked"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
