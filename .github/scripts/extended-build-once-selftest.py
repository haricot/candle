#!/usr/bin/env python3
"""Offline, dependency-free tests for the immutable SM61 build archive."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
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
        cli = built["gates"]["nemotron_asr_cuda_cli"]
        parity = built["gates"]["nemotron_asr_cuda_real_parity"]
        self.assertEqual(cli["kind"], "hosted-check")
        self.assertEqual(parity["kind"], "nemotron-parity")
        self.assertEqual(cli["binaries"], parity["binaries"])
        self.assertTrue(cli["binaries"])

    def nemotron_bundle(self):
        root, manifest = self.make_bundle()
        exemplar = manifest["gates"]["bf16"]
        manifest["gates"]["nemotron_asr_cuda_cli"] = {
            **exemplar, "kind": "hosted-check"
        }
        manifest["gates"]["nemotron_asr_cuda_real_parity"] = {
            **exemplar, "kind": "nemotron-parity"
        }
        self.write_manifest(root, manifest)
        os.environ["EXTENDED_BUILD_ONCE_MANIFEST_SHA256"] = module.sha256(root / "manifest.json")
        return root

    def test_nemotron_compilation_gate_never_executes_cli_without_model_arguments(self):
        root = self.nemotron_bundle()
        args = type("Args", (), {
            "bundle": str(root), "action": "run-gate", "name": "nemotron_asr_cuda_cli",
            "report": str(self.repo / "report"),
        })()
        with mock.patch.object(module, "exec_cmd", side_effect=AssertionError("unexpected model launch")):
            module.execute(args)

    def test_nemotron_real_parity_keeps_all_model_input_and_reference_arguments(self):
        root = self.nemotron_bundle()
        model = self.repo / "readonly-model"
        wav = self.repo / "readonly-input.wav"
        os.environ.update(CANDLE_NEMOTRON_MODEL_DIR=str(model), CANDLE_NEMOTRON_TEST_WAV=str(wav))
        args = type("Args", (), {
            "bundle": str(root), "action": "run-gate", "name": "nemotron_asr_cuda_real_parity",
            "report": str(self.repo / "report"),
        })()
        with mock.patch.object(module, "exec_cmd") as execute:
            module.execute(args)
        self.assertEqual(execute.call_count, 4)
        for call, (language, device) in zip(execute.call_args_list,
                [("fr-FR", "cpu"), ("fr-FR", "cuda-cudnn"), ("auto", "cpu"), ("auto", "cuda-cudnn")]):
            argv = call.args[0]
            options = dict(zip(argv[1::2], argv[2::2]))
            self.assertEqual(options["--model-dir"], str(model))
            self.assertEqual(options["--input"], str(wav))
            self.assertEqual(options["--mode"], "streaming")
            self.assertEqual(options["--language"], language)
            self.assertEqual(options["--lookahead"], "3")
            self.assertEqual(options["--device"], device)
            self.assertEqual(options["--dtype"], "f32")
            if device == "cuda-cudnn":
                self.assertEqual(options["--expect-result"],
                    str(self.repo / "report/nemotron-real-parity" / ("cpu-" + language) / "result.json"))
            else:
                self.assertNotIn("--expect-result", options)

    def test_nemotron_parity_failure_is_not_replaced_by_compilation_success(self):
        root = self.nemotron_bundle()
        args = type("Args", (), {
            "bundle": str(root), "action": "run-gate", "name": "nemotron_asr_cuda_real_parity",
            "report": str(self.repo / "report"),
        })()
        with mock.patch.object(module, "exec_cmd", side_effect=RuntimeError("physical parity failed")):
            with self.assertRaisesRegex(RuntimeError, "physical parity failed"):
                module.execute(args)

    def test_legacy_archive_preserves_all_original_gpu_gates(self):
        os.environ["CANDLE_BUILD_SOURCE"] = "legacy"
        source = self.repo / "candle-integration/legacy-cuda.json"
        source.write_text(json.dumps({"schema_version": 1, "kind": "legacy-cuda-five",
            "campaign": "selftest-1", "features": [{"feature": feature} for feature in FEATURES[:5]]}))
        def fake_compile(_repo, _out, _name, _argv, _env, wanted=()):
            return {name: "/bin/true" for name in wanted}
        with mock.patch.object(module, "log_cargo", side_effect=fake_compile):
            module.build(type("Args", (), {"out": str(self.repo / "built")})())
        manifest = json.loads((self.repo / "built/manifest.json").read_text())
        self.assertEqual(set(manifest["gates"]), {"combined_compile", "bf16", "fp8", "fp4_mxfp4",
                                                 "cudnn_fallback", "moe_simt_f16"})

    def test_physical_replay_does_not_require_the_build_compiler(self):
        root, manifest = self.make_bundle()
        recipe = self.repo / ".ci-tools/.github/container/Containerfile.cuda1292-ubuntu2604"
        recipe.parent.mkdir(parents=True)
        recipe.write_text("test recipe")
        image = "ghcr.io/test/ci@sha256:" + "d" * 64
        manifest["environment"] = {**ENVIRONMENT, "compiler_image": image,
                                    "container_recipe_sha256": module.sha256(recipe)}
        self.write_manifest(root, manifest)
        os.environ.update(CANDLE_BUILD_ONCE_EXECUTOR="physical-sm61", CANDLE_CUDA_IMAGE=image,
                          EXTENDED_BUILD_ONCE_MANIFEST_SHA256=module.sha256(root / "manifest.json"))
        def command(args, cwd=None):
            if args[0] == "git":
                return SHA
            if args[0] == "dpkg-query":
                return ENVIRONMENT["cudnn_package"]
            raise AssertionError("unexpected compiler invocation: " + str(args))
        with mock.patch.object(module, "output", side_effect=command), \
             mock.patch.object(module, "environment", side_effect=AssertionError("compiler inspection")):
            module.load_bundle(self.repo, root)
            os.environ["CANDLE_CUDA_IMAGE"] = "ghcr.io/test/ci@sha256:" + "e" * 64
            with self.assertRaisesRegex(RuntimeError, "trusted compiler image"):
                module.load_bundle(self.repo, root)

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


class HostedPortabilityTests(unittest.TestCase):
    def test_full_docker_wrapper_keeps_the_precheck_inside_the_container(self):
        workflow = (HERE.parent / "workflows/candle-extended-composite.yml").read_text()
        match = re.search(r"(?ms)^        id: build_once\n        run: \|\n(.*?)(?=^      -)", workflow)
        self.assertIsNotNone(match)
        command = textwrap.dedent(match.group(1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "bin"
            tools.mkdir()
            docker = tools / "docker"
            docker.write_text("#!" + os.sys.executable + "\n" +
                'import json,os,sys\nfrom pathlib import Path\n'
                'Path(os.environ["TEST_DOCKER_ARGV"]).write_text(json.dumps(sys.argv[1:]))\n'
                'out=Path(os.environ["RUNNER_TEMP"])/"extended-sm61-build-once"\n'
                'out.mkdir(parents=True)\n(out/"manifest.json").write_text("{}\\n")\n')
            docker.chmod(0o755)
            env = os.environ.copy()
            env.update(PATH=str(tools) + os.pathsep + env.get("PATH", ""),
                       RUNNER_TEMP=str(root / "runner-temp"), GITHUB_WORKSPACE=str(root / "checkout"),
                       GITHUB_OUTPUT=str(root / "output"), TEST_DOCKER_ARGV=str(root / "docker-argv.json"),
                       CANDLE_CUDA_IMAGE="ghcr.io/test/compiler@sha256:" + "a" * 64)
            result = subprocess.run(["bash", "-c", command], cwd=root, env=env,
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            argv = json.loads((root / "docker-argv.json").read_text())
            self.assertEqual(argv[-3:-1], ["bash", "-c"])
            body = argv[-1]
            self.assertIn("cargo generate-lockfile", body)
            self.assertIn("export LOCK_SHA256", body)
            self.assertIn('RUSTFLAGS="-C target-cpu=x86-64" bash ', body)
            self.assertIn('candle-extended-build-once.py"', body)
            self.assertIn('build --out "$RUNNER_TEMP/extended-sm61-build-once"', body)
            self.assertEqual((root / "output").read_text(),
                             "manifest_sha256=" + hashlib.sha256(b"{}\n").hexdigest() + "\n")

    @unittest.skipUnless(shutil.which("cargo"), "real Cargo cache portability regression")
    def test_precheck_rebuilds_native_cached_build_scripts_for_portable_x86_64(self):
        workflow = (HERE.parent / "workflows/candle-extended-composite.yml").read_text()
        command = next(line.strip() for line in workflow.splitlines()
                       if line.strip().startswith("RUSTFLAGS=") and
                       "candle-candidate-cuda-compile.sh" in line)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / ".cargo").mkdir()
            (root / "Cargo.toml").write_text(
                '[package]\nname="portable-build-script-probe"\nversion="0.1.0"\nedition="2021"\n')
            (root / "src/lib.rs").write_text("pub fn probe() {}\n")
            (root / ".cargo/config.toml").write_text(
                '[build]\nrustflags=["-C", "target-cpu=native"]\n')
            (root / "build.rs").write_text(
                'fn main() { println!("cargo:warning=HOST_RUSTFLAGS={}", '
                'std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap().replace("\\u{1f}", " ")); }\n')
            tools = root / ".ci-tools/.github/scripts"
            tools.mkdir(parents=True)
            (tools / "candle-candidate-cuda-compile.sh").write_text(
                "#!/bin/bash\nset -euo pipefail\nexec cargo check --offline -vv\n")
            env = os.environ.copy()
            for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS"):
                env.pop(key, None)
            env.update(GITHUB_WORKSPACE=str(root), CARGO_TARGET_DIR=str(root / "target"))
            native = subprocess.run(["cargo", "check", "--offline", "-vv"], cwd=root,
                                    env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(native.returncode, 0, native.stderr)
            self.assertIn("HOST_RUSTFLAGS=-C target-cpu=native", native.stderr)
            portable = subprocess.run(["bash", "-c", command], cwd=root, env=env,
                                      text=True, capture_output=True, timeout=60)
            self.assertEqual(portable.returncode, 0, portable.stderr)
            self.assertIn("HOST_RUSTFLAGS=-C target-cpu=x86-64", portable.stderr)
            self.assertNotIn("HOST_RUSTFLAGS=-C target-cpu=native", portable.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
