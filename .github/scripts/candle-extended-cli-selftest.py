#!/usr/bin/env python3
"""Offline regressions for immutable CLI sources and the single-file release."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load("cli_installer", "install-candle-extended-cli.py")
bundle = load("cli_bundle", "candle-extended-build-once.py")


class SingleApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        for name, result, parser in installer.ENTRIES:
            path = self.repo / "candle-examples/examples" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            body = "let args = Args::parse();" if parser == "clap" else (
                "let args: Vec<_> = std::env::args().skip(1).collect();")
            path.write_text(f"fn main() -> {result} {{\n{body}\nOk(())\n}}\n")

    def install(self):
        installer.install(self.repo)
        receipt = self.repo / "candle-integration/extended-cli.json"
        info = {"release_program": {"name": "candle-extended",
                                   "source_receipt_sha256": bundle.sha256(receipt)}}
        return info

    def test_sources_are_bound_and_changed_model_entrypoint_is_rejected(self):
        info = self.install()
        bundle.verify_cli_source(self.repo, info)
        path = self.repo / "candle-examples/examples/nemotron_asr.rs"
        self.assertIn("Args::parse_from(argv)", path.read_text())
        path.write_text(path.read_text() + "// different candidate\n")
        with self.assertRaisesRegex(RuntimeError, "source digest mismatch"):
            bundle.verify_cli_source(self.repo, info)

    def test_installation_failure_leaves_every_original_source_unchanged(self):
        path = self.repo / "candle-examples/examples/pockettts-sm61/main.rs"
        path.write_text(path.read_text().replace("std::env::args().skip(1).collect()", "unknown_parser()"))
        before = {p: p.read_bytes() for p in self.repo.rglob("*.rs")}
        with self.assertRaisesRegex(ValueError, "argument parser"):
            installer.install(self.repo)
        self.assertEqual(before, {p: p.read_bytes() for p in self.repo.rglob("*.rs")})

    def test_reapplying_overlay_requires_fresh_candidate(self):
        self.install()
        with self.assertRaisesRegex(ValueError, "fresh candidate"):
            installer.install(self.repo)

    def test_missing_model_entrypoint_in_receipt_is_rejected(self):
        self.install()
        path = self.repo / "candle-integration/extended-cli.json"
        data = json.loads(path.read_text())
        data["files"].pop(next(iter(data["files"])))
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "source set mismatch"):
            bundle.verify_cli_source(self.repo, {"release_program": {
                "name": "candle-extended", "source_receipt_sha256": bundle.sha256(path)}})

    def release_bundle(self):
        root = self.repo / "bundle"
        root.mkdir()
        (root / "manifest.json").write_text("immutable producer manifest\n")
        binary = root / "bin" / "executable"
        binary.parent.mkdir()
        shutil.copyfile("/bin/true", binary)
        digest = bundle.sha256(binary)
        binary.rename(root / "bin" / digest)
        manifest = {key: "selftest" for key in (
            "candidate_sha", "campaign", "run_id", "run_attempt", "lock_sha256",
            "source_manifest_sha256", "cuda_features")}
        manifest.update(cpu_profile="avx2", rustflags="-C target-cpu=x86-64-v3",
                        environment={"compiler_image": "immutable SDK"}, release_program={
                            "name": "candle-extended", "path": "bin/" + digest,
                            "sha256": digest, "bytes": (root / "bin" / digest).stat().st_size})
        return root, manifest

    def test_download_artifact_contains_one_elf_with_same_producer_hash(self):
        root, manifest = self.release_bundle()
        destination = self.repo / "release"
        bundle.export_release(self.repo, root, manifest, destination)
        self.assertEqual({p.name for p in destination.iterdir()},
                         {"candle-extended", "candle-extended.sha256", "release.json"})
        programs = [p for p in destination.iterdir() if p.read_bytes().startswith(b"\x7fELF")]
        self.assertEqual([p.name for p in programs], ["candle-extended"])
        digest = manifest["release_program"]["sha256"]
        self.assertEqual(bundle.sha256(programs[0]), digest)
        self.assertEqual((destination / "candle-extended.sha256").read_text(),
                         digest + "  candle-extended\n")
        data = json.loads((destination / "release.json").read_text())
        self.assertEqual(data["sha256"], digest)
        self.assertEqual(data["rustflags"], "-C target-cpu=x86-64-v3")
        self.assertNotEqual(data["qualification"], "GPU_PASSED")

    def test_changed_release_bytes_are_rejected(self):
        root, manifest = self.release_bundle()
        with (root / manifest["release_program"]["path"]).open("ab") as stream:
            stream.write(b"tampered executable")
        with self.assertRaisesRegex(RuntimeError, "artifact executable digest mismatch"):
            bundle.export_release(self.repo, root, manifest, self.repo / "release")

    def test_old_test_only_bundle_cannot_be_exported_as_application(self):
        root, manifest = self.release_bundle()
        del manifest["release_program"]
        with self.assertRaisesRegex(RuntimeError, "no single release executable"):
            bundle.export_release(self.repo, root, manifest, self.repo / "release")


if __name__ == "__main__":
    unittest.main(verbosity=2)
