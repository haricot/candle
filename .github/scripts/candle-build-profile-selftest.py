#!/usr/bin/env python3
"""CPU-only regressions for explicit release targets and Cargo build reuse."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_once", HERE / "candle-extended-build-once.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class CpuProfileTests(unittest.TestCase):
    def test_native_and_unadvertised_targets_are_rejected(self):
        for profile in ("native", "skylake", "x86-64-v4", "avx512"):
            with self.subTest(profile=profile), self.assertRaises(RuntimeError):
                driver.cpu_rustflags(profile)
        for flags in ("-C target-cpu=native", "-C target-cpu=x86-64-v4",
                      "-C target-cpu=x86-64 -C target-feature=+avx512f"):
            with self.subTest(flags=flags), self.assertRaises(RuntimeError):
                driver.build_rustflags({"RUSTFLAGS": flags})
        with self.assertRaises(RuntimeError):
            driver.build_rustflags({"CARGO_ENCODED_RUSTFLAGS": "-C\x1ftarget-cpu=native"})

    def test_optimized_profile_requires_every_visible_cpu_to_support_it(self):
        flags = driver.AVX2_CPU_FLAGS
        info = "flags : " + " ".join(sorted(flags)) + "\n"
        driver.check_cpu("avx2", info, "x86_64")
        aliases = (flags - {"sse3", "lzcnt", "bmi1"}) | {"pni", "abm", "bmi"}
        driver.check_cpu("avx2", "flags : " + " ".join(aliases), "x86_64")
        with self.assertRaisesRegex(RuntimeError, "missing avx2"):
            driver.check_cpu("avx2", info + "flags : " + " ".join(flags - {"avx2"}), "x86_64")
        with self.assertRaisesRegex(RuntimeError, "cannot establish"):
            driver.check_cpu("avx2", "model name : fixture", "x86_64")
        with self.assertRaisesRegex(RuntimeError, "requires x86-64"):
            driver.check_cpu("portable", machine="aarch64")

    def test_immutable_archive_binds_profile_exactly_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            source = repo / "candle-integration/extended.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"schema_version":1, "kind":"extended-multi",
                "campaign":"profile-test", "feature_count":1,
                "features":[{"feature":"bf16_candle", "cuda_features":["cuda"]}]}))
            bundle = repo / "bundle"; bundle.mkdir(); (bundle / "bin").mkdir()
            (bundle / "Cargo.lock").write_text("fixture lock")
            binary = bundle / "bin/program"; shutil.copy2("/bin/true", binary)
            digest = driver.sha256(binary); binary.rename(bundle / "bin" / digest)
            manifest = {"schema":1, "candidate_sha":"a"*40, "campaign":"profile-test",
                "run_id":"1", "run_attempt":"1", "lock_sha256":driver.sha256(bundle / "Cargo.lock"),
                "source_manifest_sha256":driver.sha256(source), "cuda_features":"cuda",
                "rustflags":"-C target-cpu=x86-64", "environment":{},
                "gates":{"bf16":{"kind":"test", "cwd":".", "argv":[], "binaries":[{
                    "path":"bin/"+digest, "sha256":digest, "bytes":(bundle / "bin" / digest).stat().st_size}]}}}
            env={"CANDIDATE_SHA":"a"*40,"GITHUB_RUN_ID":"1","GITHUB_RUN_ATTEMPT":"1"}
            def verify():
                data=json.dumps(manifest,sort_keys=True).encode()
                (bundle / "manifest.json").write_bytes(data)
                fingerprint=hashlib.sha256(data).hexdigest()
                (bundle / "manifest.sha256").write_text(fingerprint)
                with mock.patch.dict(os.environ,{**env,"EXTENDED_BUILD_ONCE_MANIFEST_SHA256":fingerprint},clear=True), \
                     mock.patch.object(driver,"output",return_value="a"*40), \
                     mock.patch.object(driver,"environment",return_value={}):
                    return driver.load_bundle(repo,bundle)
            verify()  # Historic explicitly portable archives remain valid.
            manifest["rustflags"]="-C target-cpu=x86-64-v4"
            with self.assertRaisesRegex(RuntimeError,"CPU profile/flags mismatch"):verify()
            manifest.update(cpu_profile="avx2",rustflags="-C target-cpu=x86-64-v3")
            with mock.patch.object(driver,"check_cpu") as cpu:
                verify(); cpu.assert_called_once_with("avx2")
            with mock.patch.object(driver,"check_cpu",side_effect=RuntimeError("unsupported CPU")), \
                 self.assertRaisesRegex(RuntimeError,"unsupported CPU"):
                verify()

    @unittest.skipUnless(shutil.which("rustc"), "requires the real Rust compiler")
    def test_real_rust_v3_features_fit_supplied_cpu_capabilities(self):
        result=subprocess.run(["rustc","--print=cfg","-C","target-cpu=x86-64-v3"],
                              check=True,text=True,capture_output=True)
        features={line.split('"')[1] for line in result.stdout.splitlines()
                  if line.startswith('target_feature="')}
        supplied=set("adx aes avx avx2 bmi1 bmi2 cmpxchg16b f16c fma fxsr lzcnt movbe "
                     "pclmulqdq popcnt rdrand rdseed sse sse2 sse3 sse4.1 sse4.2 ssse3 "
                     "xsave xsavec xsaveopt xsaves".split())
        self.assertTrue({"avx2","fma","f16c","bmi1","bmi2"} <= features)
        self.assertTrue(features <= supplied, "unexpected CPU requirements: "+str(features-supplied))
        self.assertFalse(any(f.startswith("avx512") for f in features))

    @unittest.skipUnless(shutil.which("cargo"), "requires real Cargo build fingerprints")
    def test_real_release_precheck_and_build_reuse_the_build_script(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root / "src").mkdir(); (root / ".cargo").mkdir()
            (root / "Cargo.toml").write_text('[package]\nname="release-reuse-probe"\nversion="0.1.0"\nedition="2021"\n')
            (root / "src/lib.rs").write_text('pub fn add(a:f32,b:f32)->f32 { a+b }\n')
            (root / ".cargo/config.toml").write_text('[build]\nrustflags=["-C","target-cpu=native"]\n')
            (root / "build.rs").write_text('''use std::{env,fs,path::PathBuf};
fn main() {
 let p=PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("build-count");
 let n=fs::read_to_string(&p).unwrap_or("0".into()).parse::<u32>().unwrap();
 fs::write(p,(n+1).to_string()).unwrap();
 println!("cargo:warning=CPU_FLAGS={}",env::var("CARGO_ENCODED_RUSTFLAGS").unwrap().replace('\\u{1f}'," "));
}
''')
            env={k:v for k,v in os.environ.items() if k not in
                 {"RUSTFLAGS","CARGO_ENCODED_RUSTFLAGS","CARGO_BUILD_RUSTFLAGS"}}
            env.update(RUSTFLAGS=driver.cpu_rustflags("portable"),CARGO_INCREMENTAL="0",
                       CARGO_TARGET_DIR=str(root / "target"))
            for verb in ("check","build","build"):
                result=subprocess.run(["cargo",verb,"--release","--offline","-vv"],cwd=root,env=env,
                                      text=True,capture_output=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn("CPU_FLAGS=-C target-cpu=x86-64",result.stderr)
                self.assertEqual((root / "build-count").read_text(),"1",result.stderr)
            # Changing the CPU profile really changes the compilation identity.
            env["RUSTFLAGS"]=driver.cpu_rustflags("avx2")
            driver.check_cpu("avx2")
            result=subprocess.run(["cargo","build","--release","--offline"],cwd=root,env=env,
                                  text=True,capture_output=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((root / "build-count").read_text(),"2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
