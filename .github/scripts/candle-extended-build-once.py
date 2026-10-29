#!/usr/bin/env python3
"""Build verified SM61 executables on hosted CI and run them without Cargo on Pascal.

The build consumes exactly the candidate checkout and CPU-qualified Cargo.lock.
The run side treats downloaded artifacts as untrusted until their SHA-256,
candidate, lockfile, campaign, source manifest and environment are verified.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys


SCHEMA = 1
FEATURE_RE = re.compile(r"^[a-z0-9_-]+$")


def fail(message):
    raise RuntimeError("EXTENDED_BUILD_ONCE=FAIL " + message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def load_source(repo):
    source = repo / "candle-integration/extended.json"
    info = json.loads(source.read_text())
    features = info.get("features")
    if (
        info.get("schema_version") != 1
        or info.get("kind") != "extended-multi"
        or info.get("feature_count") != len(features or [])
        or not features
    ):
        fail("invalid Extended source manifest")
    ids = [item["feature"] for item in features]
    if len(set(ids)) != len(ids):
        fail("duplicate source")
    cuda = sorted({f for item in features for f in item["cuda_features"]})
    if not cuda or any(not FEATURE_RE.fullmatch(f) for f in cuda):
        fail("invalid CUDA feature list")
    return source, info, set(ids), ",".join(cuda)


def environment(repo):
    if os.environ.get("CUDA_COMPUTE_CAP") != "61":
        fail("CUDA_COMPUTE_CAP must be 61")
    nvcc = os.environ.get("NVCC", "nvcc")
    nvcc_version = output([nvcc, "--version"])
    match = re.search(r"release ([0-9]+\.[0-9]+)", nvcc_version)
    if not match or match.group(1) != "12.9":
        fail("expected CUDA 12.9 NVCC")
    return {
        "rustc": output(["rustc", "--version"]),
        "nvcc_release": match.group(1),
        "gcc14": output(["gcc-14", "-dumpfullversion"]),
        "cudnn_package": output(["dpkg-query", "-W", "-f=${Version}", "libcudnn9-cuda-12"]),
        "container_recipe_sha256": sha256(
            repo / ".ci-tools/.github/container/Containerfile.cuda1292-ubuntu2604"
        ),
    }


def log_cargo(repo, out, name, argv, env, wanted=()):
    """Run cargo once; collect only compiler-artifact records with real executables."""
    print("EXTENDED_BUILD_ONCE_COMPILE=" + name, flush=True)
    log = out / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    with log.open("w") as stream:
        proc = subprocess.Popen(
            argv + ["--message-format=json"],
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        for line in proc.stdout:
            stream.write(line)
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("reason") != "compiler-artifact":
                continue
            path = record.get("executable")
            if path:
                target = record["target"]["name"]
                if target in wanted:
                    if target in artifacts and artifacts[target] != path:
                        fail("ambiguous executable for " + target)
                    artifacts[target] = path
        rc = proc.wait()
    if rc != 0:
        tail = log.read_text(errors="replace").splitlines()[-40:]
        print("\n".join(tail), file=sys.stderr)
        fail(f"{name}: Cargo exit {rc}")
    missing = set(wanted) - set(artifacts)
    if missing:
        fail(f"{name}: executable not emitted: {sorted(missing)}")
    return artifacts


def stage_binary(src, out):
    # Strip only the archived copy; never mutate Cargo's incremental outputs.
    temp = out / "bin" / ("staging-" + str(os.getpid()))
    temp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, temp)
    subprocess.run(["strip", "--strip-debug", str(temp)], check=True)
    digest = sha256(temp)
    path = out / "bin" / digest
    if not path.exists():
        temp.rename(path)
    else:
        temp.unlink()
    return {"path": f"bin/{digest}", "sha256": digest, "bytes": path.stat().st_size}


def build(args):
    repo = Path.cwd().resolve()
    source, info, features, core = load_source(repo)
    actual = output(["git", "rev-parse", "HEAD"], cwd=repo)
    if actual != os.environ.get("CANDIDATE_SHA"):
        fail("candidate SHA mismatch")
    if info["campaign"] != os.environ.get("CHECK_CAMPAIGN"):
        fail("candidate campaign mismatch")
    lock = sha256(repo / "Cargo.lock")
    if lock != os.environ.get("LOCK_SHA256"):
        fail("CPU-qualified lockfile mismatch")
    out = Path(args.out).resolve()
    if out.exists():
        fail("build directory must be fresh: " + str(out))
    out.mkdir(parents=True)
    env = os.environ.copy()
    flags = env.get("RUSTFLAGS", "")
    if "target-cpu" in flags or "target-feature" in flags:
        fail("pre-existing CPU-specific RUSTFLAGS cannot be proven portable")
    env["RUSTFLAGS"] = (flags + " -C target-cpu=x86-64").strip()
    env["CARGO_INCREMENTAL"] = "0"
    # Hosted runners have substantially less RAM than the developer's Pascal
    # machine. Bound linker/NVCC fan-out to avoid OOM during release archives.
    env["CARGO_BUILD_JOBS"] = "2"
    env["CUDA_COMPUTE_CAP"] = "61"
    gates = {}

    def gate(name, kind, binaries=(), cwd=".", argv=()):
        if name in gates:
            fail("duplicate gate: " + name)
        gates[name] = {"kind": kind, "binaries": list(binaries), "cwd": cwd, "argv": list(argv)}

    def check(name, cmd):
        log_cargo(repo, out, name, ["cargo"] + cmd, env)
        gate(name, "hosted-check")

    def tests(name, package, flags, targets, filters=(), cwd=None):
        command = ["cargo", "test", "-p", package, "--release", "--locked", "--no-run"]
        command += flags
        for kind, target in targets:
            command += ["--" + kind, target] if kind != "lib" else ["--lib"]
        wanted = [target if kind != "lib" else package.replace("-", "_") for kind, target in targets]
        bins = log_cargo(repo, out, name, command, env, wanted)
        gate(name, "test", [stage_binary(bins[target], out) for target in wanted],
             cwd or package, filters)

    def example(name, package, target, flags, argv=(), is_test=False):
        command = ["cargo", "test" if is_test else "build", "-p", package,
                   "--release", "--locked", "--example", target]
        if is_test:
            command.append("--no-run")
        command += flags
        bins = log_cargo(repo, out, name, command, env, [target])
        gate(name, "test" if is_test else "example",
             [stage_binary(bins[target], out)], package if is_test else ".", argv)

    check("combined_compile", ["check", "-p", "candle-core", "-p", "candle-nn",
                               "--release", "--locked", "--features", core])
    check("downstream_transformers_compile",
          ["check", "-p", "candle-transformers", "--lib", "--release", "--locked"])
    qwen = "cuda"
    if {"bf16_candle", "fp8_candle", "fp4_candle"} <= features:
        qwen = "cuda,cuda-legacy-bf16,cuda-legacy-fp8,cuda-legacy-fp4"
    check("downstream_qwen3_cuda_compile",
          ["check", "-p", "candle-examples", "--example", "quantized-qwen3",
           "--features", qwen, "--release", "--locked"])
    if "embeddinggemma2" in features:
        check("embeddinggemma2_cuda_compile",
              ["check", "-p", "candle-examples", "--example", "embedding-gemma2",
               "--features", "cuda", "--release", "--locked"])
    if "lean4_standalone" in features:
        check("lean4_cuda_adapter_compile",
              ["check", "--manifest-path", "tools/lean4-standalone/Cargo.toml",
               "--features", "cuda", "--release"])

    if "bf16_candle" in features:
        tests("bf16", "candle-core", ["--features", core],
              [("test", "cuda_legacy_bf16_tests")], ["--nocapture"])
    if "fp8_candle" in features:
        tests("fp8", "candle-core", ["--features", core],
              [("test", target) for target in (
                  "fp8_legacy_tests", "fp8_sm61_f8_tests", "fp8_r2_upstream_canary",
                  "fp8_r4c_dispatch_tests", "fp8_r4c_loader_tests", "fp8_r4d_native_tests"
              )], ["--nocapture"])
    if "fp4_candle" in features:
        tests("fp4_mxfp4", "candle-core", ["--features", core],
              [("test", target) for target in (
                  "mxfp4_tests", "fp4_e7c_contract", "nvfp4_loader_tests"
              )], ["--nocapture"])
    if "cudnn_fallback_candle" in features:
        tests("cudnn_fallback", "candle-core", ["--features", core],
              [("lib", "")],
              ["cuda_backend::cudnn_fallback_classifier_tests::synthetic_quarantine_routes_generic_conv1d_conv2d",
               "--exact", "--nocapture"])
    if "moe_simt_f16_candle" in features:
        tests("moe_simt_f16", "candle-nn", ["--features", "cuda"],
              [("test", "moe_simt_sm61")])
    if "asd_core_v3_standalone" in features:
        for gate_name, target in (
            ("asd_module_provider", "asd_v3_module_provider_validate"),
            ("asd_runtime_profile_swap", "asd_v3_phase_d_runtime_profile_swap_validate"),
        ):
            example(gate_name, "candle-core", target, ["--features", core])

    models = {
        "fireredvad_standalone": ("fireredvad_cuda_contracts", "models::fireredvad"),
        "smart_turn_standalone": ("smart_turn_cuda_contracts", "models::smart_turn"),
        "transcribepsy_standalone": ("transcribepsy_cuda_contracts", "models::transcribepsy"),
        "nemotron_asr_streaming_standalone": ("nemotron_asr_cuda_contracts", "nemotron3_5_asr"),
        "pockettts_sm61_standalone": ("pockettts_cuda_contracts", "models::pockettts"),
    }
    # Compile the same transformer unit-test binary once for all model gates.
    if any(f in features for f in models):
        raw = log_cargo(repo, out, "voice_cuda_unit_binary",
                        ["cargo", "test", "-p", "candle-transformers", "--lib",
                         "--release", "--locked", "--features", "cuda", "--no-run"],
                        env, ["candle_transformers"])
        binary = stage_binary(raw["candle_transformers"], out)
        for feature, (name, test_filter) in models.items():
            if feature in features:
                gate(name, "test", [binary], "candle-transformers", [test_filter])
    if "transcribepsy_standalone" in features:
        example("transcribepsy_cuda_cli", "candle-examples", "transcribepsy_tp3",
                ["--features", "cuda"], is_test=True)
    if "nemotron_asr_streaming_standalone" in features:
        example("nemotron_asr_cuda_cli", "candle-examples", "nemotron_asr",
                ["--features", "cuda,cudnn"])
        gate("nemotron_asr_cuda_real_parity", "nemotron-parity",
             gates["nemotron_asr_cuda_cli"]["binaries"])
    if "pockettts_sm61_standalone" in features:
        example("pockettts_flowlm_mimi_cuda", "candle-examples", "pockettts-sm61",
                ["--features", "cuda"], argv=["--device", "cuda"])

    manifest = {
        "schema": SCHEMA,
        "candidate_sha": actual,
        "campaign": info["campaign"],
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "lock_sha256": lock,
        "source_manifest_sha256": sha256(source),
        "cuda_features": core,
        "rustflags": env["RUSTFLAGS"],
        "environment": environment(repo),
        "gates": gates,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (out / "manifest.sha256").write_text(sha256(out / "manifest.json") + "\n")
    print(f"EXTENDED_BUILD_ONCE=PASS gates={len(gates)} binaries={len(list((out/'bin').iterdir()))}")


def load_bundle(repo, root):
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or not (root / "manifest.sha256").is_file():
        fail("missing immutable bundle manifest")
    expected = (root / "manifest.sha256").read_text().strip()
    if sha256(manifest_path) != expected:
        fail("bundle manifest digest mismatch")
    # This independently travels through the trusted *hosted job output*, not
    # through the downloadable bundle. An attacker cannot replace the binary
    # and simply regenerate the accompanying manifest.sha256.
    if os.environ.get("EXTENDED_BUILD_ONCE_MANIFEST_SHA256") != expected:
        fail("hosted build attestation digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        fail("unknown bundle schema")
    actual = output(["git", "rev-parse", "HEAD"], cwd=repo)
    if actual != manifest.get("candidate_sha") or actual != os.environ.get("CANDIDATE_SHA"):
        fail("candidate SHA mismatch")
    source, info, _, core = load_source(repo)
    if info["campaign"] != manifest.get("campaign"):
        fail("campaign mismatch")
    if os.environ.get("GITHUB_RUN_ID") != manifest.get("run_id"):
        fail("run ID mismatch")
    # GitHub can rerun just the physical job without rerunning the hosted
    # producer; the verified artifact must come from this run, never a later
    # attempt. Allow only a forward retry of an identical immutable bundle.
    try:
        if int(os.environ["GITHUB_RUN_ATTEMPT"]) < int(manifest["run_attempt"]):
            fail("archive comes from a future run attempt")
    except (TypeError, KeyError, ValueError):
        fail("invalid run attempt provenance")
    if sha256(repo / "Cargo.lock") != manifest.get("lock_sha256"):
        fail("Cargo.lock mismatch")
    if sha256(source) != manifest.get("source_manifest_sha256"):
        fail("source manifest mismatch")
    if manifest.get("cuda_features") != core:
        fail("CUDA feature set mismatch")
    if "target-cpu=x86-64" not in manifest.get("rustflags", ""):
        fail("archive is not explicitly portable")
    if environment(repo) != manifest.get("environment"):
        fail("CUDA/cuDNN/Rust/toolchain or recipe mismatch")
    gates = manifest.get("gates")
    if not isinstance(gates, dict) or not gates:
        fail("empty build gate manifest")
    referenced = set()
    for name, record in gates.items():
        if not FEATURE_RE.fullmatch(name):
            fail("invalid gate name")
        if record.get("kind") not in {"hosted-check", "test", "example", "nemotron-parity"}:
            fail("unknown gate kind")
        if record.get("cwd") not in {".", "candle-core", "candle-nn",
                                     "candle-transformers", "candle-examples"}:
            fail("unexpected executable working directory")
        for binary in record.get("binaries", []):
            digest = binary["sha256"]
            if not re.fullmatch("[0-9a-f]{64}", digest) or binary["path"] != "bin/" + digest:
                fail("unsafe executable path")
            referenced.add(digest)
            path = root / binary["path"]
            if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
                fail("nonregular executable")
            if path.stat().st_size != binary["bytes"] or sha256(path) != digest:
                fail("executable digest mismatch: " + name)
    if {p.name for p in (root / "bin").iterdir()} != referenced:
        fail("unreferenced or missing executable in archive")
    return manifest


def exec_cmd(argv, cwd, log=None):
    with subprocess.Popen(
        argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace"
    ) as proc:
        stream = open(log, "w") if log else None
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                if stream:
                    stream.write(line)
            if proc.wait() != 0:
                fail("executed program returned non-zero: " + Path(argv[0]).name)
        finally:
            if stream:
                stream.close()


def execute(args):
    repo = Path.cwd().resolve()
    root = Path(args.bundle).resolve()
    manifest = load_bundle(repo, root)
    if args.action == "verify":
        print(f"EXTENDED_BUILD_ONCE_VERIFIED=PASS gates={len(manifest['gates'])}")
        return
    record = manifest["gates"].get(args.name)
    if record is None:
        fail("gate absent from qualified archive: " + args.name)
    kind = record["kind"]
    if kind == "hosted-check":
        print("HOSTED_CUDA_COMPILE=PASS " + args.name)
        return
    binaries = []
    for binary in record["binaries"]:
        path = root / binary["path"]
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        binaries.append(str(path))
    if kind == "nemotron-parity":
        model = Path(os.environ.get("CANDLE_NEMOTRON_MODEL_DIR", "/run/haricot/models/nemotron-3.5-asr"))
        wav = Path(os.environ.get("CANDLE_NEMOTRON_TEST_WAV", "/run/haricot/fixtures/cameo-fr-16k.wav"))
        report = Path(args.report) / "nemotron-real-parity"
        if report.exists():
            fail("Nemotron report directory already exists")
        report.mkdir(parents=True)
        (report / "source-commit.txt").write_text(manifest["candidate_sha"] + "\n")
        (report / "source-index-tree.txt").write_text(output(["git", "write-tree"], cwd=repo) + "\n")
        for language in ("fr-FR", "auto"):
            cpu = report / ("cpu-" + language)
            cuda = report / ("cuda-cudnn-" + language)
            cmd = [binaries[0], "--model-dir", str(model), "--input", str(wav),
                   "--mode", "streaming", "--language", language, "--lookahead", "3"]
            exec_cmd(cmd + ["--device", "cpu", "--dtype", "f32", "--output", str(cpu)],
                     repo, str(report / (f"cpu-{language}.log")))
            exec_cmd(cmd + ["--device", "cuda-cudnn", "--dtype", "f32",
                            "--expect-result", str(cpu / "result.json"), "--output", str(cuda)],
                     repo, str(report / (f"cuda-cudnn-{language}.log")))
        print("PASS: cuda-cudnn matches native CPU tokens, text and all frame decisions (fr-FR and auto).")
        return
    if kind == "test":
        for binary in binaries:
            exec_cmd([binary] + record["argv"], repo / record["cwd"])
        return
    if kind == "example":
        if len(binaries) != 1:
            fail("expected one example executable")
        exec_cmd([binaries[0]] + record["argv"], repo / record["cwd"])
        return
    fail("unrecognized gate")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--out", required=True)
    for name in ("verify", "run-gate"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", required=True)
        p.add_argument("--report")
        if name == "run-gate":
            p.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.action == "build":
        build(args)
    else:
        if args.action == "run-gate" and not args.report:
            fail("run-gate requires --report")
        execute(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
