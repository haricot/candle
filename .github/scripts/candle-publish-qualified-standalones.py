#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
JOBS = {"prepare", "cuda_image / Independently validate CUDA environment reuse",
        "cuda_image / Resolve or publish the verified CUDA environment", "cpu_cuda_compile",
        "CPU Execute \u2014 CUDA-built artifacts", "SM61 Extended runtime", "Promote validated Extended refs"}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", "repos/haricot/candle/" + path]))


def check_run(spec, run, jobs, artifacts):
    if (run.get("id") != spec["source_run"] or run.get("head_sha") != spec["orchestrator"]
            or run.get("head_branch") != "orch/extended-v1" or run.get("run_attempt") != 1
            or run.get("path") != ".github/workflows/candle-extended-composite.yml"
            or run.get("status") != "completed" or run.get("conclusion") != "success"):
        raise ValueError("source campaign is not the exact complete successful qualification")
    if (jobs.get("total_count") != len(jobs["jobs"]) or {j["name"] for j in jobs["jobs"]} != JOBS
            or any(j.get("run_id") != spec["source_run"] or j.get("run_attempt") != 1
                   or j.get("status") != "completed" or j.get("conclusion") != "success" for j in jobs["jobs"])):
        raise ValueError("all original preparation/CPU/CUDA/GPU/promotion jobs must have passed")
    physical = next(j for j in jobs["jobs"] if j["name"] == "SM61 Extended runtime")
    if not {"self-hosted", "cuda", "sm61", "candle-extended", "extended-" + str(spec["source_run"])}.issubset(physical.get("labels", [])):
        raise ValueError("dedicated physical SM61 labels are missing")
    by_id = {a["id"]: a for a in artifacts["artifacts"]}
    for item in spec["artifacts"].values():
        remote = by_id.get(item["id"], {})
        if (remote.get("name") != item["name"] or remote.get("expired") is not False
                or remote.get("digest") != "sha256:" + item["sha256"]
                or remote.get("workflow_run", {}).get("id") != spec["source_run"]):
            raise ValueError("source artifact identity or SHA256 changed")


def verify_files(spec, directory, source):
    directory = Path(directory)
    archives = {}
    try:
        for kind, record in spec["artifacts"].items():
            path = directory / (str(record["id"]) + ".zip")
            if sha(path.read_bytes()) != record["sha256"]:
                raise ValueError("artifact ZIP SHA256 mismatch: " + kind)
            archives[kind] = zipfile.ZipFile(path)
        manifest = json.loads(source)
        if archives["promotion"].read("manifest.json") != source:
            raise ValueError("original promotion manifest differs from the immutable candidate")
        build_bytes = archives["build"].read("manifest.json")
        build = json.loads(build_bytes)
        if (manifest.get("campaign") != spec["campaign"] or manifest.get("main_sha") != spec["main"]
                or manifest.get("feature_count") != len(manifest["features"])
                or len(manifest["features"]) != 13 or manifest.get("integration_target") != "extended"
                or build.get("candidate_sha") != spec["candidate"] or build.get("campaign") != spec["campaign"]
                or build.get("run_id") != str(spec["source_run"]) or build.get("run_attempt") != "1"
                or build.get("source_manifest_sha256") != sha(source)
                or build.get("cpu_profile") != "avx2" or build.get("rustflags") != "-C target-cpu=x86-64-v3"
                or build.get("environment", {}).get("compiler_image") != spec["compiler_image"]
                or sha(archives["build"].read("Cargo.lock")) != build.get("lock_sha256")):
            raise ValueError("candidate/build/lock/environment identity mismatch")
        for gate in build["gates"].values():
            for binary in gate["binaries"]:
                if (not re.fullmatch(r"bin/[a-f0-9]{64}", binary["path"])
                        or sha(archives["build"].read(binary["path"])) != binary["sha256"]):
                    raise ValueError("hosted executable SHA256 mismatch")
        cpu = json.loads(archives["cpu"].read("report.json"))
        compile_proof = json.loads(archives["compile"].read("report.json"))
        for proof in (cpu, compile_proof):
            if (proof.get("candidate_sha") != spec["candidate"] or proof.get("campaign") != spec["campaign"]
                    or proof.get("lock_sha256") != build["lock_sha256"] or proof.get("gpu_executed") is not False):
                raise ValueError("hosted CPU/CUDA proof identity mismatch")
        if (cpu.get("status") != "CPU_CUDA_BUILT_PASSED" or cpu.get("cargo_invoked") is not False
                or cpu.get("manifest_sha256") != sha(build_bytes)
                or compile_proof.get("status") != "CUDA_COMPILE_PASSED" or compile_proof.get("exit_code") != 0
                or compile_proof.get("target") != "sm_61"):
            raise ValueError("hosted CPU/CUDA proof is incomplete")
        summary = set(archives["gpu"].read("summary.txt").decode().splitlines())
        required = {"EXTENDED_SM61_VALIDATION=PASS", "EXTENDED_SM61_CANDIDATE=" + spec["candidate"],
                    "EXTENDED_SM61_CAMPAIGN=" + spec["campaign"]}
        required.update("GATE_" + name + "=PASS" for name in build["gates"] if not name.endswith("_cpu_contracts"))
        rows = list(csv.reader(archives["gpu"].read("gpu.txt").decode().splitlines(), skipinitialspace=True))
        if not required.issubset(summary) or len(rows) != 1 or rows[0][1] != "6.1" or not rows[0][3].startswith("GPU-"):
            raise ValueError("physical SM61 qualification is incomplete")
        if "PROMOTION=PASS" not in archives["promotion"].read("summary.txt").decode().splitlines():
            raise ValueError("original Extended promotion did not pass")
        return manifest
    finally:
        for archive in archives.values():
            archive.close()


def remote_heads(repo):
    return {ref.removeprefix("refs/heads/"): oid for oid, ref in
            (line.split() for line in git(repo, "ls-remote", "--heads", "origin").decode().splitlines())}


def publication_plan(repo, spec, manifest, heads=None):
    heads = remote_heads(repo) if heads is None else heads
    if heads.get("main") != spec["main"] or heads.get("extended") != spec["candidate"]:
        raise ValueError("main or the already-qualified Extended branch moved")
    changes, leases, archives, targets = [], [], [], {}
    for item in manifest["features"]:
        target, source, prepared = item["target_ref"], item["source_sha"], item["prepared_sha"]
        subprocess.run(["git", "check-ref-format", "refs/heads/" + target], check=True)
        if target in targets or target in {"main", "extended"}:
            raise ValueError("duplicate or invalid standalone target")
        targets[target] = prepared
        if heads.get(target) not in (source, prepared) or heads.get(item["preparation_ref"]) != prepared:
            raise ValueError("standalone source/preparation lease changed: " + target)
        for ancestor, descendant in ((spec["main"], prepared), (prepared, spec["candidate"])):
            subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant], check=True)
        parents = git(repo, "rev-list", "--parents", "-n", "1", prepared).decode().split()
        if parents != [prepared, spec["main"]]:
            raise ValueError("standalone is not an isolated exact delta on the qualified main")
        patch = git(repo, "diff", "--binary", "--full-index", spec["main"], prepared)
        if sha(patch) != item["prepared_patch_sha256"]:
            raise ValueError("prepared standalone delta SHA256 mismatch")
        archive = "archive/promotion/" + spec["campaign"] + "/before/" + target
        if archive in heads and heads[archive] != source:
            raise ValueError("original standalone archive changed: " + target)
        if archive not in heads:
            archives.append(source + ":refs/heads/" + archive)
        if heads[target] == source:
            changes.append(prepared + ":refs/heads/" + target)
            leases.append("--force-with-lease=refs/heads/" + target + ":" + source)
    return changes, leases, archives, targets


def promote(repo, spec, manifest):
    changes, leases, archives, targets = publication_plan(repo, spec, manifest)
    if archives:
        subprocess.run(["git", "-C", str(repo), "push", "--atomic", "origin", *archives], check=True)
    if changes:
        subprocess.run(["git", "-C", str(repo), "push", "--atomic", *leases, "origin", *changes], check=True)
    final = remote_heads(repo)
    if any(final.get(name) != oid for name, oid in targets.items()) or final.get("extended") != spec["candidate"]:
        raise ValueError("post-publication standalone/Extended verification failed")
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / "standalone-publication.json").read_text())
    run = api("actions/runs/" + str(spec["source_run"]))
    jobs = api("actions/runs/" + str(spec["source_run"]) + "/jobs?per_page=100")
    artifacts = api("actions/runs/" + str(spec["source_run"]) + "/artifacts?per_page=100")
    check_run(spec, run, jobs, artifacts)
    args.output.mkdir(parents=True, exist_ok=True)
    for item in spec["artifacts"].values():
        with (args.output / (str(item["id"]) + ".zip")).open("wb") as stream:
            subprocess.run(["gh", "api", "repos/haricot/candle/actions/artifacts/" + str(item["id"]) + "/zip"], check=True, stdout=stream)
    subprocess.run(["git", "-C", str(args.repo), "fetch", "--no-tags", "origin", spec["candidate"]], check=True)
    source = git(args.repo, "show", spec["candidate"] + ":candle-integration/extended.json")
    manifest = verify_files(spec, args.output, source)
    receipt = {"source_qualification": spec, "original_source_policy": manifest["update_canonical_feature_refs"],
               "publication_policy": "publish_each_qualified_standalone_on_main", "compiled": False,
               "gpu_executed_by_publisher": False, "candidate": spec["candidate"], "main": spec["main"]}
    receipt["standalones"] = promote(args.repo, spec, manifest)
    receipt["status"] = "PASS"
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("QUALIFIED_STANDALONES_PUBLISHED=PASS count=" + str(len(receipt["standalones"])))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        raise SystemExit("QUALIFIED_STANDALONE_PUBLICATION_REJECT=" + str(error))
