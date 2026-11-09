#!/usr/bin/env python3
"""Fail-closed provenance gate for a previously qualified Candle Unified run.

This performs no compilation and no Git writes. The existing atomic promotion
script consumes the resulting pinned environment only if every check passes.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPOSITORY = "haricot/candle"
RUN_ID = 42518965385
RUN_ATTEMPT = 1
SOURCE_WORKFLOW_SHA = "d744bb0c544b2a0e5e270bc456a45ae2904c68ba"
SOURCE_WORKFLOW_ID = 411422568
BASE_SHA = "cfcbd7186d13271843ce7eb41d25659bd0835850"
EXTENDED_SHA = "a0d83786cb9448dc6a36942f35a3e616c06c97d9"
LEGACY_SHA = "417fce15f880868e9e6fc069a4fc0bdae514e59d"
CAMPAIGN = f"extended-{RUN_ID}-{RUN_ATTEMPT}"
EXTENDED_REF = f"integration/extended/candidate/{CAMPAIGN}"
LEGACY_REF = f"integration/legacy-cuda/candidate/{CAMPAIGN}"
SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")

REQUIRED_JOBS = (
    "Static contracts (no Cargo, no CUDA)",
    "prepare",
    "cuda_image / Independently validate CUDA environment reuse",
    "cuda_image / Resolve or publish the verified CUDA environment",
    "cpu_cuda_compile",
    "CPU Execute — CUDA-built artifacts",
    "SM61 Extended runtime",
)
REQUIRED_ARTIFACTS = (
    f"extended-preparation-{RUN_ID}",
    f"legacy-preparation-{RUN_ID}",
    f"cuda-environment-{RUN_ID}-{RUN_ATTEMPT}-candle",
    f"extended-{CAMPAIGN}-cuda-compile",
    f"extended-{CAMPAIGN}-build-once",
    f"extended-{CAMPAIGN}-cpu-execute",
    f"extended-{CAMPAIGN}-sm61",
    f"candle-extended-{CAMPAIGN}-avx2",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def api(path):
    return json.loads(command("gh", "api", "-X", "GET", path))


def remote_head(ref):
    require(ref and ref not in (".", ".."), f"unsafe ref: {ref!r}")
    require(re.fullmatch(r"[A-Za-z0-9_.\-/]+", ref) is not None, f"invalid ref: {ref!r}")
    require(".." not in ref and not ref.endswith(".lock"), f"unsafe ref: {ref!r}")
    rows = command("git", "ls-remote", "--heads", "origin", f"refs/heads/{ref}").splitlines()
    require(len(rows) == 1, f"missing or duplicate remote ref: {ref}")
    sha, returned_ref = rows[0].split("\t", 1)
    require(returned_ref == f"refs/heads/{ref}" and SHA_RE.fullmatch(sha), f"unexpected ref: {ref}")
    return sha


def verify_run(run):
    for key, value in (
        ("id", RUN_ID), ("run_attempt", RUN_ATTEMPT),
        ("head_sha", SOURCE_WORKFLOW_SHA), ("workflow_id", SOURCE_WORKFLOW_ID),
        ("head_branch", "orch/unified-candle-v1"),
        ("path", ".github/workflows/candle-unified-orchestrator.yml"),
        ("event", "push"), ("status", "completed"), ("conclusion", "success"),
    ):
        require(run.get(key) == value, f"run mismatch: {key}={run.get(key)!r}, expected {value!r}")


def verify_jobs(payload):
    jobs = payload.get("jobs")
    require(isinstance(jobs, list), "missing run jobs")
    require(payload.get("total_count") == len(jobs), "paginated job evidence is incomplete")
    for name in REQUIRED_JOBS:
        matches = [job for job in jobs if job.get("name") == name]
        require(len(matches) == 1, f"job missing or duplicated: {name}")
        job = matches[0]
        require(job.get("status") == "completed" and job.get("conclusion") == "success", f"job not PASS: {name}")
        require(job.get("run_id") == RUN_ID, f"job from different run: {name}")
    promote = [job for job in jobs if job.get("name") == "Promote Extended-qualified Legacy, Extended and standalone refs"]
    require(len(promote) == 1 and promote[0].get("conclusion") == "skipped", "source promotion already ran or was not skipped")


def verify_artifacts(payload):
    artifacts = payload.get("artifacts")
    require(isinstance(artifacts, list), "missing run artifacts")
    require(payload.get("total_count") == len(artifacts), "paginated artifacts are incomplete")
    for name in REQUIRED_ARTIFACTS:
        matches = [artifact for artifact in artifacts if artifact.get("name") == name]
        require(len(matches) == 1, f"artifact missing or duplicated: {name}")
        item = matches[0]
        origin = item.get("workflow_run") or {}
        require(not item.get("expired") and item.get("size_in_bytes", 0) > 0, f"artifact unavailable: {name}")
        require(origin.get("id") == RUN_ID and origin.get("head_sha") == SOURCE_WORKFLOW_SHA, f"artifact provenance mismatch: {name}")
        require(DIGEST_RE.fullmatch(item.get("digest") or ""), f"artifact digest invalid: {name}")


def verify_manifests(extended, legacy, configured):
    require(extended.get("kind") == "extended-multi", "not an Extended manifest")
    require(legacy.get("kind") == "legacy-cuda-five", "not a Legacy manifest")
    require(legacy.get("promotion_target") == "cuda_legacy", "Legacy target mismatch")
    for name, manifest in (("Extended", extended), ("Legacy", legacy)):
        require(manifest.get("campaign") == CAMPAIGN, f"{name} campaign mismatch")
        require(manifest.get("main_sha") == BASE_SHA, f"{name} base mismatch")
        require(manifest.get("strategy") == "exact-delta", f"{name} strategy mismatch")
        require(SHA_RE.fullmatch(manifest.get("old_integration_sha") or ""), f"{name} old integration SHA missing")
    require(extended.get("update_canonical_feature_refs") is True, "Extended standalone promotion disabled")
    features = extended.get("features")
    require(isinstance(features, list) and len(features) == 13, "expected 13 Extended standalones")
    expected = configured.get("features")
    require(isinstance(expected, list), "Extended config malformed")
    require([x.get("feature") for x in features] == [x.get("feature") for x in expected], "feature order or set differs from approved source config")
    require(len({x.get("target_ref") for x in features}) == 13, "duplicate target refs")
    for feat, config in zip(features, expected):
        feature = feat["feature"]
        require(feat.get("target_ref") not in ("main", "extended", "cuda_legacy", "orch/unified-candle-v1"), f"{feature}: forbidden target ref")
        for key in ("source_ref", "target_ref", "review_source_sha"):
            require(feat.get(key) == config.get(key), f"{feature}: {key} differs from reviewed config")
        require(feat.get("source_ref") == feat.get("target_ref"), f"{feature}: source and target not identical")
        for key in ("source_sha", "prepared_sha", "review_source_sha", "common_base_sha"):
            require(SHA_RE.fullmatch(feat.get(key) or ""), f"{feature}: invalid {key}")
        require(feat.get("common_base_sha") == BASE_SHA, f"{feature}: source is not based on qualified main")
        require(feat.get("preparation_ref") == f"prepare/extended/{CAMPAIGN}/{feature}", f"{feature}: preparation ref mismatch")
        for key in ("source_patch_sha256", "prepared_patch_sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", feat.get(key) or ""), f"{feature}: invalid {key}")
    shared = features[:5]
    require([x.get("feature") for x in legacy.get("features", [])] == [x["feature"] for x in shared], "Legacy feature set mismatch")
    for lf, ef in zip(legacy["features"], shared):
        for key in ("feature", "source_ref", "target_ref", "source_sha", "review_source_sha", "prepared_sha", "common_base_sha", "source_patch_sha256", "prepared_patch_sha256"):
            require(lf.get(key) == ef.get(key), f"shared Legacy mismatch: {lf['feature']} {key}")
        require(lf.get("preparation_ref") == f"prepare/legacy/{CAMPAIGN}/{lf['feature']}", f"Legacy preparation ref mismatch: {lf['feature']}")


def selftest():
    run = dict(id=RUN_ID, run_attempt=RUN_ATTEMPT, head_sha=SOURCE_WORKFLOW_SHA,
               workflow_id=SOURCE_WORKFLOW_ID, head_branch="orch/unified-candle-v1",
               path=".github/workflows/candle-unified-orchestrator.yml", event="push",
               status="completed", conclusion="success")
    verify_run(run)
    for bad in (dict(run, conclusion="failure"), dict(run, head_sha="0" * 40)):
        try:
            verify_run(bad)
        except ValueError:
            continue
        raise AssertionError("accepted unqualified source run")
    jobs = [dict(name=name, status="completed", conclusion="success", run_id=RUN_ID) for name in REQUIRED_JOBS]
    jobs.append(dict(name="Promote Extended-qualified Legacy, Extended and standalone refs", conclusion="skipped"))
    verify_jobs(dict(jobs=jobs, total_count=len(jobs)))
    jobs[0]["conclusion"] = "failure"
    try:
        verify_jobs(dict(jobs=jobs, total_count=len(jobs)))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted failed job")
    artifacts = [dict(name=name, expired=False, size_in_bytes=1, digest="sha256:" + "a" * 64,
                      workflow_run=dict(id=RUN_ID, head_sha=SOURCE_WORKFLOW_SHA)) for name in REQUIRED_ARTIFACTS]
    verify_artifacts(dict(artifacts=artifacts, total_count=len(artifacts)))
    artifacts[0]["expired"] = True
    try:
        verify_artifacts(dict(artifacts=artifacts, total_count=len(artifacts)))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted expired artifact")
    print("CANDLE_R2_PROVENANCE_SELFTEST=PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY, "repository mismatch")
    require(bool(os.environ.get("GH_TOKEN")), "GH_TOKEN required")
    run = api(f"repos/{REPOSITORY}/actions/runs/{RUN_ID}")
    verify_run(run)
    verify_jobs(api(f"repos/{REPOSITORY}/actions/runs/{RUN_ID}/attempts/{RUN_ATTEMPT}/jobs?per_page=100"))
    verify_artifacts(api(f"repos/{REPOSITORY}/actions/runs/{RUN_ID}/artifacts?per_page=100"))

    for ref, expected in (("main", BASE_SHA), (EXTENDED_REF, EXTENDED_SHA), (LEGACY_REF, LEGACY_SHA)):
        require(remote_head(ref) == expected, f"ref moved: {ref}")

    command("git", "fetch", "--no-tags", "origin", f"+refs/heads/{EXTENDED_REF}:refs/remotes/origin/r2-extended")
    command("git", "fetch", "--no-tags", "origin", f"+refs/heads/{LEGACY_REF}:refs/remotes/origin/r2-legacy")
    extended = json.loads(command("git", "show", f"{EXTENDED_SHA}:candle-integration/extended.json"))
    legacy = json.loads(command("git", "show", f"{LEGACY_SHA}:candle-integration/legacy-cuda.json"))
    config = json.loads(Path(".github/extended-sources.json").read_text())
    verify_manifests(extended, legacy, config)
    for item in extended["features"]:
        feature = item["feature"]
        require(remote_head(item["target_ref"]) == item["source_sha"], f"source branch advanced: {feature}")
        require(remote_head(item["preparation_ref"]) == item["prepared_sha"], f"prepared ref moved: {feature}")
        command("git", "merge-base", "--is-ancestor", item["prepared_sha"], EXTENDED_SHA)
    for item in legacy["features"]:
        # Unified assembly intentionally reuses the Extended preparation refs.
        # candle-legacy-assemble.sh does NOT publish prepare/legacy/* in this mode.
        # The Extended refs were verified above, and the candidate must contain
        # those exact same commits as ancestors.
        command("git", "merge-base", "--is-ancestor", item["prepared_sha"], LEGACY_SHA)
    require(remote_head("extended") == extended["old_integration_sha"], "Extended stable ref changed")
    require(remote_head("cuda_legacy") == legacy["old_integration_sha"], "Legacy stable ref changed")
    command("python3", "-B", ".github/scripts/candle-unified-pair-verify.py", "--selftest")

    variables = dict(CANDIDATE_SHA=EXTENDED_SHA, CANDIDATE_REF=EXTENDED_REF,
                     LEGACY_CANDIDATE_SHA=LEGACY_SHA, LEGACY_CANDIDATE_REF=LEGACY_REF,
                     CAMPAIGN=CAMPAIGN, BASE_SHA=BASE_SHA, TARGET_BRANCH="extended",
                     MANIFEST_PATH="candle-integration/extended.json", PROMOTION_KIND="extended",
                     RUN_ID=str(RUN_ID), RUN_ATTEMPT=str(RUN_ATTEMPT))
    target = os.environ.get("GITHUB_ENV")
    require(target, "GITHUB_ENV is required to transfer pinned promotion refs")
    with open(target, "a", encoding="utf-8") as f:
        for key, value in variables.items():
            f.write(f"{key}={value}\n")
    print(f"CANDLE_R2_PROOF=PASS run={RUN_ID} attempt={RUN_ATTEMPT} main={BASE_SHA} features=13 legacy=5")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"CANDLE_R2_PROOF=FAIL reason={exc}", file=sys.stderr)
        sys.exit(1)
