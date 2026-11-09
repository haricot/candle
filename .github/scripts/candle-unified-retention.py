#!/usr/bin/env python3
"""Prune only refs backed by a successful, verified Unified promotion.

Keep one rolling pre-promotion branch per canonical target.  All writes use
atomic pushes and exact-SHA leases; an unproved historical ref is left alone.
No CUDA build or additional qualification is performed by this script.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SHA = re.compile(r"^[0-9a-f]{40}$")
CAMPAIGN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
PREVIOUS_PREFIX = "archive/unified/previous/"
PRUNABLE_PREFIXES = (
    "prepare/extended/", "prepare/legacy/",
    "integration/extended/candidate/", "integration/legacy-cuda/candidate/",
    "archive/promotion/", "archive/extended/sm61-validated-",
    "archive/legacy/sm61-validated-",
)
DUPLICATE_TEMP_PREFIXES = PRUNABLE_PREFIXES[:4]


def require(condition, why):
    if not condition:
        raise ValueError(why)


def sha(value, label):
    require(isinstance(value, str) and SHA.fullmatch(value), f"invalid {label} SHA")
    return value


def validate_ref(name):
    require(isinstance(name, str) and name and not name.startswith("-") and ".." not in name,
            f"invalid ref: {name!r}")
    result = subprocess.run(["git", "check-ref-format", f"refs/heads/{name}"], capture_output=True)
    require(result.returncode == 0, f"invalid git branch: {name!r}")
    return name


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def remote_heads():
    found = {}
    for line in git("ls-remote", "--heads", "origin").splitlines():
        commit, ref = line.split("\t", 1)
        require(ref.startswith("refs/heads/"), "unexpected remote ref")
        found[ref[len("refs/heads/"):]] = sha(commit, ref)
    return found


def open_pr_refs(repository):
    # A failed or truncated PR query must stop cleanup, not relax protection.
    out = subprocess.check_output([
        "gh", "api", "--paginate", "--jq", ".[] | [.head.ref, .base.ref] | @tsv",
        f"repos/{repository}/pulls?state=open&per_page=100",
    ], text=True)
    return {ref for line in out.splitlines() for ref in line.split("\t") if ref}


def validated_inputs(report_dir, env):
    campaign = env["CAMPAIGN"]
    require(CAMPAIGN.fullmatch(campaign), "unsafe campaign name")
    summary = (report_dir / "summary.txt").read_text()
    lines = summary.splitlines()
    for exact in ("PROMOTION=PASS", f"CAMPAIGN={campaign}",
                  f"CANDIDATE_SHA={sha(env['CANDIDATE_SHA'], 'Extended candidate')}",
                  "TARGET_BRANCH=extended"):
        require(lines.count(exact) == 1, f"missing or repeated promotion proof: {exact}")
    extended = json.loads((report_dir / "manifest.json").read_text())
    legacy = json.loads((report_dir / "legacy-manifest.json").read_text())
    for name, manifest in (("Extended", extended), ("Legacy", legacy)):
        require(manifest.get("campaign") == campaign, f"{name} campaign mismatch")
        require(manifest.get("main_sha") == env["BASE_SHA"], f"{name} main mismatch")
    require(extended.get("promotion_target", extended.get("integration_target")) == "extended",
            "not an Extended integration manifest")
    require(legacy.get("promotion_target", legacy.get("integration_target")) == "cuda_legacy",
            "not a Legacy integration manifest")
    require(extended.get("update_canonical_feature_refs") is True,
            "R3 retention requires promoted standalone refs")
    sha(env["LEGACY_CANDIDATE_SHA"], "Legacy candidate")
    sha(env["BASE_SHA"], "qualified main")
    return extended, legacy


def build_plan(heads, protected_pr_refs, extended, legacy, env):
    """Pure decision logic: refuse stale promotion, preserve unproved history."""
    campaign = env["CAMPAIGN"]
    run_id = str(env["RUN_ID"])
    attempt = str(env["RUN_ATTEMPT"])
    require(CAMPAIGN.fullmatch(campaign), "unsafe campaign")
    require(run_id.isdecimal() and attempt.isdecimal(), "unsafe run identity")
    require(env["CANDIDATE_REF"] == f"integration/extended/candidate/{campaign}",
            "unbound Extended candidate ref")
    require(env["LEGACY_CANDIDATE_REF"] == f"integration/legacy-cuda/candidate/{campaign}",
            "unbound Legacy candidate ref")

    expected_targets = {"extended": sha(env["CANDIDATE_SHA"], "Extended candidate"),
                        "cuda_legacy": sha(env["LEGACY_CANDIDATE_SHA"], "Legacy candidate")}
    previous = {}
    pending = {}
    previous["extended"] = sha(extended.get("old_integration_sha"), "previous Extended")
    previous["cuda_legacy"] = sha(legacy.get("old_integration_sha"), "previous Legacy")

    features = extended.get("features")
    require(isinstance(features, list) and features, "missing Extended features")
    for item in features:
        target = validate_ref(item["target_ref"])
        require(target not in expected_targets, f"duplicate or reserved target {target}")
        expected_targets[target] = sha(item["prepared_sha"], f"prepared {target}")
        previous[target] = sha(item["source_sha"], f"previous {target}")
        preparation = validate_ref(item["preparation_ref"])
        require(preparation == f"prepare/extended/{campaign}/{item['feature']}",
                f"unexpected Extended preparation ref {preparation}")
        pending[preparation] = expected_targets[target]

    legacy_features = legacy.get("features")
    require(isinstance(legacy_features, list) and legacy_features, "missing Legacy features")
    for item in legacy_features:
        target = validate_ref(item["target_ref"])
        require(target in expected_targets and expected_targets[target] == sha(item["prepared_sha"], target),
                f"Legacy and Extended preparation diverged for {target}")
        preparation = validate_ref(item["preparation_ref"])
        require(preparation == f"prepare/legacy/{campaign}/{item['feature']}",
                f"unexpected Legacy preparation ref {preparation}")
        pending[preparation] = expected_targets[target]

    for target, value in expected_targets.items():
        require(heads.get(target) == value,
                f"canonical target moved or was not promoted: {target} (expected {value}, got {heads.get(target)})")

    pending[env["CANDIDATE_REF"]] = expected_targets["extended"]
    pending[env["LEGACY_CANDIDATE_REF"]] = expected_targets["cuda_legacy"]
    pending[f"archive/extended/sm61-validated-{run_id}-{attempt}"] = expected_targets["extended"]
    pending[f"archive/legacy/sm61-validated-{run_id}-{attempt}"] = expected_targets["cuda_legacy"]

    # Promotion created per-campaign backups.  A missing backup on retry is fine
    # only when the rolling previous ref already contains the same SHA.
    updates = {}
    for target, old in previous.items():
        rolling = validate_ref(PREVIOUS_PREFIX + target)
        archive = validate_ref(f"archive/promotion/{campaign}/before/{target}")
        if heads.get(archive) is not None:
            require(heads[archive] == old, f"old preimage archive moved: {archive}")
            pending[archive] = old
        else:
            require(heads.get(rolling) == old, f"missing last preimage evidence: {archive}")
        if old != expected_targets[target] and heads.get(rolling) != old:
            updates[rolling] = old
    require(not (set(updates) & protected_pr_refs),
            "an open PR protects a rolling previous-state branch")

    # Older preparation/integration duplicates may be retired by SHA identity.
    # Historical archive *names* may still be used as audit evidence even if
    # the same object is reachable; only this campaign's archives are removed.
    reachable = set(expected_targets.values()) | set(previous.values())
    for ref, current in heads.items():
        if (ref.startswith(DUPLICATE_TEMP_PREFIXES) and ref not in pending
                and current in reachable):
            pending[ref] = current

    deletions = {}
    skipped = []
    for ref, commit in sorted(pending.items()):
        validate_ref(ref)
        if ref not in heads:
            continue  # Idempotent cleanup or unused Legacy preparation ref.
        require(heads[ref] == commit, f"temporary ref changed unexpectedly: {ref}")
        if ref in protected_pr_refs:
            skipped.append({"ref": ref, "reason": "OPEN_PR_HEAD_OR_BASE"})
        else:
            deletions[ref] = commit

    # No rolling retention branch is ever deleted, even by the duplicate sweep.
    require(not (set(updates) & set(deletions)), "conflicting retention operations")
    return {"canonical": expected_targets, "previous": previous, "updates": updates,
            "deletions": deletions, "skipped": skipped,
            "unproved_remaining": sorted(ref for ref in heads if ref.startswith(PRUNABLE_PREFIXES)
                                         and ref not in deletions)}


def apply_plan(plan, expected_heads, repository):
    # Recheck stable heads and PRs immediately before the one atomic transaction.
    latest = remote_heads()
    for target, commit in plan["canonical"].items():
        require(latest.get(target) == commit, f"stable target moved before cleanup: {target}")
    active = open_pr_refs(repository)
    for ref in set(plan["deletions"]) | set(plan["updates"]):
        require(ref not in active, f"new PR protects {ref}")
    specs = []
    leases = []
    for ref, commit in sorted(plan["updates"].items()):
        require(latest.get(ref) == expected_heads.get(ref), f"rolling ref moved: {ref}")
        specs.append(f"+{commit}:refs/heads/{ref}")
        leases.append(f"--force-with-lease=refs/heads/{ref}:{expected_heads.get(ref, '')}")
    for ref, commit in sorted(plan["deletions"].items()):
        require(latest.get(ref) == commit, f"deletion ref moved: {ref}")
        specs.append(f":refs/heads/{ref}")
        leases.append(f"--force-with-lease=refs/heads/{ref}:{commit}")
    if specs:
        git("push", "--atomic", *leases, "origin", *specs)
    after = remote_heads()
    for target, commit in plan["canonical"].items():
        require(after.get(target) == commit, f"stable target moved during cleanup: {target}")
    for ref, commit in plan["updates"].items():
        require(after.get(ref) == commit, f"rolling archive postcheck failed: {ref}")
    for ref in plan["deletions"]:
        require(ref not in after, f"deleted ref appeared after push: {ref}")
    return len(specs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="opt-in mutating cleanup after PROMOTION=PASS")
    args = parser.parse_args()
    required = ("GITHUB_REPOSITORY", "CAMPAIGN", "CANDIDATE_SHA", "CANDIDATE_REF",
                "LEGACY_CANDIDATE_SHA", "LEGACY_CANDIDATE_REF", "BASE_SHA", "RUN_ID", "RUN_ATTEMPT")
    env = {name: os.environ[name] for name in required}
    extended, legacy = validated_inputs(args.report_dir, env)
    current = remote_heads()
    protected = open_pr_refs(env["GITHUB_REPOSITORY"])
    plan = build_plan(current, protected, extended, legacy, env)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    record = args.report_dir / "retention.json"
    report = {"schema": "candle-unified-retention-r3.v1", "campaign": env["CAMPAIGN"],
              "run": env["RUN_ID"], "attempt": env["RUN_ATTEMPT"],
              "mode": "apply" if args.apply else "dry-run", **plan}
    report["status"] = "PLANNED"
    record.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"UNIFIED_RETENTION=PLAN updates={len(plan['updates'])} deletes={len(plan['deletions'])} "
          f"open_pr_skips={len(plan['skipped'])} unproved={len(plan['unproved_remaining'])}")
    if args.apply:
        try:
            count = apply_plan(plan, current, env["GITHUB_REPOSITORY"])
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            report["status"] = "FAIL_SAFE"
            report["reason"] = str(exc)
            record.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            raise
        report["status"] = "PASS"
        record.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"UNIFIED_RETENTION=PASS applied={count} previous_states={len(plan['previous'])}")
    else:
        report["status"] = "DRY_RUN"
        record.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("UNIFIED_RETENTION=DRY_RUN no refs modified")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(f"\n## Unified R3 branch retention ({'apply' if args.apply else 'dry-run'})\n\n")
            summary.write(f"- Rolling previous states: {len(plan['previous'])}\n")
            summary.write(f"- Planned deletions: {len(plan['deletions'])}\n")
            summary.write(f"- Protected by PR: {len(plan['skipped'])}\n")
            summary.write(f"- Unproved historical refs retained: {len(plan['unproved_remaining'])}\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        print(f"UNIFIED_RETENTION=FAIL_SAFE reason={exc}", file=sys.stderr)
        sys.exit(1)
