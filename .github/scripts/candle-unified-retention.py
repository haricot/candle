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
import tempfile

SHA = re.compile(r"^[0-9a-f]{40}$")
CAMPAIGN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
PREVIOUS_PREFIX = "archive/unified/previous/"
VALIDATED_PREFIXES = ("archive/extended/sm61-validated-", "archive/legacy/sm61-validated-")
PRUNABLE_PREFIXES = (
    "prepare/extended/", "prepare/legacy/",
    "integration/extended/candidate/", "integration/legacy-cuda/candidate/",
    "archive/promotion/",
)
HISTORY_TAG_PREFIX = "candle-retention/"


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


def other_active_runs(repository, current_run_id):
    """Fail closed if another Actions campaign might still need temporary refs."""
    active = set()
    for status in ("queued", "in_progress", "waiting", "requested", "pending"):
        out = subprocess.check_output([
            "gh", "api", "--paginate", "--jq", ".workflow_runs[] | .id",
            f"repos/{repository}/actions/runs?status={status}&per_page=100",
        ], text=True)
        for line in out.splitlines():
            require(line.isdecimal(), "invalid active GitHub Actions run evidence")
            if line != str(current_run_id):
                active.add(line)
    return sorted(active)


def retained_graph(roots):
    """Exact commit ancestry of refs that will survive this transaction."""
    commits = sorted(set(roots))
    require(commits and all(SHA.fullmatch(commit) for commit in commits),
            "invalid preserved commit roots")
    # A missing object, shallow clone or truncated output must block pruning.
    output = subprocess.check_output(["git", "rev-list", "--stdin"],
                                     input="\n".join(commits) + "\n", text=True)
    ancestors = set(output.splitlines())
    require(set(commits).issubset(ancestors), "incomplete retained commit graph")
    return ancestors


def history_tag(campaign):
    require(CAMPAIGN.fullmatch(campaign), "unsafe retention history campaign")
    return HISTORY_TAG_PREFIX + campaign


def remote_tag_exists(tag):
    return bool(git("ls-remote", "--refs", "--tags", "origin", f"refs/tags/{tag}"))


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


def build_plan(heads, protected_pr_refs, extended, legacy, env,
               reachable_commits=None, active_run_ids=()):
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
    # SM61 qualification refs are permanent evidence, including this campaign.
    # They must never be swept, even when identical to the promoted integration.
    for kind, target in (("extended", "extended"), ("legacy", "cuda_legacy")):
        proof = f"archive/{kind}/sm61-validated-{run_id}-{attempt}"
        require(heads.get(proof) == expected_targets[target],
                f"missing preserved SM61 qualification ref: {proof}")

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

    # Only commit objects reachable from refs that survive the entire atomic
    # transaction may lose a temporary branch name. No history is reconstructed
    # from timestamps or GitHub's garbage-collection assumptions.
    survivors = {ref: value for ref, value in heads.items()
                 if (not ref.startswith(PRUNABLE_PREFIXES) or ref in protected_pr_refs)}
    survivors.update({PREVIOUS_PREFIX + target: old for target, old in previous.items()})
    reachable = set(survivors.values())
    if reachable_commits is not None:
        reachable.update(reachable_commits)
    for ref, current in heads.items():
        if (ref.startswith(PRUNABLE_PREFIXES) and ref not in pending
                and current in reachable):
            pending[ref] = current

    deletions = {}
    skipped = []
    for ref, commit in sorted(pending.items()):
        validate_ref(ref)
        if ref not in heads:
            continue  # Idempotent cleanup or unused Legacy preparation ref.
        require(heads[ref] == commit, f"temporary ref changed unexpectedly: {ref}")
        if ref in protected_pr_refs or active_run_ids:
            reason = "OPEN_PR_HEAD_OR_BASE" if ref in protected_pr_refs else "OTHER_ACTIVE_CAMPAIGN"
            skipped.append({"ref": ref, "reason": reason})
        else:
            deletions[ref] = commit

    # No rolling retention branch is ever deleted, even by the duplicate sweep.
    require(not (set(updates) & set(deletions)), "conflicting retention operations")
    require(not any(ref.startswith(VALIDATED_PREFIXES) for ref in deletions),
            "SM61 qualification evidence must never be deleted")
    protected_skips = {item["ref"] for item in skipped}
    return {"canonical": expected_targets, "previous": previous, "updates": updates,
            "deletions": deletions, "skipped": skipped,
            "proof_roots": {ref: value for ref, value in heads.items()
                            if ref in survivors and not ref.startswith(PREVIOUS_PREFIX)},
            "active_run_ids": list(active_run_ids),
            "unproved_remaining": sorted(ref for ref in heads if ref.startswith(PRUNABLE_PREFIXES)
                                         and ref not in deletions and ref not in protected_skips)}


def apply_plan(plan, expected_heads, repository):
    # Recheck stable heads and PRs immediately before the one atomic transaction.
    latest = remote_heads()
    for target, commit in plan["canonical"].items():
        require(latest.get(target) == commit, f"stable target moved before cleanup: {target}")
    for ref, commit in plan["proof_roots"].items():
        require(latest.get(ref) == commit, f"preserved ancestry proof moved: {ref}")
    active = open_pr_refs(repository)
    for ref in set(plan["deletions"]) | set(plan["updates"]):
        require(ref not in active, f"new PR protects {ref}")
    if plan["deletions"]:
        require(not other_active_runs(repository, plan["run_id"]),
                "another active campaign prevents temporary-ref cleanup")
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
    tag = history_tag(plan["campaign"])
    if plan["deletions"]:
        require(not remote_tag_exists(tag), f"immutable retention history already exists: {tag}")
        record = {"schema": "candle-unified-retention-history.v1",
                  "campaign": plan["campaign"], "run": plan["run_id"],
                  "attempt": plan["run_attempt"],
                  "deleted": [{"ref": ref, "sha": commit}
                              for ref, commit in sorted(plan["deletions"].items())]}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json") as fd:
            fd.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            fd.flush()
            # This annotated tag is a durable provenance record; unlike the
            # 30-day Actions artifact it is retained by GitHub Git refs.
            subprocess.check_call(["git", "tag", "-a", tag, plan["canonical"]["extended"],
                                   "-F", fd.name])
        specs.append(f"refs/tags/{tag}:refs/tags/{tag}")
    if specs:
        git("push", "--atomic", *leases, "origin", *specs)
    after = remote_heads()
    for target, commit in plan["canonical"].items():
        require(after.get(target) == commit, f"stable target moved during cleanup: {target}")
    for ref, commit in plan["updates"].items():
        require(after.get(ref) == commit, f"rolling archive postcheck failed: {ref}")
    for ref in plan["deletions"]:
        require(ref not in after, f"deleted ref appeared after push: {ref}")
    if plan["deletions"]:
        require(remote_tag_exists(tag), f"retention history tag missing: {tag}")
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
    active = other_active_runs(env["GITHUB_REPOSITORY"], env["RUN_ID"])
    preserved = {ref: value for ref, value in current.items()
                 if not ref.startswith(PRUNABLE_PREFIXES)}
    preserved.update({"archive/unified/previous/extended":
                      sha(extended["old_integration_sha"], "previous Extended"),
                      "archive/unified/previous/cuda_legacy":
                      sha(legacy["old_integration_sha"], "previous Legacy")})
    for item in extended["features"]:
        preserved[PREVIOUS_PREFIX + validate_ref(item["target_ref"])] = sha(
            item["source_sha"], "previous standalone")
    graph = retained_graph(preserved.values())
    plan = build_plan(current, protected, extended, legacy, env, graph, active)
    plan.update(campaign=env["CAMPAIGN"], run_id=env["RUN_ID"],
                run_attempt=env["RUN_ATTEMPT"])
    args.report_dir.mkdir(parents=True, exist_ok=True)
    record = args.report_dir / "retention.json"
    report = {"schema": "candle-unified-retention-r4.v1", "campaign": env["CAMPAIGN"],
              "run": env["RUN_ID"], "attempt": env["RUN_ATTEMPT"],
              "mode": "apply" if args.apply else "dry-run", **plan}
    report["status"] = "PLANNED"
    record.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"UNIFIED_RETENTION=PLAN updates={len(plan['updates'])} deletes={len(plan['deletions'])} "
          f"protected_skips={len(plan['skipped'])} unproved={len(plan['unproved_remaining'])}")
    if args.apply:
        try:
            count = apply_plan(plan, current, env["GITHUB_REPOSITORY"])
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            report["status"] = "FAIL_SAFE"
            report["reason"] = str(exc)
            record.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            raise
        report["status"] = "PASS"
        report["history_tag"] = history_tag(env["CAMPAIGN"]) if plan["deletions"] else None
        record.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"UNIFIED_RETENTION=PASS applied={count} previous_states={len(plan['previous'])}")
    else:
        report["status"] = "DRY_RUN"
        record.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("UNIFIED_RETENTION=DRY_RUN no refs modified")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(f"\n## Unified branch retention ({'apply' if args.apply else 'dry-run'})\n\n")
            summary.write(f"- Rolling previous states: {len(plan['previous'])}\n")
            summary.write(f"- Planned deletions: {len(plan['deletions'])}\n")
            summary.write(f"- Protected by PR or active run: {len(plan['skipped'])}\n")
            summary.write(f"- Unproved historical refs retained: {len(plan['unproved_remaining'])}\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        print(f"UNIFIED_RETENTION=FAIL_SAFE reason={exc}", file=sys.stderr)
        sys.exit(1)
