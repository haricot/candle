#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REQUEST = ".github/cleanup-sm61-20261030.json"
WORKFLOW = ".github/workflows/candle-sm61-hygiene.yml"
TAG_PREFIX = "archive/candle-pruned-20261030/"
SHA = re.compile(r"[0-9a-f]{40}\Z")
PREPARE = re.compile(r"prepare/(?:legacy|extended)/(?:legacy|extended)-(\d+)-\d+/[A-Za-z0-9_.-]+\Z")
BACKUP = re.compile(r"archive/promotion/(?:legacy|extended)-(\d+)-\d+/before/[A-Za-z0-9_.-]+\Z")
MERGED_CI = {"ci/extended-examples-legacy-alias-validation"}


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def api(path):
    return json.loads(command("gh", "api", "repos/haricot/candle" + path))


def fingerprint(plan):
    scope = {k: v for k, v in plan.items() if k not in ("phase", "preview_run_id")}
    return hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()


def validate(plan):
    version = plan.get("schema_version")
    expected_scope = {1: "sm61-hygiene-20261030", 2: "sm61-hygiene-20261030-followup"}.get(version)
    if (expected_scope is None or plan.get("repository") != "haricot/candle"
            or plan.get("scope") != expected_scope
            or plan.get("phase") not in ("preview", "apply")):
        raise RuntimeError("invalid hygiene request")
    refs = [t["ref"] for t in plan["targets"]]
    if not refs or len(refs) != len(set(refs)):
        raise RuntimeError("empty or duplicate target refs")
    for item in plan["targets"]:
        if not SHA.fullmatch(item["sha"]) or item["ref"] in plan["protected_heads"]:
            raise RuntimeError("target SHA or protected ref is invalid")
        match = PREPARE.fullmatch(item["ref"])
        if match and int(match[1]) in plan["completed_runs"]:
            continue
        backup = BACKUP.fullmatch(item["ref"]) if version == 2 else None
        if backup and str(backup[1]) in plan.get("promoted_runs", {}):
            continue
        if (version == 2 and item["ref"] in MERGED_CI
                and item.get("merged_into") in plan["protected_heads"]):
            continue
        raise RuntimeError("target is not a pinned completed preparation, promotion backup or merged CI ref")
    if version == 2:
        for proof in plan["promoted_runs"].values():
            if not SHA.fullmatch(proof["candidate"]):
                raise RuntimeError("invalid promoted candidate")
        if not set(plan["orphan_workflows"]).issubset(plan["retired_workflows"]):
            raise RuntimeError("orphan workflows must be explicitly retired")
    for path in plan["retired_workflows"]:
        if not re.fullmatch(r"\.github/workflows/[A-Za-z0-9_.-]+\.yml", path):
            raise RuntimeError("invalid retired workflow path")
        if path in (WORKFLOW, ".github/workflows/candle-legacy-cuda.yml",
                    ".github/workflows/candle-extended-composite.yml", ".github/workflows/candle-cuda-ci-image.yml"):
            raise RuntimeError("current workflow cannot be retired")


class Refs:
    def heads(self):
        return {ref.removeprefix("refs/heads/"): sha
                for sha, ref in (line.split() for line in command("git", "ls-remote", "--heads", "origin").splitlines())}

    def check(self, plan, *, deleted=False):
        heads = self.heads()
        for name, expected in plan["protected_heads"].items():
            if heads.get(name) != expected:
                raise RuntimeError("protected head changed: " + name)
        for target in plan["targets"]:
            expected = None if deleted else target["sha"]
            if heads.get(target["ref"]) != expected:
                raise RuntimeError("preparation head changed: " + target["ref"])

    def archive(self, plan):
        tags = {ref: sha for sha, ref in (line.split() for line in
                command("git", "ls-remote", "--tags", "origin", "refs/tags/" + TAG_PREFIX + "*").splitlines())}
        specs = []
        for target in plan["targets"]:
            tag = TAG_PREFIX + target["ref"]
            ref = "refs/tags/" + tag
            command("git", "check-ref-format", ref)
            if ref in tags:
                if tags.get(ref + "^{}") != target["sha"]:
                    raise RuntimeError("archive tag collision: " + tag)
                continue
            command("git", "cat-file", "-e", target["sha"] + "^{commit}")
            command("git", "tag", "-a", tag, target["sha"], "-m",
                    "Archive " + target["ref"] + " at " + target["sha"] + " before branch cleanup")
            specs.append(ref + ":" + ref)
        if specs:
            command("git", "push", "--atomic", "origin", *specs)
        tags = {ref: sha for sha, ref in (line.split() for line in
                command("git", "ls-remote", "--tags", "origin", "refs/tags/" + TAG_PREFIX + "*").splitlines())}
        for target in plan["targets"]:
            if tags.get("refs/tags/" + TAG_PREFIX + target["ref"] + "^{}") != target["sha"]:
                raise RuntimeError("annotated archive not verified: " + target["ref"])

    def delete(self, plan):
        leases = ["--force-with-lease=refs/heads/" + t["ref"] + ":" + t["sha"] for t in plan["targets"]]
        deletions = [":refs/heads/" + t["ref"] for t in plan["targets"]]
        command("git", "push", "--atomic", *leases, "origin", *deletions)


def preflight(plan, refs):
    refs.check(plan)
    run = api("/actions/runs/" + str(plan["validation"]["run_id"]))
    jobs = api("/actions/runs/" + str(run["id"]) + "/jobs?per_page=100")["jobs"]
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise RuntimeError("Candle validation workflow is not successful")
    for key in ("gpu_job_id", "promotion_job_id"):
        if not any(j["id"] == plan["validation"][key] and j["conclusion"] == "success" for j in jobs):
            raise RuntimeError("physical validation or promotion is missing")
    for run_id in plan["completed_runs"]:
        if api("/actions/runs/" + str(run_id)).get("status") != "completed":
            raise RuntimeError("preparation campaign is still active")
    if plan["schema_version"] == 2:
        for run_id, proof in plan["promoted_runs"].items():
            run = api("/actions/runs/" + run_id)
            jobs = api("/actions/runs/" + run_id + "/jobs?per_page=100")["jobs"]
            if run.get("status") != "completed" or run.get("conclusion") != "success":
                raise RuntimeError("promotion backup still belongs to an unfinished campaign")
            for key in ("gpu_job_id", "promotion_job_id"):
                if not any(j["id"] == proof[key] and j["conclusion"] == "success" for j in jobs):
                    raise RuntimeError("promotion backup lacks physical validation or promotion")
            if refs.heads().get("archive/extended/sm61-validated-" + run_id + "-1") != proof["candidate"]:
                raise RuntimeError("physical validation archive changed")
        for item in plan["targets"]:
            if item["ref"] in MERGED_CI:
                command("git", "merge-base", "--is-ancestor", item["sha"],
                        plan["protected_heads"][item["merged_into"]])
        # A registered one-shot workflow is retired only when no current head
        # contains its file. Historical commits, runs and artifacts are retained.
        for sha in set(refs.heads().values()):
            for path in plan["orphan_workflows"]:
                if command("git", "ls-tree", "--name-only", sha, "--", path):
                    raise RuntimeError("orphan workflow still exists on a current head: " + path)
    protected = {t["ref"] for t in plan["targets"]}
    page = 1
    while True:
        prs = api("/pulls?state=open&per_page=100&page=" + str(page))
        if any(p[side]["ref"] in protected for p in prs for side in ("base", "head")):
            raise RuntimeError("an open PR uses a target branch")
        if len(prs) < 100:
            break
        page += 1
    retired = set(plan["retired_workflows"])
    for state in ("queued", "in_progress", "waiting", "requested", "pending"):
        page = 1
        while True:
            runs = api("/actions/runs?status=" + state + "&per_page=100&page=" + str(page))["workflow_runs"]
            targets = {t["ref"] for t in plan["targets"]}
            if any(r.get("path") in retired or r.get("head_branch") in targets for r in runs):
                raise RuntimeError("a target branch or retired workflow is still active")
            if len(runs) < 100:
                break
            page += 1


def verify_preview(plan):
    run = api("/actions/runs/" + str(int(plan["preview_run_id"])))
    if (run.get("status") != "completed" or run.get("conclusion") != "success"
            or run.get("head_branch") != "mirror_orch" or run.get("path") != WORKFLOW):
        raise RuntimeError("matching successful preview required")
    content = api("/contents/" + REQUEST + "?ref=" + run["head_sha"])
    previous = json.loads(base64.b64decode(content["content"]))
    if previous.get("phase") != "preview" or fingerprint(previous) != fingerprint(plan):
        raise RuntimeError("preview covered another scope")


def retire_workflows(plan):
    page = 1
    retired = []
    while True:
        workflows = api("/actions/workflows?per_page=100&page=" + str(page))["workflows"]
        for workflow in workflows:
            if workflow["path"] in plan["retired_workflows"]:
                command("gh", "api", "--method", "PUT", "repos/haricot/candle/actions/workflows/" + str(workflow["id"]) + "/disable")
                current = api("/actions/workflows/" + str(workflow["id"]))
                if current.get("state") != "disabled_manually":
                    raise RuntimeError("workflow retirement not confirmed")
                retired.append({"id": workflow["id"], "path": workflow["path"]})
        if len(workflows) < 100:
            break
        page += 1
    return retired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(Path(REQUEST).read_text())
    validate(plan)
    if (os.environ.get("GITHUB_REPOSITORY") != "haricot/candle"
            or os.environ.get("GITHUB_REF") != "refs/heads/mirror_orch"
            or command("git", "rev-parse", "HEAD") != os.environ["GITHUB_SHA"]):
        raise RuntimeError("wrong repository or orchestration commit")
    refs = Refs()
    preflight(plan, refs)
    report = {"phase": plan["phase"], "fingerprint": fingerprint(plan), "targets": plan["targets"],
              "protected_heads": plan["protected_heads"], "deleted_branches": 0, "retired_workflows": []}
    if plan["phase"] == "apply":
        verify_preview(plan)
        command("git", "config", "user.name", "github-actions[bot]")
        command("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
        command("gh", "auth", "setup-git")
        refs.archive(plan)
        preflight(plan, refs)
        refs.delete(plan)
        refs.check(plan, deleted=True)
        report["deleted_branches"] = len(plan["targets"])
        report["retired_workflows"] = retire_workflows(plan)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("CANDLE_HYGIENE=PASS phase=" + plan["phase"] + " targets=" + str(len(plan["targets"]))
          + " deleted=" + str(report["deleted_branches"]) + " retired=" + str(len(report["retired_workflows"])))


if __name__ == "__main__":
    main()
