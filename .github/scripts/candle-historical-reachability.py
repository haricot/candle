#!/usr/bin/env python3
"""R4.3: audit abandoned Unified temporary branches; prune only with explicit approval.

Preserve independent commits by immutable annotated tags in the *same atomic
push* that removes their branch refs. Never prune current refs, PR branches,
qualification evidence, or refs in use by any other active Actions run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

RETENTION_PATH = Path(__file__).with_name("candle-unified-retention.py")
spec = importlib.util.spec_from_file_location("candle_unified_retention", RETENTION_PATH)
retention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retention)

ARCHIVE_TAG_PREFIX = "candle-historical/commit/"
HISTORY_TAG_PREFIX = "candle-historical/audit-"
SHA = retention.SHA
PRUNABLE_PREFIXES = retention.PRUNABLE_PREFIXES


def campaign_for_ref(ref):
    parts = ref.split("/")
    if len(parts) >= 4 and parts[:2] == ["prepare", "extended"]:
        return parts[2]
    if len(parts) >= 4 and parts[:2] == ["prepare", "legacy"]:
        return parts[2]
    if len(parts) >= 4 and parts[:3] == ["integration", "extended", "candidate"]:
        return parts[3]
    if len(parts) >= 4 and parts[:3] == ["integration", "legacy-cuda", "candidate"]:
        return parts[3]
    if len(parts) >= 5 and parts[:2] == ["archive", "promotion"] and parts[3] == "before":
        return parts[2]
    return "unknown"


def is_prunable(ref):
    if not ref.startswith(PRUNABLE_PREFIXES):
        return False
    parts = ref.split("/")
    if ref.startswith("prepare/"):
        return len(parts) == 4 and parts[1] in ("extended", "legacy")
    if ref.startswith("integration/"):
        return len(parts) == 4 and parts[2] == "candidate"
    if ref.startswith("archive/promotion/"):
        return len(parts) == 5 and parts[3] == "before"
    return False


def plan(heads, protected, reachable, active_run_ids):
    """Pure plan: no ambiguous prefixes, no fresh-ref writes, no force updates."""
    if not any(ref.startswith("archive/extended/sm61-validated-") for ref in heads):
        raise ValueError("missing Extended SM61 qualification evidence")
    if not any(ref.startswith("archive/legacy/sm61-validated-") for ref in heads):
        raise ValueError("missing Legacy SM61 qualification evidence")
    if not heads.get("extended") or not heads.get("cuda_legacy"):
        raise ValueError("missing canonical integration refs")
    deletions, skips, tags = {}, [], defaultdict(list)
    for ref, commit in sorted(heads.items()):
        if not ref.startswith(PRUNABLE_PREFIXES):
            continue
        retention.validate_ref(ref)
        retention.sha(commit, ref)
        if not is_prunable(ref):
            skips.append({"ref": ref, "sha": commit, "reason": "UNKNOWN_SHAPE"})
            continue
        if ref in protected:
            skips.append({"ref": ref, "sha": commit, "reason": "OPEN_PR"})
            continue
        if active_run_ids:
            skips.append({"ref": ref, "sha": commit, "reason": "ACTIVE_ACTIONS_RUN"})
            continue
        kind = "REACHABLE" if commit in reachable else "TAGGED_UNIQUE"
        deletions[ref] = {"sha": commit, "campaign": campaign_for_ref(ref),
                          "preservation": kind}
        if kind == "TAGGED_UNIQUE":
            tags[commit].append(ref)
    return {
        "deletions": deletions, "preservation_tags": dict(tags), "skipped": skips,
        "active_run_ids": list(active_run_ids),
        "survivor_roots": {ref: commit for ref, commit in heads.items()
                           if ref not in deletions},
    }


def fetch_all_heads(heads):
    # Head objects needed for exact reachability checks and tag creation.
    retention.git("fetch", "--no-tags", "--filter=blob:none", "origin",
                  "+refs/heads/*:refs/remotes/origin/*")
    for ref, commit in heads.items():
        local = retention.git("rev-parse", f"refs/remotes/origin/{ref}")
        if local != commit:
            raise ValueError(f"remote ref changed during object fetch: {ref}")
        if subprocess.call(["git", "cat-file", "-e", commit + "^{commit}"]) != 0:
            raise ValueError(f"commit unavailable for {ref}")


def compute_reachable(heads):
    preserved = [sha for ref, sha in heads.items() if not is_prunable(ref)]
    return retention.retained_graph(preserved)


def annotated_tag(tag, commit, message):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json") as tmp:
        tmp.write(json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n")
        tmp.flush()
        subprocess.check_call(["git", "tag", "-a", tag, commit, "-F", tmp.name])


def apply(audit, original_heads, repo, run_id):
    if not audit["deletions"]:
        return 0
    if not re.fullmatch(r"[0-9]+", run_id):
        raise ValueError("invalid run ID")
    # One operation: no branch can disappear without its preserving tags.
    latest = retention.remote_heads()
    if latest != original_heads:
        raise ValueError("branch inventory changed after preflight")
    if retention.open_pr_refs(repo) & set(audit["deletions"]):
        raise ValueError("a new PR protects a candidate")
    if retention.other_active_runs(repo, run_id):
        raise ValueError("another campaign became active")
    history_tag = HISTORY_TAG_PREFIX + run_id
    if retention.remote_tag_exists(history_tag):
        raise ValueError("immutable history tag already exists")
    specs, leases = [], []
    for commit, refs in sorted(audit["preservation_tags"].items()):
        # Run-specific tag: even if an older archive moves, this transaction
        # itself independently pins every distinct commit it is deleting.
        tag = ARCHIVE_TAG_PREFIX + run_id + "/" + commit
        if retention.remote_tag_exists(tag):
            raise ValueError(f"preservation tag already exists: {tag}")
        annotated_tag(tag, commit, {"schema": "candle-historical-commit.v1",
                                    "sha": commit, "refs": sorted(refs)})
        specs.append(f"refs/tags/{tag}:refs/tags/{tag}")
    record = {
        "schema": "candle-historical-audit.v1", "run": run_id,
        "deleted": [{"ref": ref, **info}
                    for ref, info in sorted(audit["deletions"].items())],
    }
    annotated_tag(history_tag, original_heads["extended"], record)
    specs.append(f"refs/tags/{history_tag}:refs/tags/{history_tag}")
    for ref, info in sorted(audit["deletions"].items()):
        specs.append(f":refs/heads/{ref}")
        leases.append(f"--force-with-lease=refs/heads/{ref}:{info['sha']}")
    retention.git("push", "--atomic", *leases, "origin", *specs)
    after = retention.remote_heads()
    if any(ref in after for ref in audit["deletions"]):
        raise ValueError("deleted ref reappeared")
    if not retention.remote_tag_exists(history_tag):
        raise ValueError("retention history tag was not preserved")
    for commit in audit["preservation_tags"]:
        tag = ARCHIVE_TAG_PREFIX + run_id + "/" + commit
        peeled = retention.git("ls-remote", "--tags", "origin",
                               f"refs/tags/{tag}^{{}}")
        if peeled.split("\t", 1)[0] != commit:
            raise ValueError(f"preservation tag missing or changed for {commit}")
    return len(audit["deletions"])


def summary(report):
    lines = [
        "## R4.3 — historical branch reachability",
        "",
        f"- Status: **{report['status']}**",
        f"- Planned deletions: **{len(report['deletions'])}**",
        f"- Unique commits needing immutable tags: **{len(report['preservation_tags'])}**",
        f"- Protected or unmatched: **{len(report['skipped'])}**",
        f"- Other active campaigns: **{len(report['active_run_ids'])}**",
        f"- To apply this exact inventory: `{report['confirmation_token']}`",
        "",
        "| Ref kind | Count |",
        "| --- | ---: |",
    ]
    counts = defaultdict(int)
    for ref in report["deletions"]:
        counts["/".join(ref.split("/")[:2])] += 1
    for kind, count in sorted(counts.items()):
        lines.append(f"| `{kind}` | {count} |")
    return "\n".join(lines) + "\n"


def confirmation_token(audit):
    # Bind approval to the ENTIRE remote branch inventory, including preserved
    # qualification refs, canonical heads and PR-protected refs, not only the
    # candidate deletion set. Any intervening ref movement demands a fresh preview.
    snapshot = sorted(
        [(ref, info["sha"]) for ref, info in audit["deletions"].items()]
        + list(audit["survivor_roots"].items())
    )
    digest = hashlib.sha256(json.dumps(snapshot, separators=(",", ":")).encode()).hexdigest()
    return "PRUNE_HISTORICAL_" + digest[:16].upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", repo):
        raise ValueError("unsafe GitHub repository")
    current = retention.remote_heads()
    open_prs = retention.open_pr_refs(repo)
    active = retention.other_active_runs(repo, run_id)
    fetch_all_heads(current)
    reachable = compute_reachable(current)
    audit = plan(current, open_prs, reachable, active)
    token = confirmation_token(audit)
    if args.apply and (not audit["deletions"] or args.confirm != token):
        raise ValueError("confirmation does not match the current historical ref inventory")
    report = {"schema": "candle-historical-audit.v1", "run": run_id,
              "mode": "apply" if args.apply else "dry-run",
              "status": "PLANNED", "confirmation_token": token, **audit}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    target = args.report_dir / "historical-audit.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.apply:
        try:
            count = apply(audit, current, repo, run_id)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            report["status"] = "FAIL_SAFE"
            report["reason"] = str(exc)
            target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            raise
        report["status"] = "PASS"
        report["deleted_count"] = count
        report["history_tag"] = HISTORY_TAG_PREFIX + run_id if count else None
    else:
        report["status"] = "DRY_RUN"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    result = summary(report)
    print(result)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as out:
            out.write(result)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"CANDLE_HISTORICAL_AUDIT=FAIL_SAFE reason={exc}", file=sys.stderr)
        sys.exit(1)
