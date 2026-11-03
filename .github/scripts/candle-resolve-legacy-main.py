#!/usr/bin/env python3
"""Replay one reviewed Legacy/Zip upstream migration, bound to the entire tree."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SPEC = Path(__file__).resolve().parent.parent / "resolutions/legacy-main-zip913-v1/manifest.json"


def git(work, *args):
    return subprocess.check_output(["git", "-C", str(work), *args])


def resolve(action, work, main, source, patch):
    spec = json.loads(SPEC.read_text())
    if main != spec["main"] or source not in spec["sources"]:
        raise ValueError("unreviewed Legacy source/main pair")
    selected = spec["sources"][source]
    if hashlib.sha256(Path(patch).read_bytes()).hexdigest() != selected["source_patch_sha256"]:
        raise ValueError("Legacy source delta changed")
    if action == "resolve":
        stages = {}
        for line in git(work, "ls-files", "-u").decode().splitlines():
            fields, path = line.split("\t", 1)
            _, blob, stage = fields.split()
            stages.setdefault(path, {})[{"1":"base", "2":"ours", "3":"theirs"}[stage]] = blob
        if stages != selected.get("files", spec["files"]):
            raise ValueError("unreviewed Legacy conflict paths or stages")
        # Replay only reviewed source/main conflicts: preserve upstream Zip APIs
        # and combine ASD exports/dependencies from frozen reviewed files.
        for path in sorted(stages):
            reviewed = selected.get("resolved_files", {}).get(path)
            if reviewed:
                (Path(work) / path).write_bytes((SPEC.parent / reviewed).read_bytes())
                subprocess.run(["git", "-C", str(work), "add", "--", path], check=True)
            else:
                subprocess.run(["git", "-C", str(work), "checkout", main, "--", path], check=True)
    elif action != "verify":
        raise ValueError("unknown action")
    if git(work, "ls-files", "-u").strip():
        raise ValueError("unresolved Legacy conflicts")
    if git(work, "write-tree").decode().strip() != selected["prepared_tree"]:
        raise ValueError("Legacy prepared tree differs from reviewed migration")
    print("LEGACY_MAIN_ZIP913_REVIEWED=PASS")


if __name__ == "__main__":
    try:
        resolve(*sys.argv[1:])
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print("LEGACY_MAIN_ZIP913_REJECT=" + str(error), file=sys.stderr)
        sys.exit(1)
