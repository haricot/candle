#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SPEC = Path(__file__).resolve().parent.parent / "resolutions/legacy-main-zip913-v1/manifest.json"


def git(*args):
    return subprocess.check_output(["git", *args])


def require_commit(sha):
    if subprocess.run(["git", "cat-file", "-e", sha + "^{commit}"], stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["git", "fetch", "--no-tags", "origin", sha], check=True, stdout=subprocess.DEVNULL)


def verify(feature, review, observed, main, allowed, spec_path=SPEC):
    for value in (review, observed, main):
        if not re.fullmatch(r"[a-f0-9]{40}", value):
            raise ValueError("invalid source commit identity")
        require_commit(value)
    if observed == review:
        return "EXACT"
    if not allowed:
        raise ValueError("source head differs from its reviewed SHA")
    spec = json.loads(Path(spec_path).read_text()) if Path(spec_path).is_file() else {}
    current, seen = observed, set()
    while current != review:
        if current in seen:
            raise ValueError("cyclic prepared source provenance")
        seen.add(current)
        parents = git("rev-list", "--parents", "-n", "1", current).decode().split()
        if len(parents) != 2:
            raise ValueError("relocated source must be a single-parent prepared commit")
        parent = parents[1]
        if subprocess.run(["git", "merge-base", "--is-ancestor", parent, main]).returncode:
            raise ValueError("prepared source parent is not an ancestor of current main")
        subject = git("log", "-1", "--format=%s", current).decode().strip()
        match = re.fullmatch(r"prepare\(" + re.escape(feature) + r"\): exact source delta ([a-f0-9]{40}) on main " + parent, subject)
        if not match:
            raise ValueError("prepared source identity differs from its feature/source/main")
        source = match[1]
        require_commit(source)
        base = git("merge-base", parent, source).decode().strip()
        original = git("diff", "--binary", "--full-index", base, source)
        actual = git("diff", "--binary", "--full-index", parent, current)
        if not original:
            raise ValueError("reviewed source delta is unexpectedly empty")
        if original != actual:
            migration = spec.get("sources", {}).get(source, {})
            tree = git("rev-parse", current + "^{tree}").decode().strip()
            if (parent != spec.get("main") or migration.get("feature") != feature
                    or migration.get("source_patch_sha256") != hashlib.sha256(original).hexdigest()
                    or migration.get("prepared_tree") != tree):
                raise ValueError("prepared source delta differs from its reviewed source or exact migration")
        current = source
    return "REVIEWED_EQUIVALENT"


if __name__ == "__main__":
    try:
        result = verify(os.environ["FEATURE"], os.environ["REVIEW_SOURCE_SHA"],
                        os.environ["OBSERVED_SOURCE_SHA"], os.environ["CURRENT_MAIN_SHA"],
                        os.environ.get("ALLOW_PROMOTED_EQUIVALENT_SOURCE") == "true")
        print("EXTENDED_SOURCE_PIN=" + result + " feature=" + os.environ["FEATURE"])
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit("EXTENDED_SOURCE_PIN_REJECT=" + str(error))
