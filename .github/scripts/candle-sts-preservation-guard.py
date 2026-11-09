#!/usr/bin/env python3
"""Fail closed if Unified reconstruction would silently lose experimental STS."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


STS_FILES = (
    "docs/STS_INTEGRATION.md",
    "candle-core/src/diagnostics.rs",
    "candle-core/src/stateful.rs",
    "candle-core/src/stateful/cuda.rs",
    "candle-core/src/stateful/validation.rs",
)


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=False
    )


def blob(repo: Path, revision: str, path: str) -> str | None:
    result = git(repo, "rev-parse", "--verify", f"{revision}:{path}")
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
        raise RuntimeError(f"Unexpected git blob identity for {revision}:{path}")
    return sha


def check(repo: Path, existing: str, candidate: str) -> str:
    for ref in (existing, candidate):
        if git(repo, "cat-file", "-e", f"{ref}^{{commit}}").returncode:
            raise RuntimeError(f"STS guard cannot resolve commit {ref}")

    if blob(repo, existing, STS_FILES[0]) is None:
        return "STS_PRESERVATION=NOT_PRESENT_IN_STABLE_EXTENDED"

    changes = []
    for path in STS_FILES:
        prior = blob(repo, existing, path)
        later = blob(repo, candidate, path)
        if prior is None or later is None:
            changes.append(f"missing {path}")
        elif prior != later:
            changes.append(f"changed {path}")
    if changes:
        raise RuntimeError(
            "STS_PRESERVATION=FAIL_SAFE: candidate would discard or change the "
            "unqualified STS integration: " + ", ".join(changes)
            + "; add a reviewed STS overlay to the Unified composition first"
        )
    return "STS_PRESERVATION=PASS_IDENTICAL_EXPERIMENTAL_STS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--existing", required=True)
    parser.add_argument("--candidate", default="HEAD")
    args = parser.parse_args()
    try:
        print(check(args.repo, args.existing, args.candidate))
    except RuntimeError as error:
        print(f"::error::{error}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
