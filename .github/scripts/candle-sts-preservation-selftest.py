#!/usr/bin/env python3
"""Real-git contract tests for the experimental STS preservation guard."""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("candle-sts-preservation-guard.py")
spec = importlib.util.spec_from_file_location("sts_guard", SCRIPT)
assert spec and spec.loader
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


with tempfile.TemporaryDirectory(prefix="sts-guard-") as root:
    repo = Path(root)
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "STS test")
    git(repo, "commit", "--allow-empty", "-qm", "pre-STs")
    pre = git(repo, "rev-parse", "HEAD")
    assert guard.check(repo, pre, pre) == "STS_PRESERVATION=NOT_PRESENT_IN_STABLE_EXTENDED"

    for path in guard.STS_FILES:
        dest = repo / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"STS content for {path}\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "integrate experimental STS")
    complete = git(repo, "rev-parse", "HEAD")
    assert guard.check(repo, complete, complete) == "STS_PRESERVATION=PASS_IDENTICAL_EXPERIMENTAL_STS"

    (repo / guard.STS_FILES[2]).write_text("changed stateful implementation\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "change stateful implementation")
    changed = git(repo, "rev-parse", "HEAD")
    try:
        guard.check(repo, complete, changed)
    except RuntimeError as error:
        assert "changed candle-core/src/stateful.rs" in str(error)
    else:
        raise AssertionError("STS guard accepted an unreviewed change")

    (repo / guard.STS_FILES[0]).unlink()
    git(repo, "add", "-u")
    git(repo, "commit", "-qm", "drop STS marker")
    removed = git(repo, "rev-parse", "HEAD")
    try:
        guard.check(repo, complete, removed)
    except RuntimeError as error:
        assert "missing docs/STS_INTEGRATION.md" in str(error)
    else:
        raise AssertionError("STS guard accepted missing integration metadata")

print("STS_PRESERVATION_SELFTEST=PASS")
