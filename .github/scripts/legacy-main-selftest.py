#!/usr/bin/env python3
"""Reproduce the actual upstream conflict and reject modified inputs/outputs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

here = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location("resolver", here / "candle-resolve-legacy-main.py")
resolver = importlib.util.module_from_spec(loader)
loader.loader.exec_module(resolver)
spec = json.loads(resolver.SPEC.read_text())
prepared = {}
for source in spec["sources"]:
    with tempfile.TemporaryDirectory(prefix="bf16-main-regression-") as temporary:
        root = Path(temporary)
        work = root / "source"
        patch = root / "source.patch"
        base = subprocess.check_output(["git", "merge-base", spec["main"], source], text=True).strip()
        patch.write_bytes(subprocess.check_output(["git", "diff", "--binary", "--full-index", base, source]))
        subprocess.run(["git", "worktree", "add", "--detach", str(work), spec["main"]], check=True, capture_output=True)
        try:
            applied = subprocess.run(["git", "-C", str(work), "apply", "--3way", "--index", str(patch)], capture_output=True)
            expected_conflict = spec["sources"][source].get("conflicted", True)
            assert (applied.returncode != 0) == expected_conflict, "original apply behavior changed"
            if expected_conflict:
                resolver.resolve("resolve", work, spec["main"], source, patch)
            resolver.resolve("verify", work, spec["main"], source, patch)
            subprocess.run(["git", "-C", str(work), "-c", "user.name=CI", "-c", "user.email=ci@example.invalid", "commit", "-m", "Reviewed Legacy upstream compatibility"], check=True, capture_output=True)
            prepared[spec["sources"][source]["feature"]] = subprocess.check_output(["git", "-C", str(work), "rev-parse", "HEAD"], text=True).strip()
            rejected = 0
            for main, changed_source in (("f"*40, source), (spec["main"], "f"*40)):
                try: resolver.resolve("verify", work, main, changed_source, patch)
                except ValueError: rejected += 1
            kernel = work / "candle-kernels/build.rs"
            kernel.write_text(kernel.read_text() + "\n// changed BF16 kernel\n")
            subprocess.run(["git", "-C", str(work), "add", str(kernel)], check=True)
            try: resolver.resolve("verify", work, spec["main"], source, patch)
            except ValueError: rejected += 1
            assert rejected == 3, "modified migration was accepted"
            print("LEGACY_MAIN_REGRESSION=PASS original-conflict resolved-tree rejected-main rejected-source rejected-kernel")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(work)], check=True, capture_output=True)

# Exercise the real, exact-stage aggregate resolvers without network or publication.
with tempfile.TemporaryDirectory(prefix="legacy-union-regression-") as temporary:
    root = Path(temporary)
    work = root / "aggregate"
    report = root / "report"
    (report / "logs").mkdir(parents=True)
    (report / "conflicts").mkdir()
    config_path = here.parent / "extended-sources.json"
    if not config_path.exists():
        config_path = here.parent / "legacy-cuda-sources.json"
    config = json.loads(config_path.read_text())
    reviews = [x["review_source_sha"] for x in config["features"][:5]]
    prepared = [prepared[x["feature"]] for x in config["features"][:5]]
    names = [None, "bf16-fp8", "bf16-fp8-fp4", "bf16-fp8-fp4-cudnn", "bf16-fp8-fp4-moe"]
    subprocess.run(["git", "worktree", "add", "--detach", str(work), spec["main"]], check=True, capture_output=True)
    try:
        for i, commit in enumerate(prepared):
            merge = subprocess.run(["git", "-C", str(work), "merge", "--no-ff", "--no-edit", commit], capture_output=True)
            if merge.returncode:
                assert names[i], "unexpected BF16 integration conflict"
                subprocess.run(["bash", str(here / ("candle-resolve-" + names[i] + ".sh")), str(work), *reviews[:i+1], str(report)], check=True)
        for path in spec["files"]:
            assert (work / path).read_bytes() == resolver.git(work, "show", spec["main"] + ":" + path), path
        assert 'version = "0.11.1"' in (work / "candle-kernels/Cargo.toml").read_text()
        assert "Metal indexed MoE requires Metal input and expert ids" in (work / "candle-core/src/quantized/mod.rs").read_text()
        for commit in prepared:
            subprocess.run(["git", "-C", str(work), "merge-base", "--is-ancestor", commit, "HEAD"], check=True)
        print("LEGACY_UNION_REGRESSION=PASS exact-stages upstream-zip upstream-metal feature-ancestry")
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(work)], check=True, capture_output=True)
