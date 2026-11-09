#!/usr/bin/env python3
"""Offline R3.1 contracts: Unified owns legacy workflow dispatch and evidence.

No GitHub writes, CUDA build or network access. These tests intentionally do
not relax the old SHA-pinned CI/GPU evidence gates.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
UNIFIED = "orch/unified-candle-v1"
MIRROR = "mirror_orch"
MIGRATED = (
    "candle-asd-extended.yml",
    "candle-candidate-ci.yml",
    "candle-cleanup.yml",
    "candle-cuda1292-container-smoke.yml",
    "candle-gpu-ci.yml",
    "candle-promote.yml",
    "candle-sync.yml",
)


def workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


class DefaultBranchMigration(unittest.TestCase):
    def test_unified_is_the_only_active_orchestration_branch(self) -> None:
        for path in WORKFLOWS.glob("*.yml"):
            with self.subTest(path=path.name):
                self.assertNotIn(MIRROR, path.read_text(encoding="utf-8"))
        for name in MIGRATED:
            with self.subTest(workflow=name):
                self.assertIn(UNIFIED, workflow(name))

    def test_every_job_guard_moves_to_unified(self) -> None:
        for name in (
            "candle-asd-extended.yml",
            "candle-candidate-ci.yml",
            "candle-cleanup.yml",
            "candle-gpu-ci.yml",
            "candle-promote.yml",
            "candle-sync.yml",
        ):
            with self.subTest(workflow=name):
                self.assertIn(f"if: github.ref == 'refs/heads/{UNIFIED}'", workflow(name))

    def test_dispatch_chain_pins_new_orchestrator(self) -> None:
        chain = (
            ("candle-sync.yml", "candle-candidate-ci.yml"),
            ("candle-gpu-ci.yml", "candle-promote.yml"),
            ("candle-promote.yml", "candle-cleanup.yml"),
        )
        for source, destination in chain:
            with self.subTest(source=source):
                self.assertIn(f"gh workflow run {destination}", workflow(source))
                self.assertIn(f"--ref {UNIFIED}", workflow(source))

    def test_ci_and_gpu_evidence_stays_branch_bound(self) -> None:
        for name in ("candle-gpu-ci.yml", "candle-promote.yml"):
            with self.subTest(workflow=name):
                text = workflow(name)
                self.assertIn(f'.head_branch=="{UNIFIED}"', text)
                self.assertIn(".conclusion==\"success\"", text)
        self.assertIn(".gpu_executed==true", workflow("candle-promote.yml"))
        self.assertIn(".gpu_compute_cap==\"6.1\"", workflow("candle-promote.yml"))

    def test_cleanup_is_opt_in_and_confirmation_bound(self) -> None:
        text = workflow("candle-cleanup.yml")
        events = text.split("permissions:", 1)[0]
        self.assertIn("  workflow_dispatch:", events)
        self.assertNotIn("  push:", events)
        self.assertIn("default: true", events)  # dry_run
        self.assertIn('"${CONFIRM}" != DELETE_TEMP_REFS', text)
        self.assertIn("--force-with-lease", text)
        self.assertNotIn(".github/cleanup-request.json", text)

    def test_completed_historical_workflows_are_not_exposed(self) -> None:
        self.assertFalse((WORKFLOWS / "candle-unified-promote-r2.yml").exists())
        self.assertFalse((WORKFLOWS / "candle-sm61-hygiene.yml").exists())
        # The old shell maintenance helper stays available for audit, but no
        # live workflow references or invokes it.
        self.assertTrue((ROOT / ".github/scripts/candle-six-temporary-cleanup.sh").exists())
        for path in WORKFLOWS.glob("*.yml"):
            self.assertNotIn(
                "candle-six-temporary-cleanup.sh", path.read_text(encoding="utf-8")
            )

    def test_unified_push_does_not_automatically_qualify_gpu(self) -> None:
        text = workflow("candle-unified-orchestrator.yml")
        self.assertIn(f"branches: [{UNIFIED}]", text)
        self.assertIn("contains(github.event.head_commit.message, '[run-unified]')", text)
        self.assertIn("github.event_name == 'workflow_dispatch'", text)

    def test_container_pr_smoke_follows_default(self) -> None:
        self.assertIn(
            f"branches: [{UNIFIED}]", workflow("candle-cuda1292-container-smoke.yml")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
