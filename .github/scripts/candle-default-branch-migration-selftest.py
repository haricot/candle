#!/usr/bin/env python3
"""Offline R3.2 contracts: Unified owns qualification; legacy sync only prepares.

No GitHub writes, CUDA build or network access. These tests protect the Unified
SM61 promotion gate and the separate manual sync/rebase cleanup.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
UNIFIED = "orch/unified-candle-v1"
MIRROR = "mirror_orch"
MIGRATED = (
    "candle-cleanup.yml",
    "candle-cuda1292-container-smoke.yml",
    "candle-sync.yml",
)
RETIRED = (
    "candle-build-once-contracts.yml",
    "candle-hosted-disk-contracts.yml",
    "candle-standalone-publication.yml",
    "candle-candidate-ci.yml",
    "candle-gpu-ci.yml",
    "candle-promote.yml",
    "candle-asd-extended.yml",
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
        for name in ("candle-cleanup.yml", "candle-sync.yml"):
            with self.subTest(workflow=name):
                self.assertIn(f"if: github.ref == 'refs/heads/{UNIFIED}'", workflow(name))

    def test_retired_workflows_are_absent_from_default(self) -> None:
        for name in RETIRED:
            with self.subTest(workflow=name):
                self.assertFalse((WORKFLOWS / name).exists())

    def test_no_active_workflow_calls_retired_workflows(self) -> None:
        for path in WORKFLOWS.glob("*.yml"):
            text = path.read_text(encoding="utf-8")
            for name in RETIRED:
                with self.subTest(source=path.name, retired=name):
                    self.assertNotIn(name, text)

    def test_sync_creates_unqualified_candidates_without_dispatch(self) -> None:
        text = workflow("candle-sync.yml")
        self.assertIn("  workflow_dispatch:", text)
        self.assertIn("PREPARED_UNQUALIFIED", text)
        self.assertNotIn("      validation:", text)
        self.assertNotIn("actions: write", text)
        self.assertNotIn("gh workflow run", text)

    def test_sm61_evidence_remains_mandatory_for_unified_promotion(self) -> None:
        text = workflow("candle-unified-orchestrator.yml")
        self.assertIn("SM61 Extended runtime", text)
        self.assertIn("needs.physical_sm61.result == 'success'", text)
        self.assertIn("needs.cpu_execute.result == 'success'", text)
        self.assertIn("inputs.promote == true", text)
        self.assertIn("candle-promote-validated.sh", text)
        self.assertIn("candle-unified-retention.py", text)

    def test_cleanup_is_opt_in_and_confirmation_bound(self) -> None:
        text = workflow("candle-cleanup.yml")
        events = text.split("permissions:", 1)[0]
        self.assertIn("  workflow_dispatch:", events)
        self.assertNotIn("  push:", events)
        self.assertIn("default: true", events)  # dry_run
        self.assertIn('"${CONFIRM}" != DELETE_TEMP_REFS', text)
        self.assertIn("--force-with-lease", text)
        self.assertIn("refs/heads/sync/*", text)
        self.assertIn("refs/heads/rebase/*", text)
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
