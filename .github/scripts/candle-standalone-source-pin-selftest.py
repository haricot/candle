#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("candle-standalone-source-pin.py")
loader = importlib.util.spec_from_file_location("source_pin", SCRIPT)
pin = importlib.util.module_from_spec(loader)
loader.loader.exec_module(pin)


class SourcePinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.before = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.before)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Source pin test")
        self.git("config", "user.email", "pin@example.invalid")
        (self.root / "feature.txt").write_text("base\n")
        self.base = self.commit("base")
        (self.root / "feature.txt").write_text("reviewed feature\n")
        self.review = self.commit("reviewed feature")
        self.git("checkout", "-q", "--detach", self.base)
        (self.root / "upstream.txt").write_text("upstream\n")
        self.main = self.commit("new main")
        self.spec = Path(self.temp.name) / "migration.json"

    def git(self, *args):
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()

    def commit(self, subject):
        self.git("add", ".")
        self.git("commit", "-qm", subject)
        return self.git("rev-parse", "HEAD")

    def prepared(self, source, content="reviewed feature\n"):
        self.git("checkout", "-q", "--detach", self.main)
        (self.root / "feature.txt").write_text(content)
        return self.commit("prepare(feature): exact source delta " + source + " on main " + self.main)

    def verify(self, observed, allowed=True):
        return pin.verify("feature", self.review, observed, self.main, allowed, self.spec)

    def test_exact_review_is_still_accepted(self):
        self.assertEqual(self.verify(self.review, False), "EXACT")

    def test_successive_rebases_preserve_the_same_reviewed_delta(self):
        first = self.prepared(self.review)
        second = self.prepared(first)
        self.assertEqual(self.verify(second), "REVIEWED_EQUIVALENT")

    def test_unapproved_rebase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reviewed SHA"):
            self.verify(self.prepared(self.review), False)

    def test_new_functional_changes_cannot_hide_behind_a_prepared_subject(self):
        with self.assertRaisesRegex(ValueError, "delta differs"):
            self.verify(self.prepared(self.review, "unreviewed implementation\n"))

    def test_exact_registered_migration_preserves_upstream_compatibility(self):
        observed = self.prepared(self.review, "reviewed feature migrated to current upstream\n")
        source = pin.git("diff", "--binary", "--full-index", self.base, self.review)
        self.spec.write_text(json.dumps({"main": self.main, "sources": {self.review: {
            "feature": "feature", "source_patch_sha256": hashlib.sha256(source).hexdigest(),
            "prepared_tree": self.git("rev-parse", observed + "^{tree}")}}}))
        self.assertEqual(self.verify(observed), "REVIEWED_EQUIVALENT")
        second = self.prepared(observed, "reviewed feature migrated to current upstream\n")
        self.assertEqual(self.verify(second), "REVIEWED_EQUIVALENT")
        changed = self.prepared(self.review, "changed kernel after migration\n")
        with self.assertRaisesRegex(ValueError, "delta differs"):
            self.verify(changed)


if __name__ == "__main__":
    unittest.main()
