#!/usr/bin/env python3
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch


here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("driver", here / "candle-extended-build-once.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class LegacyArchiveTests(unittest.TestCase):
    def test_every_original_legacy_gpu_gate_has_a_hosted_executable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "candle-integration/legacy-cuda.json"
            source.parent.mkdir()
            features = ["bf16_candle", "fp8_candle", "fp4_candle", "cudnn_fallback_candle", "moe_simt_f16_candle"]
            source.write_text(json.dumps({"schema_version": 1, "kind": "legacy-cuda-five",
                "campaign": "test", "features": [{"feature": name} for name in features]}))
            (root / "Cargo.lock").write_text("test lock")
            env = {"CANDLE_BUILD_SOURCE": "legacy", "CANDIDATE_SHA": "a" * 40,
                   "CHECK_CAMPAIGN": "test", "LOCK_SHA256": driver.sha256(root / "Cargo.lock")}
            def fake_compile(_repo, _out, _name, _argv, _env, wanted=()):
                return {name: "/bin/true" for name in wanted}
            old = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, env), patch.object(driver, "output", return_value="a" * 40), \
                     patch.object(driver, "environment", return_value={}), \
                     patch.object(driver, "log_cargo", side_effect=fake_compile):
                    driver.build(type("Args", (), {"out": str(root / "bundle")})())
            finally:
                os.chdir(old)
            built = json.loads((root / "bundle/manifest.json").read_text())
            expected = set(re.findall(r"\brun_gate\s+([a-z][a-z0-9_]+)\b", (here / "candle-legacy-sm61.sh").read_text()))
            self.assertEqual(set(built["gates"]), expected)
            self.assertTrue(all(record["kind"] == "hosted-check" or record["binaries"] for record in built["gates"].values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
