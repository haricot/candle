#!/usr/bin/env python3
"""Independent, host-only tests for the Candle CUDA link-stub CPU wrapper."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WRAPPER = Path(__file__).with_name("candle-cpu-cuda-link-stub.py")


class CpuCudaLinkStubTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="candle-stub-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_stub(self, name, body):
        path = self.root / name
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-x", "c", "-o", str(path), "-"],
            input=body,
            text=True,
            check=True,
            capture_output=True,
        )
        return path

    def invoke(self, stub, program=None, *, env=None):
        command = [sys.executable, str(WRAPPER), "--stub", str(stub), "--"]
        if program is not None:
            command.extend(program)
        return subprocess.run(command, capture_output=True, text=True, env=env)

    def test_good_stub_reaches_cpu_child_and_preserves_existing_loader(self):
        stub = self.make_stub(
            "valid.so", "int cuInit(unsigned int flags) { (void)flags; return 34; }"
        )
        code = (
            "import ctypes, os, pathlib; "
            "driver=ctypes.CDLL('libcuda.so.1'); "
            "assert driver.cuInit(0)==34; "
            "entries=os.environ['LD_LIBRARY_PATH'].split(':'); "
            "assert pathlib.Path(entries[0], 'libcuda.so.1').resolve()==pathlib.Path("
            + repr(str(stub)) + ").resolve(); "
            "assert entries[1:]==['/original/loader']; "
            "print('CPU_TEST_CHILD=PASS')"
        )
        env = dict(os.environ, LD_LIBRARY_PATH="/original/loader")
        original_loader = os.environ.get("LD_LIBRARY_PATH")
        result = self.invoke(stub, [sys.executable, "-c", code], env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CANDLE_CPU_CUDA_LINK_STUB=PASS", result.stdout)
        self.assertIn("CPU_TEST_CHILD=PASS", result.stdout)
        self.assertEqual(os.environ.get("LD_LIBRARY_PATH"), original_loader)

    def test_real_driver_or_incompatible_stub_fails_closed(self):
        stub = self.make_stub(
            "not-a-stub.so", "int cuInit(unsigned int flags) { (void)flags; return 0; }"
        )
        marker = self.root / "should-not-exist"
        result = self.invoke(
            stub, [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("instead of CUDA_ERROR_STUB_LIBRARY=34", result.stderr)
        self.assertFalse(marker.exists())

    def test_missing_cuinit_fails_closed(self):
        stub = self.make_stub("bad.so", "int unrelated(void) { return 34; }")
        result = self.invoke(stub, [sys.executable, "-c", "pass"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cuInit is missing", result.stderr)

    def test_absent_stub_fails_closed(self):
        result = self.invoke(
            self.root / "missing.so", [sys.executable, "-c", "pass"]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA link stub absent", result.stderr)

    def test_requires_a_cpu_command(self):
        result = self.invoke(self.root / "missing.so")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("a CPU test command is required", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
