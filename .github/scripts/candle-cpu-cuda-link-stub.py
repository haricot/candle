#!/usr/bin/env python3
"""Run CPU-only contracts from CUDA-linked binaries using NVIDIA's link stub.

The temporary loader path is given only to the child CPU process. Physical
SM61 jobs must never invoke this wrapper or load the link-time driver stub.
"""

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CUDA_ERROR_STUB_LIBRARY = 34
CUDA_STUB_PATHS = (
    Path("/usr/local/cuda/lib64/stubs/libcuda.so"),
    Path("/usr/local/cuda-12.9/targets/x86_64-linux/lib/stubs/libcuda.so"),
    Path("/usr/local/cuda-12.9/lib64/stubs/libcuda.so"),
)


def select_stub(explicit):
    if explicit is not None:
        candidates = (explicit,)
    else:
        candidates = CUDA_STUB_PATHS
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise RuntimeError("CUDA link stub absent: " + ", ".join(map(str, candidates)))


def run_cpu(stub, command):
    driver = ctypes.CDLL(str(stub))
    try:
        initializer = driver.cuInit
    except AttributeError as exc:
        raise RuntimeError("invalid CUDA link stub: cuInit is missing") from exc
    initializer.argtypes = [ctypes.c_uint]
    initializer.restype = ctypes.c_int
    status = initializer(0)
    if status != CUDA_ERROR_STUB_LIBRARY:
        raise RuntimeError(
            f"refusing CPU stub: cuInit returned {status} instead of "
            f"CUDA_ERROR_STUB_LIBRARY={CUDA_ERROR_STUB_LIBRARY}"
        )

    with tempfile.TemporaryDirectory(prefix="candle-cpu-driver-") as directory:
        (Path(directory) / "libcuda.so.1").symlink_to(stub)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = directory + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        print("CANDLE_CPU_CUDA_LINK_STUB=PASS GPU_AVAILABLE=false", flush=True)
        return subprocess.run(command, env=env, check=False).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a CPU test command is required")
    return run_cpu(select_stub(args.stub), command)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as exc:
        print(f"CANDLE_CPU_CUDA_LINK_STUB=FAIL {exc}", file=sys.stderr)
        sys.exit(1)
