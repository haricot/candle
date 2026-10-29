#!/usr/bin/env bash
set -euo pipefail
test "$(dpkg-query -W -f='${Version}' cuda-toolkit-12-9)" = 12.9.2-1
test "$(dpkg-query -W -f='${Version}' libcublas-12-9)" = 12.9.2.10-1
test "$(dpkg-query -W -f='${Version}' libcudnn9-cuda-12)" = 9.10.2.21-1
test "$(gcc-14 -dumpversion)" = 14
nvcc --version | grep -F 'release 12.9'
rustup run 1.99.0 rustc --version | grep -F 'rustc 1.99.0'
rustup run 1.99.0 cargo --version | grep -F 'cargo 1.99.0'
printf '__global__ void probe() {}\n' > /tmp/sm61-image-smoke.cu
nvcc -ccbin=/usr/bin/g++-14 -arch=sm_61 -c /tmp/sm61-image-smoke.cu -o /tmp/sm61-image-smoke.o
test -s /tmp/sm61-image-smoke.o
