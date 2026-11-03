#!/usr/bin/env bash
# Only the disposable GitHub-hosted VM; never a physical/self-hosted runner.
set -euo pipefail
if [[ "${GITHUB_ACTIONS:-}" != true || "${HARICOT_RUNNER_ENVIRONMENT:-}" != github-hosted ]]; then
  echo 'HOSTED_DISK_CLEANUP=REFUSED runner must be github-hosted' >&2
  exit 2
fi
minimum="${HARICOT_MIN_FREE_GIB-25}"
if [[ ! "$minimum" =~ ^[1-9][0-9]*$ ]]; then
  echo 'HOSTED_DISK_CLEANUP=REFUSED invalid free-space threshold' >&2
  exit 2
fi
available() { df --output=avail --block-size=1 / | tail -n 1 | tr -d '[:space:]'; }
before="$(available)"
echo "HOSTED_DISK_BEFORE_BYTES=$before"
df -h /
# These preinstalled SDKs are unused by the pinned CUDA compiler container.
# Keep Python, Node, Rust, Docker/containerd, the workspace, Cargo caches and
# all artifacts. In particular, do not prune the verified CUDA image.
sudo rm -rf -- \
  /usr/local/lib/android \
  /usr/share/dotnet \
  /opt/ghc \
  /usr/local/.ghcup \
  /usr/share/swift \
  /usr/local/share/powershell \
  /opt/az \
  /opt/hostedtoolcache/CodeQL
after="$(available)"
echo "HOSTED_DISK_AFTER_BYTES=$after"
echo "HOSTED_DISK_FREED_BYTES=$((after - before))"
df -h /
if (( after < minimum * 1024 * 1024 * 1024 )); then
  echo "HOSTED_DISK_CLEANUP=FAIL need ${minimum} GiB free before CUDA image extraction; available_bytes=$after" >&2
  exit 1
fi
echo "HOSTED_DISK_CLEANUP=PASS minimum_free_gib=$minimum"
