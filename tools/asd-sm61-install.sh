#!/usr/bin/env bash
set -euo pipefail

# Install the seven ASD exact-grouped sm61 CUDA implementations into the
# user-scoped ASD artifact store. Builtin PTX remains untouched (Phase A).
#
# Runtime resolution after installation:
#   external CUBIN -> external PTX -> builtin raw PTX -> qualified provider fallback
#
# Environment:
#   CANDLE_ASD_HOME  ASD root (default: ${XDG_DATA_HOME:-$HOME/.local/share}/asd)
#   NVCC             nvcc path (default: /opt/cuda/bin/nvcc)
#   NVCC_CCBIN       host compiler (default: /usr/bin/g++-14)
#   CUOBJDUMP        cuobjdump path (default: sibling of nvcc)
#
# Usage:
#   bash tools/asd-sm61-install.sh
#   bash tools/asd-sm61-install.sh --verify-only
#   bash tools/asd-sm61-install.sh --prune-external-ptx

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASD_HOME="${CANDLE_ASD_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/asd}"
ART="$ASD_HOME/artifacts/sm61"
NVCC="${NVCC:-/opt/cuda/bin/nvcc}"
CCBIN="${NVCC_CCBIN:-/usr/bin/g++-14}"
CUOBJDUMP="${CUOBJDUMP:-$(dirname "$NVCC")/cuobjdump}"

VERIFY_ONLY=0
PRUNE_EXTERNAL_PTX=0
for arg in "$@"; do
  case "$arg" in
    --verify-only) VERIFY_ONLY=1 ;;
    --prune-external-ptx) PRUNE_EXTERNAL_PTX=1 ;;
    -h|--help)
      sed -n '1,24p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

for tool in sha256sum awk grep; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done
if [[ "$VERIFY_ONLY" -eq 0 ]]; then
  [[ -x "$NVCC" ]] || { echo "nvcc not executable: $NVCC" >&2; exit 1; }
  command -v "$CCBIN" >/dev/null 2>&1 || [[ -x "$CCBIN" ]] || {
    echo "NVCC_CCBIN not found: $CCBIN" >&2
    exit 1
  }
fi
[[ -x "$CUOBJDUMP" ]] || { echo "cuobjdump not executable: $CUOBJDUMP" >&2; exit 1; }

# source-file|implementation-id|entry-symbol
CATALOGUE=(
  "sm61_exact_grouped_k00.cu|candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256|flow_v0322_ct1d_s32_g2_u1_b256"
  "sm61_exact_grouped_k01.cu|candle.sm61-exact-grouped.ct1d-s32-g4-u1-b256|flow_v0322_ct1d_s32_g4_u1_b256"
  "sm61_exact_grouped_k02.cu|candle.sm61-exact-grouped.ct1d-s32-g8-u1-b256|flow_v0322_ct1d_s32_g8_u1_b256"
  "sm61_exact_grouped_k03.cu|candle.sm61-exact-grouped.ct1d-s32-g16-u1-b256|flow_v0322_ct1d_s32_g16_u1_b256"
  "sm61_exact_grouped_k04.cu|candle.sm61-exact-grouped.ct2d-s32-g16-u4-b128|flow_v0322_ct2d_s32_g16_u4_b128"
  "sm61_exact_grouped_k05.cu|candle.sm61-exact-grouped.ct2d-s32-g32-u4-b64|flow_v0322_ct2d_s32_g32_u4_b64"
  "sm61_exact_grouped_k06.cu|candle.sm61-exact-grouped.gc1d-l128-g8-u1-b256|flow_v0322_gc1d_l128_g8_u1_b256"
)

SOURCE_DIR="$ROOT/candle-kernels/src/sm61_exact_grouped"
mkdir -p "$ART"

verify_one() {
  local source_file="$1" impl="$2" entry="$3"
  local cu="$ART/$impl.cu"
  local cubin="$ART/$impl.cubin"
  local manifest="$ART/$impl.manifest"

  for f in "$cu" "$cubin" "$manifest"; do
    [[ -f "$f" ]] || { echo "VERIFY_FAIL missing=$f" >&2; return 1; }
  done

  local repo_source="$SOURCE_DIR/$source_file"
  local cubin_sha source_sha repo_source_sha
  [[ -f "$repo_source" ]] || { echo "VERIFY_FAIL missing=$repo_source" >&2; return 1; }
  cubin_sha="$(sha256sum "$cubin" | awk '{print $1}')"
  source_sha="$(sha256sum "$cu" | awk '{print $1}')"
  repo_source_sha="$(sha256sum "$repo_source" | awk '{print $1}')"
  [[ "$source_sha" == "$repo_source_sha" ]] || {
    echo "VERIFY_FAIL impl=$impl reason=source_sha256 installed=$source_sha repo=$repo_source_sha" >&2
    return 1
  }

  grep -qx 'ASD-CUDA-MODULE-V1' <(head -n1 "$manifest") || {
    echo "VERIFY_FAIL impl=$impl reason=manifest_header" >&2; return 1;
  }
  grep -qx 'abi_version=1' "$manifest" || {
    echo "VERIFY_FAIL impl=$impl reason=abi" >&2; return 1;
  }
  grep -qx "implementation_id=$impl" "$manifest" || {
    echo "VERIFY_FAIL impl=$impl reason=implementation" >&2; return 1;
  }
  grep -qx 'architecture=sm61' "$manifest" || {
    echo "VERIFY_FAIL impl=$impl reason=architecture" >&2; return 1;
  }
  grep -qx 'artifact_kind=cubin' "$manifest" || {
    echo "VERIFY_FAIL impl=$impl reason=artifact_kind" >&2; return 1;
  }
  grep -qx "entry=$entry" "$manifest" || {
    echo "VERIFY_FAIL impl=$impl reason=entry" >&2; return 1;
  }
  grep -qx "artifact_sha256=$cubin_sha" "$manifest" || {
    echo "VERIFY_FAIL impl=$impl reason=sha256" >&2; return 1;
  }

  "$CUOBJDUMP" --dump-resource-usage "$cubin" |
    grep -Fq "Function $entry:" || {
      echo "VERIFY_FAIL impl=$impl reason=entry_not_in_cubin" >&2; return 1;
    }

  printf 'VERIFY_OK implementation=%s source_sha256=%s cubin_sha256=%s entry=%s\n' \
    "$impl" "$source_sha" "$cubin_sha" "$entry"
}

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  failures=0
  for row in "${CATALOGUE[@]}"; do
    IFS='|' read -r source_file impl entry <<<"$row"
    verify_one "$source_file" "$impl" "$entry" || failures=$((failures + 1))
  done
  [[ "$failures" -eq 0 ]] || exit 1
  echo "STATUS=PASS artifacts=7 mode=verify_only"
  exit 0
fi

STAGE="$(mktemp -d "$ART/.install.XXXXXX")"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

CATALOGUE_FILE="$STAGE/ASD-SM61-EXACT-GROUPED-CATALOGUE-V1"
{
  echo "ASD-SM61-EXACT-GROUPED-CATALOGUE-V1"
  echo "architecture=sm61"
  echo "artifact_kind=cubin"
  echo "compiler=$NVCC"
  "$NVCC" --version | sed 's/^/compiler_version=/'
  echo "ccbin=$CCBIN"
} > "$CATALOGUE_FILE"

for row in "${CATALOGUE[@]}"; do
  IFS='|' read -r source_file impl entry <<<"$row"
  src="$SOURCE_DIR/$source_file"
  [[ -f "$src" ]] || { echo "missing source: $src" >&2; exit 1; }

  stage_cu="$STAGE/$impl.cu"
  stage_cubin="$STAGE/$impl.cubin"
  stage_manifest="$STAGE/$impl.manifest"

  cp "$src" "$stage_cu"

  "$NVCC" \
    -gencode=arch=compute_61,code=sm_61 \
    --cubin \
    --default-stream per-thread \
    --expt-relaxed-constexpr \
    -std=c++17 \
    -O3 \
    -allow-unsupported-compiler \
    -Wno-deprecated-gpu-targets \
    -ccbin "$CCBIN" \
    "$stage_cu" \
    -o "$stage_cubin"

  cubin_sha="$(sha256sum "$stage_cubin" | awk '{print $1}')"
  source_sha="$(sha256sum "$stage_cu" | awk '{print $1}')"

  "$CUOBJDUMP" --dump-resource-usage "$stage_cubin" |
    grep -Fq "Function $entry:" || {
      echo "compiled CUBIN missing expected entry: $entry" >&2
      exit 1
    }

  cat > "$stage_manifest" <<EOF
ASD-CUDA-MODULE-V1
abi_version=1
implementation_id=$impl
architecture=sm61
artifact_kind=cubin
entry=$entry
artifact_sha256=$cubin_sha
EOF

  printf 'artifact|implementation=%s|source_file=%s|source_sha256=%s|entry=%s|cubin_sha256=%s\n' \
    "$impl" "$source_file" "$source_sha" "$entry" "$cubin_sha" >> "$CATALOGUE_FILE"

  printf 'STAGED implementation=%s cubin_sha256=%s\n' "$impl" "$cubin_sha"
done

# Publish only after all seven CUBINs, entry checks and manifests succeeded.
# Staging lives inside ART, so each final rename stays on the same filesystem.
for row in "${CATALOGUE[@]}"; do
  IFS='|' read -r source_file impl entry <<<"$row"
  mv -f "$STAGE/$impl.cu" "$ART/$impl.cu"
  mv -f "$STAGE/$impl.cubin" "$ART/$impl.cubin"
  mv -f "$STAGE/$impl.manifest" "$ART/$impl.manifest"
  if [[ "$PRUNE_EXTERNAL_PTX" -eq 1 ]]; then
    rm -f "$ART/$impl.ptx"
  fi
done
mv -f "$CATALOGUE_FILE" "$ART/ASD-SM61-EXACT-GROUPED-CATALOGUE-V1"

# Known reference for K00 on the current CUDA 12.9 toolchain. This is informative,
# not a portability gate: manifests bind the actual compiler output.
K00_IMPL='candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256'
K00_KNOWN_SHA='c5613e589dffc62e1c5094276eb6fdcf52cd0cc7f60316a4f7c7a1bfe50b3e62'
K00_ACTUAL_SHA="$(sha256sum "$ART/$K00_IMPL.cubin" | awk '{print $1}')"
if [[ "$K00_ACTUAL_SHA" == "$K00_KNOWN_SHA" ]]; then
  echo "K00_REFERENCE_MATCH sha256=$K00_ACTUAL_SHA"
else
  echo "K00_REFERENCE_DIFFERENT expected=$K00_KNOWN_SHA actual=$K00_ACTUAL_SHA"
fi

failures=0
for row in "${CATALOGUE[@]}"; do
  IFS='|' read -r source_file impl entry <<<"$row"
  verify_one "$source_file" "$impl" "$entry" || failures=$((failures + 1))
done
[[ "$failures" -eq 0 ]] || exit 1

echo "ASD_ARTIFACT_DIR=$ART"
echo "CATALOGUE=$ART/ASD-SM61-EXACT-GROUPED-CATALOGUE-V1"
echo "BUILTIN_RAW=preserved"
echo "STATUS=PASS artifacts=7 mode=install"
