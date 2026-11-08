#!/usr/bin/env python3
"""Bind the five Legacy source identities to a single Extended candidate.

This is an identity/provenance gate. Runtime qualification is performed by the
CUDA-built Extended archive and its physical SM61 tests, not a Legacy rebuild.
"""
import argparse
import json
import sys
from pathlib import Path

LEGACY = (
    "bf16_candle", "fp8_candle", "fp4_candle",
    "cudnn_fallback_candle", "moe_simt_f16_candle",
)
SOURCE_KEYS = (
    "feature", "source_ref", "target_ref", "source_sha",
    "review_source_sha", "prepared_sha", "common_base_sha",
    "source_patch_sha256", "prepared_patch_sha256",
)


def verify(extended: dict, legacy: dict) -> None:
    if extended.get("kind") != "extended-multi":
        raise ValueError("not an Extended candidate")
    if legacy.get("kind") != "legacy-cuda-five":
        raise ValueError("not a five-feature Legacy candidate")
    if legacy.get("promotion_target") != "cuda_legacy":
        raise ValueError("Legacy promotion target changed")
    if any(extended.get(k) != legacy.get(k) for k in ("campaign", "main_sha", "strategy")):
        raise ValueError("different campaign, common main SHA or preparation strategy")
    if extended["strategy"] != "exact-delta":
        raise ValueError("shared preparation must be exact-delta")
    shared = extended.get("features")
    old = legacy.get("features")
    if not isinstance(shared, list) or not isinstance(old, list):
        raise ValueError("missing feature lists")
    if [x.get("feature") for x in old] != list(LEGACY):
        raise ValueError("Legacy feature order or set changed")
    by_name = {x["feature"]: x for x in shared}
    if len(by_name) != len(shared):
        raise ValueError("duplicate Extended feature")
    for item in old:
        feature = item["feature"]
        source = by_name.get(feature)
        if source is None:
            raise ValueError(f"Extended does not contain {feature}")
        for key in SOURCE_KEYS:
            if item.get(key) != source.get(key) or not item.get(key):
                raise ValueError(f"{feature}: inconsistent {key}")
        if item["prepared_sha"] == item["source_sha"] and item["common_base_sha"] != extended["main_sha"]:
            raise ValueError(f"{feature}: unexpected non-rebased standalone")


def selftest() -> None:
    features = [{
        "feature": f, "source_ref": f, "target_ref": f,
        "source_sha": "1" * 40, "review_source_sha": "2" * 40,
        "prepared_sha": "3" * 40, "common_base_sha": "4" * 40,
        "source_patch_sha256": "5" * 64,
        "prepared_patch_sha256": "6" * 64,
    } for f in LEGACY]
    common = {"campaign": "campaign", "main_sha": "7" * 40, "strategy": "exact-delta"}
    extended = dict(common, kind="extended-multi", features=features)
    legacy = dict(common, kind="legacy-cuda-five", promotion_target="cuda_legacy", features=features)
    verify(extended, legacy)
    for mutation in ("source_sha", "prepared_sha", "source_patch_sha256", "prepared_patch_sha256"):
        damaged = json.loads(json.dumps(legacy))
        damaged["features"][0][mutation] = "0" * len(damaged["features"][0][mutation])
        try:
            verify(extended, damaged)
        except ValueError:
            continue
        raise AssertionError(f"accepted mismatched {mutation}")
    print("CANDLE_UNIFIED_PAIR_SELFTEST=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", type=Path)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    try:
        if args.selftest:
            selftest()
        else:
            if not args.extended or not args.legacy:
                parser.error("--extended and --legacy are required")
            verify(json.loads(args.extended.read_text()), json.loads(args.legacy.read_text()))
            print("CANDLE_UNIFIED_PAIR=PASS provenance=shared-extended-preparation")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"CANDLE_UNIFIED_PAIR=FAIL reason={exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
