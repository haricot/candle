#!/usr/bin/env python3
"""Auditable prepare summary: the entire Extended source set, not only Legacy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

SHA = re.compile(r"[a-f0-9]{40}\Z")
LEGACY = (
    "bf16_candle", "fp8_candle", "fp4_candle",
    "cudnn_fallback_candle", "moe_simt_f16_candle",
)


def summarize(config: dict, extended: dict, legacy: dict) -> str:
    configured = config.get("features")
    entries = extended.get("features")
    legacy_entries = legacy.get("features")
    if (not isinstance(configured, list) or not isinstance(entries, list) or
        not isinstance(legacy_entries, list)):
        raise ValueError("source configuration or manifest has no feature list")
    expected = [f.get("feature") for f in configured]
    selected = [f.get("feature") for f in entries]
    shared = [f.get("feature") for f in legacy_entries]
    if (not all(isinstance(name, str) for name in expected + selected + shared) or
        len(expected) != len(set(expected)) or selected != expected or
        shared != list(LEGACY) or not set(shared).issubset(set(selected))):
        raise ValueError("Extended is not an exact ordered composition of all configured sources")
    if (extended.get("kind") != "extended-multi" or
        legacy.get("kind") != "legacy-cuda-five" or
        extended.get("feature_count") != len(expected) or
        legacy.get("feature_count") != len(LEGACY) or
        extended.get("campaign") != legacy.get("campaign") or
        extended.get("main_sha") != legacy.get("main_sha") or
        not SHA.fullmatch(str(extended.get("main_sha", "")))):
        raise ValueError("candidate count, campaign or base-main identity mismatch")
    for cfg, item in zip(configured, entries):
        if (item.get("source_ref") != cfg.get("source_ref") or
            item.get("review_source_sha") != cfg.get("review_source_sha") or
            not SHA.fullmatch(str(item.get("source_sha", ""))) or
            not SHA.fullmatch(str(item.get("prepared_sha", "")))):
            raise ValueError(f"{item.get('feature')}: source or prepared SHA does not match its contract")
    by_name = {entry["feature"]: entry for entry in entries}
    for item in legacy_entries:
        if any(item.get(key) != by_name[item["feature"]].get(key)
               for key in ("source_ref", "source_sha", "prepared_sha", "review_source_sha")):
            raise ValueError(f"{item['feature']}: Legacy differs from Extended")
    lines = [
        "## Unified preparation — all standalone sources",
        "",
        f"**Extended: {len(expected)}/{len(expected)} configured standalone sources prepared.** "
        f"**cuda_legacy: {len(LEGACY)}/{len(LEGACY)} shared sources** "
        "(a subset of Extended, not five additional sources).",
        "",
        f"- Campaign: `{extended['campaign']}`",
        f"- Shared `main`: `{extended['main_sha']}`",
        "- Qualification: **UNVALIDATED_CANDIDATE** — hosted CPU/CUDA and physical SM61 gates "
        "are separate; preparation alone does not promote any stable ref.",
        "",
        "| # | Standalone source | Source branch | Source SHA | Prepared SHA | Used by |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for index, entry in enumerate(entries, 1):
        feature = entry["feature"]
        # Names are code-controlled, but Markdown must remain safe if a manifest changes.
        safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        source = entry["source_sha"][:12]
        prepared = entry["prepared_sha"][:12]
        used_by = "Extended + cuda_legacy" if feature in LEGACY else "Extended"
        lines.append(f"| {index} | `{safe(feature)}` | `{safe(entry['source_ref'])}` | "
                     f"`{source}` | `{prepared}` | {used_by} |")
    lines += ["", f"**Total independent standalone sources: {len(expected)}.**", ""]
    return "\n".join(lines)


def selftest() -> None:
    features = [*LEGACY, *(f"standalone_{i}" for i in range(9))]
    entries = [dict(feature=name, source_ref=name, source_sha="a" * 40,
                    prepared_sha="b" * 40, review_source_sha="c" * 40)
               for name in features]
    config = {"features": [{"feature": f, "source_ref": f,
                            "review_source_sha": "c" * 40} for f in features]}
    ext = dict(kind="extended-multi", campaign="test-14", main_sha="d" * 40,
               feature_count=14, features=entries)
    legacy = dict(kind="legacy-cuda-five", campaign="test-14", main_sha="d" * 40,
                  feature_count=5, features=entries[:5])
    report = summarize(config, ext, legacy)
    assert "Extended: 14/14" in report and "cuda_legacy: 5/5" in report
    assert report.count("| Extended") == 14
    for change in ("missing", "different-prepared-sha", "bad-base"):
        variant = json.loads(json.dumps(ext))
        if change == "missing":
            variant["features"].pop()
        elif change == "different-prepared-sha":
            variant["features"][0]["prepared_sha"] = "0" * 40
        else:
            variant["main_sha"] = "f" * 40
        try:
            summarize(config, variant, legacy)
        except ValueError:
            continue
        raise AssertionError("Accepted inconsistent source summary: " + change)
    print("CANDLE_UNIFIED_SOURCE_SUMMARY_SELFTEST=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(".github/extended-sources.json"))
    parser.add_argument("--extended", type=Path)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    try:
        if args.selftest:
            selftest()
        else:
            if not all((args.extended, args.legacy, args.summary)):
                parser.error("--extended, --legacy and --summary are required")
            report = summarize(json.loads(args.config.read_text()),
                               json.loads(args.extended.read_text()),
                               json.loads(args.legacy.read_text()))
            with args.summary.open("a", encoding="utf-8") as f:
                f.write(report)
            print("CANDLE_UNIFIED_SOURCE_SUMMARY=PASS source_count=" +
                  str(len(json.loads(args.config.read_text())["features"])))
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        print(f"CANDLE_UNIFIED_SOURCE_SUMMARY=FAIL reason={exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
