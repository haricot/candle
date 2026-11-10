#!/usr/bin/env python3
"""Read-only Unified campaign map for GITHUB_STEP_SUMMARY.

The job graph comes from the workflow YAML, and the source groups come from
the two source contracts. GitHub's API adds live workflow states/run history;
an API failure never suppresses the local graph or masks a qualification gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


WORKFLOW = Path(".github/workflows/candle-unified-orchestrator.yml")
EXTENDED = Path(".github/extended-sources.json")
LEGACY = Path(".github/legacy-cuda-sources.json")
JOB_HEADER = re.compile(r"^  ([a-zA-Z][a-zA-Z_0-9-]*):\s*(?:#.*)?$", re.M)
JOB_ID = re.compile(r"[a-zA-Z][a-zA-Z_0-9-]*\Z")
REF_NAME = re.compile(r"[a-zA-Z0-9_.-]+\Z")
STATUS = {
    "success": "✅ success",
    "failure": "❌ failure",
    "cancelled": "⛔ cancelled",
    "skipped": "⏭️ skipped",
    "in_progress": "🔄 running",
    "queued": "⏳ queued",
    "waiting": "⏳ waiting",
    "pending": "⏳ pending",
}


def safe(value: object) -> str:
    return html.escape(str(value), quote=False).replace("|", "\\|").replace(
        "\r", " ").replace("\n", " ")


def jobs_from_workflow(contents: str) -> dict[str, dict]:
    """Read just the root jobs mapping; no installed YAML library required."""
    marker = re.search(r"^jobs:\s*$", contents, re.M)
    if not marker:
        raise ValueError("workflow has no jobs mapping")
    content = contents[marker.end():]
    matches = list(JOB_HEADER.finditer(content))
    if not matches:
        raise ValueError("workflow has no jobs")
    jobs = {}
    for index, match in enumerate(matches):
        body = content[match.end():matches[index + 1].start() if index + 1 < len(matches) else None]
        name_match = re.search(r"^    name:\s*(.+)$", body, re.M)
        name = name_match.group(1).strip().strip("'\"") if name_match else match.group(1)
        needs_match = re.search(r"^    needs:\s*(.*)$", body, re.M)
        deps = []
        if needs_match:
            raw = needs_match.group(1).strip().split("#", 1)[0].strip()
            if raw.startswith("[") and raw.endswith("]"):
                deps = [part.strip().strip("'\"") for part in raw[1:-1].split(",") if part.strip()]
            elif raw:
                deps = [raw.strip("'\"")]
            else:
                tail = body[needs_match.end():]
                for line in tail.splitlines():
                    item = re.match(r"^      -\s*([a-zA-Z][\w-]*)\s*$", line)
                    if item:
                        deps.append(item.group(1))
                    elif line and not line.startswith("      "):
                        break
        jobs[match.group(1)] = {"name": name, "needs": deps}
    for key, data in jobs.items():
        if not JOB_ID.fullmatch(key) or any(dep not in jobs for dep in data["needs"]):
            raise ValueError(f"unknown dependency in job {key}")
    return jobs


def source_groups(extended: dict, legacy: dict) -> tuple[list[dict], set[str]]:
    ext = extended.get("features", [])
    leg = legacy.get("features", [])
    names = [entry.get("feature") for entry in ext]
    shared = [entry.get("feature") for entry in leg]
    if (not ext or not leg or len(names) != len(set(names))
            or len(shared) != len(set(shared)) or not set(shared).issubset(names)
            or not all(isinstance(name, str) and REF_NAME.fullmatch(name) for name in names + shared)):
        raise ValueError("source configuration is inconsistent")
    return ext, set(shared)


def state_for(job: str, statuses: dict) -> str:
    value = statuses.get(job, {})
    if isinstance(value, dict):
        value = value.get("result") or value.get("status")
    return STATUS.get(str(value), "—")


def node_label(value: str) -> str:
    # Mermaid labels are quoted; strip syntax that could inject new diagram nodes.
    return html.escape(re.sub(r"[\[\]{}\"\x00-\x1f]", " ", value), quote=False)[:110]


def diagram(jobs: dict, statuses: dict, ext_count: int, legacy_count: int) -> str:
    def upstream(job: str) -> set[str]:
        visited = set()
        stack = list(jobs[job]["needs"])
        while stack:
            dep = stack.pop()
            if dep not in visited:
                visited.add(dep)
                stack.extend(jobs[dep]["needs"])
        return visited

    lines = [
        "flowchart TD",
        f'  sources["Standalone sources ({ext_count})"]',
        f'  shared["Legacy subset ({legacy_count})"]',
    ]
    for key, info in jobs.items():
        label = node_label(info["name"])
        result = state_for(key, statuses)
        lines.append(f'  j_{key}["{label} — {node_label(result)}"]')
    if "prepare" in jobs:
        lines.extend(["  sources --> j_prepare", "  sources -.->|shared subset| shared",
                      "  shared --> j_prepare"])
    for key, info in jobs.items():
        for dep in info["needs"]:
            # The table retains every declared need; show only non-redundant
            # edges in the picture, otherwise large fan-ins hide the stages.
            if not any(dep in upstream(other) for other in info["needs"] if other != dep):
                lines.append(f"  j_{dep} --> j_{key}")
    return "\n".join(lines)


def github_api(url: str) -> dict | list:
    token = os.environ.get("GH_TOKEN", "")
    req = Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "candle-unified-topology",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    with urlopen(req, timeout=9) as response:
        return json.loads(response.read(1_000_000).decode("utf-8"))


def api_evidence(repo: str, query=github_api) -> tuple[dict, list, list, list[str]]:
    """API errors affect display only; callers must not convert them into PASS."""
    base = f"https://api.github.com/repos/{repo}"
    errors = []
    workflows, runs, tags = {}, [], []
    for label, path in (
        ("workflows", "/actions/workflows?per_page=100"),
        ("runs", "/actions/workflows/candle-unified-orchestrator.yml/runs?per_page=12"),
        ("retention", "/git/matching-refs/tags/candle-retention/"),
    ):
        try:
            data = query(base + path)
            if label == "workflows":
                workflows = {w["path"]: w for w in data.get("workflows", []) if "path" in w}
            elif label == "runs":
                runs = data.get("workflow_runs", [])
            else:
                tags = data if isinstance(data, list) else []
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
            # No history tags yet is a normal 404.
            if not (label == "retention" and getattr(exc, "code", None) == 404):
                errors.append(label)
    # Annotated tags store compact, immutable {campaign, run, deleted:[ref,sha]}.
    # Read a bounded sample; the map must never wait indefinitely on history.
    for item in sorted(tags, key=lambda value: value.get("ref", ""), reverse=True)[:8]:
        obj = item.get("object", {})
        tag_sha = obj.get("sha", "")
        if obj.get("type") != "tag" or not re.fullmatch(r"[a-f0-9]{40}", tag_sha):
            continue
        try:
            annotation = query(base + "/git/tags/" + tag_sha)
            message = json.loads(annotation.get("message", "{}"))
            if message.get("schema") == "candle-unified-retention-history.v1":
                item["history"] = message
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
            errors.append("retention-details")
            break
    return workflows, runs, tags, errors


def timestamp(raw: object) -> str:
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(
            timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return "—"


def render(jobs: dict, sources: list[dict], shared: set[str], statuses: dict,
           repo: str, run_id: str, campaign: str, workflows: dict,
           runs: list, tags: list, api_errors: list[str]) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", repo):
        raise ValueError("invalid repository")
    url = f"https://github.com/{repo}/actions/runs/{run_id}" if run_id.isdecimal() else ""
    lines = [
        "## Unified CI — live topology",
        "",
        f"**Campaign:** `{safe(campaign or 'not resolved')}` · "
        f"**Sources:** {len(sources)} Extended, {len(shared)} also in cuda_legacy "
        "(shared, not additional).",
        "",
        "```mermaid",
        diagram(jobs, statuses, len(sources), len(shared)),
        "```",
        "",
        "| Gate | Result | Depends on |",
        "| --- | --- | --- |",
    ]
    for key, info in jobs.items():
        lines.append(f"| `{safe(key)}` | {state_for(key, statuses)} | "
                     f"{', '.join('`' + dep + '`' for dep in info['needs']) or '—'} |")
    if url:
        lines.extend(["", f"[Current workflow run]({url})"])

    lines.extend([
        "",
        "<details><summary>Configured standalone source map</summary>",
        "",
        "| Source branch | Consumers |",
        "| --- | --- |",
    ])
    for item in sources:
        name = item["feature"]
        lines.append(f"| `{safe(item['source_ref'])}` | "
                     f"{'Extended + cuda_legacy' if name in shared else 'Extended'} |")
    lines.extend(["", "</details>", "", "### Workflow inventory (this branch)", ""])
    active_files = sorted(Path(".github/workflows").glob("*.yml"))
    names = []
    for file in active_files:
        name = re.search(r"^name:\s*(.+)$", file.read_text(encoding="utf-8"), re.M)
        registered = workflows.get(file.as_posix())
        state = registered.get("state", "not registered") if registered else "unknown"
        names.append((file.name, name.group(1).strip() if name else file.stem, state))
    lines.extend([
        f"{len(names)} YAML files in the checked-out branch. "
        "GitHub's sidebar can also display historical workflows.",
        "",
        "<details><summary>Workflow files and GitHub state</summary>",
        "",
        "| Workflow | GitHub state |",
        "| --- | --- |",
    ])
    for filename, name, state in names:
        lines.append(f"| `{safe(filename)}` — {safe(name)} | {safe(state)} |")
    lines.extend(["", "</details>", "", "### Recent Unified campaigns", "",
                  "| Run | Campaign (if available) | Event | Result | Started (UTC) |",
                  "| --- | --- | --- | --- | --- |"])
    campaigns_from_tags = {
        str(tag["history"]["run"]): tag["history"]["campaign"]
        for tag in tags if isinstance(tag, dict) and isinstance(tag.get("history"), dict)
        and "run" in tag["history"] and "campaign" in tag["history"]
    }
    if runs:
        for run in runs[:12]:
            identity = run.get("id")
            if not isinstance(identity, int):
                continue
            inputs = run.get("inputs") or {}
            name = inputs.get("campaign") if isinstance(inputs, dict) else None
            name = name or campaigns_from_tags.get(str(identity), "—")
            result = run.get("conclusion") or run.get("status") or "—"
            lines.append(f"| [#{identity}](https://github.com/{repo}/actions/runs/{identity}) "
                         f"| `{safe(name)}` | {safe(run.get('event', '—'))} "
                         f"| {STATUS.get(result, safe(result))} "
                         f"| {timestamp(run.get('created_at'))} |")
    else:
        lines.append("| — | — | — | No history available | — |")
    relevant_tags = []
    for item in tags:
        ref = item.get("ref", "") if isinstance(item, dict) else ""
        if ref.startswith("refs/tags/candle-retention/"):
            relevant_tags.append(ref.removeprefix("refs/tags/"))
    if relevant_tags:
        lines.extend(["", "<details><summary>Immutable retention records (Git tags)</summary>",
                      "", "| Campaign | Run | Deleted refs | Tag |",
                      "| --- | --- | ---: | --- |"])
        for item in sorted(tags, key=lambda x: x.get("ref", ""), reverse=True)[:12]:
            tag = item.get("ref", "").removeprefix("refs/tags/")
            if not tag.startswith("candle-retention/"):
                continue
            history = item.get("history", {})
            lines.append(f"| `{safe(history.get('campaign', '—'))}` "
                         f"| {safe(history.get('run', '—'))} "
                         f"| {len(history.get('deleted', [])) if isinstance(history.get('deleted'), list) else '—'} "
                         f"| [`{safe(tag)}`](https://github.com/{repo}/tree/"
                         f"{quote(tag, safe='/')}) |")
        lines.extend(["", "</details>"])
    if api_errors:
        lines.extend(["", f"> GitHub API unavailable for: {', '.join(api_errors)}. "
                      "Topology is from checked-out files; missing statuses are not PASS."])
    lines.append("")
    return "\n".join(lines)


def selftest() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    jobs = jobs_from_workflow(text)
    assert "prepare" in jobs and "physical_sm61" in jobs
    assert "prepare" in jobs["cpu_cuda_compile"]["needs"]
    assert "physical_sm61" in jobs["promote_validated"]["needs"]
    assert "campaign_overview" in jobs
    assert "promote_validated" in jobs["campaign_overview"]["needs"]
    ext, shared = source_groups(json.loads(EXTENDED.read_text()), json.loads(LEGACY.read_text()))
    assert len(ext) >= len(shared) > 0
    report = render(jobs, ext, shared, {"prepare": {"result": "failure"}},
                    "haricot/candle", "123", "test", {}, [], [], [])
    assert "❌ failure" in report and "⏭️ skipped" not in report
    assert report.count("| Extended + cuda_legacy |") == len(shared)
    assert "flowchart TD" in report and "j_cuda_image --> j_cpu_cuda_compile" in report
    extra = {"feature": "future_standalone", "source_ref": "future_standalone"}
    growing, reused = source_groups(
        {"features": ext + [extra]},
        {"features": [item for item in ext if item["feature"] in shared]})
    assert f"**Sources:** {len(ext) + 1} Extended" in render(
        jobs, growing, reused, {}, "haricot/candle", "124",
        "next", {}, [], [], [])

    def fixture_api(url: str):
        if "/actions/workflows?per_page" in url:
            return {"workflows": [{"path": WORKFLOW.as_posix(), "state": "active"}]}
        if "/actions/workflows/candle-unified-orchestrator.yml/runs" in url:
            return {"workflow_runs": [{"id": 123, "event": "workflow_dispatch",
                                       "conclusion": "success",
                                       "created_at": "2026-11-10T09:00:00Z"}]}
        if "/git/matching-refs/tags/" in url:
            return [{"ref": "refs/tags/candle-retention/test",
                     "object": {"type": "tag", "sha": "a" * 40}}]
        if "/git/tags/" in url:
            return {"message": json.dumps({
                "schema": "candle-unified-retention-history.v1", "campaign": "test",
                "run": "123", "deleted": [{"ref": "prepare/extended/test/a",
                                          "sha": "b" * 40}]})}
        raise AssertionError(url)

    workflows, runs, tags, errors = api_evidence("haricot/candle", fixture_api)
    assert not errors and len(tags) == 1
    report = render(jobs, ext, shared, {}, "haricot/candle", "123", "",
                    workflows, runs, tags, errors)
    assert "| `test` | workflow_dispatch | ✅ success" in report
    assert "| 1 |" in report and "candle-retention/test" in report
    small = jobs_from_workflow("jobs:\n  a:\n    needs: [b]\n  b:\n    name: Second\n")
    assert small["a"]["needs"] == ["b"]
    try:
        jobs_from_workflow("jobs:\n  a:\n    needs: [missing]\n")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted unknown dependency")
    # Exercise the *production* jq filter, not a duplicated fixture.
    shell = Path(".github/scripts/candle-legacy-assemble.sh").read_text()
    filters = re.findall(r"'(\{schema_version:1,kind:\"legacy-cuda-five\".*?features:\.\})'",
                         shell, re.S)
    assert len(filters) == 1, "Legacy assembly filter not found"
    selected = [item for item in ext if item["feature"] in shared]
    fixture = [dict(feature=item["feature"], source_ref=item["source_ref"],
                    review_source_sha=item["review_source_sha"],
                    source_sha="c" * 40, prepared_sha="d" * 40)
               for item in selected]
    data = "\n".join(json.dumps(item) for item in fixture) + "\n"
    cmd = ["jq", "-s", "--arg", "main_sha", "a" * 40, "--arg", "campaign", "test",
           "--arg", "strategy", "exact-delta", "--arg", "old_integration_sha", "",
           "--arg", "config_sha256", "b" * 64, filters[0]]
    manifest = json.loads(subprocess.check_output(cmd, input=data, text=True))
    assert manifest["feature_count"] == len(shared) == len(manifest["features"])
    ext_fixture = {"kind": "extended-multi", "main_sha": "a" * 40,
                   "campaign": "test", "feature_count": len(fixture),
                   "features": fixture}
    config_fixture = {"features": selected}
    source_summary = runpy.run_path(".github/scripts/candle-unified-source-summary.py")
    assert "cuda_legacy" in source_summary["summarize"](
        config_fixture, ext_fixture, manifest)
    print("CANDLE_UNIFIED_TOPOLOGY_SELFTEST=PASS")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--no-api", action="store_true")
    p.add_argument("--output-json", type=Path)
    args = p.parse_args()
    if args.selftest:
        selftest()
        return 0
    jobs = jobs_from_workflow(WORKFLOW.read_text(encoding="utf-8"))
    sources, shared = source_groups(json.loads(EXTENDED.read_text()), json.loads(LEGACY.read_text()))
    statuses = json.loads(os.environ.get("NEEDS_JSON", "{}"))
    repo = os.environ.get("GITHUB_REPOSITORY", "haricot/candle")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    campaign = os.environ.get("CAMPAIGN", "")
    wf, runs, tags, errors = ({}, [], [], []) if args.no_api else api_evidence(repo)
    summary = render(jobs, sources, shared, statuses, repo, run_id, campaign,
                     wf, runs, tags, errors)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as out:
            out.write(summary)
    else:
        print(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({
            "schema": "candle-unified-topology.v1",
            "campaign": campaign or None, "run_id": run_id,
            "source_count": len(sources), "legacy_shared_count": len(shared),
            "jobs": {key: {**value, "result": state_for(key, statuses)}
                     for key, value in jobs.items()},
            "api_errors": errors,
        }, indent=2) + "\n")
    print(f"CANDLE_UNIFIED_TOPOLOGY=PASS jobs={len(jobs)} sources={len(sources)} "
          f"legacy_shared={len(shared)} api_unavailable={','.join(errors) or 'none'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, AssertionError, subprocess.CalledProcessError) as exc:
        print(f"CANDLE_UNIFIED_TOPOLOGY=FAIL reason={exc}", file=sys.stderr)
        raise SystemExit(1)
