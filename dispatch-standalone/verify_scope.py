import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
meta = json.loads((ROOT / "dispatch-standalone/sources.json").read_text())
base = meta["base"]["sha"]
subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", base, "HEAD"], check=True)
changed = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--name-only", base], text=True).splitlines()
untracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
assert set(changed + untracked) <= set(meta["current_allowed_paths"])
assert meta["implementation_gate"]["state"] == "source_audit_complete"
for path, expected in meta["payload_sha256"].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
assert not any(meta["qualification"][key] for key in ["gpu_validated", "models_executed", "performance_qualified"])
rust = [str(ROOT / path) for path in sorted(meta["current_allowed_paths"]) if path.endswith(".rs")]
subprocess.run(["rustfmt", "--check", "--edition", "2021", "--config", "skip_children=true", *rust], check=True)
print("DISPATCH_S0_SOURCE_SCOPE=PASS")
