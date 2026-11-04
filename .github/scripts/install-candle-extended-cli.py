#!/usr/bin/env python3
"""Install callable model entrypoints and one linked CLI before candidate commit."""
import argparse
import hashlib
import json
from pathlib import Path

PAYLOAD = Path(__file__).resolve().parents[1] / "extended-cli"
ENTRIES = (
    ("quantized-qwen3/main.rs", "anyhow::Result<()>", "clap"),
    ("embedding-gemma2/main.rs", "Result<()>", "clap"),
    ("nemotron_asr.rs", "Result<()>", "clap"),
    ("transcribepsy_tp3.rs", "Result<()>", "clap"),
    ("pockettts-sm61/main.rs", "Result<()>", "manual"),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(repo):
    repo = Path(repo).resolve()
    examples = repo / "candle-examples/examples"
    receipt = repo / "candle-integration/extended-cli.json"
    if receipt.exists():
        raise ValueError("CLI overlay already installed; prepare a fresh candidate")
    writes = {}
    for name, result, parser in ENTRIES:
        path = examples / name
        source = path.read_text()
        marker = f"fn main() -> {result} {{"
        if source.count(marker) != 1 or "pub(crate) fn run_from" in source:
            raise ValueError("unsupported entrypoint structure: " + name)
        wrapper = (marker + "\n    run_from(std::env::args_os().collect())\n}\n\n"
                   + f"pub(crate) fn run_from(argv: Vec<std::ffi::OsString>) -> {result} {{")
        source = source.replace(marker, wrapper, 1)
        old = "Args::parse()" if parser == "clap" else "std::env::args().skip(1).collect()"
        new = "Args::parse_from(argv)" if parser == "clap" else "argv.into_iter().skip(1).map(|a| a.into_string().map_err(|_| Error::Msg(\"non-UTF8 argument\".into()))).collect::<Result<Vec<_>>>()?"
        if source.count(old) != 1:
            raise ValueError("unsupported argument parser: " + name)
        writes[path] = source.replace(old, new, 1)
    for name in ("main.rs", "acoustic.rs"):
        writes[examples / "candle-extended" / name] = (PAYLOAD / name).read_text()
    # Validate the complete edit plan before modifying any source.
    for path, source in writes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    data = {"schema": "candle-extended-cli-source-v1", "program": "candle-extended",
            "installer_sha256": digest(Path(__file__)),
            "files": {str(path.relative_to(repo)): digest(path) for path in writes}}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    print("CANDLE_EXTENDED_CLI_SOURCE=PASS files=" + str(len(writes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    install(parser.parse_args().repo)
