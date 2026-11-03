#!/usr/bin/env python3
"""Export readable names from an attested CI archive without executing binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat

NAMES = {
    "nemotron_asr_cuda_cli": ("applications", "nemotron_asr"),
    "pockettts_flowlm_mimi_cuda": ("diagnostics", "pockettts-sm61"),
    "asd_module_provider": ("diagnostics", "asd_v3_module_provider_validate"),
    "asd_runtime_profile_swap": ("diagnostics", "asd_v3_phase_d_runtime_profile_swap_validate"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(bundle, destination, expected):
    bundle = Path(bundle).resolve()
    destination = Path(destination)
    if destination.exists():
        raise ValueError("destination must be new")
    manifest_path = bundle / "manifest.json"
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or digest(manifest_path) != expected:
        raise ValueError("trusted manifest SHA256 mismatch")
    if (bundle / "manifest.sha256").read_text().strip() != expected:
        raise ValueError("archive manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1 or not manifest.get("gates"):
        raise ValueError("unknown or empty archive")
    records = []
    for gate, record in sorted(manifest["gates"].items(), key=lambda item: (item[0] not in NAMES, item[0])):
        if not re.fullmatch(r"[a-z0-9_-]+", gate):
            raise ValueError("unsafe gate name")
        for number, binary in enumerate(record.get("binaries", []), 1):
            fingerprint = binary["sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", fingerprint) or binary["path"] != "bin/" + fingerprint:
                raise ValueError("unsafe binary path")
            source = bundle / binary["path"]
            if (source.is_symlink() or not stat.S_ISREG(source.stat().st_mode)
                    or source.stat().st_size != binary["bytes"] or digest(source) != fingerprint):
                raise ValueError("binary SHA256/size mismatch: " + gate)
            with source.open("rb") as stream:
                magic = stream.read(4)
            if magic != b"\x7fELF":
                raise ValueError("not an ELF executable")
            if gate in NAMES:
                if len(record["binaries"]) != 1:
                    raise ValueError("application must have exactly one executable")
                category, name = NAMES[gate]
            elif record["kind"] == "test":
                category, name = "tests", gate + ("-" + str(number) if len(record["binaries"]) > 1 else "")
            else:
                category, name = "diagnostics", gate + "-" + str(number)
            # One file per original SHA; shared test/model gates remain aliases
            # in the index. A parity gate can refer to its application binary.
            prior = next((r for r in records if r["sha256"] == fingerprint), None)
            path = prior["path"] if prior else category + "/" + name
            records.append({"gate":gate,"kind":record["kind"],"path":path,
                            "sha256":fingerprint,"bytes":binary["bytes"],"source":source,
                            "argv":record.get("argv",[]),"cwd":record.get("cwd",".")})
    destination.mkdir(parents=True)
    for record in records:
        target = destination / record["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record["source"], target)
            target.chmod(0o755)
    provenance = destination / "provenance"
    provenance.mkdir()
    shutil.copyfile(manifest_path, provenance / "ci-manifest.json")
    index = {"schema":"candle-readable-programs-v1", "candidate":manifest["candidate_sha"],
             "run_id":manifest["run_id"], "run_attempt":manifest["run_attempt"],
             "cpu_profile":manifest.get("cpu_profile", "portable"), "rustflags":manifest["rustflags"],
             "original_manifest_sha256":expected,
             "executables":[{k:v for k,v in r.items() if k != "source"} for r in records]}
    (destination / "index.json").write_text(json.dumps(index,indent=2,sort_keys=True)+"\n")
    rows = "\n".join("| `"+r["gate"]+"` | `"+r["path"]+"` |" for r in records)
    (destination / "README.md").write_text(
        "# Candle - executables nommes\n\n"
        "Ces fichiers sont copies octet pour octet depuis l'archive CI, sans recompilation.\n\n"
        "- `applications/` : programmes utilisateur. `nemotron_asr` exige ses poids et un WAV; "
        "les dossiers absents ne sont pas produits par cette campagne.\n"
        "- `diagnostics/` : validation ASD et `pockettts-sm61`, exemple de qualification "
        "d'operateurs, pas le moteur PocketTTS C3.\n"
        "- `tests/` : executables du harnais de tests Rust, pas des applications de modele.\n"
        "- `index.json` : correspondance complete des gates, fichiers et SHA256; "
        "plusieurs gates peuvent partager le meme executable.\n\n"
        "Verifier les fichiers : `sha256sum -c SHA256SUMS`.\n\n"
        "Ce paquet nomme ne constitue pas une nouvelle preuve GPU. Etat et preuves : "
        "https://github.com/haricot/candle/actions/runs/"+str(manifest["run_id"])+"\n\n"
        "Candidat : `"+manifest["candidate_sha"]+"`. Profil CPU : `"+index["cpu_profile"]+"`.\n\n"
        "Le manifeste original dans provenance conserve ses chemins internes `bin/<sha256>`; "
        "index.json relie ces empreintes aux nouveaux noms. L'archive CI originale reste intacte.\n\n"
        "| Gate CI | Fichier dans ce paquet |\n|---|---|\n"+rows+"\n")
    paths = sorted(p for p in destination.rglob("*") if p.is_file())
    (destination / "SHA256SUMS").write_text("".join(digest(p)+"  "+p.relative_to(destination).as_posix()+"\n" for p in paths))
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-sha256",required=True)
    args = parser.parse_args()
    result = export(args.bundle,args.out,args.expected_sha256)
    print("CANDLE_READABLE_PROGRAMS=PASS gates="+str(len(result["executables"])))
