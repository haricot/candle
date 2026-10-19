#!/usr/bin/env python3
"""Clone a pinned FastConformer/TDT Candle source and adapt its CLI for a
local QVAC model directory. All changes stay in this standalone checkout."""
import argparse
from pathlib import Path
import re
import subprocess
from hdb2a_conformer_patch import patch_conformer

PIN_PARAKEET = '53c0d3065f9dd8de0356b25391e60ed3a471192b'
PIN_CANDLE = 'f5ce7f6f8b7a68d14a55419f9f221e1ae3537fd3'
PARAKEET_URL = 'https://github.com/n0madic/parakeet-candle.git'
CANDLE_URL = 'https://github.com/haricot/candle.git'


def run(*args):
    print('+', *map(str,args), flush=True)
    return subprocess.run([str(x) for x in args], check=True, capture_output=False)


def modify_cargo(text: str, candle_source: str) -> str:
    for name, pkg in [('candle','candle-core'),('candle-nn','candle-nn')]:
        if candle_source == 'git':
            new = f'{name} = {{ package = "{pkg}", git = "{CANDLE_URL}", rev = "{PIN_CANDLE}", version = "0.11", default-features = false }}'
        else:
            new = f'{name} = {{ package = "{pkg}", path = "{candle_source}/{pkg}", version = "0.11", default-features = false }}'
        if name == 'candle':
            pattern = rf'^{re.escape(name)}\s*=\s*\{{\s*package\s*=\s*"{re.escape(pkg)}",\s*version\s*=\s*"0\.10",\s*default-features\s*=\s*false\s*\}}\s*$'
        else:
            pattern = rf'^{re.escape(name)}\s*=\s*\{{\s*version\s*=\s*"0\.10",\s*default-features\s*=\s*false\s*\}}\s*$'
        text, n = re.subn(pattern, lambda _:new, text, flags=re.MULTILINE)
        if n!=1:
            raise ValueError(f'Expected one Candle 0.10 dependency for {name}, found {n}; upstream revision changed')
    # Prevent Cargo from enrolling this vendored runtime in the HDB0 workspace.
    text += '\n[workspace]\nresolver = "2"\n'
    return text


def patch_main(source: str) -> str:
    after = '    model_id: String,\n'
    if source.count(after)!=1:
        raise ValueError('No unique model_id field in pinned parakeet-candle')
    source=source.replace(after, after + '''\n    /// Local converted NeMo checkpoint directory. Disables HF model loading.\n    #[arg(long)]\n    model_dir: Option<PathBuf>,\n''')
    start = source.index('    let api = Api::new()?;\n', source.index('fn main() -> Result<()>'))
    end = source.index('    if args.debug {\n        let total_bytes:', start)
    old=source[start:end]
    if '    let safetensors_files = match repo.get("model.safetensors.index.json")' not in old:
        raise ValueError('Unexpected pinned Candle loader contents: abort patch')
    # Preserve original remote-model logic behind the else branch, but add
    # explicit disk assets first and never silently revert to teacher weights.
    remote='\n'.join('    '+line if line else '' for line in old.rstrip().split('\n'))
    replacement='''    let config_path: PathBuf;\n    let safetensors_files: Vec<PathBuf>;\n    if let Some(dir) = args.model_dir.as_ref() {\n        config_path = dir.join("config.json");\n        safetensors_files = vec![dir.join("model.safetensors")];\n        anyhow::ensure!(config_path.is_file(), "missing local NeMo-derived config.json: {}", config_path.display());\n        anyhow::ensure!(safetensors_files[0].is_file(), "missing local NeMo-derived model.safetensors: {}", safetensors_files[0].display());\n        eprintln!("HDB2A_CANDLE_LOCAL_MODEL={}", dir.display());\n    } else {\n''' + remote + '''\n        // Convert the original locally determined variables to branch outputs.\n        config_path = config_path_remote;\n        safetensors_files = safetensors_files_remote;\n    }\n    let config: serde_json::Value = serde_json::from_reader(File::open(&config_path)?)?;\n'''
    # Rename local declarations in old block for scope-safe assignment after branch.
    remote=remote.replace('let config_path = repo.get("config.json")','let config_path_remote = repo.get("config.json")')
    remote=remote.replace('File::open(&config_path)', 'File::open(&config_path_remote)')
    remote=remote.replace('config_path.display()', 'config_path_remote.display()')
    remote=remote.replace('    let config: serde_json::Value = serde_json::from_reader(File::open(&config_path_remote)?)?;','    let _config_unused: serde_json::Value = serde_json::from_reader(File::open(&config_path_remote)?)?;')
    remote=remote.replace('let safetensors_files = match repo.get(', 'let safetensors_files_remote = match repo.get(')
    replacement='''    let config_path: PathBuf;\n    let safetensors_files: Vec<PathBuf>;\n    if let Some(dir) = args.model_dir.as_ref() {\n        config_path = dir.join("config.json");\n        safetensors_files = vec![dir.join("model.safetensors")];\n        anyhow::ensure!(config_path.is_file(), "missing local config.json: {}", config_path.display());\n        anyhow::ensure!(safetensors_files[0].is_file(), "missing local model.safetensors: {}", safetensors_files[0].display());\n        eprintln!("HDB2A_CANDLE_LOCAL_MODEL={}", dir.display());\n    } else {\n''' + remote + '''\n        config_path = config_path_remote;\n        safetensors_files = safetensors_files_remote;\n    }\n    let config: serde_json::Value = serde_json::from_reader(File::open(&config_path)?)?;\n'''
    result=source[:start]+replacement+source[end:]
    if result.count('let config: serde_json::Value')!=1:
        raise ValueError('Config declaration was not patched correctly')
    return result


def repair_runtime(out: Path) -> None:
    """Upgrade a pinned, already bootstrapped v0.2.0 runtime in place."""
    out = out.resolve()
    cargo, main_file, conformer = out / 'Cargo.toml', out / 'src/main.rs', out / 'src/parakeet/conformer.rs'
    if not all(p.is_file() for p in (cargo, main_file, conformer)):
        raise ValueError(f'Existing Candle runtime is incomplete: {out}')
    head = subprocess.check_output(['git', '-C', str(out), 'rev-parse', 'HEAD'], text=True).strip()
    if head != PIN_PARAKEET:
        raise ValueError(f'Unexpected runtime HEAD: {head}')
    if 'HDB2A_CANDLE_LOCAL_MODEL' not in main_file.read_text():
        raise ValueError('Runtime is not bootstrapped for local QVAC weights')
    if 'candle-core' not in cargo.read_text() or '0.11' not in cargo.read_text():
        raise ValueError('Runtime Candle dependency is not adapted to 0.11')
    old = conformer.read_text()
    patched = patch_conformer(old)
    if patched != old:
        conformer.write_text(patched)
        print('HDB2A_RUNTIME_REPAIR=PATCHED_LAYER_NORM_UNBUILT')
    else:
        print('HDB2A_RUNTIME_REPAIR=ALREADY_PATCHED')
    print(f'HDB2A_RUNTIME_DIR={out}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime-dir', type=Path,required=True)
    p.add_argument('--repair-existing', action='store_true')
    p.add_argument('--candle-root',type=Path,default=None,help='Use local Candle source instead of pinned remote Git revision')
    args=p.parse_args()
    out=args.runtime_dir.resolve()
    if args.repair_existing:
        repair_runtime(out)
        return
    if out.exists():
        raise SystemExit(f'Refusing to alter existing runtime: {out}. Remove it and re-run bootstrap if needed.')
    out.parent.mkdir(parents=True,exist_ok=True)
    run('git','clone','--no-checkout',PARAKEET_URL,out)
    run('git','-C',out,'checkout','--detach',PIN_PARAKEET)
    observed=subprocess.check_output(['git','-C',str(out),'rev-parse','HEAD'],text=True).strip()
    if observed!=PIN_PARAKEET:
        raise ValueError(f'Pinned runtime SHA mismatch: {observed}')
    candle_source='git'
    if args.candle_root:
        candle_root=args.candle_root.resolve()
        if not (candle_root/'candle-core/Cargo.toml').is_file():
            raise ValueError(f'Not a Candle source tree: {candle_root}')
        candle_source=str(candle_root)
        print(f'HDB2A_CANDLE_LOCAL_SOURCE={candle_root} (revision not pinned, report local git HEAD)')
    cargo=out/'Cargo.toml'
    main_file=out/'src/main.rs'
    patched_cargo=modify_cargo(cargo.read_text(),candle_source)
    patched_main=patch_main(main_file.read_text())
    conformer_file=out/'src/parakeet/conformer.rs'
    patched_conformer=patch_conformer(conformer_file.read_text())
    cargo.write_text(patched_cargo)
    main_file.write_text(patched_main)
    conformer_file.write_text(patched_conformer)
    print('HDB2A_RUNTIME_PATCH=APPLIED_PINNED_SOURCE_UNBUILT')
    print(f'HDB2A_RUNTIME_DIR={out}')
    print('Next step: ./run-hdb2a.sh build cpu')

if __name__=='__main__':main()
