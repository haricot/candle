"""Install a pinned standalone uv binary into a local work directory, if absent."""
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

VERSION = '0.12.23'
WHEEL = f'uv-{VERSION}-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'


def install(directory):
    destination = Path(directory)/'uv'
    if destination.is_file():
        return
    with urllib.request.urlopen(f'https://pypi.org/pypi/uv/{VERSION}/json', timeout=30) as response:
        metadata = json.load(response)
    matches = [a for a in metadata['urls'] if a['filename'] == WHEEL]
    if len(matches) != 1:
        raise RuntimeError('No unique pinned Linux x86-64 uv wheel')
    asset = matches[0]
    with urllib.request.urlopen(asset['url'], timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != asset['digests']['sha256']:
        raise RuntimeError('uv wheel checksum differs from the pinned package metadata')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        binaries = [p for p in archive.namelist() if p.endswith('.data/scripts/uv')]
        if len(binaries) != 1:
            raise RuntimeError('Unexpected standalone uv package layout')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(binaries[0]))
        destination.chmod(0o755)


if __name__ == '__main__':
    install(sys.argv[1])
