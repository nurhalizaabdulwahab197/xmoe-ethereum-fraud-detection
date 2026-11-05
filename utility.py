import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_code(root):
    manifest = read(root / 'release_manifest.json')
    for name, expected in manifest['files_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError('Shared pipeline file changed: ' + name)
    return manifest
