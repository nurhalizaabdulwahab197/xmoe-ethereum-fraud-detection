"""Safely extract and fingerprint the audited ScamSweeper RQ4 release."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

EXPECTED_ARCHIVE_SHA256 = "74b3e49f84aebf5111af66b4cc0c5ee779b462ecb95b6b5a06a728f835dc006a"
EXPECTED = {"MulDiGraph.pkl", "normal_account.txt", "phish_account.txt", "scams_account.txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive_hash = sha256(args.archive)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"Unexpected RQ4 archive SHA-256: {archive_hash}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    with zipfile.ZipFile(args.archive) as source:
        members = {Path(item.filename).name: item for item in source.infolist() if not item.is_dir()}
        if set(members) != EXPECTED:
            raise ValueError(f"Unexpected RQ4 members: {sorted(members)}")
        for name, item in members.items():
            target = args.output_dir / name
            if not target.exists():
                with source.open(item) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 16 * 1024 * 1024)
            if target.stat().st_size != item.file_size:
                raise ValueError(f"Size mismatch: {target}")
            records[name] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
    manifest = {"archive": str(args.archive.resolve()), "archive_sha256": archive_hash, "files": records, "pickle_loaded": False}
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "pass", "manifest": str(path.resolve()), "files": records}, indent=2))


if __name__ == "__main__":
    main()
