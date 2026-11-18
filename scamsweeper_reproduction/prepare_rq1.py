"""Safely extract the three released ScamSweeper RQ1 graph pickles."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path


EXPECTED = {
    "normal_pkl.zip": ("normal", "xlabel_normal.pickle"),
    "phish_pkl.zip": ("phish", "phish.pickle"),
    "scams_pkl.zip": ("scams", "scams.pickle"),
}
EXPECTED_OUTER_SHA256 = "4481bddf4cc75cd35e8ecebf51eb20d3cd0154c1dbfa899848467219b55b8c87"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new path")

    args.output_dir.mkdir(parents=True)
    outer_sha256 = digest(args.archive)
    if outer_sha256 != EXPECTED_OUTER_SHA256:
        raise ValueError(
            "RQ1 archive SHA-256 does not match the audited release: "
            f"{outer_sha256}"
        )
    manifest = {
        "outer_archive": str(args.archive.resolve()),
        "outer_archive_bytes": args.archive.stat().st_size,
        "outer_archive_sha256": outer_sha256,
        "files": {},
    }
    with zipfile.ZipFile(args.archive) as outer:
        if set(outer.namelist()) != set(EXPECTED):
            raise ValueError(f"Unexpected outer members: {outer.namelist()}")
        for nested_name, (folder, pickle_name) in EXPECTED.items():
            nested_bytes = outer.read(nested_name)
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                if nested.namelist() != [pickle_name]:
                    raise ValueError(
                        f"Unexpected members in {nested_name}: {nested.namelist()}"
                    )
                destination = args.output_dir / folder / pickle_name
                destination.parent.mkdir(parents=True)
                with nested.open(pickle_name) as source, destination.open("wb") as target:
                    while block := source.read(8 * 1024 * 1024):
                        target.write(block)
            manifest["files"][folder] = {
                "path": str(destination.resolve()),
                "bytes": destination.stat().st_size,
                "sha256": digest(destination),
            }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir.resolve()), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()

