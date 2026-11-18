"""Create and verify the complete RQ4 normal-account extraction plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def addresses(path: Path) -> list[str]:
    rows = [row.strip().lower() for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate addresses in {path}")
    if any(not (row.startswith("0x") and len(row) == 42) for row in rows):
        raise ValueError(f"Invalid Ethereum address in {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-list", type=Path, required=True)
    parser.add_argument("--rq1-source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    normal = addresses(args.normal_list)
    source_report = json.loads(args.rq1_source_report.read_text(encoding="utf-8"))
    rq1_normal = sorted({row.lower() for row in source_report["normal"]["isp_1_nodes"]})
    if len(normal) != 636:
        raise ValueError(f"Expected 636 RQ4 normal accounts, found {len(normal)}")
    if set(normal) != set(rq1_normal):
        raise ValueError("RQ4 normal list is not exactly the RQ1-normal isp=1 source set")

    plan = {
        "selection_version": "rq4-all-normal-targets-v1",
        "selection": "all released RQ4 normal_account.txt addresses; no sampling",
        "source_normal_list_sha256": sha256(args.normal_list),
        "source_rq1_report_sha256": sha256(args.rq1_source_report),
        "target_count": len(normal),
        "targets": [
            {
                "address": address,
                "source_declared_class": "normal",
                "source_member": "RQ4/normal_account.txt",
                "graphs_present": ["rq1_normal"],
            }
            for address in sorted(normal)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "target_count": len(normal), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
