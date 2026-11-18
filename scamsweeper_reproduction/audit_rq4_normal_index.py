"""Audit a complete static RQ1 index for the 636 RQ4 normal accounts."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.targets.read_text(encoding="utf-8"))
    targets = {row["address"] for row in plan["targets"]}
    summary = json.loads((args.index / "summary.json").read_text(encoding="utf-8"))
    if summary["target_plan_sha256"] != sha256(args.targets):
        raise ValueError("Target plan hash differs from index")
    if len(summary["graphs"]) != 1:
        raise ValueError("Expected exactly one indexed graph")
    graph = summary["graphs"][0]
    required = {
        "graph": "normal",
        "complete": True,
        "all_target_incident_pair_blocks": True,
        "node_and_adjacency_source_counts_match": True,
        "edges_seen": 142750370,
        "committed_source_blocks": 12042066,
    }
    for key, expected in required.items():
        if graph.get(key) != expected:
            raise ValueError(f"Unexpected {key}: {graph.get(key)!r}")
    found = set(graph["target_source_addresses_found"])
    if found != targets or len(found) != 636:
        raise ValueError("Not all 636 targets were found as RQ1-normal sources")

    visits = collections.Counter()
    edge_visits = collections.Counter()
    unique: dict[tuple[int, int], int] = {}
    duplicate_visits = 0
    blocks_path = args.index / "normal" / "selected_blocks.jsonl"
    with blocks_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            target = row["target"]
            direction = row["direction"]
            if target not in targets or direction not in {"in", "out"}:
                raise ValueError(f"Invalid visit at line {line_number}")
            if row["scalar_snapshot"] is not None or row["edge_records"] < 1:
                raise ValueError(f"Invalid block payload at line {line_number}")
            key = (row["start"], row["end"])
            if key in unique:
                duplicate_visits += 1
                if unique[key] != row["edge_records"]:
                    raise ValueError(f"Edge count disagreement for physical block {key}")
            else:
                unique[key] = row["edge_records"]
            visits[(target, direction)] += 1
            edge_visits[(target, direction)] += row["edge_records"]

    missing_incidence = sorted(target for target in targets if not visits[(target, "in")] and not visits[(target, "out")])
    report = {
        "status": "pass" if not missing_incidence else "fail",
        "scope": "complete static index of every RQ1-normal pair block incident to all 636 RQ4 normal targets",
        "target_count": len(targets),
        "targets_found_as_sources": len(found),
        "targets_without_incident_blocks": missing_incidence,
        "block_visits": sum(visits.values()),
        "unique_physical_pair_blocks": len(unique),
        "duplicate_target_visits_to_physical_blocks": duplicate_visits,
        "unique_edge_records": sum(unique.values()),
        "incoming_block_visits": sum(count for (target, direction), count in visits.items() if direction == "in"),
        "outgoing_block_visits": sum(count for (target, direction), count in visits.items() if direction == "out"),
        "incoming_edge_record_visits": sum(count for (target, direction), count in edge_visits.items() if direction == "in"),
        "outgoing_edge_record_visits": sum(count for (target, direction), count in edge_visits.items() if direction == "out"),
        "index_invariants": required,
        "input_sha256": {
            "target_plan": sha256(args.targets),
            "index_summary": sha256(args.index / "summary.json"),
            "selected_blocks": sha256(blocks_path),
        },
        "code_sha256": {
            "indexer_recorded_by_index": summary["code_sha256"],
            "auditor": sha256(Path(__file__)),
        },
        "training_performed": False,
        "table_iii_reproduced": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
