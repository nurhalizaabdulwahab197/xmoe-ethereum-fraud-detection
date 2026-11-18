"""Verify ScamSweeper RQ1 Table I against completed static pickle scans.

This verifier consumes reports produced by the bounded opcode readers in the
parent workspace.  It does not unpickle the multi-gigabyte NetworkX objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_TABLE_I = {
    "normal": {"nodes": 12_042_066, "source_nodes": 636, "edges": 142_750_370, "sd_degree": 3555.35},
    "phish": {"nodes": 10_159_847, "source_nodes": 4_905, "edges": 62_011_219, "sd_degree": 1285.25},
    "scams": {"nodes": 8_736_430, "source_nodes": 3_125, "edges": 64_265_586, "sd_degree": 541.10},
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(node_report_path: Path, edge_report_path: Path) -> dict:
    node_report = load_json(node_report_path)
    edge_report = load_json(edge_report_path)
    edge_graphs = {item["graph"]: item for item in edge_report["graphs"]}

    datasets = {}
    for name, expected in PAPER_TABLE_I.items():
        nodes = node_report[name]
        edges = edge_graphs[name]
        edge_count = edges.get("edges_seen", edges.get("edge_records"))
        source_blocks = edges.get("committed_source_blocks", edges.get("source_blocks"))
        scan_complete = edges.get("complete", edge_report.get("status") == "complete")
        degree_statistics = edges.get("degrees_adjacency_sources")
        observed_sd = (
            degree_statistics["total"]["population_sd"]
            if degree_statistics is not None
            else None
        )
        if edge_count is None or source_blocks is None:
            raise ValueError(f"Unsupported edge report schema for {name}")
        observed = {
            "nodes": int(nodes["sampled_nodes"]),
            "source_marker_isp_1": int(nodes["node_attribute_values"]["isp"].get("1", 0)),
            "edges": int(edge_count),
            "adjacency_source_blocks": int(source_blocks),
            "total_degree_population_sd": observed_sd,
        }
        checks = {
            "node_table_complete": bool(nodes["node_table_complete"]),
            "adjacency_scan_complete": bool(scan_complete),
            "nodes_match_paper": observed["nodes"] == expected["nodes"],
            "edges_match_paper": observed["edges"] == expected["edges"],
            "source_marker_matches_paper_source_nodes": (
                observed["source_marker_isp_1"] == expected["source_nodes"]
            ),
            "sd_degree_matches_paper": (
                observed_sd is not None
                and round(observed_sd, 2) == expected["sd_degree"]
            ),
        }
        datasets[name] = {
            "paper": expected,
            "observed": observed,
            "checks": checks,
            "source_marker_difference": observed["source_marker_isp_1"] - expected["source_nodes"],
            "degree_distribution_reproduced": degree_statistics is not None,
            "paper_sd_degree_reproduced": checks["sd_degree_matches_paper"],
        }

    structural_pass = all(
        item["checks"][key]
        for item in datasets.values()
        for key in (
            "node_table_complete",
            "adjacency_scan_complete",
            "nodes_match_paper",
            "edges_match_paper",
        )
    )
    source_marker_pass = all(
        item["checks"]["source_marker_matches_paper_source_nodes"]
        for item in datasets.values()
    )
    degree_pass = all(
        item["checks"]["sd_degree_matches_paper"] for item in datasets.values()
    )
    if structural_pass and degree_pass:
        overall_status = (
            "table_i_reproduction_complete"
            if source_marker_pass
            else "table_i_reproduction_complete_with_documented_source_count_discrepancy"
        )
    else:
        overall_status = "partial_reproduction"
    return {
        "target": "ScamSweeper RQ1 Table I artifact reproduction",
        "datasets": datasets,
        "structural_counts_status": "pass" if structural_pass else "fail",
        "source_marker_status": "pass" if source_marker_pass else "unresolved",
        "degree_statistics_status": "pass" if degree_pass else "pending",
        "overall_status": overall_status,
        "success_rule": (
            "Mark complete only when structural counts pass, the 4,701 versus 4,905 "
            "Phishing source discrepancy is resolved or explicitly scoped, and the "
            "degree distributions plus the paper's SD Degree statistic are reproduced."
        ),
        "independent_blockchain_collection_reproduced": False,
        "model_training_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-report", type=Path, required=True)
    parser.add_argument("--edge-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.node_report, args.edge_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "structural_counts_status": result["structural_counts_status"],
        "source_marker_status": result["source_marker_status"],
        "degree_statistics_status": result["degree_statistics_status"],
        "overall_status": result["overall_status"],
    }, indent=2))


if __name__ == "__main__":
    main()

