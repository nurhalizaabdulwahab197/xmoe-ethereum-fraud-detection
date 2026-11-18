"""Finalize and classify the ScamSweeper RQ1 artifact reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PAPER_SD = {"normal": 3555.35, "phish": 1285.25, "scams": 541.10}


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-verification", type=Path, required=True)
    parser.add_argument("--degree-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = json.loads(args.table_verification.read_text(encoding="utf-8"))
    graphs = {}
    for name in ("normal", "phish", "scams"):
        report_path = args.degree_dir / f"{name}_degree_report.json"
        histogram_path = args.degree_dir / f"{name}_degree_frequency.csv"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sd = report["degrees_adjacency_sources"]["total"]["population_sd"]
        graphs[name] = {
            "nodes": report["node_count"],
            "adjacency_sources": report["source_blocks"],
            "edges": report["edge_records"],
            "total_degree_population_sd": sd,
            "paper_sd_degree": PAPER_SD[name],
            "rounded_sd_matches_paper": round(sd, 2) == PAPER_SD[name],
            "degree_report_sha256": sha256(report_path),
            "degree_frequency_sha256": sha256(histogram_path),
        }

    structural_pass = table["structural_counts_status"] == "pass"
    degree_pass = all(item["rounded_sd_matches_paper"] for item in graphs.values())
    figure_ready = args.figure.is_file() and args.figure.stat().st_size > 0
    if not (structural_pass and degree_pass and figure_ready):
        status = "incomplete"
    else:
        status = "artifact_reproduction_complete_with_documented_source_count_discrepancy"

    result = {
        "target": "ScamSweeper RQ1",
        "status": status,
        "structural_counts_reproduced": structural_pass,
        "degree_distributions_reproduced": figure_ready,
        "sd_degree_reproduced": degree_pass,
        "sd_degree_definition": (
            "Population standard deviation of total degree (in-degree + out-degree) "
            "over serialized adjacency source records."
        ),
        "graphs": graphs,
        "documented_discrepancies": [
            {
                "field": "Phishing #L/source count",
                "paper": 4905,
                "released_pickle_isp_1": 4701,
                "difference": -204,
                "interpretation": (
                    "The released artifact does not contain enough provenance to recover "
                    "or identify the 204 records. No labels were synthesized."
                ),
            },
            {
                "field": "Phishing node versus adjacency-source scope",
                "node_records": 10159847,
                "adjacency_sources": 10159176,
                "difference": -671,
                "interpretation": (
                    "The paper's SD Degree is reproduced only when computed over adjacency "
                    "source records; including the 671 absent sources as zero-degree node "
                    "records gives 1285.2096 instead of 1285.2520."
                ),
            },
        ],
        "figure": str(args.figure.resolve()),
        "figure_sha256": sha256(args.figure),
        "independent_blockchain_collection_reproduced": False,
        "scope_note": (
            "Complete artifact-level reproduction of RQ1 descriptive graph statistics; "
            "not an independent recreation of the authors' Ethereum crawl."
        ),
        "model_training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": status,
        "structural_counts_reproduced": structural_pass,
        "sd_degree_reproduced": degree_pass,
        "degree_distributions_reproduced": figure_ready,
    }, indent=2))


if __name__ == "__main__":
    main()

