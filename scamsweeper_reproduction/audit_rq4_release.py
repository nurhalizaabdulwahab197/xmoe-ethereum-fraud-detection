"""Restricted-load audit of the ScamSweeper RQ4 NetworkX release."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
from pathlib import Path

import networkx as nx


class RestrictedGraphUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("networkx.classes.multidigraph", "MultiDiGraph"): nx.MultiDiGraph,
        ("builtins", "dict"): dict,
    }

    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(f"Forbidden global: {module}.{name}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path: Path) -> set[str]:
    return {line.strip().lower() for line in path.read_text().splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rq1-node-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for name, record in manifest["files"].items():
        path = args.root / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Manifest verification failed: {path}")
    with (args.root / "MulDiGraph.pkl").open("rb") as stream:
        graph = RestrictedGraphUnpickler(stream).load()
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError(type(graph))

    labels = {name: load_labels(args.root / f"{name}_account.txt") for name in ("normal", "phish", "scams")}
    graph_nodes = {str(node).strip().lower() for node in graph}
    isp_nodes = {str(node).strip().lower() for node, data in graph.nodes(data=True) if data.get("isp") == 1}
    node_keys = collections.Counter(key for _, data in graph.nodes(data=True) for key in data)
    edge_keys = collections.Counter()
    for _, _, _, data in graph.edges(keys=True, data=True):
        edge_keys.update(data.keys())
    label_incidence = {}
    for name, values in labels.items():
        present = values & graph_nodes
        incoming = sum(graph.in_degree(node) for node in present)
        outgoing = sum(graph.out_degree(node) for node in present)
        label_incidence[name] = {
            "present_nodes": len(present),
            "nodes_with_incident_edges": sum(graph.degree(node) > 0 for node in present),
            "incoming_edge_visits": incoming,
            "outgoing_edge_visits": outgoing,
        }

    report = {
        "status": "partial_artifact_reproduction",
        "graph": {"type": type(graph).__name__, "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "directed": graph.is_directed(), "multigraph": graph.is_multigraph()},
        "label_counts": {name: len(values) for name, values in labels.items()},
        "label_overlaps": {"normal_phish": len(labels["normal"] & labels["phish"]), "normal_scams": len(labels["normal"] & labels["scams"]), "phish_scams": len(labels["phish"] & labels["scams"])},
        "graph_coverage": {name: {"present": len(values & graph_nodes), "missing": len(values - graph_nodes), "missing_first_10": sorted(values - graph_nodes)[:10]} for name, values in labels.items()},
        "isp": {"count": len(isp_nodes), "equals_phish_list": isp_nodes == labels["phish"], "isp_not_phish": len(isp_nodes - labels["phish"]), "phish_not_isp": len(labels["phish"] - isp_nodes)},
        "node_attribute_keys": dict(node_keys),
        "edge_attribute_keys": dict(edge_keys),
        "label_incidence": label_incidence,
        "exact_table3_reproduction_ready": False,
        "blockers": ["422 of 636 released normal labels are absent from MulDiGraph.pkl", "complete author model/baseline code and configuration are absent", "exact split and metric implementation are absent"],
    }
    if args.rq1_node_report:
        rq1 = json.loads(args.rq1_node_report.read_text())
        rq1_normal_sources = {str(value).lower() for value in rq1["normal"]["isp_1_nodes"]}
        report["rq1_normal_cross_release"] = {
            "rq1_normal_isp_1_count": len(rq1_normal_sources),
            "rq4_normal_list_count": len(labels["normal"]),
            "intersection": len(rq1_normal_sources & labels["normal"]),
            "sets_equal": rq1_normal_sources == labels["normal"],
            "recovery_candidate": rq1_normal_sources == labels["normal"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
