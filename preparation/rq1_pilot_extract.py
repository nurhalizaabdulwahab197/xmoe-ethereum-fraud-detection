"""Extract a bounded technical RQ1 pilot with exact sparse memo resolution.

This reads primitive pickle opcodes, never pickle.load or serialized globals.
Scope: at most N records from the first outgoing source block of each graph.
It is not a random sample or a complete account history.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import pickletools
import sqlite3
from pathlib import Path

from rq1_node_prefix_audit import ADDRESS, BoundedReader, LimitReached

SCALARS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4", "BINFLOAT", "FLOAT"}
UNKNOWN = object()


class Ref:
    def __init__(self, index):
        self.index = index


class Frame:
    def __init__(self, kind="generic", target=None):
        self.kind, self.target = kind, target
        self.values = {}


class Mark:
    def __init__(self, owner):
        self.owner = owner


class PilotEnd(Exception):
    pass


def typed(value):
    if isinstance(value, Ref):
        return {"type": "unresolved_memo", "index": value.index}
    if value is None or type(value) in (str, int, float, bool):
        return {"type": type(value).__name__, "text": str(value) if value is not None else None}
    raise ValueError("Nonprimitive value in pilot edge")


def scan(boundary, limit, extra=None, max_bytes=16 * 1024**2):
    if not boundary["node_table_complete"]:
        raise ValueError("Incomplete node boundary")
    path = Path(boundary["file"])
    if path.stat().st_size != boundary["file_bytes"]:
        raise ValueError("Source size changed")
    base_memo = boundary["next_memo_index_at_boundary"]
    memo = {int(k): v for k, v in boundary["scalar_memo_at_boundary"].items()}
    memo.update({int(k): v for k, v in boundary["address_memo_sample"].items()})
    memo.update(extra or {})
    next_memo = base_memo
    needed = set()
    stack, records = [], []
    root, neighbors, source = None, None, None
    require_neighbor_mark = False
    last = UNKNOWN
    offset = boundary["last_opcode_position"]
    stop_reason = "unexpected end of stream"

    def assign(frame, key, value, position):
        if not isinstance(frame, Frame):
            raise ValueError("Invalid dictionary target")
        if frame is root and value is neighbors:
            raise PilotEnd("first source block completed")
        if frame.kind == "pair":
            if not isinstance(value, Frame) or any(not isinstance(k, str) for k in value.values):
                raise ValueError("Edge attributes are not a primitive keyed dictionary")
            attributes = {k: v for k, v in value.values.items()}
            for v in attributes.values():
                typed(v)
            records.append({"source": source, "target": frame.target, "key": key,
                            "attributes": attributes, "pickle_assignment_offset": position})
            if len(records) >= limit:
                raise PilotEnd("record limit reached within first source block")
        elif frame.kind not in {"root", "neighbors"}:
            frame.values[key] = value

    with path.open("rb") as handle:
        handle.seek(offset)
        reader = BoundedReader(handle, offset + max_bytes)
        try:
            for op, arg, position in pickletools.genops(reader):
                name = op.name
                if name in {"PROTO", "FRAME"}:
                    continue
                if name in {"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"}:
                    idx = next_memo if name == "MEMOIZE" else int(arg)
                    if len(memo) >= 250000:
                        raise ValueError("Pilot memo capacity exceeded")
                    memo[idx] = stack[-1] if root is not None else last
                    next_memo = max(next_memo, idx + 1)
                    continue
                if root is None:
                    if name in SCALARS and arg == "_adj":
                        last = arg
                    elif name == "EMPTY_DICT" and last == "_adj":
                        root = Frame("root")
                        stack.append(root)
                    else:
                        raise ValueError("Unexpected _adj prefix")
                    continue
                if require_neighbor_mark and name != "MARK":
                    raise ValueError("Pilot requires a multi-neighbor first source block; this prefix is unsupported")
                if name in SCALARS:
                    stack.append(arg)
                elif name in {"BINGET", "LONG_BINGET", "GET"}:
                    idx = int(arg)
                    value = memo.get(idx, UNKNOWN)
                    if value is UNKNOWN:
                        if idx >= base_memo:
                            raise ValueError(f"Missing newly-created memo entry {idx}")
                        needed.add(idx)
                        value = Ref(idx)
                    stack.append(value)
                elif name == "MARK":
                    if not stack or not isinstance(stack[-1], Frame):
                        raise ValueError("Unexpected dictionary MARK context")
                    stack.append(Mark(stack[-1]))
                    require_neighbor_mark = False
                elif name == "EMPTY_DICT":
                    if neighbors is None:
                        source = stack[-1]
                        if not isinstance(source, (str, Ref)):
                            raise ValueError("Source key is not an address/memo reference")
                        neighbors = Frame("neighbors")
                        stack.append(neighbors)
                        require_neighbor_mark = True
                    else:
                        owner = next((x.owner for x in reversed(stack) if isinstance(x, Mark)), None)
                        if owner is root:
                            raise PilotEnd("next source block began")
                        if owner is neighbors and isinstance(stack[-1], (str, Ref)):
                            target = stack[-1]
                            stack.append(Frame("pair", target))
                        else:
                            stack.append(Frame())
                elif name in {"NONE", "NEWTRUE", "NEWFALSE"}:
                    stack.append(None if name == "NONE" else name == "NEWTRUE")
                elif name in {"SETITEM", "SETITEMS"}:
                    if name == "SETITEM":
                        value, key = stack.pop(), stack.pop()
                        pairs = [(key, value)]
                    else:
                        mark = next(i for i in range(len(stack) - 1, -1, -1) if isinstance(stack[i], Mark))
                        values = stack[mark + 1:]
                        del stack[mark:]
                        if len(values) % 2:
                            raise ValueError("Odd dictionary batch")
                        pairs = list(zip(values[::2], values[1::2]))
                    for key, value in pairs:
                        assign(stack[-1], key, value, position)
                else:
                    raise ValueError(f"Unsupported pilot opcode {name}")
        except PilotEnd as exc:
            stop_reason = str(exc)
        except LimitReached as exc:
            raise ValueError(f"Pilot byte bound reached before its record/source limit: {exc}") from exc
        bytes_read = reader.tell() - offset
    if not records:
        raise ValueError("No edge records extracted")
    return records, needed, {"stop_reason": stop_reason, "adjacency_bytes_read": bytes_read}


def resolve_prior_scalars(boundary, requested):
    """Replay only enough of the node prefix to resolve requested memo IDs."""
    if not requested:
        return {}, 0
    requested = set(requested)
    resolved, next_memo, last = {}, 0, UNKNOWN
    path = Path(boundary["file"])
    with path.open("rb") as handle:
        reader = BoundedReader(handle, boundary["last_opcode_position"])
        for op, arg, position in pickletools.genops(reader):
            name = op.name
            if name in {"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"}:
                idx = next_memo if name == "MEMOIZE" else int(arg)
                if idx in requested:
                    if not isinstance(last, str):
                        raise ValueError(f"Requested endpoint memo {idx} is not a string literal")
                    resolved[idx] = last
                    if len(resolved) == len(requested):
                        return resolved, reader.tell()
                next_memo = max(next_memo, idx + 1)
            elif name in SCALARS:
                last = arg
            elif name in {"PROTO", "FRAME"}:
                continue
            else:
                last = UNKNOWN
    raise ValueError(f"Could not resolve endpoint memo IDs: {sorted(requested - resolved.keys())[:10]}")


def normalize_record(record, group, ordinal):
    source, target = record["source"], record["target"]
    if isinstance(source, Ref) or isinstance(target, Ref):
        raise ValueError("Unresolved endpoint reached normalization")
    attrs = record["attributes"]
    reasons = []
    valid_source = isinstance(source, str) and ADDRESS.fullmatch(source)
    valid_target = isinstance(target, str) and ADDRESS.fullmatch(target)
    if not valid_source or not valid_target:
        reasons.append("non_address_endpoint")
    time_source = "timestamp_attribute" if "timestamp" in attrs else "edge_key"
    value = attrs.get("timestamp", record["key"])
    time_value = None
    if type(value) in (int, float) and math.isfinite(value) and float(value).is_integer():
        time_value = int(value)
        # Plausibility check only; this does not prove the source unit.
        if not 1420070400 <= time_value <= 1793318400:
            reasons.append("timestamp_outside_pilot_plausibility_window")
    else:
        reasons.append("timestamp_not_finite_integer")
    key = record["key"]
    if "timestamp" in attrs and type(key) in (int, float) and key != value:
        reasons.append("timestamp_key_mismatch")
    return {"pilot_record_id": f"{group}:{ordinal}", "source_graph": group,
            "source_address_raw": source, "target_address_raw": target,
            "source_address": source.lower() if valid_source else None,
            "target_address": target.lower() if valid_target else None,
            "original_edge_key": typed(key), "attributes_raw": {k: typed(v) for k, v in attrs.items()},
            "event_time_seconds_candidate": time_value, "event_time_source": time_source,
            "timestamp_unit_verified": False, "amount_unit_verified": False,
            "pickle_assignment_offset": record["pickle_assignment_offset"],
            "valid_for_structural_temporal_pilot": not reasons, "quarantine_reasons": reasons}


def make_nodes(records, account_manifest):
    labels = {r["address"]: r for r in account_manifest["accounts"]}
    state = {}
    def node(address, spelling):
        result = state.setdefault(address, {"incoming": set(), "outgoing": set(), "times": [], "days": set(), "spellings": set()})
        result["spellings"].add(spelling)
        return result
    for row in records:
        if not row["valid_for_structural_temporal_pilot"]:
            continue
        u, v, time = row["source_address"], row["target_address"], row["event_time_seconds_candidate"]
        a, b = node(u, row["source_address_raw"]), node(v, row["target_address_raw"])
        if u != v:
            a["outgoing"].add(v)
            b["incoming"].add(u)
        for entry in (a, b):
            entry["times"].append(time)
            entry["days"].add(time // 86400)
    output = []
    for address, row in sorted(state.items()):
        label = labels.get(address)
        output.append({"address": address, "original_spellings": sorted(row["spellings"]),
                       "source_declared_class": label["source_declared_class"] if label else None,
                       "label_review_status": label["status"] if label else "unlabeled_context",
                       "has_candidate_label": bool(label and label["status"] == "candidate_without_detected_conflict"),
                       "observation_scope": "bounded_first_source_prefix_pilot_only",
                       "features": {"distinct_in_counterparties": len(row["incoming"]), "distinct_out_counterparties": len(row["outgoing"]),
                                    "reciprocal_counterparties": len(row["incoming"] & row["outgoing"]),
                                    "active_days": len(row["days"]), "activity_span_seconds": max(row["times"]) - min(row["times"])},
                       "first_event_seconds_candidate": min(row["times"]), "last_event_seconds_candidate": max(row["times"])})
    return output


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--records-per-graph", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.records_per_graph <= 10000:
        parser.error("records per graph must be in 1..10000")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose a new empty output directory")
    boundaries = json.loads(args.boundaries.read_text(encoding="utf-8"))
    label_manifest = json.loads(args.labels.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.output_dir / "memo_endpoints.sqlite")
    db.execute("CREATE TABLE memo (graph TEXT, memo_index INTEGER, raw_string TEXT, PRIMARY KEY(graph,memo_index))")
    all_records, groups = [], {}
    try:
        for group, boundary in boundaries.items():
            print(f"Discovering {group} prefix references...", flush=True)
            first, required, first_meta = scan(boundary, args.records_per_graph)
            resolved, node_bytes = resolve_prior_scalars(boundary, required)
            replay, unresolved, meta = scan(boundary, args.records_per_graph, resolved)
            if unresolved or len(first) != len(replay):
                raise ValueError("Exact endpoint replay failed")
            cached = {int(k): v for k, v in boundary["address_memo_sample"].items()}
            cached.update(resolved)
            db.executemany("INSERT INTO memo VALUES (?,?,?)", [(group, k, v) for k, v in sorted(cached.items())])
            db.commit()
            rows = [normalize_record(row, group, i) for i, row in enumerate(replay)]
            all_records.extend(rows)
            groups[group] = {"file": boundary["file"], "source_file_bytes": boundary["file_bytes"], "record_count": len(rows),
                             "resolved_additional_memos": len(resolved), "node_prefix_bytes_replayed": node_bytes,
                             "valid_records": sum(r["valid_for_structural_temporal_pilot"] for r in rows),
                             "quarantined_records": sum(not r["valid_for_structural_temporal_pilot"] for r in rows), **meta}
            print(json.dumps({"graph": group, **groups[group]}), flush=True)
    finally:
        db.close()
    nodes = make_nodes(all_records, label_manifest)
    def write_jsonl(name, items):
        with (args.output_dir / name).open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, allow_nan=False) + "\n")
    write_jsonl("pilot_edges.jsonl", all_records)
    write_jsonl("pilot_nodes.jsonl", nodes)
    write_jsonl("quarantined_edges.jsonl", [r for r in all_records if not r["valid_for_structural_temporal_pilot"]])
    candidate_counts = collections.Counter(r["source_declared_class"] for r in nodes if r["has_candidate_label"])
    schema = {"predictive_features": list(nodes[0]["features"]) if nodes else [],
              "excluded_from_features": ["address", "isp", "source_graph", "source_declared_class", "label_review_status", "event_time_source", "attributes_raw"],
              "monetary_features_enabled": False, "partial_observation_features_only": True,
              "unique_transaction_counts_claimed": False, "normal_class_assigned_to_unknown_nodes": False}
    (args.output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    report = {"status": "technical_pilot_only", "sampling": "at most requested records from first outgoing source block per graph; not random/target-balanced",
              "graphs": groups, "total_records": len(all_records), "canonical_nodes": len(nodes),
              "candidate_labeled_nodes_observed": dict(candidate_counts),
              "review_nodes_observed": sum(r["label_review_status"] == "review_required" for r in nodes),
              "unlabeled_context_nodes": sum(r["source_declared_class"] is None for r in nodes),
              "unresolved_endpoint_memos": 0, "parallel_records_preserved": True,
              "transaction_deduplication_performed": False, "training_or_splits_created": False,
              "timestamp_interpretation": "candidate Unix seconds; original unit declaration unavailable",
              "input_sha256": {"boundary_manifest": sha256(args.boundaries), "label_manifest": sha256(args.labels)},
              "code_sha256": {"pilot": sha256(Path(__file__)), "primitive_reader": sha256(Path(__file__).with_name("rq1_node_prefix_audit.py"))},
              "source_archive_sha256_reference": "4481bddf4cc75cd35e8ecebf51eb20d3cd0154c1dbfa899848467219b55b8c87"}
    (args.output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "total_records", "canonical_nodes", "candidate_labeled_nodes_observed", "unlabeled_context_nodes", "unresolved_endpoint_memos")}, indent=2))


if __name__ == "__main__":
    main()
