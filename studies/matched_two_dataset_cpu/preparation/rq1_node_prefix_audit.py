"""Bounded static inspection of primitive node attributes in NetworkX pickles.

Interprets only primitive pickle opcodes inside the leading _node dictionary.
Never imports pickle globals, invokes reducers, constructs NetworkX objects or
unpickles the graph. Prefix counts cannot establish full label semantics.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickletools
import re
from pathlib import Path


ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}\Z")
MARK = object()
UNKNOWN = object()


class LimitReached(Exception):
    pass


class BoundedReader:
    def __init__(self, handle, limit):
        self.handle, self.limit = handle, limit
        self.position = handle.tell()

    def read(self, count=-1):
        remaining = self.limit - self.position
        if count < 0 or count > remaining:
            raise LimitReached("byte limit reached")
        value = self.handle.read(count)
        self.position += len(value)
        return value

    def readline(self):
        remaining = self.limit - self.position
        if remaining <= 0:
            raise LimitReached("byte limit reached")
        line = self.handle.readline(remaining)
        self.position += len(line)
        if not line.endswith(b"\n"):
            raise LimitReached("line limit reached")
        return line

    def tell(self):
        return self.position


class NodeTable:
    def __init__(self, limit, reference=None):
        self.limit = limit
        self.count = 0
        self.keys = collections.Counter()
        self.values = collections.defaultdict(collections.Counter)
        self.examples = []
        self.nonstandard_node_keys = 0
        self.nonstandard_examples = []
        self.reference = reference or {}
        self.reference_nodes = collections.defaultdict(list)

    def add(self, address, attributes):
        if not isinstance(address, str) or not ADDRESS.fullmatch(address):
            self.nonstandard_node_keys += 1
            if len(self.nonstandard_examples) < 10:
                self.nonstandard_examples.append({"type": type(address).__name__, "repr": repr(address)[:160]})
        if not isinstance(attributes, dict):
            raise ValueError("Non-dictionary node attributes")
        if isinstance(address, str):
            normalized = address.lower()
            source_class = self.reference.get(normalized)
            if source_class is not None:
                self.reference_nodes[source_class].append({"address": normalized, "isp_present": "isp" in attributes, "isp_value": attributes.get("isp")})
        self.count += 1
        if self.limit is None and self.count % 1000000 == 0:
            print(f"  {self.count:,} node records inspected", flush=True)
        for key, value in attributes.items():
            self.keys[str(key)] += 1
            self.values[str(key)][json.dumps(value, sort_keys=True)] += 1
        if len(self.examples) < 8 or (attributes and len(self.examples) < 20):
            self.examples.append({"address": address, "attributes": attributes})
        if self.limit is not None and self.count >= self.limit:
            raise LimitReached("node sample limit reached")


def inspect(path, nodes, byte_limit, reference=None):
    memo, memo_count = {}, 0
    address_memo_sample = {}
    last = UNKNOWN
    table = NodeTable(nodes, reference)
    stack = []
    found = False
    waiting = False
    started = False
    reason = "end of stream"
    position = 0
    node_table_complete = False
    scalar_ops = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4", "BINFLOAT", "FLOAT"}
    with path.open("rb") as handle:
        reader = BoundedReader(handle, byte_limit)
        try:
            for op, arg, position in pickletools.genops(reader):
                name = op.name
                if name in {"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"}:
                    value = stack[-1] if started else last
                    index = memo_count if name == "MEMOIZE" else int(arg)
                    if isinstance(value, str) and ADDRESS.fullmatch(value) and len(address_memo_sample) < 256:
                        address_memo_sample[index] = value
                    # Node IDs and attribute dictionaries can number in the
                    # millions. Retain only reusable scalar attribute keys or
                    # values, while still counting every memo index. Any later
                    # reference to an omitted object is rejected explicitly.
                    if value is not UNKNOWN and isinstance(value, (str, int, float, bool, type(None))) and not (isinstance(value, str) and ADDRESS.fullmatch(value)):
                        if len(memo) >= 100000:
                            raise LimitReached("scalar memo capacity reached")
                        memo[index] = value
                    memo_count = max(memo_count, index + 1)
                    continue
                if not started:
                    if name in scalar_ops:
                        last = arg
                        if arg == "_node":
                            found, waiting = True, True
                    elif waiting and name == "EMPTY_DICT":
                        stack = [table]
                        started, waiting = True, False
                    else:
                        last = UNKNOWN
                    continue
                if name in {"PROTO", "FRAME"}:
                    continue
                if name in scalar_ops:
                    if len(stack) == 1 and stack[0] is table and isinstance(arg, str) and arg in {"_adj", "_succ", "_pred", "adj"}:
                        reason = "node dictionary ended before next graph field"
                        node_table_complete = arg in {"_adj", "_succ", "_pred", "adj"}
                        break
                    stack.append(arg)
                elif name == "MARK":
                    stack.append(MARK)
                elif name == "EMPTY_DICT":
                    stack.append({})
                elif name in {"BINGET", "LONG_BINGET", "GET"}:
                    value = memo.get(int(arg), UNKNOWN)
                    if value is UNKNOWN:
                        raise ValueError(f"Unknown memo reference {arg}")
                    stack.append(value)
                elif name == "NONE":
                    stack.append(None)
                elif name in {"NEWTRUE", "NEWFALSE"}:
                    stack.append(name == "NEWTRUE")
                elif name in {"SETITEM", "SETITEMS"}:
                    if name == "SETITEM":
                        value, key = stack.pop(), stack.pop()
                        pairs = [(key, value)]
                    else:
                        marker_index = next(i for i in range(len(stack) - 1, -1, -1) if stack[i] is MARK)
                        items = stack[marker_index + 1:]
                        del stack[marker_index:]
                        if len(items) % 2:
                            raise ValueError("Odd dictionary item count")
                        pairs = list(zip(items[::2], items[1::2]))
                    target = stack[-1]
                    for key, value in pairs:
                        if isinstance(target, NodeTable):
                            target.add(key, value)
                        elif isinstance(target, dict):
                            target[key] = value
                        else:
                            raise ValueError("Unexpected dictionary target")
                else:
                    raise ValueError(f"Unsupported node-attribute opcode {name}")
        except (LimitReached, ValueError, StopIteration) as exc:
            reason = str(exc)
        bytes_read = handle.tell()
    return {"file": str(path), "file_bytes": path.stat().st_size,
            "node_dictionary_found": found, "sampled_nodes": table.count,
            "node_table_complete": node_table_complete,
            "last_opcode_position": position, "bytes_read": bytes_read,
            "next_memo_index_at_boundary": memo_count,
            "scalar_memo_at_boundary": memo,
            "address_memo_sample": address_memo_sample,
            "stop_reason": reason, "node_attribute_counts": dict(table.keys),
            "node_attribute_values": {k: dict(v) for k, v in table.values.items()},
            "reference_nodes_found": dict(table.reference_nodes),
            "nonstandard_node_keys": table.nonstandard_node_keys,
            "nonstandard_node_key_examples": table.nonstandard_examples,
            "missing_isp_in_sample": table.count - table.keys.get("isp", 0),
            "examples": table.examples,
            "label_semantics_verified": False,
            "scope": "Leading node sample only; no unpickling or graph construction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--max-mib", type=int, default=16)
    parser.add_argument("--full-node-table", action="store_true", help="Scan the node table only, stopping before adjacency; --nodes is ignored")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.nodes <= 0 or args.max_mib <= 0:
        parser.error("Limits must be positive")
    if args.output.exists():
        parser.error("Output exists; use a new file to preserve the report")
    files = {"normal": args.root / "normal/xlabel_normal.pickle",
             "phish": args.root / "phish/phish.pickle",
             "scams": args.root / "scams/scams.pickle"}
    report = {}
    for group, path in files.items():
        print(f"Inspecting {group} node table...", flush=True)
        report[group] = inspect(path, None if args.full_node_table else args.nodes, args.max_mib * 1024**2)
        print(f"{group}: {report[group]['sampled_nodes']:,} nodes; complete={report[group]['node_table_complete']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for group, row in report.items():
        print(group, json.dumps({key: row[key] for key in ("sampled_nodes", "node_attribute_counts", "node_attribute_values", "missing_isp_in_sample", "stop_reason")}))
    print("Label semantics remain unverified. No graphs were loaded or labelled.")


if __name__ == "__main__":
    main()
