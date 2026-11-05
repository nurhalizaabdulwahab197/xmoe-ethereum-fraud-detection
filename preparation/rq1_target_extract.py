"""Extract a bounded target pilot from committed static RQ1 dictionary indexes.

No pickle objects are deserialized. Unknown context accounts stay unlabeled.
Sampling is by serialized counterparty block and edge-key hash, not transaction ID.
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import heapq
import json
import mmap
import pickletools
import struct
from pathlib import Path

from rq1_pilot_extract import SCALARS, make_nodes, normalize_record, sha256


class StringIndex:
    def __init__(self, folder):
        self.files = []
        self.maps = []
        for name in ('strings.idx', 'strings.dat', 'strings.aliases'):
            f = (folder / name).open('rb')
            self.files.append(f)
            self.maps.append(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if f.seek(0, 2) else b'')
        self.index, self.data, self.aliases = self.maps
        if len(self.index) % 16 or len(self.aliases) % 8:
            raise ValueError('Truncated string index')

    @staticmethod
    def find(data, width, key):
        lo, hi = 0, len(data) // width
        while lo < hi:
            mid = (lo + hi) // 2
            found = struct.unpack_from('<I', data, mid * width)[0]
            if found < key:
                lo = mid + 1
            else:
                hi = mid
        return lo * width if lo < len(data) // width and struct.unpack_from('<I', data, lo * width)[0] == key else None

    @functools.lru_cache(maxsize=8192)
    def get(self, key):
        for _ in range(32):
            at = self.find(self.index, 16, key)
            if at is not None:
                _, size, offset = struct.unpack_from('<IIQ', self.index, at)
                if offset + size > len(self.data):
                    raise ValueError('String index offset outside data')
                return self.data[offset:offset + size].decode('utf-8', errors='strict')
            at = self.find(self.aliases, 8, key)
            if at is None:
                raise ValueError(f'Unresolved non-string or absent memo {key}')
            key = struct.unpack_from('<II', self.aliases, at)[1]
        raise ValueError('Cyclic or excessive string memo alias chain')

    def resolve(self, ref):
        if ref['kind'] == 'literal':
            return ref['value']
        if ref['kind'] == 'memo':
            return self.get(ref['index'])
        raise ValueError('Unknown endpoint reference type')

    def close(self):
        self.get.cache_clear()
        for m in self.maps:
            if isinstance(m, mmap.mmap):
                m.close()
        for f in self.files:
            f.close()


class SliceReader:
    """Expose one verified byte slice plus a synthetic STOP to pickletools."""
    def __init__(self, file, start, end):
        self.file, self.end, self.stop = file, end, False
        file.seek(start)

    def tell(self):
        return self.file.tell() + int(self.stop)

    def read(self, size):
        if not 0 <= size <= 16 * 1024**2:
            raise ValueError('Oversized primitive in selected block')
        remaining = self.end - self.file.tell()
        if remaining == 0 and size == 1 and not self.stop:
            self.stop = True
            return b'.'
        if size > remaining:
            raise ValueError('Selected block ends inside a primitive')
        result = self.file.read(size)
        if len(result) != size:
            raise ValueError('Source truncated')
        return result

    def readline(self):
        b = bytearray()
        for _ in range(1024**2):
            c = self.read(1)
            b.extend(c)
            if c == b'\n':
                return bytes(b)
        raise ValueError('Oversized line')


ROOT, MARK = object(), object()


def sample_block(file, block, strings, limit=8, seed=20261029):
    reader = SliceReader(file, block['start'], block['end'])
    source, target = strings.resolve(block['source']), strings.resolve(block['peer'])
    expected = block['target']
    if (source if block['direction'] == 'out' else target).lower() != expected:
        raise ValueError('Indexed target is not the promised incident endpoint')
    stack, local, next_memo, selected, seen = [], {}, block['memo_start'], [], 0

    def assign(parent, key, value, position):
        nonlocal seen
        if parent is ROOT:
            if not isinstance(value, dict) or any(isinstance(v, (dict, list, set)) for v in value.values()):
                raise ValueError('Non-flat edge attributes')
            if not all(isinstance(k, str) for k in value):
                raise ValueError('Non-string attribute name')
            record = {'source': source, 'target': target, 'key': key, 'attributes': value,
                      'pickle_assignment_offset': position}
            payload = f'{seed}|{block["start"]}|{type(key).__name__}|{key!r}'.encode()
            rank = int.from_bytes(hashlib.sha256(payload).digest(), 'big')
            item = (-rank, -seen, record)
            if len(selected) < limit:
                heapq.heappush(selected, item)
            elif item[:2] > selected[0][:2]:
                heapq.heapreplace(selected, item)
            seen += 1
        elif isinstance(parent, dict):
            if key in parent:
                raise ValueError('Repeated edge attribute key')
            parent[key] = value
        else:
            raise ValueError('Malformed dictionary assignment')

    for op, arg, pos in pickletools.genops(reader):
        name = op.name
        if name in {'PROTO', 'FRAME'}:
            continue
        if name == 'EMPTY_DICT':
            stack.append(ROOT if not stack else {})
        elif name in SCALARS:
            stack.append(arg)
        elif name in {'NEWTRUE', 'NEWFALSE', 'NONE'}:
            stack.append({'NEWTRUE': True, 'NEWFALSE': False, 'NONE': None}[name])
        elif name == 'MARK':
            stack.append(MARK)
        elif name in {'MEMOIZE', 'BINPUT', 'LONG_BINPUT', 'PUT'}:
            idx = next_memo if name == 'MEMOIZE' else int(arg)
            value = stack[-1]
            # Dictionary references are forbidden by the native schema pass;
            # strings are resolved from disk. Do not retain all edge dictionaries.
            if value is not ROOT and not isinstance(value, (dict, str)):
                local[idx] = value
            next_memo = max(next_memo, idx + 1)
        elif name in {'BINGET', 'LONG_BINGET', 'GET'}:
            idx = int(arg)
            stack.append(local[idx] if idx in local else strings.get(idx))
        elif name == 'SETITEM':
            v, k = stack.pop(), stack.pop()
            assign(stack[-1], k, v, pos)
        elif name == 'SETITEMS':
            marker = max(i for i, x in enumerate(stack) if x is MARK)
            parent, values = stack[marker - 1], stack[marker + 1:]
            if len(values) % 2:
                raise ValueError('Odd dictionary batch')
            for i in range(0, len(values), 2):
                assign(parent, values[i], values[i + 1], pos)
            del stack[marker:]
        elif name == 'STOP':
            if stack != [ROOT] or seen != block['edge_records']:
                raise ValueError(f'Pair-block count or stack mismatch: {seen} vs {block["edge_records"]}')
            break
        else:
            raise ValueError(f'Unsupported selected-block opcode {name} at {pos}')
    return [record for _, _, record in sorted(selected, key=lambda x: -x[1])], seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--records-per-block', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not 1 <= args.records_per_block <= 64:
        parser.error('Use a new output directory and 1..64 records per block')
    plan = json.loads(args.targets.read_text(encoding='utf-8'))
    labels = json.loads(args.labels.read_text(encoding='utf-8'))
    summary = json.loads((args.index / 'summary.json').read_text())
    if summary['target_plan_sha256'] != sha256(args.targets):
        raise ValueError('Target selection differs from index')
    all_rows, groups, block_logs = [], {}, []
    for report in summary['graphs']:
        if not report['complete'] or not report['strings_logged']:
            raise ValueError('Complete adjacency and string index required')
        group = report['graph']
        folder = args.index / group
        blocks = [json.loads(line) for line in (folder / 'selected_blocks.jsonl').read_text().splitlines()]
        unique = {}
        for block in blocks:
            unique.setdefault((block['start'], block['end']), []).append(block)
        count_before = len(all_rows)
        strings = StringIndex(folder)
        try:
            source_path = Path(report['source_file'])
            stat = source_path.stat()
            if stat.st_size != report['source_file_bytes'] or abs(stat.st_mtime * 1000 - report['source_mtime_ms']) > 1:
                raise ValueError('Source size or modification time differs from index')
            with source_path.open('rb') as f:
                for _, visits in sorted(unique.items()):
                    block = visits[0]
                    records, seen = sample_block(f, block, strings, args.records_per_block)
                    for record in records:
                        row = normalize_record(record, group, len(all_rows))
                        row['pair_block_start'] = block['start']
                        row['pilot_target_incidence'] = sorted({v['target'] for v in visits})
                        row['observation_scope'] = 'bounded_target_counterparty_sample'
                        all_rows.append(row)
                    block_logs.append({'graph': group, 'start': block['start'], 'end': block['end'],
                                       'edge_records_seen': seen, 'records_selected': len(records),
                                       'target_visits': [{'target': v['target'], 'direction': v['direction']} for v in visits]})
        finally:
            strings.close()
        groups[group] = {'source_edge_records_indexed': report['edges_seen'],
                         'source_blocks_indexed': report['committed_source_blocks'],
                         'node_minus_adjacency_source_count': report['node_minus_adjacency_source_count'],
                         'structural_audit_required': report['structural_audit_required'],
                         'selected_unique_pair_blocks': len(unique), 'selected_records': len(all_rows) - count_before,
                         'selected_blocks_sha256': sha256(folder / 'selected_blocks.jsonl')}
        print(f'{group}: {len(unique)} pair blocks, {len(all_rows) - count_before} pilot records', flush=True)
    nodes = make_nodes(all_rows, labels)
    for node in nodes:
        node['observation_scope'] = 'bounded_target_counterparty_sample'
    by_node = {node['address']: node for node in nodes}
    coverage = []
    for target in plan['targets']:
        address = target['address']
        incident = [row for row in all_rows if address in (row['source_address'], row['target_address'])]
        coverage.append({'address': address, 'declared_class': target['source_declared_class'],
                         'sample_records': len(incident),
                         'valid_sample_records': sum(row['valid_for_structural_temporal_pilot'] for row in incident),
                         'incoming_sample_records': sum(row['target_address'] == address for row in incident),
                         'outgoing_sample_records': sum(row['source_address'] == address for row in incident),
                         'observed_as_valid_node': address in by_node})
    report = {'status': 'target_centered_technical_pilot_only', 'graphs': groups,
              'target_accounts': len(coverage), 'targets_observed_valid': sum(r['observed_as_valid_node'] for r in coverage),
              'total_records': len(all_rows), 'canonical_nodes': len(nodes),
              'quarantined_records': sum(not r['valid_for_structural_temporal_pilot'] for r in all_rows),
              'candidate_labels_observed': dict(collections.Counter(n['source_declared_class'] for n in nodes if n['has_candidate_label'])),
              'unlabeled_context_nodes': sum(n['source_declared_class'] is None for n in nodes),
              'training_performed': False, 'full_history_features': False, 'monetary_features_enabled': False,
              'training_or_splits_created': False, 'unresolved_endpoint_memos': 0,
              'transaction_hash_available_for_deduplication': False,
              'cross_archive_transaction_deduplication_performed': False,
              'duplicate_visits_to_same_physical_pair_block_merged': True,
              'records_per_block_limit': args.records_per_block,
              'sampling': 'fixed 5 targets per class; bottom-k counterparty serialization offsets per target/direction/source graph; bottom-k edge-key hashes within each selected pair',
              'timestamp_units_verified': False, 'amount_units_verified': False,
              'input_sha256': {'target_plan': sha256(args.targets), 'labels': sha256(args.labels), 'index_summary': sha256(args.index / 'summary.json')},
              'code_sha256': {'extractor': sha256(Path(__file__)), 'normalization': sha256(Path(__file__).with_name('rq1_pilot_extract.py'))}}
    args.output_dir.mkdir(parents=True)
    for name, data in [('pilot_edges.jsonl', all_rows), ('pilot_nodes.jsonl', nodes),
                       ('quarantined_edges.jsonl', [r for r in all_rows if not r['valid_for_structural_temporal_pilot']]),
                       ('sampled_blocks.jsonl', block_logs)]:
        (args.output_dir / name).write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in data), encoding='utf-8')
    (args.output_dir / 'target_coverage.json').write_text(json.dumps(coverage, indent=2))
    (args.output_dir / 'pilot_report.json').write_text(json.dumps(report, indent=2))
    schema = {'predictive_features': list(nodes[0]['features']) if nodes else [],
              'excluded_from_features': ['address', 'isp', 'source_graph', 'source_declared_class', 'label_review_status',
                                         'pilot_target_incidence', 'target_selection_rank', 'attributes_raw'],
              'scope': 'sample-derived partial observations only; not full-history model features',
              'monetary_features_enabled': False, 'normal_class_assigned_to_unknown_nodes': False,
              'unique_blockchain_transaction_counts_claimed': False}
    (args.output_dir / 'feature_schema.json').write_text(json.dumps(schema, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
