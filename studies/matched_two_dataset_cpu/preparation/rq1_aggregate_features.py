"""Aggregate duplicate-invariant cohort features from complete RQ1 indexes.

Standard library only. No unpickling, amount conversion, split creation or training.
The graph ledger covers cohort-incident pairs; context-node features are not built.
"""
from __future__ import annotations
import argparse
import collections
import csv
import functools
import hashlib
import json
import math
import mmap
import pickletools
import sqlite3
import struct
import time
from pathlib import Path
import re

ADDRESS = re.compile(r'0x[0-9a-fA-F]{40}\Z')
MARK = object()
SCALARS = {'SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8', 'UNICODE', 'BININT', 'BININT1', 'BININT2', 'INT', 'LONG', 'LONG1', 'LONG4', 'BINFLOAT', 'FLOAT'}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


class Strings:
    def __init__(self, folder, expected_entries):
        self.files, self.maps = [], []
        for name in ('strings.idx', 'strings.dat', 'strings.aliases'):
            f = (folder / name).open('rb')
            self.files.append(f)
            self.maps.append(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if f.seek(0, 2) else b'')
        self.index, self.data, self.aliases = self.maps
        if len(self.index) != expected_entries * 16 or len(self.aliases) % 8:
            raise ValueError('String index is incomplete')

    @staticmethod
    def find(data, width, key):
        lo, hi = 0, len(data) // width
        while lo < hi:
            mid = (lo + hi) // 2
            if struct.unpack_from('<I', data, mid * width)[0] < key:
                lo = mid + 1
            else:
                hi = mid
        at = lo * width
        return at if at < len(data) and struct.unpack_from('<I', data, at)[0] == key else None

    @functools.lru_cache(maxsize=16384)
    def get(self, key):
        for _ in range(32):
            at = self.find(self.index, 16, key)
            if at is not None:
                _, size, offset = struct.unpack_from('<IIQ', self.index, at)
                if offset + size > len(self.data):
                    raise ValueError('String offset outside data')
                return self.data[offset:offset + size].decode('utf-8')
            at = self.find(self.aliases, 8, key)
            if at is None:
                raise ValueError(f'Absent or non-string memo {key}')
            key = struct.unpack_from('<II', self.aliases, at)[1]
        raise ValueError('String alias cycle/depth')

    def resolve(self, ref):
        if ref['kind'] == 'literal':
            return ref['value']
        if ref['kind'] == 'memo':
            return self.get(ref['index'])
        raise ValueError('Unknown reference')

    def close(self):
        self.get.cache_clear()
        for m in self.maps:
            if isinstance(m, mmap.mmap):
                m.close()
        for f in self.files:
            f.close()


class Slice:
    def __init__(self, f, start, end):
        self.f, self.end, self.stop = f, end, False
        f.seek(start)

    def tell(self):
        return self.f.tell() + int(self.stop)

    def read(self, n):
        if not 0 <= n <= 16 * 1024**2:
            raise ValueError('Oversized primitive')
        remaining = self.end - self.f.tell()
        if not remaining and n == 1 and not self.stop:
            self.stop = True
            return b'.'
        if n > remaining:
            raise ValueError('Truncated range')
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError('Source truncated')
        return data

    def readline(self):
        data = bytearray()
        for _ in range(1024**2):
            c = self.read(1)
            data.extend(c)
            if c == b'\n':
                return bytes(data)
        raise ValueError('Oversized line')


def event_time(key, attrs):
    value = attrs.get('timestamp', key)
    if type(value) not in (int, float) or (isinstance(value, float) and (not math.isfinite(value) or not value.is_integer())):
        return None, 'nonintegral_or_nonfinite_time'
    if not 1420070400 <= value <= 1793318400:
        return None, 'outside_declared_plausibility_window'
    if 'timestamp' in attrs and type(key) in (int, float) and key != value:
        return None, 'timestamp_key_mismatch'
    return int(value), None


class Frame:
    def __init__(self):
        self.tag, self.values, self.days = 'empty', {}, {}
        self.records, self.pairs, self.bad_time = 0, 0, 0


def decode_block(f, task, strings, emit_pair, stats, quarantine):
    """Decode each physical range once, retaining only per-pair day aggregates."""
    root = Frame()
    stack, local, next_memo = [], {}, task['memo_start']
    source = strings.resolve(task['source'])
    target = strings.resolve(task['target']) if task['kind'] == 'pair' else None
    reader = Slice(f, task['start'], task['end'])

    def assign(parent, key, value, position):
        if not isinstance(parent, Frame):
            raise ValueError('Invalid dictionary parent')
        if parent is root and task['kind'] == 'source':
            if not isinstance(key, str) or not isinstance(value, Frame) or value.tag not in {'empty', 'pair'}:
                raise ValueError('Invalid source pair shape')
            emit_pair(source, key, value)
            root.records += value.records
            root.pairs += 1
        elif isinstance(value, Frame):
            if value.tag not in {'empty', 'leaf'} or parent.tag not in {'empty', 'pair'}:
                raise ValueError('Unsupported edge shape')
            parent.tag = 'pair'
            parent.records += 1
            stats['decoded_edge_record_occurrences'] += 1
            attrs = value.values
            for name, raw in attrs.items():
                stats['attribute:' + name] += 1
                if name == 'amount':
                    stats['amount_type:' + type(raw).__name__] += 1
            stamp, reason = event_time(key, attrs)
            if reason:
                parent.bad_time += 1
                stats['time_error:' + reason] += 1
                quarantine({'reason': reason, 'pickle_assignment_offset': position,
                            'edge_key_raw': {'type': type(key).__name__, 'text': repr(key)},
                            'attributes_raw': {k: {'type': type(v).__name__, 'text': repr(v)} for k, v in attrs.items()}})
            else:
                day = stamp // 86400
                old = parent.days.get(day)
                parent.days[day] = (min(stamp, old[0]), max(stamp, old[1])) if old else (stamp, stamp)
        else:
            if parent.tag not in {'empty', 'leaf'} or not isinstance(key, str) or key in parent.values:
                raise ValueError('Nonflat or duplicated attributes')
            parent.tag = 'leaf'
            parent.values[key] = value

    stopped = False
    for op, arg, position in pickletools.genops(reader):
        name = op.name
        if name in {'PROTO', 'FRAME'}:
            continue
        if name == 'EMPTY_DICT':
            stack.append(root if not stack else Frame())
        elif name in SCALARS:
            stack.append(arg)
        elif name in {'NEWTRUE', 'NEWFALSE', 'NONE'}:
            stack.append({'NEWTRUE': True, 'NEWFALSE': False, 'NONE': None}[name])
        elif name == 'MARK':
            stack.append(MARK)
        elif name in {'MEMOIZE', 'BINPUT', 'LONG_BINPUT', 'PUT'}:
            idx = next_memo if name == 'MEMOIZE' else int(arg)
            value = stack[-1]
            if not isinstance(value, (str, Frame)):
                local[idx] = value
            next_memo = max(next_memo, idx + 1)
        elif name in {'BINGET', 'LONG_BINGET', 'GET'}:
            idx = int(arg)
            stack.append(local[idx] if idx in local else strings.get(idx))
        elif name == 'SETITEM':
            value, key = stack.pop(), stack.pop()
            assign(stack[-1], key, value, position)
        elif name == 'SETITEMS':
            at = max(i for i, x in enumerate(stack) if x is MARK)
            values = stack[at + 1:]
            if at < 1 or len(values) % 2:
                raise ValueError('Malformed dictionary batch')
            for i in range(0, len(values), 2):
                assign(stack[at - 1], values[i], values[i + 1], position)
            del stack[at:]
        elif name == 'STOP':
            if stack != [root] or root.records != task['edge_records']:
                raise ValueError(f'Range count mismatch: decoded={root.records}, indexed={task["edge_records"]}')
            if task['kind'] == 'source' and root.pairs != task['pair_blocks']:
                raise ValueError('Source pair-block count differs')
            if task['kind'] == 'pair':
                if target.lower() != task['address']:
                    raise ValueError('Incoming target reference differs')
                emit_pair(source, target, root)
            stopped = True
            break
        else:
            raise ValueError(f'Unsupported primitive opcode {name} at {position}')
    if not stopped:
        raise ValueError('Range did not close')


def initialize(db):
    db.executescript('''
    PRAGMA journal_mode=DELETE;
    PRAGMA synchronous=NORMAL;
    PRAGMA cache_size=-65536;
    CREATE TABLE context(graph TEXT, start INTEGER, ref TEXT, PRIMARY KEY(graph,start)) WITHOUT ROWID;
    CREATE TABLE tasks(graph TEXT, start INTEGER, payload TEXT, PRIMARY KEY(graph,start)) WITHOUT ROWID;
    CREATE TABLE edges(src TEXT, dst TEXT, source_mask INTEGER, PRIMARY KEY(src,dst)) WITHOUT ROWID;
    CREATE TABLE days(address TEXT, day INTEGER, first INTEGER, last INTEGER, PRIMARY KEY(address,day)) WITHOUT ROWID;
    CREATE TABLE observed(address TEXT PRIMARY KEY, valid_endpoint_records INTEGER, invalid_time_records INTEGER) WITHOUT ROWID;
    ''')


def prepare_tasks(db, index, graphs, pairs_only=False, task_limit=0):
    totals = collections.Counter()
    for report in graphs:
        graph = report['graph']
        folder = index / graph
        outgoing = {}
        with (folder / 'outgoing_sources.jsonl').open(encoding='utf-8-sig') as f:
            for line in f:
                row = json.loads(line)
                outgoing[row['start']] = row
                if not pairs_only:
                    db.execute('INSERT INTO tasks VALUES(?,?,?)', (graph, row['start'], json.dumps({**row, 'kind': 'source'})))
        print(f'{graph}: preparing context and range lookups...', flush=True)
        with (folder / 'source_context.jsonl').open(encoding='utf-8-sig') as f:
            for number, line in enumerate(f, 1):
                row = json.loads(line)
                db.execute('INSERT INTO context VALUES(?,?,?)', (graph, row['source_block_start'], json.dumps(row['source'])))
                if number % 100000 == 0:
                    print(f'{graph}: {number:,} context source mappings prepared', flush=True)
        with (folder / 'incoming_pairs.jsonl').open(encoding='utf-8-sig') as f:
            for number, line in enumerate(f, 1):
                if number % 100000 == 0:
                    db.commit()
                    print(f'{graph}: {number:,} incoming range entries inspected', flush=True)
                row = json.loads(line)
                owner = outgoing.get(row['source_block_start'])
                if owner and not pairs_only:
                    if not owner['start'] <= row['start'] < row['end'] <= owner['end']:
                        raise ValueError('Incoming pair range lies outside its selected source range')
                    totals['incoming_ranges_already_covered_by_outgoing'] += 1
                    continue
                source = db.execute('SELECT ref FROM context WHERE graph=? AND start=?', (graph, row['source_block_start'])).fetchone()
                if not source:
                    raise ValueError('Missing source-context mapping')
                row.update(kind='pair', source=json.loads(source[0]))
                db.execute('INSERT INTO tasks VALUES(?,?,?)', (graph, row['start'], json.dumps(row)))
                if pairs_only and task_limit:
                    count = db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]
                    if count >= task_limit:
                        db.commit()
                        totals['tasks'] = count
                        return dict(totals)
        db.commit()
    totals['tasks'] = db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]
    return dict(totals)


def pair_sink(db, cohort, mask, stats, quarantine):
    def emit(source, target, pair):
        stats['decoded_pair_blocks'] += 1
        if not pair.records:
            return
        if not ADDRESS.fullmatch(source) or not ADDRESS.fullmatch(target):
            stats['invalid_endpoint_record_occurrences'] += pair.records
            quarantine({'reason': 'non_address_endpoint', 'source_raw': source, 'target_raw': target, 'edge_record_occurrences': pair.records})
            return
        u, v = source.lower(), target.lower()
        incident = {u, v} & cohort.keys()
        if not incident:
            raise ValueError('Decoded range has no cohort endpoint')
        db.execute('INSERT INTO edges VALUES(?,?,?) ON CONFLICT(src,dst) DO UPDATE SET source_mask=source_mask|excluded.source_mask', (u, v, mask))
        for address in incident:
            db.execute('INSERT INTO observed VALUES(?,?,?) ON CONFLICT(address) DO UPDATE SET valid_endpoint_records=valid_endpoint_records+excluded.valid_endpoint_records, invalid_time_records=invalid_time_records+excluded.invalid_time_records', (address, pair.records, pair.bad_time))
            db.executemany('INSERT INTO days VALUES(?,?,?,?) ON CONFLICT(address,day) DO UPDATE SET first=MIN(first,excluded.first), last=MAX(last,excluded.last)',
                           ((address, day, first, last) for day, (first, last) in pair.days.items()))
    return emit


def feature_rows(db, cohort):
    db.execute('CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst,src)')
    db.commit()
    for address, label in sorted(cohort.items()):
        incoming = db.execute('SELECT COUNT(*) FROM edges WHERE dst=? AND src<>dst', (address,)).fetchone()[0]
        outgoing = db.execute('SELECT COUNT(*) FROM edges WHERE src=? AND src<>dst', (address,)).fetchone()[0]
        reciprocal = db.execute('SELECT COUNT(*) FROM edges a JOIN edges b ON a.dst=b.src AND a.src=b.dst WHERE a.src=? AND a.src<>a.dst', (address,)).fetchone()[0]
        days, first, last = db.execute('SELECT COUNT(*),MIN(first),MAX(last) FROM days WHERE address=?', (address,)).fetchone()
        observation = db.execute('SELECT valid_endpoint_records,invalid_time_records FROM observed WHERE address=?', (address,)).fetchone() or (0, 0)
        yield {'address': address, 'class_id': label['class_id'], 'class_name': label['class_name'],
               'distinct_in_counterparties': incoming, 'distinct_out_counterparties': outgoing,
               'reciprocal_counterparties': reciprocal, 'active_days': days,
               'activity_span_seconds': last - first if first is not None else 0,
               'first_event_seconds_candidate': first, 'last_event_seconds_candidate': last,
               'has_valid_endpoint_observations': bool(observation[0]), 'has_valid_temporal_observations': bool(days),
               'invalid_time_record_occurrences': observation[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--accounts', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preflight-tasks', type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() or args.preflight_tasks < 0:
        parser.error('Use a new output directory and a nonnegative preflight limit')
    summary = load_json(args.index / 'summary.json')
    if summary['status'] != 'complete_observation_ranges_indexed' or not all(g['complete'] for g in summary['graphs']):
        raise ValueError('A complete observation index is required')
    if digest(args.accounts).lower() != summary['input_sha256']['--accounts'].lower():
        raise ValueError('Account CSV differs from indexed cohort')
    with args.accounts.open(encoding='utf-8-sig', newline='') as f:
        cohort = {r['address']: r for r in csv.DictReader(f)}
    if len(cohort) != summary['cohort_accounts']:
        raise ValueError('Cohort count differs')
    args.output_dir.mkdir(parents=True)
    db = sqlite3.connect(args.output_dir / 'graph_ledger.sqlite')
    started, done, last_log = time.monotonic(), 0, 0
    stats = collections.Counter()
    quarantined_examples = 0
    graph_reports = []
    def quarantine(row):
        nonlocal quarantined_examples
        if quarantined_examples < 1000:
            quarantine_file.write(json.dumps({'graph': current_graph, **row}, allow_nan=False) + '\n')
            quarantined_examples += 1
    try:
        initialize(db)
        print('Preparing on-disk range tasks; preserving incoming/outgoing physical-range overlap rules...', flush=True)
        tasks = prepare_tasks(db, args.index, summary['graphs'], pairs_only=bool(args.preflight_tasks), task_limit=args.preflight_tasks)
        print(f'Range tasks prepared: {tasks["tasks"]:,}', flush=True)
        with (args.output_dir / 'quarantine_examples.jsonl').open('w', encoding='utf-8') as quarantine_file:
            for graph_report in summary['graphs']:
                current_graph = graph_report['graph']
                source = Path(graph_report['source_file'])
                stat = source.stat()
                if stat.st_size != graph_report['source_file_bytes'] or abs(stat.st_mtime * 1000 - graph_report['source_mtime_ms']) > 1:
                    raise ValueError('Raw source changed since indexing')
                strings = Strings(args.index / current_graph, graph_report['string_entries'])
                before = stats['decoded_edge_record_occurrences']
                try:
                    mask = {'normal': 1, 'phish': 2, 'scams': 4}.get(current_graph, 8)
                    emit = pair_sink(db, cohort, mask, stats, quarantine)
                    with source.open('rb') as f:
                        cursor = db.execute('SELECT payload FROM tasks WHERE graph=? ORDER BY start', (current_graph,))
                        while True:
                            batch = cursor.fetchmany(64)
                            if not batch:
                                break
                            for (payload,) in batch:
                                if args.preflight_tasks and done >= args.preflight_tasks:
                                    break
                                decode_block(f, json.loads(payload), strings, emit, stats, quarantine)
                                done += 1
                                if time.monotonic() - last_log > 10:
                                    print(f'{current_graph}: {done:,}/{tasks["tasks"]:,} ranges; {stats["decoded_edge_record_occurrences"]:,} observation records decoded', flush=True)
                                    last_log = time.monotonic()
                            db.commit()
                            if args.preflight_tasks and done >= args.preflight_tasks:
                                break
                finally:
                    strings.close()
                after = source.stat()
                if after.st_size != stat.st_size or after.st_mtime_ns != stat.st_mtime_ns:
                    raise ValueError('Raw source changed during aggregation')
                graph_reports.append({'graph': current_graph, 'decoded_edge_record_occurrences': stats['decoded_edge_record_occurrences'] - before})
                if args.preflight_tasks and done >= args.preflight_tasks:
                    break
        rows = list(feature_rows(db, cohort))
        features = ['distinct_in_counterparties', 'distinct_out_counterparties', 'reciprocal_counterparties', 'active_days', 'activity_span_seconds']
        with (args.output_dir / 'cohort_features.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = {'status': 'partial_preflight' if args.preflight_tasks else 'complete_cohort_feature_aggregation',
                  'cohort_accounts': len(rows), 'processed_range_tasks': done, 'task_preparation': tasks,
                  'graphs': graph_reports, 'statistics': dict(stats),
                  'unique_canonical_directed_pairs': db.execute('SELECT COUNT(*) FROM edges').fetchone()[0],
                  'accounts_with_valid_endpoint_observations': sum(r['has_valid_endpoint_observations'] for r in rows),
                  'accounts_with_valid_temporal_observations': sum(r['has_valid_temporal_observations'] for r in rows),
                  'accounts_with_invalid_time_records': sum(r['invalid_time_record_occurrences'] > 0 for r in rows),
                  'quarantine_example_limit': 1000, 'quarantine_examples_saved': quarantined_examples,
                  'temporal_units_verified': False, 'temporal_checks': 'integral, matching timestamp/key when both numeric, plausible Unix seconds in 2015-01-01..2026-10-30',
                  'amount_features_enabled': False, 'event_frequency_features_enabled': False,
                  'cross_archive_transaction_deduplication_claimed': False,
                  'features': features, 'predictive_feature_policy': 'five duplicate-invariant cohort features only; flags, labels, address, source mask and provenance excluded',
                  'observation_scope': 'all indexed available cohort-incident source observations, not complete blockchain history',
                  'context_node_features_built': False, 'full_gnn_training_graph_ready': False,
                  'splits_created': False, 'model_training_performed': False,
                  'elapsed_seconds': time.monotonic() - started,
                  'input_sha256': {'accounts': digest(args.accounts), 'index_summary': digest(args.index / 'summary.json')},
                  'code_sha256': digest(Path(__file__))}
        if not args.preflight_tasks and done != tasks['tasks']:
            raise ValueError('Not all prepared tasks were processed')
        (args.output_dir / 'aggregation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        db.rollback()
        (args.output_dir / 'ERROR.json').write_text(json.dumps({'complete': False, 'error': str(error), 'processed_tasks': done}, indent=2))
        raise
    finally:
        db.close()


if __name__ == '__main__':
    main()
