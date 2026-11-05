"""Build common, unscaled RQ1 graph arrays with real context-node features.

Standard library only. Reuses the frozen primitive decoder. Existing inputs are
read-only. Unknown labels stay -1; no split, fitted preprocessing or training.
"""
from __future__ import annotations
import argparse
from array import array
import collections
import csv
import json
import os
from pathlib import Path
import platform
import sqlite3
import struct
import sys
import time

import rq1_aggregate_features as aggregation

FEATURES = ['distinct_in_counterparties', 'distinct_out_counterparties',
            'reciprocal_counterparties', 'active_days', 'activity_span_seconds']


def write_npy(path, values, shape, descriptor):
    if len(values) != math_product(shape):
        raise ValueError('Array size/shape mismatch')
    header = repr({'descr': descriptor, 'fortran_order': False, 'shape': tuple(shape)})
    padding = (-(10 + len(header) + 1)) % 64
    data = (header + ' ' * padding + '\n').encode('ascii')
    if len(data) >= 65536:
        raise ValueError('Oversized NPY header')
    if sys.byteorder != 'little' and values.itemsize > 1:
        values.byteswap()
    with Path(path).open('xb') as f:
        f.write(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(data)) + data)
        values.tofile(f)


def math_product(shape):
    result = 1
    for n in shape:
        result *= n
    return result


def context_sink(db, cohort, stats, quarantine):
    def emit(source, target, pair):
        stats['decoded_pair_blocks'] += 1
        if not pair.records:
            return
        if not aggregation.ADDRESS.fullmatch(source) or not aggregation.ADDRESS.fullmatch(target):
            stats['invalid_endpoint_record_occurrences'] += pair.records
            quarantine({'reason': 'non_address_endpoint', 'source_raw': source,
                        'target_raw': target, 'edge_record_occurrences': pair.records})
            return
        endpoints = {source.lower(), target.lower()}
        if not endpoints & cohort.keys():
            raise ValueError('Observation is not incident to the frozen cohort')
        for address in endpoints - cohort.keys():
            db.executemany('''INSERT INTO context_days VALUES(?,?,?,?)
                ON CONFLICT(address,day) DO UPDATE SET
                first=MIN(first,excluded.first),last=MAX(last,excluded.last)''',
                ((address, day, first, last) for day, (first, last) in pair.days.items()))
    return emit


def prepare_structure(db, cohort):
    db.executescript('''
    CREATE TABLE cohort(address TEXT PRIMARY KEY,label INTEGER NOT NULL) WITHOUT ROWID;
    CREATE TABLE nodes(address TEXT PRIMARY KEY,node_id INTEGER UNIQUE NOT NULL,label INTEGER NOT NULL) WITHOUT ROWID;
    CREATE TABLE features(node_id INTEGER PRIMARY KEY,in_degree INTEGER NOT NULL DEFAULT 0,
        out_degree INTEGER NOT NULL DEFAULT 0,reciprocal INTEGER NOT NULL DEFAULT 0,
        active_days INTEGER NOT NULL DEFAULT 0,first INTEGER,last INTEGER);
    ''')
    db.executemany('INSERT INTO cohort VALUES(?,?)', ((a, int(row['class_id'])) for a, row in cohort.items()))
    print('Building deterministic node IDs from canonical edge endpoints and the frozen cohort...', flush=True)
    db.execute('''INSERT INTO nodes
        SELECT all_nodes.address,ROW_NUMBER() OVER(ORDER BY all_nodes.address)-1,COALESCE(c.label,-1)
        FROM (SELECT src AS address FROM raw.edges UNION SELECT dst AS address FROM raw.edges
              UNION SELECT address FROM cohort) AS all_nodes
        LEFT JOIN cohort c ON c.address=all_nodes.address''')
    db.execute('INSERT INTO features(node_id) SELECT node_id FROM nodes')
    db.commit()
    queries = [
        ('out_degree', 'SELECT src,COUNT(*) FROM raw.edges WHERE src<>dst GROUP BY src'),
        ('in_degree', 'SELECT dst,COUNT(*) FROM raw.edges WHERE src<>dst GROUP BY dst'),
        ('reciprocal', '''SELECT a.src,COUNT(*) FROM raw.edges a JOIN raw.edges b
                         ON a.src=b.dst AND a.dst=b.src WHERE a.src<>a.dst GROUP BY a.src''')]
    for column, query in queries:
        print('Computing structural feature: ' + column, flush=True)
        cursor = db.execute(query)
        while batch := cursor.fetchmany(5000):
            db.executemany(f'UPDATE features SET {column}=? WHERE node_id=(SELECT node_id FROM nodes WHERE address=?)',
                           ((count, address) for address, count in batch))
            db.commit()
    return db.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]


def collect_context_days(db, index, summary, cohort, expected_tasks, limit, out):
    stats = collections.Counter()
    done, examples, last_progress = 0, 0, time.monotonic()
    with (out / 'context_pass_quarantine_examples.jsonl').open('w', encoding='utf-8') as qf:
        for graph in summary['graphs']:
            name = graph['graph']
            def quarantine(row):
                nonlocal examples
                if examples < 1000:
                    qf.write(json.dumps({'graph': name, **row}, allow_nan=False) + '\n')
                    examples += 1
            source = Path(graph['source_file'])
            before = source.stat()
            if before.st_size != graph['source_file_bytes'] or abs(before.st_mtime * 1000 - graph['source_mtime_ms']) > 1:
                raise ValueError('Raw source changed since indexing')
            strings = aggregation.Strings(index / name, graph['string_entries'])
            try:
                emit = context_sink(db, cohort, stats, quarantine)
                with source.open('rb') as f:
                    cursor = db.execute('SELECT payload FROM raw.tasks WHERE graph=? ORDER BY start', (name,))
                    while batch := cursor.fetchmany(64):
                        for (payload,) in batch:
                            task = json.loads(payload)
                            if limit and task['kind'] != 'pair':
                                continue
                            if limit and done >= limit:
                                break
                            aggregation.decode_block(f, task, strings, emit, stats, quarantine)
                            done += 1
                            if time.monotonic() - last_progress >= 10:
                                print(f'{name}: {done:,}/{expected_tasks:,} ranges; {stats["decoded_edge_record_occurrences"]:,} observations decoded for context', flush=True)
                                last_progress = time.monotonic()
                        db.commit()
                        if limit and done >= limit:
                            break
            finally:
                strings.close()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Raw source changed during the context pass')
            if limit and done >= limit:
                break
    return done, dict(stats), examples


def finish_temporal_features(db):
    if db.execute('''SELECT EXISTS(SELECT 1 FROM context_days d
                     LEFT JOIN nodes n ON n.address=d.address WHERE n.address IS NULL)''').fetchone()[0]:
        raise ValueError('A temporal context address is absent from the structural graph')
    if db.execute('''SELECT EXISTS(SELECT 1 FROM context_days d JOIN cohort c ON d.address=c.address)''').fetchone()[0]:
        raise ValueError('Context-day aggregation overlapped copied cohort-day data')
    print('Combining original cohort days with newly aggregated context days...', flush=True)
    cursor = db.execute('''SELECT n.node_id,t.days,t.first,t.last FROM
        (SELECT address,COUNT(*) AS days,MIN(first) AS first,MAX(last) AS last FROM
            (SELECT address,day,first,last FROM raw.days UNION ALL
             SELECT address,day,first,last FROM context_days) GROUP BY address) t
        JOIN nodes n ON n.address=t.address''')
    while batch := cursor.fetchmany(5000):
        db.executemany('UPDATE features SET active_days=?,first=?,last=? WHERE node_id=?',
                       ((days, first, last, node_id) for node_id, days, first, last in batch))
        db.commit()


def verify_cohort_features(db, expected):
    checked = 0
    for address, original in expected.items():
        row = db.execute('''SELECT f.in_degree,f.out_degree,f.reciprocal,f.active_days,
            CASE WHEN f.first IS NULL THEN 0 ELSE f.last-f.first END,n.label
            FROM nodes n JOIN features f ON n.node_id=f.node_id WHERE n.address=?''', (address,)).fetchone()
        reference = tuple(int(original[key]) for key in FEATURES) + (int(original['class_id']),)
        if row != reference:
            raise ValueError(f'Existing cohort feature/label changed for {address}: {row} vs {reference}')
        checked += 1
    return checked


def export_arrays(db, out):
    count = db.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]
    features, labels, available, mask = array('q'), array('q'), array('B'), array('B')
    with (out / 'node_mapping.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'address', 'label', 'is_candidate_labeled'])
        cursor = db.execute('''SELECT n.node_id,n.address,n.label,f.in_degree,f.out_degree,
            f.reciprocal,f.active_days,f.first,f.last FROM nodes n JOIN features f
            ON n.node_id=f.node_id ORDER BY n.node_id''')
        for number, (node_id, address, label, incoming, outgoing, reciprocal, days, first, last) in enumerate(cursor):
            if node_id != number or not re_full_canonical(address):
                raise ValueError('Noncanonical address or nondeterministic node IDs')
            features.extend((incoming, outgoing, reciprocal, days, last-first if first is not None else 0))
            labels.append(label)
            available.append(first is not None)
            mask.append(label >= 0)
            writer.writerow((node_id, address, label, int(label >= 0)))
    if len(labels) != count:
        raise ValueError('Feature/node count mismatch')
    write_npy(out / 'features_raw.npy', features, (count, 5), '<i8')
    write_npy(out / 'labels.npy', labels, (count,), '<i8')
    write_npy(out / 'temporal_available.npy', available, (count,), '|u1')
    write_npy(out / 'candidate_labeled_mask.npy', mask, (count,), '|u1')
    del features, labels, available, mask
    edges = db.execute('SELECT COUNT(*) FROM raw.edges').fetchone()[0]
    source_ids, target_ids = array('q', [0]) * edges, array('q', [0]) * edges
    print(f'Exporting {edges:,} directed canonical pairs...', flush=True)
    cursor = db.execute('''SELECT s.node_id,t.node_id FROM raw.edges e
        JOIN nodes s ON s.address=e.src JOIN nodes t ON t.address=e.dst ORDER BY e.src,e.dst''')
    written = 0
    for i, (source, target) in enumerate(cursor):
        source_ids[i], target_ids[i] = source, target
        written += 1
    if written != edges:
        raise ValueError('An edge endpoint was lost during node mapping')
    source_ids.extend(target_ids)
    del target_ids
    write_npy(out / 'edge_index.npy', source_ids, (2, edges), '<i8')
    return count, edges


def re_full_canonical(address):
    return bool(aggregation.ADDRESS.fullmatch(address)) and address == address.lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--accounts', type=Path, required=True)
    parser.add_argument('--aggregation', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preflight-tasks', type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() or args.preflight_tasks < 0:
        parser.error('Use a new output directory and a nonnegative preflight limit')
    index_summary = aggregation.load_json(args.index / 'summary.json')
    old_report = aggregation.load_json(args.aggregation / 'aggregation_report.json')
    if index_summary['status'] != 'complete_observation_ranges_indexed' or old_report['status'] != 'complete_cohort_feature_aggregation':
        raise ValueError('Complete indexing and cohort aggregation are required')
    if aggregation.digest(args.index / 'summary.json') != old_report['input_sha256']['index_summary']:
        raise ValueError('Index differs from the cohort aggregation input')
    if aggregation.digest(args.accounts) != old_report['input_sha256']['accounts']:
        raise ValueError('Candidate account CSV changed')
    if aggregation.digest(Path(aggregation.__file__)) != old_report['code_sha256']:
        raise ValueError('Frozen aggregation decoder differs from the completed cohort run')
    with args.accounts.open(encoding='utf-8-sig', newline='') as f:
        cohort_rows = list(csv.DictReader(f))
    cohort = {r['address']: r for r in cohort_rows}
    with (args.aggregation / 'cohort_features.csv').open(encoding='utf-8-sig', newline='') as f:
        original_rows = list(csv.DictReader(f))
    original = {r['address']: r for r in original_rows}
    if len(cohort) != len(cohort_rows) or len(original) != len(original_rows) or set(original) != set(cohort) or len(cohort) != old_report['cohort_accounts']:
        raise ValueError('Cohort/feature identities differ or duplicate rows exist')
    for address, row in cohort.items():
        if original[address]['class_id'] != row['class_id'] or original[address]['class_name'] != row['class_name']:
            raise ValueError('Cohort labels differ')
    ledger = (args.aggregation / 'graph_ledger.sqlite').resolve()
    before_ledger = ledger.stat()
    args.output_dir.mkdir(parents=True)
    db = sqlite3.connect(args.output_dir / 'graph_inputs_ledger.sqlite', uri=True)
    started = time.monotonic()
    done = 0
    try:
        db.execute('ATTACH DATABASE ? AS raw', (ledger.as_uri() + '?mode=ro',))
        db.executescript('''PRAGMA cache_size=-65536; PRAGMA raw.cache_size=-65536;
            PRAGMA journal_mode=DELETE; PRAGMA synchronous=NORMAL;
            CREATE TABLE context_days(address TEXT,day INTEGER,first INTEGER,last INTEGER,
                PRIMARY KEY(address,day)) WITHOUT ROWID;''')
        expected_tasks = db.execute('SELECT COUNT(*) FROM raw.tasks').fetchone()[0]
        if expected_tasks != old_report['processed_range_tasks']:
            raise ValueError('Observation range task count changed')
        if db.execute('SELECT COUNT(*) FROM raw.edges').fetchone()[0] != old_report['unique_canonical_directed_pairs']:
            raise ValueError('Structural edge count changed')
        if not args.preflight_tasks:
            prepare_structure(db, cohort)
        print('Additional pass: aggregating temporal observations for context addresses...', flush=True)
        done, stats, quarantine_examples = collect_context_days(db, args.index, index_summary, cohort,
                                                               expected_tasks, args.preflight_tasks, args.output_dir)
        report = {'status': 'partial_context_preflight', 'processed_range_tasks': done,
                  'cohort_accounts': len(cohort), 'context_pass_statistics': stats,
                  'quarantine_examples_saved': quarantine_examples, 'splits_created': False,
                  'model_training_performed': False, 'elapsed_seconds': time.monotonic() - started}
        if not args.preflight_tasks:
            if done != expected_tasks or stats['decoded_edge_record_occurrences'] != old_report['statistics']['decoded_edge_record_occurrences']:
                raise ValueError('Context pass did not cover the same complete observation ranges')
            finish_temporal_features(db)
            checked = verify_cohort_features(db, original)
            count, edges = export_arrays(db, args.output_dir)
            label_counts = {str(label): n for label, n in db.execute('SELECT label,COUNT(*) FROM nodes GROUP BY label')}
            missing = db.execute('SELECT COUNT(*) FROM features WHERE first IS NULL').fetchone()[0]
            report.update(status='common_graph_inputs_prepared', total_nodes=count, directed_edges=edges,
                          labeled_nodes=sum(v for k, v in label_counts.items() if k != '-1'),
                          unlabeled_context_nodes=label_counts.get('-1', 0), label_counts=label_counts,
                          original_cohort_rows_verified_unchanged=checked, nodes_without_valid_temporal_observations=missing,
                          feature_names=FEATURES, feature_shape=[count, 5], edge_index_shape=[2, edges],
                          directed=True, self_loops='observed loops retained; excluded from counterparty counts',
                          feature_storage='raw int64; no scaling, log transform, imputation or train-fitted operation applied',
                          label_policy='Normal=0, Phishing=1, Scams=2, unlabeled context=-1; candidate mask is not a train/test split',
                          temporal_units_verified=False, amount_features_enabled=False,
                          observation_scope='cohort-incident graph only; context features describe this observed graph, not full blockchain history',
                          no_missing_context_features_fabricated=True,
                          artifact_sha256={p.name: aggregation.digest(p) for p in args.output_dir.glob('*.npy')})
        after_ledger = ledger.stat()
        if (before_ledger.st_size, before_ledger.st_mtime_ns) != (after_ledger.st_size, after_ledger.st_mtime_ns):
            raise ValueError('Original ledger changed during graph preparation')
        report['elapsed_seconds'] = time.monotonic() - started
        report['input_sha256'] = {'accounts': aggregation.digest(args.accounts),
                                 'index_summary': aggregation.digest(args.index / 'summary.json'),
                                 'cohort_features': aggregation.digest(args.aggregation / 'cohort_features.csv'),
                                 'aggregation_report': aggregation.digest(args.aggregation / 'aggregation_report.json'),
                                 'graph_ledger': aggregation.digest(ledger)}
        report['code_sha256'] = {'builder': aggregation.digest(Path(__file__)), 'decoder': aggregation.digest(Path(aggregation.__file__))}
        report['runtime'] = {'python': sys.version, 'platform': platform.platform(), 'sqlite': sqlite3.sqlite_version, 'logical_processors': os.cpu_count()}
        db.commit()
        (args.output_dir / 'graph_inputs_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        db.rollback()
        (args.output_dir / 'ERROR.json').write_text(json.dumps({'complete': False, 'error': str(error), 'processed_tasks': done}, indent=2))
        raise
    finally:
        db.close()


if __name__ == '__main__':
    main()
