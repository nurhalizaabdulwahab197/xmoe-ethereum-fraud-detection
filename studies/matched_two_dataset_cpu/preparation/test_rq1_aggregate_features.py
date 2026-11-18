import collections
import csv
import json
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path
import test_rq1_full_observation_index as fixture
from rq1_aggregate_features import Frame, event_time, feature_rows, initialize, pair_sink


class AggregationTests(unittest.TestCase):
    def test_complete_fixture_aggregation_without_double_visiting_ranges(self):
        helper = fixture.FullIndexTests()
        helper.setUp()
        try:
            proc, root, (a, b, c, context, upper_b) = helper.run_fixture()
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            run = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('rq1_aggregate_features.py')),
                                  '--index', str(root / 'index'), '--accounts', str(root / 'accounts.csv'),
                                  '--output-dir', str(root / 'features')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            report = json.loads((root / 'features' / 'aggregation_report.json').read_text())
            self.assertEqual(report['status'], 'complete_cohort_feature_aggregation')
            self.assertEqual(report['statistics']['decoded_edge_record_occurrences'], 103)
            self.assertEqual(report['unique_canonical_directed_pairs'], 3)
            self.assertEqual(report['task_preparation']['incoming_ranges_already_covered_by_outgoing'], 1)
            self.assertFalse(report['full_gnn_training_graph_ready'])
            with (root / 'features' / 'cohort_features.csv').open(newline='') as f:
                rows = {r['address']: r for r in csv.DictReader(f)}
            self.assertEqual(int(rows[a]['distinct_out_counterparties']), 2)
            self.assertEqual(int(rows[b]['distinct_in_counterparties']), 2)
            self.assertEqual(int(rows[b]['activity_span_seconds']), 400)
            self.assertEqual(rows[c]['has_valid_endpoint_observations'], 'False')
        finally:
            helper.tearDown()

    def test_duplicate_invariant_aggregation_direction_selfloops_and_invalid_times(self):
        db = sqlite3.connect(':memory:')
        initialize(db)
        a, b = '0x' + 'a' * 40, '0x' + 'b' * 40
        cohort = {a: {'class_id': '0', 'class_name': 'normal'}}
        stats, quarantine = collections.Counter(), []
        pair = Frame()
        pair.records = 10
        pair.days = {17361: (1500000000, 1500000100)}
        first = pair_sink(db, cohort, 1, stats, quarantine.append)
        second = pair_sink(db, cohort, 2, stats, quarantine.append)
        first(a, b, pair)
        second(a.upper().replace('0X', '0x'), b, pair)
        first(b, a, pair)
        first(a, a, pair)
        bad = Frame()
        bad.records, bad.bad_time = 1, 1
        first(a, b, bad)
        rows = list(feature_rows(db, cohort))
        self.assertEqual(rows[0]['distinct_in_counterparties'], 1)
        self.assertEqual(rows[0]['distinct_out_counterparties'], 1)
        self.assertEqual(rows[0]['reciprocal_counterparties'], 1)
        self.assertEqual(rows[0]['active_days'], 1)
        self.assertEqual(rows[0]['activity_span_seconds'], 100)
        self.assertEqual(rows[0]['invalid_time_record_occurrences'], 1)
        self.assertEqual(db.execute('SELECT source_mask FROM edges WHERE src=? AND dst=?', (a, b)).fetchone()[0], 3)
        first('GENESIS', a, pair)
        self.assertEqual(stats['invalid_endpoint_record_occurrences'], 10)
        db.close()

    def test_temporal_policy_does_not_round_or_accept_boolean(self):
        self.assertEqual(event_time(1500000000, {}), (1500000000, None))
        self.assertEqual(event_time(1500000000, {'timestamp': 1500000000.0}), (1500000000, None))
        self.assertIsNone(event_time(1500000000, {'timestamp': 1500000000.5})[0])
        self.assertEqual(event_time(1500000000, {'timestamp': 1500000001})[1], 'timestamp_key_mismatch')
        self.assertIsNone(event_time(True, {})[0])
        self.assertIsNone(event_time(2**120 + 1, {})[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
