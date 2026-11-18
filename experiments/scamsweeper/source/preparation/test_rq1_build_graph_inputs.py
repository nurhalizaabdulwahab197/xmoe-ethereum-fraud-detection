import json
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
import test_rq1_full_observation_index as fixture


class GraphInputTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.FullIndexTests()
        self.fixture.setUp()
        proc, self.root, self.addresses = self.fixture.run_fixture()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        proc = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('rq1_aggregate_features.py')),
                               '--index', str(self.root / 'index'), '--accounts', str(self.root / 'accounts.csv'),
                               '--output-dir', str(self.root / 'features')], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def tearDown(self):
        self.fixture.tearDown()

    def run_builder(self, output):
        return subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('rq1_build_graph_inputs.py')),
                               '--index', str(self.root / 'index'), '--accounts', str(self.root / 'accounts.csv'),
                               '--aggregation', str(self.root / 'features'), '--output-dir', str(self.root / output)],
                              capture_output=True, text=True)

    def test_real_context_features_unlabeled_mask_and_exact_numpy_shapes(self):
        proc = self.run_builder('graph')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        a, b, c, context, _ = self.addresses
        folder = self.root / 'graph'
        report = json.loads((folder / 'graph_inputs_report.json').read_text())
        self.assertEqual(report['status'], 'common_graph_inputs_prepared')
        self.assertEqual(report['total_nodes'], 4)
        self.assertEqual(report['directed_edges'], 3)
        self.assertEqual(report['original_cohort_rows_verified_unchanged'], 3)
        self.assertEqual(report['unlabeled_context_nodes'], 1)
        features = np.load(folder / 'features_raw.npy', mmap_mode='r')
        labels = np.load(folder / 'labels.npy')
        mask = np.load(folder / 'candidate_labeled_mask.npy')
        available = np.load(folder / 'temporal_available.npy')
        edges = np.load(folder / 'edge_index.npy')
        self.assertEqual(features.shape, (4, 5))
        self.assertEqual(features.dtype, np.dtype('<i8'))
        self.assertEqual(labels.tolist(), [0, 1, 2, -1])
        self.assertEqual(mask.tolist(), [1, 1, 1, 0])
        self.assertEqual(available.tolist(), [1, 1, 0, 1])
        self.assertEqual(features[3].tolist(), [1, 1, 0, 1, 100])
        self.assertEqual(features[2].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(edges.tolist(), [[0, 0, 3], [1, 3, 1]])
        self.assertFalse(report['splits_created'])
        self.assertFalse(report['model_training_performed'])

    def test_cohort_feature_change_blocks_final_artifact_report(self):
        original = self.root / 'features' / 'cohort_features.csv'
        # Modify a known field explicitly, independent of line-ending layout.
        import csv
        with original.open(newline='') as f:
            rows = list(csv.DictReader(f))
        rows[0]['distinct_out_counterparties'] = '99'
        with original.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        proc = self.run_builder('tampered')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('Existing cohort feature/label changed', proc.stderr)
        self.assertFalse((self.root / 'tampered' / 'graph_inputs_report.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
