"""Checks for directed counting and training-only feature preparation."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from common_structure import compute, CONTRACT, FEATURE_NAMES
from prepare_common_structure import adapt, write, ROOT
from utility import sha


class CommonTests(unittest.TestCase):
    def test_direction_parallel_self_loops_and_isolates(self):
        edges = np.array([[0, 0, 1, 1, 2], [1, 1, 0, 2, 2]], dtype=np.int64)
        x, _ = compute(edges, 4)
        np.testing.assert_array_equal(x, [[1, 1, 1], [1, 2, 1], [1, 0, 0], [0, 0, 0]])

    def test_node_permutation_equivariance(self):
        edges = np.array([[0, 1, 1, 2], [1, 0, 2, 3]], dtype=np.int64)
        p = np.array([2, 0, 3, 1])
        a, _ = compute(edges, 4)
        b, _ = compute(p[edges], 4)
        np.testing.assert_array_equal(a, b[p])

    def test_holdout_and_context_cannot_fit_scaler(self):
        raw = np.arange(270, dtype=np.int64).reshape(90, 3)
        labels = np.array([0, 1, 2] * 25 + [-1] * 15)
        _, masks, scaler = adapt(raw, labels, 100)
        changed = raw.copy()
        changed[~masks['train']] += 1000000
        _, other_masks, other_scaler = adapt(changed, labels, 100)
        np.testing.assert_array_equal(scaler.mean_, other_scaler.mean_)
        np.testing.assert_array_equal(scaler.scale_, other_scaler.scale_)
        for n in masks:
            np.testing.assert_array_equal(masks[n], other_masks[n])
            self.assertFalse(masks[n][labels < 0].any())

    def test_graph_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            compute(np.array([[0], [5]], dtype=np.int64), 4)

    def fixture(self, root):
        folder = root / 'rq1'
        seed = folder / 'seed100'
        seed.mkdir(parents=True)
        write(root / 'feature_contract.json', CONTRACT)
        labels = np.array([0, 1, 2] * 20 + [-1] * 3, dtype=np.int64)
        edges = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
        raw, _ = compute(edges, len(labels))
        x, masks, _ = adapt(raw, labels, 100)
        for name, a in [('features_raw', raw), ('labels', labels), ('edge_index', edges)]:
            np.save(folder / (name + '.npy'), a)
        np.save(seed / 'features_normalized.npy', x)
        for n, m in masks.items():
            np.save(seed / (n + '_mask.npy'), m)
        meta = {'array_sha256': {p.name: sha(p) for p in seed.glob('*.npy')}}
        write(seed / 'preprocessing_and_split.json', meta)
        meta['manifest_sha256'] = sha(seed / 'preprocessing_and_split.json')
        write(folder / 'common_input_report.json', {
            'status': 'common_structure_inputs_prepared', 'dataset': 'rq1',
            'contract_sha256': sha(root / 'feature_contract.json'),
            'shared_config_sha256': sha(ROOT / 'common_config.json'),
            'code_sha256': {p: sha(ROOT / p) for p in ('common_structure.py', 'prepare_common_structure.py')},
            'feature_names': FEATURE_NAMES, 'seeds': {'100': meta},
            'graph_files': {p.name: sha(p) for p in folder.glob('*.npy')}})
        return {'rq1': {'prepared_common': str(folder),
            'prepared_common_sha256': sha(folder / 'common_input_report.json'),
            'input_dim': 3, 'profile': 'common_structure_3', 'label_policy': 'synthetic'}}

    def test_prepared_adapter_alignment_and_context_exclusion(self):
        from adapters import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            reg = self.fixture(Path(tmp))
            values, meta = load_dataset('rq1', reg, 100, None)
            self.assertEqual(meta['feature_names'], FEATURE_NAMES)
            self.assertEqual(meta['representation'], 'common_structure_3')
            self.assertEqual(tuple(values[0].shape), (63, 3))
            for mask in values[3:]:
                self.assertFalse(mask[values[2] < 0].any())

    def test_corrupt_prepared_array_rejected(self):
        from adapters import checked_entry
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = self.fixture(root)
            with (root / 'rq1' / 'seed100' / 'features_normalized.npy').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Common seed artifact differs'):
                checked_entry('rq1', reg, 100)


if __name__ == '__main__':
    unittest.main()
