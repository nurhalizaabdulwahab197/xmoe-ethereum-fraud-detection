import csv
import json
import pickle
import subprocess
import tempfile
import unittest
from pathlib import Path
from rq1_node_prefix_audit import inspect
from rq1_target_extract import StringIndex, sample_block


class FullIndexTests(unittest.TestCase):
    def run_fixture(self, tamper=False):
        root = Path(self.temp.name)
        a, b, c, context = ['0x' + v * 40 for v in 'abcd']
        upper_b = '0x' + 'B' * 40
        edges = {1500000000 + i: {'amount': i} for i in range(101)}
        obj = {'G': {'_node': {a: {}, b: {'isp': 0}, upper_b: {'isp': 1}, c: {}, context: {}},
                     '_adj': {a: {b: edges, context: {1500000300: {'amount': 0}}},
                              b: {}, c: {}, context: {upper_b: {1500000400: {'amount': 2}}}}, '_succ': {}}}
        source = root / 'source.pkl'
        source.write_bytes(pickle.dumps(obj, protocol=5))
        (root / 'bounds.json').write_text(json.dumps({'fixture': inspect(source, None, 1024**2)}))
        with (root / 'accounts.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['address', 'class_id', 'class_name', 'label_status'])
            for i, (addr, name) in enumerate(zip([a, b, c], ['normal', 'phish', 'scams'])):
                writer.writerow([addr, i if not tamper else 2, name, 'candidate_without_detected_conflict'])
        (root / 'labels.json').write_text(json.dumps({'accounts': [
            {'address': addr, 'source_declared_class': name, 'status': 'candidate_without_detected_conflict'}
            for addr, name in zip([a, b, c], ['normal', 'phish', 'scams'])]}))
        proc = subprocess.run(['node', str(Path(__file__).with_name('rq1_full_observation_index.mjs')),
                               '--accounts', str(root / 'accounts.csv'), '--labels', str(root / 'labels.json'),
                               '--boundaries', str(root / 'bounds.json'), '--expected-accounts', '3',
                               '--output-dir', str(root / 'index')], capture_output=True, text=True)
        return proc, root, [a, b, c, context, upper_b]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def test_all_records_indexed_without_pilot_caps_and_context_resolves(self):
        proc, root, (a, b, c, context, upper_b) = self.run_fixture()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        folder = root / 'index' / 'fixture'
        summary = json.loads((root / 'index' / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'complete_observation_ranges_indexed')
        self.assertFalse(summary['full_features_extracted'])
        report = summary['graphs'][0]
        self.assertEqual(report['source_edge_records_scanned'], 103)
        self.assertEqual(report['incoming_target_edge_record_occurrences'], 102)
        self.assertEqual(report['outgoing_target_source_blocks'], 3)
        pairs = [json.loads(line) for line in (folder / 'incoming_pairs.jsonl').read_text().splitlines()]
        sources = {r['source_block_start']: r['source'] for r in map(json.loads, (folder / 'source_context.jsonl').read_text().splitlines())}
        strings = StringIndex(folder)
        try:
            self.assertEqual({strings.resolve(sources[p['source_block_start']]) for p in pairs}, {a, context})
            with (root / 'source.pkl').open('rb') as f:
                pair = next(p for p in pairs if p['edge_records'] == 101)
                block = {**pair, 'source': sources[pair['source_block_start']], 'peer': pair['target'],
                         'target': pair['address'], 'direction': 'in'}
                _, count = sample_block(f, block, strings, limit=8)
                self.assertEqual(count, 101)
        finally:
            strings.close()
        coverage = {r['address']: r for r in json.loads((folder / 'coverage.json').read_text())}
        self.assertEqual(coverage[b]['incoming_edge_records'], 102)
        self.assertEqual(coverage[a]['outgoing_edge_records'], 102)
        self.assertEqual(coverage[c]['raw_source_entries'], 1)

    def test_cohort_label_tampering_is_rejected_before_scanning(self):
        proc, _, _ = self.run_fixture(tamper=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('Cohort/explicit-label mismatch', proc.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
