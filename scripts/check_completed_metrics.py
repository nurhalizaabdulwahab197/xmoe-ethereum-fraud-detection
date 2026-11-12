"""Verify published aggregate metric hashes, identities and ten-seed summaries."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/metrics_completion'
REQUIRED = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1',
            'macro_roc_auc_ovr', 'weighted_roc_auc_ovr', 'balanced_accuracy', 'mcc') + tuple(
                c + '_' + m for c in ('normal', 'phishing', 'scamming')
                for m in ('precision', 'recall', 'f1', 'roc_auc_ovr', 'ap', 'support'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    manifest = json.loads((BASE / 'MANIFEST.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Published metric artifact changed: ' + name)
    total = 0
    for item in manifest['series']:
        rows = read_csv(BASE / item['id'] / 'per_seed.csv')
        summary = read_csv(BASE / item['id'] / 'summary.csv')
        if len(rows) != item['expected_rows']:
            raise ValueError('Wrong row count: ' + item['id'])
        groups = {}
        for row in rows:
            key = tuple(row[k] for k in item['group_by'])
            groups.setdefault(key, []).append(row)
            for metric in REQUIRED:
                value = float(row[metric])
                if not math.isfinite(value):
                    raise ValueError('Non-finite metric: ' + metric)
                if metric.endswith('_support'):
                    if value < 0 or value != int(value):
                        raise ValueError('Invalid class support')
                elif metric == 'mcc':
                    if not -1 <= value <= 1:
                        raise ValueError('Invalid MCC')
                elif not -1e-12 <= value <= 1 + 1e-12:
                    raise ValueError('Metric outside 0-1 range: ' + metric)
            if sum(float(row[c + '_support']) for c in ('normal', 'phishing', 'scamming')) != item['test_nodes']:
                raise ValueError('Test class support mismatch')
            fingerprint = row['prediction_sha256']
            if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
                raise ValueError('Invalid prediction fingerprint')
        if set(groups) != {tuple(v) for v in item['expected_groups']}:
            raise ValueError('Unexpected model/configuration/variant identity')
        for group in groups.values():
            if sorted(int(r['seed']) for r in group) != list(range(100, 110)):
                raise ValueError('Expected exactly seeds 100-109')
        identities = set()
        for record in summary:
            key = tuple(record[k] for k in item['group_by'])
            identity = key + (record['metric'],)
            if identity in identities:
                raise ValueError('Duplicate summary metric')
            identities.add(identity)
            values = [float(r[record['metric']]) for r in groups[key]]
            if int(record['n_seeds']) != 10:
                raise ValueError('Wrong summary seed count')
            for field, value in [('mean', statistics.mean(values)), ('sample_sd', statistics.stdev(values))]:
                if not math.isclose(float(record[field]), value, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Summary mismatch: ' + item['id'] + ' ' + record['metric'])
        expected = {key + (metric,) for key in groups for metric in REQUIRED}
        if not expected.issubset(identities):
            raise ValueError('Incomplete required metric summaries')
        total += len(rows)
    print(f'PASS: {len(manifest["series"])} series, {total} metric records; hashes, identities, required metrics and sample SD verified.')
    print('Series overlap; this is not a count of independent experiments. No training or model inference performed.')


if __name__ == '__main__':
    main()
