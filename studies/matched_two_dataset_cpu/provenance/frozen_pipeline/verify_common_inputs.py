"""Independently recompute prepared features and train-only scaling; no model inference."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from adapters import checked_entry, load_dataset
from common_structure import compute, FEATURE_NAMES
from prepare_common_structure import write
from utility import read
from run import configuration


def verify(registry_path, output):
    if output.exists():
        raise ValueError('Preserve existing audit output')
    config = configuration()
    registry = read(registry_path)
    runs = []
    for dataset in ('blte', 'rq1'):
        entry = checked_entry(dataset, registry, 100)
        root = Path(entry['prepared_common'])
        raw = np.load(root / 'features_raw.npy', allow_pickle=False)
        edge = np.load(root / 'edge_index.npy', allow_pickle=False)
        recomputed, _ = compute(edge, len(raw))
        if not np.array_equal(raw, recomputed):
            raise ValueError('Feature recomputation differs')
        del recomputed, edge
        for seed in config['seeds']:
            values, meta = load_dataset(dataset, registry, seed, None)
            x, _, _, train, _, _ = values
            train_ids = train.numpy()
            scaler = StandardScaler().fit(np.asarray(raw[train_ids], dtype=np.float64))
            expected = scaler.transform(np.asarray(raw, dtype=np.float64)).astype(np.float32)
            if not np.array_equal(x.numpy(), expected) or meta['feature_names'] != FEATURE_NAMES:
                raise ValueError('Train-only normalized features differ')
            runs.append({'dataset': dataset, 'seed': seed, 'shape': list(x.shape),
                'all_node_normalization_exact': True, 'partition_counts': meta['partition_counts']})
            del values, expected, x
        print(f'{dataset}: all ten seed arrays and scaler fits verified', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, {'status': 'both_common_structure_inputs_verified', 'seed_inputs_checked': len(runs),
        'feature_names': FEATURE_NAMES, 'datasets': runs, 'training_performed': False,
        'validation_evaluated': False, 'test_evaluated': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    verify(a.registry.resolve(), a.output.resolve())
