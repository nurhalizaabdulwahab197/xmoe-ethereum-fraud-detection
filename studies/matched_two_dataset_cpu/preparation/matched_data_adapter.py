"""RQ1 input adapter for a BLTE-clean-matched supplementary study.

Five dataset-native raw features are an explicit input adaptation. This module
does not reconstruct 23 BLTE features, train models or score predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def adapt(raw, labels, candidates, seed):
    if raw.ndim != 2 or raw.shape[1] != 5 or labels.shape != (len(raw),) or candidates.shape != labels.shape:
        raise ValueError('Five-feature array and aligned label/candidate arrays required')
    if not np.isfinite(raw).all() or not np.isin(labels, [-1, 0, 1, 2]).all():
        raise ValueError('Invalid raw features or labels')
    if not np.isin(candidates, [0, 1]).all() or not np.array_equal(candidates.astype(bool), labels >= 0):
        raise ValueError('Candidate mask must identify exactly the available labeled accounts')
    ids = np.flatnonzero(candidates)
    # Same two sklearn calls and seed usage as BLTE clean_data; global IDs preserve graph alignment.
    train, remainder = train_test_split(ids, test_size=.30, random_state=seed, stratify=labels[ids])
    validation, test = train_test_split(remainder, test_size=.50, random_state=seed, stratify=labels[remainder])
    scaler = StandardScaler().fit(np.asarray(raw[train], dtype=np.float64))
    x = scaler.transform(np.asarray(raw, dtype=np.float64)).astype(np.float32)
    masks = {}
    for name, index in [('train', train), ('validation', validation), ('test', test)]:
        mask = np.zeros(len(labels), dtype=np.uint8)
        mask[index] = 1
        masks[name] = mask
    if not np.array_equal(sum(masks.values()), candidates):
        raise ValueError('Partition masks do not form a disjoint labeled cover')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite scaled values')
    return x, masks, scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seeds', default=','.join(map(str, range(100, 110))))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if len(seeds) != len(set(seeds)) or not seeds or any(s not in range(100, 110) for s in seeds):
        raise ValueError('Unique seeds in 100–109 required')
    report = json.loads((args.graph / 'graph_inputs_report.json').read_text(encoding='utf-8-sig'))
    if report['status'] != 'common_graph_inputs_prepared':
        raise ValueError('Completed graph required')
    inputs = {}
    for name in ('features_raw.npy', 'labels.npy', 'candidate_labeled_mask.npy', 'edge_index.npy'):
        inputs[name] = sha(args.graph / name)
        if inputs[name] != report['artifact_sha256'][name]:
            raise ValueError('Source graph artifact changed: ' + name)
    if args.check_only:
        print(json.dumps({'status': 'raw_RQ1_input_hashes_verified', 'features': report['feature_names'],
            'planned_split_and_initialization_seeds': seeds,
            'preprocessing': 'StandardScaler fit labeled training rows per seed; no log1p',
            'input_adaptation': 'Five RQ1-native features; not exact BLTE 23-feature reconstruction',
            'splits_created': False, 'training_performed': False, 'test_evaluated': False}, indent=2))
        return
    if args.output_dir is None or args.output_dir.exists():
        raise ValueError('Choose a new --output-dir; existing outputs are preserved')
    raw = np.load(args.graph / 'features_raw.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(args.graph / 'labels.npy', mmap_mode='r', allow_pickle=False)
    candidates = np.load(args.graph / 'candidate_labeled_mask.npy', mmap_mode='r', allow_pickle=False)
    args.output_dir.mkdir(parents=True)
    index = {'status': 'supplementary_matched_seed_inputs_prepared', 'seeds': seeds,
        'source_graph': str(args.graph.resolve()), 'source_input_sha256': inputs,
        'feature_names': report['feature_names'], 'input_dimension': 5,
        'input_adaptation': 'Dataset-native RQ1 features; no claim of BLTE feature semantic equivalence',
        'split_population': 'Labeled accounts only; global ascending node-ID order',
        'topology_policy': 'Original full graph retained without modification or copying',
        'preprocessing': 'Train-labeled-only StandardScaler; no log1p',
        'label_policy': 'Dataset-native source annotations retained; existing review exclusions unchanged',
        'script_sha256': sha(Path(__file__)), 'seed_artifacts': {},
        'training_performed': False, 'test_evaluated': False}
    for seed in seeds:
        x, masks, scaler = adapt(raw, labels, candidates, seed)
        folder = args.output_dir / f'seed{seed}'
        folder.mkdir()
        np.save(folder / 'features_normalized.npy', x, allow_pickle=False)
        for name, mask in masks.items():
            np.save(folder / (name + '_mask.npy'), mask, allow_pickle=False)
        counts = {name: {'total': int(mask.sum()), 'classes': np.bincount(labels[mask.astype(bool)], minlength=3).tolist()}
                  for name, mask in masks.items()}
        details = {'seed': seed, 'partition_counts': counts, 'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(), 'variance': scaler.var_.tolist(),
            'fit_rows': int(masks['train'].sum()), 'feature_names': report['feature_names'],
            'array_sha256': {p.name: sha(p) for p in folder.glob('*.npy')},
            'training_performed': False, 'test_evaluated': False}
        (folder / 'preprocessing_and_split.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
        index['seed_artifacts'][str(seed)] = {'partition_counts': counts,
            'manifest_sha256': sha(folder / 'preprocessing_and_split.json')}
        print(f'Seed {seed}: train={counts["train"]["total"]}, validation={counts["validation"]["total"]}, test={counts["test"]["total"]}', flush=True)
    (args.output_dir / 'matched_input_report.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': index['status'], 'seeds': seeds, 'training_performed': False, 'test_evaluated': False}, indent=2))


if __name__ == '__main__':
    main()
