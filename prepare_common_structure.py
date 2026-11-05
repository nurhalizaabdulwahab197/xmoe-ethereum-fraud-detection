"""Prepare both datasets under one declared structural feature contract; no models run."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adapters import checked_entry, blte_graph
from common_structure import compute, CONTRACT, FEATURE_NAMES
from utility import read, sha

ROOT = Path(__file__).resolve().parent


def adapt(raw, labels, seed):
    if raw.shape != (len(labels), 3) or not np.isfinite(raw).all():
        raise ValueError('Expected aligned finite three-feature input')
    if not np.isin(labels, [-1, 0, 1, 2]).all():
        raise ValueError('Invalid source labels')
    ids = np.flatnonzero(labels >= 0)
    train, rest = train_test_split(ids, test_size=.30, random_state=seed, stratify=labels[ids])
    val, test = train_test_split(rest, test_size=.50, random_state=seed, stratify=labels[rest])
    scaler = StandardScaler().fit(np.asarray(raw[train], dtype=np.float64))
    x = scaler.transform(np.asarray(raw, dtype=np.float64)).astype(np.float32)
    masks = {}
    for name, indices in zip(('train', 'validation', 'test'), (train, val, test)):
        mask = np.zeros(len(labels), dtype=bool)
        mask[indices] = True
        masks[name] = mask
    if not np.array_equal(sum(m.astype(np.int8) for m in masks.values()), labels >= 0):
        raise ValueError('Invalid partition cover')
    return x, masks, scaler


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prepare(registry_path, output):
    if output.exists():
        raise ValueError('Preserve existing output; select a new folder')
    registry = read(registry_path)
    config = read(ROOT / 'common_config.json')
    # Validate source inventories before creating outputs.
    for dataset in ('blte', 'rq1'):
        for seed in config['seeds']:
            checked_entry(dataset, registry, seed)
    output.mkdir(parents=True)
    write(output / 'feature_contract.json', CONTRACT)
    prepared = {}
    for dataset in ('blte', 'rq1'):
        entry = registry[dataset]
        folder = output / dataset
        folder.mkdir()
        if dataset == 'blte':
            graph = blte_graph(entry['graph'])
            edges, labels = graph.edge_index.cpu().numpy(), graph.y.cpu().numpy()
            from shared_engine import reference
            ref = reference()
            # Verify graph-to-CSV label alignment before using graph node IDs.
            csv_labels = ref.pd.read_csv(entry['csv'])['attack_encoded'].to_numpy()
            if not np.array_equal(labels, csv_labels):
                raise ValueError('BLTE graph/CSV label alignment differs')
        else:
            lock = read(entry['study_lock'])
            source = Path(lock['graph'])
            edges = np.load(source / 'edge_index.npy', mmap_mode='r', allow_pickle=False)
            labels = np.load(source / 'labels.npy', mmap_mode='r', allow_pickle=False)
        raw, stats = compute(edges, len(labels))
        if dataset == 'rq1':
            old = np.load(source / 'features_raw.npy', mmap_mode='r', allow_pickle=False)
            if not np.array_equal(raw, old[:, :3]):
                raise ValueError('Independent structural computation differs from audited RQ1 features')
        for name, array in [('features_raw', raw), ('edge_index', edges), ('labels', labels)]:
            np.save(folder / (name + '.npy'), array, allow_pickle=False)
        report = {'status': 'common_structure_inputs_prepared', 'dataset': dataset,
            'profile': CONTRACT['profile'], 'feature_names': FEATURE_NAMES,
            'contract_sha256': sha(output / 'feature_contract.json'),
            'shared_config_sha256': sha(ROOT / 'common_config.json'),
            'source_registry_sha256': sha(registry_path), 'source_entry': entry,
            'code_sha256': {p: sha(ROOT / p) for p in ('common_structure.py', 'prepare_common_structure.py')},
            'nodes': len(labels), 'labeled_nodes': int((labels >= 0).sum()),
            'edges': edges.shape[1], 'feature_statistics': stats,
            'all_zero_labeled_rows': int(((raw.sum(axis=1) == 0) & (labels >= 0)).sum()),
            'graph_files': {p.name: sha(p) for p in folder.glob('*.npy')}, 'seeds': {},
            'training_performed': False, 'validation_evaluated': False, 'test_evaluated': False}
        for seed in config['seeds']:
            x, masks, scaler = adapt(raw, labels, seed)
            if dataset == 'blte':
                old_masks = [m.cpu().numpy() for m in ref.clean_data(graph, Path(entry['csv']), seed)[3:]]
            else:
                old_folder = Path(lock['inputs']) / f'seed{seed}'
                old_masks = [np.load(old_folder / (n + '_mask.npy'), allow_pickle=False).astype(bool)
                             for n in masks]
            if not all(np.array_equal(a, b) for a, b in zip(masks.values(), old_masks)):
                raise ValueError('Shared split differs from previously audited seed masks')
            target = folder / f'seed{seed}'
            target.mkdir()
            np.save(target / 'features_normalized.npy', x, allow_pickle=False)
            for name, mask in masks.items():
                np.save(target / (name + '_mask.npy'), mask, allow_pickle=False)
            details = {'seed': seed, 'feature_names': FEATURE_NAMES,
                'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'variance': scaler.var_.tolist(),
                'same_masks_as_audited_native_profile': True,
                'partition_counts': {n: {'total': int(m.sum()), 'classes': np.bincount(labels[m], minlength=3).tolist()}
                                     for n, m in masks.items()},
                'array_sha256': {p.name: sha(p) for p in target.glob('*.npy')}}
            write(target / 'preprocessing_and_split.json', details)
            report['seeds'][str(seed)] = {'manifest_sha256': sha(target / 'preprocessing_and_split.json'), **details}
            print(f'{dataset}: common features and training-only scaler prepared, seed {seed}', flush=True)
        write(folder / 'common_input_report.json', report)
        prepared[dataset] = {'prepared_common': str(folder.resolve()),
            'prepared_common_sha256': sha(folder / 'common_input_report.json'),
            'profile': CONTRACT['profile'], 'input_dim': 3, 'label_policy': entry['label_policy']}
    write(output / 'datasets.common.local.json', prepared)
    write(output / 'preparation_summary.json', {
        'status': 'both_common_structure_datasets_prepared', 'seeds': config['seeds'],
        'profile': CONTRACT['profile'], 'feature_names': FEATURE_NAMES,
        'penalty': config['training_options']['x_moe_normal_phishing_penalty'],
        'training_performed': False, 'validation_evaluated': False, 'test_evaluated': False,
        'limitations': ['BLTE23 replaced by three structural counts for this separate study.',
            'Source graph scopes and source annotations remain different.',
            'BLTE has many all-zero feature rows; no performance improvement is assumed.']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=ROOT / 'datasets.local.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.registry.resolve(), args.output_dir.resolve())
