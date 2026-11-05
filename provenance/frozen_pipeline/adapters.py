"""Dataset I/O only. Both adapters feed the same original training engine."""
from pathlib import Path

import numpy as np
import torch

from utility import read, sha


def checked_entry(dataset, registry, seed):
    entry = registry[dataset]
    if 'prepared_common' in entry:
        from common_structure import CONTRACT
        folder = Path(entry['prepared_common'])
        path = folder / 'common_input_report.json'
        if sha(path) != entry['prepared_common_sha256']:
            raise ValueError('Prepared common input inventory differs')
        report = read(path)
        contract = folder.parent / 'feature_contract.json'
        if (sha(contract) != report['contract_sha256'] or read(contract) != CONTRACT or
                report['dataset'] != dataset or report['status'] != 'common_structure_inputs_prepared' or
                entry['input_dim'] != 3 or entry['profile'] != CONTRACT['profile'] or
                report['shared_config_sha256'] != sha(Path(__file__).with_name('common_config.json'))):
            raise ValueError('Common feature contract or training configuration differs')
        for name, expected in report['code_sha256'].items():
            if sha(Path(__file__).parent / name) != expected:
                raise ValueError('Common preparation implementation differs: ' + name)
        meta = report['seeds'][str(seed)]
        if sha(folder / f'seed{seed}' / 'preprocessing_and_split.json') != meta['manifest_sha256']:
            raise ValueError('Common seed manifest differs')
        for name, expected in report['graph_files'].items():
            if sha(folder / name) != expected:
                raise ValueError('Common graph artifact differs: ' + name)
        for name, expected in meta['array_sha256'].items():
            if sha(folder / f'seed{seed}' / name) != expected:
                raise ValueError('Common seed artifact differs: ' + name)
        return entry
    if dataset == 'blte':
        for key in ('csv', 'graph'):
            if sha(entry[key]) != entry[key + '_sha256']:
                raise ValueError('BLTE artifact hash mismatch: ' + key)
        return entry
    if sha(entry['study_lock']) != entry['study_lock_sha256']:
        raise ValueError('RQ1 audited study lock differs')
    lock = read(entry['study_lock'])
    if lock['status'] != 'matched_supplement_data_audited_protocol_locked' or seed not in lock['seeds']:
        raise ValueError('RQ1 seed or audited state invalid')
    if sha(Path(lock['inputs']) / 'matched_input_report.json') != lock['input_index_sha256']:
        raise ValueError('RQ1 prepared-input index changed')
    for name, expected in lock['graph_file_sha256'].items():
        if sha(Path(lock['graph']) / name) != expected:
            raise ValueError('RQ1 graph changed: ' + name)
    folder = Path(lock['inputs']) / f'seed{seed}'
    meta = lock['seed_inputs'][str(seed)]
    if sha(folder / 'preprocessing_and_split.json') != meta['manifest_sha256']:
        raise ValueError('RQ1 seed manifest differs')
    for name, expected in meta['array_sha256'].items():
        if sha(folder / name) != expected:
            raise ValueError('RQ1 prepared seed array differs: ' + name)
    return entry


def blte_graph(path):
    # Narrow allowlist for the already hash-verified released PyG Data artifact.
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
        return torch.load(path, map_location='cpu', weights_only=True)


def validate(values, dimension):
    x, edge, y, train, validation, test = values
    if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != dimension or not torch.isfinite(x).all():
        raise ValueError('Invalid dataset features')
    if edge.dtype != torch.long or edge.shape[0] != 2 or y.shape != (len(x),):
        raise ValueError('Invalid aligned graph or labels')
    if edge.min() < 0 or edge.max() >= len(x):
        raise ValueError('Invalid graph endpoint')
    total = train.long() + validation.long() + test.long()
    if (total > 1).any() or not torch.equal(total.bool(), y >= 0):
        raise ValueError('Supervised masks must be disjoint and exactly cover labeled nodes')
    if set(y[y >= 0].tolist()) != {0, 1, 2}:
        raise ValueError('Expected three encoded classes')


def load_dataset(dataset, registry, seed, ref):
    entry = checked_entry(dataset, registry, seed)
    if 'prepared_common' in entry:
        graph = Path(entry['prepared_common'])
        folder = graph / f'seed{seed}'
        report = read(graph / 'common_input_report.json')
        paths = [folder / 'features_normalized.npy', graph / 'edge_index.npy', graph / 'labels.npy',
                 folder / 'train_mask.npy', folder / 'validation_mask.npy', folder / 'test_mask.npy']
        values = [torch.from_numpy(np.load(p, allow_pickle=False)) for p in paths]
        names = report['feature_names']
        fingerprints = {'common_input_report': entry['prepared_common_sha256'],
                        'contract': report['contract_sha256'], **report['seeds'][str(seed)]['array_sha256']}
    elif dataset == 'blte':
        graph = blte_graph(entry['graph'])
        values = ref.clean_data(graph, Path(entry['csv']), seed)
        if not torch.equal(values[2], graph.y.cpu()):
            raise ValueError('Released BLTE graph and CSV label alignment differs')
        names = ref.pd.read_csv(entry['csv'], nrows=0).drop(columns=ref.NON_FEATURE_COLUMNS).columns.tolist()
        fingerprints = {key: entry[key + '_sha256'] for key in ('csv', 'graph')}
    else:
        lock = read(entry['study_lock'])
        folder, graph = Path(lock['inputs']) / f'seed{seed}', Path(lock['graph'])
        paths = [folder / 'features_normalized.npy', graph / 'edge_index.npy', graph / 'labels.npy',
                 folder / 'train_mask.npy', folder / 'validation_mask.npy', folder / 'test_mask.npy']
        values = []
        for i, path in enumerate(paths):
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            values.append(torch.from_numpy(array.astype(bool) if i >= 3 else array.copy()))
        names = read(folder / 'preprocessing_and_split.json')['feature_names']
        fingerprints = {'study_lock': entry['study_lock_sha256'], **lock['seed_inputs'][str(seed)]['array_sha256']}
    validate(values, entry['input_dim'])
    if len(names) != entry['input_dim']:
        raise ValueError('Feature names/dimension mismatch')
    meta = {'dataset': dataset, 'seed': seed, 'feature_names': names, 'input_dim': entry['input_dim'],
        'representation': entry.get('profile', 'dataset_native'),
        'nodes': len(values[0]), 'edges': values[1].shape[1], 'label_policy': entry['label_policy'],
        'partition_counts': {name: torch.bincount(values[2][mask], minlength=3).tolist()
             for name, mask in zip(('train','validation','test'), values[3:])},
        'input_fingerprints': fingerprints}
    return values, meta
