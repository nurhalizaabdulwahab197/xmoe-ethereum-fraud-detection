"""Rebuild and verify each frozen RQ1 seed from its exact raw artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from protocol import masks_for

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ROOT / 'provenance/rq1_expected_arrays.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    import io
    stream = io.BytesIO();np.save(stream, array, allow_pickle=False)
    return hashlib.sha256(stream.getbuffer()).hexdigest()


def inspect_graph(graph):
    expected = json.loads(EXPECTED.read_text())
    for name, digest in expected['graph_file_sha256'].items():
        path = Path(graph) / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError('Missing or changed original raw RQ1 artifact: ' + str(path))
    return expected


def load(graph, seed):
    graph = Path(graph);expected = inspect_graph(graph)
    raw = np.load(graph / 'features_raw.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(graph / 'labels.npy', allow_pickle=False)
    edges = np.load(graph / 'edge_index.npy', allow_pickle=False)
    masks = masks_for(labels, seed)
    scaler = StandardScaler().fit(np.asarray(raw[masks[0]], dtype=np.float64))
    x = scaler.transform(np.asarray(raw, dtype=np.float64)).astype(np.float32)
    arrays = {'features_normalized.npy': x,
              **{name+'_mask.npy': mask.astype(np.uint8) for name, mask in zip(('train','validation','test'), masks)}}
    for name, array in arrays.items():
        if array_sha(array) != expected['seed_inputs'][str(seed)]['array_sha256'][name]:
            raise ValueError('Rebuilt seed input differs from frozen study: ' + name)
    if not np.array_equal(np.load(graph/'candidate_labeled_mask.npy').astype(bool), labels >= 0):
        raise ValueError('Unlabeled context must never enter supervised splits.')
    metadata = {'dataset': 'rq1', 'seed': seed, 'input_sha256': expected['graph_file_sha256'],
                'seed_array_sha256': expected['seed_inputs'][str(seed)]['array_sha256'],
                'feature_names': expected['feature_names'], 'scaler_fit_rows': int(masks[0].sum()),
                'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                'raw_snapshot_features': True, 'test_labels_used_in_training': False}
    return (torch.from_numpy(x), torch.from_numpy(edges), torch.from_numpy(labels),
            *[torch.from_numpy(mask) for mask in masks]), metadata
