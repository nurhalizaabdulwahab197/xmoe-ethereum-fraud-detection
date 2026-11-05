"""Label-free structural counts on the supplied observed directed graph."""
import numpy as np

FEATURE_NAMES = ['distinct_in_counterparties', 'distinct_out_counterparties', 'reciprocal_counterparties']
CONTRACT = {
    'profile': 'common_structure_3',
    'feature_names': FEATURE_NAMES,
    'definitions': [
        'Number of distinct source neighbors u != v with observed pair u -> v.',
        'Number of distinct destination neighbors u != v with observed pair v -> u.',
        'Number of distinct neighbors u != v with both observed directions.'],
    'units': ['count', 'count', 'count'],
    'feature_graph_policy': 'Exclude self loops; collapse parallel directed pairs; preserve direction.',
    'message_passing_policy': 'Retain each source edge_index unchanged, including its self loops.',
    'fit_policy': 'Raw counts, no log1p; StandardScaler fit on labeled training nodes only per seed.',
    'scope': 'Common structural representation; not original BLTE23 reproduction. Source graph scopes and dataset-native annotations differ.',
}


def compute(edge_index, nodes):
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != np.int64:
        raise ValueError('Expected 2xE int64 edge index')
    if nodes < 1 or nodes * nodes >= np.iinfo(np.int64).max:
        raise ValueError('Unsupported graph size')
    if edges.size and (edges.min() < 0 or edges.max() >= nodes):
        raise ValueError('Endpoint outside node range')
    src, dst = edges
    keep = src != dst
    codes = np.unique(src[keep] * nodes + dst[keep])
    src, dst = codes // nodes, codes % nodes
    incoming = np.bincount(dst, minlength=nodes)
    outgoing = np.bincount(src, minlength=nodes)
    reciprocal_pair = np.isin(dst * nodes + src, codes, assume_unique=True, kind='sort')
    reciprocal = np.bincount(src[reciprocal_pair], minlength=nodes)
    features = np.stack([incoming, outgoing, reciprocal], axis=1).astype(np.int64)
    return features, {'directed_pairs_excluding_self_loops': len(codes),
        'directed_reciprocal_pairs': int(reciprocal_pair.sum()),
        'isolated_or_self_loop_only_nodes': int((features.sum(axis=1) == 0).sum())}
